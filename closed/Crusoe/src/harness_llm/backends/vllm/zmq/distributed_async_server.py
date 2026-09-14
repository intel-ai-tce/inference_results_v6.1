import logging
import multiprocessing as mp
import os
import sys
import asyncio
import logging
import threading
import queue
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
import harness_llm.common.numa_helpers as nh
from harness_llm.common.config_parser import HarnessCfg
from harness_llm.common.rpd_trace_utils import (
    rpd_trace_range,
    rpd_trace_range_non_timed,
)
from harness_llm.backends.common.constants import HarnessStates, WarmUp
from harness_llm.backends.common.utils import (
    check_parallelism_configuration,
    get_visible_device_indices,
    is_sglang_llm_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


class DistributedAsyncServerBase:
    """ZMQ streaming worker for the server scenario. The ZMQ plumbing, async
    event loop, warmup, ready handshake and the send protocol (first token, then
    the remaining tokens, then a None sentinel — matching DistributedServerSUT's
    buffer/extend semantics) all live here. The engine-specific bits (env, engine
    construction, token streaming, shutdown) are provided by the vllm/sglang
    subclasses so a single transport serves both backends.
    """

    def __init__(
        self,
        node_id,
        headnode_address,
        devices,
        llm_config: dict,
        sampling_params: dict,
        benchmark: str,
        warmup_enabled: bool,
    ):
        self.node_id = node_id
        self.headnode_address = headnode_address
        self.devices = devices
        self.llm_config = llm_config
        self.sampling_params_config = sampling_params
        self.sampling_params = None
        self.engine = None
        self.benchmark = benchmark
        self.warmup_enabled = warmup_enabled

    # --- engine specific hooks -------------------------------------------------

    def set_engine_env(self):
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join([str(d) for d in self.devices])

    def init_engine(self):
        """Build self.engine and self.sampling_params (runs in the worker proc)."""
        raise NotImplementedError

    async def _stream(self, request_id, prompt_token_ids):
        """Async-generate for one prompt, yielding the *cumulative* output token
        ids at each streaming step."""
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator

    def shutdown_engine(self):
        raise NotImplementedError

    def apply_forwarded_sampling_params(self, forwarded_sampling_params):
        raise NotImplementedError

    # --- lifecycle -------------------------------------------------------------

    @rpd_trace_range_non_timed("SUT:Worker")
    def start(self):
        self.set_engine_env()
        self.process = mp.Process(target=self.launch)
        self.process.start()

    @rpd_trace_range_non_timed("SUT:Worker")
    def launch(self):
        self.context = zmq.Context()
        self.receiver = self.context.socket(zmq.DEALER)
        self.send_queue = queue.Queue()

        self.identity = (
            f"{self.node_id}-gpu{'_'.join([str(d) for d in self.devices])}".encode()
        )
        self.receiver.setsockopt(zmq.IDENTITY, self.identity)
        # Connect to router
        self.address, self.port = self.headnode_address.split(":")
        self.receiver.connect(f"tcp://{self.address}:{self.port}")

        nh.set_affinity_by_device(self.devices[0])

        self.log(f"llm_config={self.llm_config}")
        # TODO handle stop_seq_id_config properly
        self.sampling_params_config.pop("stop_seq_ids_config", None)
        self.log(f"sampling_params={self.sampling_params_config}")

        self.init_engine()

        async_event_loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(
            target=run_async_event_loop, args=([async_event_loop]), daemon=True
        )
        self.async_thread.start()

        future = asyncio.run_coroutine_threadsafe(self.run_warmup(), async_event_loop)
        # Blocking until we can start
        result = future.result()
        # This will register the server with the router
        self.sender_thread = threading.Thread(target=self.sender_loop, daemon=True)
        self.sender_thread.start()

        ack = self.receiver.recv_pyobj()
        self.log(f"Got {ack=}")
        # The head forwards its effective sampling params in the ready ack. These carry
        # any runtime overrides applied on the head (notably the gpt-oss accuracy override
        # that forces max_tokens=32768). The worker parsed its own copy from the YAML and
        # would otherwise serve queries with the stale value, so rebuild the sampling
        # params from the forwarded params here (warmup above used the local copy).
        if isinstance(ack, dict):
            ack_identity = ack.get("identity")
            forwarded_sampling_params = ack.get("sampling_params")
        else:
            ack_identity = ack
            forwarded_sampling_params = None
        assert (
            ack_identity == self.identity
        ), f"Expected ack {self.identity}, got {ack_identity}"

        if forwarded_sampling_params is not None:
            forwarded_sampling_params = dict(forwarded_sampling_params)
            forwarded_sampling_params.pop("stop_seq_ids_config", None)
            self.log(f"Applying forwarded sampling_params={forwarded_sampling_params}")
            self.apply_forwarded_sampling_params(forwarded_sampling_params)

        self.run(async_event_loop)

    async def run_warmup(self):
        if self.warmup_enabled:
            if WarmUp.ENCODED_SAMPLES.get(self.benchmark, None) is None:
                self.log(
                    f"No warmup sample for benchmark={self.benchmark}, skipping warmup"
                )
                return
            self.log("Started warmup")
            await self.warmup_generate()
            self.log("Warmup completed")

    async def warmup_generate(self):
        tasks = [self._warmup_generate(str(i)) for i in range(10)]
        await asyncio.gather(*tasks)

    async def _warmup_generate(self, request_id: str):
        prompt_token_ids = WarmUp.ENCODED_SAMPLES.get(self.benchmark, None)
        async for _ in self._stream(request_id, prompt_token_ids):
            pass

    def sender_loop(self):
        # ZMQ requires dedicated context per thread
        ctx = zmq.Context()
        sender = ctx.socket(zmq.DEALER)
        sender.setsockopt(zmq.IDENTITY, self.identity)
        sender.connect(f"tcp://{self.address}:{int(self.port) + 1}")
        sender.setsockopt(zmq.LINGER, 0)
        sender.send_pyobj(self.identity)

        self.log("Sender loop started...")
        while True:
            data = self.send_queue.get()
            sender.send_pyobj(data)
            if data is None:
                break
        self.log("Sender loop ended.")
        sender.close()
        ctx.term()

    @rpd_trace_range("SUT:Worker")
    def run(self, async_event_loop):
        self.log(f"Processing started...")
        while True:
            try:
                sample = self.receiver.recv_pyobj()
                if sample is None:
                    self.shutdown_engine()
                    self.error(f"recv got end signal...")
                    self.send_queue.put(None)
                    break
                asyncio.run_coroutine_threadsafe(
                    self.generate_v2(sample), async_event_loop
                )
            except Exception as e:
                self.error(f"{e=}")
                break
        self.close()

    async def generate_v2(self, samples):
        await asyncio.wait(
            [asyncio.create_task(self.generate(sample)) for sample in samples]
        )

    async def generate(self, sample):
        try:
            request_id = sample[0]
            prompt_token_ids = sample[1]
            output_token_ids = []
            first_token_id_count = 0
            async for cumulative_token_ids in self._stream(request_id, prompt_token_ids):
                output_token_ids = cumulative_token_ids
                if 0 == first_token_id_count and len(output_token_ids) > 0:
                    first_token_id_count = len(output_token_ids)
                    self.send_queue.put([request_id, output_token_ids])
            self.send_queue.put([request_id, output_token_ids[first_token_id_count:]])
            self.send_queue.put([request_id, None])
        except Exception as e:
            self.error(f"generate {e=}")

    def log(self, message):
        log.info(f"Server {self.identity} - {message}")

    def error(self, message):
        log.error(f"Server {self.identity} - {message}")

    def close(self):
        self.sender_thread.join()
        self.receiver.close()
        self.context.term()


class DistributedAsyncServer(DistributedAsyncServerBase):
    """vllm-backed ZMQ streaming worker."""

    def set_engine_env(self):
        super().set_engine_env()
        import harness_llm.backends.vllm.vllm_utils as utils

        os.environ["VLLM_CACHE_ROOT"] = utils.generate_vllm_cache_dir(
            str(self.devices[0])
        )
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = utils.generate_torch_inductor_cache_dir(
            str(self.devices[0])
        )

    def init_engine(self):
        from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs
        import harness_llm.backends.vllm.vllm_utils as utils

        self.llm_config = utils.validate_and_correct(
            utils.populate_compile_config(self.llm_config)
        )
        self.sampling_params = SamplingParams(**self.sampling_params_config)
        engine_args = AsyncEngineArgs(**self.llm_config)
        self.engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args, start_engine_loop=True
        )

    async def _stream(self, request_id, prompt_token_ids):
        results_generator = self.engine.generate(
            {"prompt_token_ids": prompt_token_ids}, self.sampling_params, request_id
        )
        async for request_output in results_generator:
            yield request_output.outputs[0].token_ids

    def shutdown_engine(self):
        del self.engine

    def apply_forwarded_sampling_params(self, forwarded_sampling_params):
        from vllm import SamplingParams

        self.sampling_params = SamplingParams(**forwarded_sampling_params)


class SGLangDistributedAsyncServer(DistributedAsyncServerBase):
    """sglang-backed ZMQ streaming worker."""

    def set_engine_env(self):
        super().set_engine_env()
        dev_tag = "_".join(str(d) for d in self.devices)
        for _env, _path in (
            ("AITER_JIT_DIR", f"/tmp/aiter_jit_gpu{dev_tag}"),
            ("TRITON_CACHE_DIR", f"/tmp/triton_gpu{dev_tag}"),
            ("TORCHINDUCTOR_CACHE_DIR", f"/tmp/inductor_gpu{dev_tag}"),
        ):
            os.makedirs(_path, exist_ok=True)
            os.environ[_env] = _path

    def init_engine(self):
        import harness_llm.backends.sglang.engine_factory as engine_factory
        from harness_llm.common.container_utils import remove_none_from_dict
        import harness_llm.common.logging as harness_logging

        self.engine = engine_factory.create_from(llm_config=self.llm_config)
        # sglang overwrites logging config during init; restore harness logging.
        harness_logging.set_level()
        self.sampling_params = remove_none_from_dict(self.sampling_params_config)

    async def _stream(self, request_id, prompt_token_ids):
        results_generator = await self.engine.async_generate(
            input_ids=prompt_token_ids,
            sampling_params=self.sampling_params,
            stream=True,
        )
        async for request_output in results_generator:
            yield request_output["output_ids"]

    def shutdown_engine(self):
        self.engine.shutdown()

    def apply_forwarded_sampling_params(self, forwarded_sampling_params):
        from harness_llm.common.container_utils import remove_none_from_dict

        self.sampling_params = remove_none_from_dict(forwarded_sampling_params)


def run_async_event_loop(async_event_loop):
    asyncio.set_event_loop(async_event_loop)
    async_event_loop.run_forever()


def set_mlperf_envs(env_config: dict):
    print(f"{env_config=}", flush=True)
    for env, val in env_config.items():
        if val is not None:
            os.environ[env] = str(val)
            log.info(f"Setting {env} to {val}")


def create_workers(conf):
    node_id = conf.get_with_default("node_id", None)
    headnode_address = conf.get_with_default("headnode_address", None)
    if node_id is None:
        print(f"Provide node_id=<unique_name> via command line")
        sys.exit(1)
    if headnode_address is None:
        print(f"Provide headnode_address=<ip:port> via command line")
        sys.exit(1)

    set_mlperf_envs(conf["env_config"])
    assert conf.scenario.lower() == "server"
    llm_config = conf["llm_config"]

    # The zmq transport drives either the vllm or the sglang engine. sglang's
    # data parallelism is intra-engine (dp-attention) so it is NOT counted in
    # the per-instance device size; a single engine instance spans tp*pp GPUs.
    use_sglang = is_sglang_llm_config(llm_config)
    if use_sglang:
        tp = llm_config.get("tp_size", 1)
        pp = llm_config.get("pp_size", 1)
        dp = 1
        worker_cls = SGLangDistributedAsyncServer
    else:
        assert conf.backend.lower() in ("vllm", "zmq")
        assert conf.engine_version.lower() == "async"
        tp = llm_config["tensor_parallel_size"]
        pp = llm_config["pipeline_parallel_size"]
        dp = llm_config["data_parallel_size"]
        worker_cls = DistributedAsyncServer

    conf["harness_config"]["tensor_parallelism"] = tp
    conf["harness_config"]["pipeline_parallelism"] = pp
    conf["harness_config"]["data_parallelism"] = dp

    sampling_params = conf["sampling_params"]
    harness_config = conf["harness_config"]

    dc = harness_config.get("device_count", 8)
    visible_devices = get_visible_device_indices(dc)
    engine_device_size = dp * tp * pp
    instance_count = dc // engine_device_size
    check_parallelism_configuration(instance_count, dp, tp, pp, dc)

    for i in range(0, instance_count):
        devices = visible_devices[engine_device_size * i : engine_device_size * (i + 1)]
        server = worker_cls(
            node_id,
            headnode_address,
            devices,
            llm_config,
            sampling_params,
            conf["benchmark_name"],
            harness_config["enable_warmup"],
        )
        server.start()


def run_from_cli() -> None:
    harnessCfg = HarnessCfg().create_from_cli()
    create_workers(harnessCfg)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    try:
        run_from_cli()
    except Exception as e:
        raise e
