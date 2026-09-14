import logging
import multiprocessing as mp
import os
import sys
import asyncio
import logging
import threading
import queue
import zmq
import gc
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
import harness_llm.common.numa_helpers as nh
from harness_llm.common.config_parser import HarnessCfg
from harness_llm.common.rpd_trace_utils import (
    rpd_trace_range,
    rpd_trace_range_non_timed,
)
from harness_llm.backends.common.constants import WarmUp
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

HARNESS_GC_LIMIT = int(os.getenv("HARNESS_GC_LIMIT", 0))
HARNESS_ZMQ_PERF_SAMPLE_COUNT = int(os.getenv("HARNESS_ZMQ_PERF_SAMPLE_COUNT", 3000))


class DistributedSyncOfflineBase:
    """ZMQ worker that owns an inference engine and answers prompt batches from
    the head SUT. All ZMQ plumbing, warmup, perf timing and the request loop
    live here; the engine-specific bits (env, engine construction, generation,
    shutdown) are provided by the vllm/sglang subclasses so a single transport
    serves both backends.
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
        """Set per-instance env (cache dirs etc). Runs in the parent before the
        worker process is spawned. HIP_VISIBLE_DEVICES is common to all
        engines."""
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join([str(d) for d in self.devices])

    def init_engine(self):
        """Build self.engine and self.sampling_params. Runs inside the worker
        process."""
        raise NotImplementedError

    def do_generate(self, prompt_token_ids):
        """Run generation for a list of prompt token-id lists and return a list
        of output token-id lists (one per prompt)."""
        raise NotImplementedError

    def shutdown_engine(self):
        raise NotImplementedError

    def apply_forwarded_sampling_params(self, forwarded_sampling_params):
        """Rebuild the worker's sampling params from the head's forwarded copy
        (carries runtime overrides applied on the head)."""
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
        self.identity = (
            f"{self.node_id}-gpu{'_'.join([str(d) for d in self.devices])}".encode()
        )
        self.receiver.setsockopt(zmq.IDENTITY, self.identity)
        self.receiver.setsockopt(zmq.LINGER, 0)
        self.address, self.port = self.headnode_address.split(":")
        self.receiver.connect(f"tcp://{self.address}:{self.port}")

        nh.set_affinity_by_device(self.devices[0])

        self.log(f"llm_config={self.llm_config}")
        # TODO handle stop_seq_id_config properly
        self.sampling_params_config.pop("stop_seq_ids_config", None)
        self.log(f"sampling_params={self.sampling_params_config}")

        self.init_engine()

        self.run_warmup()

        perf = self.time_perf()

        self.sender = self.context.socket(zmq.DEALER)
        self.sender.setsockopt(zmq.IDENTITY, self.identity)
        self.sender.connect(f"tcp://{self.address}:{int(self.port) + 1}")
        self.sender.setsockopt(zmq.LINGER, 0)
        self.sender.send_pyobj([self.identity, perf])

        ack = self.receiver.recv_pyobj()
        self.log(f"Got {ack=}")
        # The head forwards its effective sampling params in the ready ack, carrying any
        # runtime overrides applied on the head (notably the gpt-oss accuracy override that
        # forces max_tokens=32768). Rebuild the sampling params from the forwarded params so
        # the worker doesn't serve queries with the stale value it parsed from the YAML
        # (warmup / perf timing above intentionally used the local copy).
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

        self.run()

    def run_warmup(self):
        if self.warmup_enabled:
            warmup_sample = WarmUp.ENCODED_SAMPLES.get(self.benchmark, None)
            if warmup_sample is None:
                self.log(
                    f"No warmup sample for benchmark={self.benchmark}, skipping warmup"
                )
                return
            self.log("Started warmup")
            self.generate_dummy(num_samples=10)
            self.log("Warmup completed")

    def time_perf(self):
        total_time = 1
        if HARNESS_ZMQ_PERF_SAMPLE_COUNT > 0:
            if WarmUp.ENCODED_SAMPLES.get(self.benchmark, None) is None:
                self.log(
                    f"No perf sample for benchmark={self.benchmark}, "
                    "skipping perf run (even workload split)"
                )
                return total_time
            self.log("Starting perf run...")
            start_time = time.time()
            self.generate_dummy(num_samples=HARNESS_ZMQ_PERF_SAMPLE_COUNT)
            end_time = time.time()
            total_time = end_time - start_time
            self.log(f"Perf run completed in {total_time} seconds")
        return total_time

    def generate_dummy(self, num_samples=10):
        prompt_token_ids = [
            WarmUp.ENCODED_SAMPLES.get(self.benchmark, None)
        ] * num_samples
        self.do_generate(prompt_token_ids)

    @rpd_trace_range("SUT:Worker")
    def run(self):
        # The GC is going to be called after certain number of steps
        sample_count = 0
        is_gc_limit_specified = HARNESS_GC_LIMIT > 0
        if is_gc_limit_specified:
            gc.collect()
            gc.disable()
        self.log(f"Processing started...")
        while True:
            try:
                item = self.receiver.recv_pyobj()
                if item is None:
                    self.shutdown_engine()
                    self.error(f"recv got end signal...")
                    self.sender.send_pyobj(None)
                    break
                start, end, prompt_token_ids, stop_token_ids = item
                sample_count += len(prompt_token_ids)
                if is_gc_limit_specified and sample_count >= HARNESS_GC_LIMIT:
                    gc.collect()
                    sample_count = 0
                processed_output = self.do_generate(prompt_token_ids)
                self.sender.send_pyobj((start, end, processed_output))
            except Exception as e:
                self.error(f"{e=}")
                break
        self.close()

    def log(self, message):
        log.info(f"Server {self.identity} - {message}")

    def error(self, message):
        log.error(f"Server {self.identity} - {message}")

    def close(self):
        self.receiver.close()
        self.sender.close()
        self.context.term()


class DistributedSyncOffline(DistributedSyncOfflineBase):
    """vllm-backed ZMQ worker."""

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
        from vllm import LLM, SamplingParams
        import harness_llm.backends.vllm.vllm_utils as utils

        self.llm_config = utils.validate_and_correct(
            utils.populate_compile_config(self.llm_config)
        )
        self.sampling_params = SamplingParams(**self.sampling_params_config)
        self.engine = LLM(**self.llm_config)

    def do_generate(self, prompt_token_ids):
        from vllm.inputs import TokensInput

        pred_output_tokens = self.engine.generate(
            prompts=[
                TokensInput(type="token", prompt_token_ids=token_ids)
                for token_ids in prompt_token_ids
            ],
            sampling_params=self.sampling_params,
            use_tqdm=(not (os.getenv("HARNESS_DISABLE_VLLM_LOGS", "0") == "1")),
        )
        return [output.outputs[0].token_ids for output in pred_output_tokens]

    def shutdown_engine(self):
        del self.engine

    def apply_forwarded_sampling_params(self, forwarded_sampling_params):
        from vllm import SamplingParams

        self.sampling_params = SamplingParams(**forwarded_sampling_params)


class SGLangDistributedSyncOffline(DistributedSyncOfflineBase):
    """sglang-backed ZMQ worker."""

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

    def do_generate(self, prompt_token_ids):
        pred_output_tokens = self.engine.generate(
            input_ids=prompt_token_ids,
            sampling_params=self.sampling_params,
        )
        return [output["output_ids"] for output in pred_output_tokens]

    def shutdown_engine(self):
        self.engine.shutdown()

    def apply_forwarded_sampling_params(self, forwarded_sampling_params):
        from harness_llm.common.container_utils import remove_none_from_dict

        self.sampling_params = remove_none_from_dict(forwarded_sampling_params)


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
    assert conf.scenario.lower() == "offline"
    llm_config = conf["llm_config"]

    # The zmq transport drives either the vllm or the sglang engine. sglang's
    # data parallelism is intra-engine (dp-attention) so it is NOT counted in
    # the per-instance device size; a single engine instance spans tp*pp GPUs.
    use_sglang = is_sglang_llm_config(llm_config)
    if use_sglang:
        tp = llm_config.get("tp_size", 1)
        pp = llm_config.get("pp_size", 1)
        dp = 1
        worker_cls = SGLangDistributedSyncOffline
    else:
        assert conf.backend.lower() in ("vllm", "zmq")
        assert conf.engine_version.lower() == "sync"
        tp = llm_config["tensor_parallel_size"]
        pp = llm_config["pipeline_parallel_size"]
        dp = llm_config["data_parallel_size"]
        worker_cls = DistributedSyncOffline

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
        devices = visible_devices[engine_device_size * i: engine_device_size * (i + 1)]
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
