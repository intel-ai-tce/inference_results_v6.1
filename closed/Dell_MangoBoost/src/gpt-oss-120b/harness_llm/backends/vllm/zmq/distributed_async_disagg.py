"""
SUT-driven disaggregated prefill workers (server and offline scenarios).

Workers are scenario-agnostic engine runners. ALL routing lives in the SUT:
the SUT assigns each worker a globally-unique kv_rank at registration time,
pairs prefills with decodes itself, and bakes the prefill/decode KV markers
into each request_id via `format_request_id` before dispatching work. Workers
therefore have NO worker-to-worker wiring: they only talk to the SUT and run
the vLLM engine. The actual KV-cache transfer is performed by vLLM's
P2pNcclConnector, keyed off the addresses embedded in the request_id.

Port layout (host/base derived from `headnode_address`):
  base + 0                 : SUT ROUTER (SUT -> workers); worker `receiver` (DEALER) connects.
  base + 1                 : SUT ROUTER (workers -> SUT); worker `sender` (DEALER) connects.
  base + 2 + global_kv_rank: KV cache port for this worker (see disagg_utils.kv_port_for_rank).

Handshake (in launch(), before the engine is built):
  worker -> SUT (base+1): REGISTER
  SUT  -> worker (base):  [assigned_rank, N_p, N_d]
  (engine built here using the assigned kv_rank / kv_port)
  worker -> SUT (base+1): KVADDR
  SUT  -> worker (base):  identity (go-ahead ack, once all workers are ready)

Run phase:
  SUT -> worker (base): None (shutdown) or a work batch (list of items), each
    item = (formatted_request_id, original_id, prompt_token_ids, stop_ids).
  worker -> SUT (base+1):
    PrefillServer: one PREFILL_DONE(formatted_request_id) per sample.
    DecodeServer:  [original_id, tokens] chunks then [original_id, None] terminal.
"""

import logging
import multiprocessing as mp
import os
import sys
import asyncio
import threading
import queue
import gc
import zmq
from zmq.utils.monitor import recv_monitor_message

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
import harness_llm.common.numa_helpers as nh
from harness_llm.common.config_parser import HarnessCfg
from harness_llm.common.rpd_trace_utils import (
    rpd_trace_range,
    rpd_trace_range_non_timed,
)
from harness_llm.backends.common.utils import (
    check_parallelism_configuration,
    get_visible_device_indices,
)
import harness_llm.backends.vllm.vllm_utils as utils

from disagg_utils import (
    ROLE_PREFILL,
    ROLE_DECODE,
    detect_local_address,
    kv_port_for_rank,
    make_register_msg,
    make_kvaddr_msg,
    make_prefill_done_msg,
    setup_kv_transfer_config,
)

from omegaconf import OmegaConf

from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


# When > 0, the run loop disables automatic GC and forces a collection every
# HARNESS_GC_LIMIT processed samples. Mainly useful for the offline scenario's
# large in-flight counts; harmless (never triggers) when unset.
HARNESS_GC_LIMIT = int(os.getenv("HARNESS_GC_LIMIT", 0))


class BaseDisaggServer:
    """Shared, scenario-agnostic engine runner.

    Subclasses (`PrefillServer` / `DecodeServer`) only customize `kv_role`,
    sampling-param tweaks, and the per-sample `generate` behavior.
    """

    # Overridden by subclasses: "kv_producer" (prefill) / "kv_consumer" (decode).
    kv_role = None

    def __init__(
        self,
        node_id,
        headnode_address,
        devices,
        llm_config: dict,
        sampling_params: dict,
        benchmark: str,
        kv_connector: str,
        kv_buffer_size: float,
        role: str,
        stream_output: bool = True,
    ):
        self.node_id = node_id
        self.headnode_address = headnode_address
        self.devices = devices
        self.llm_config = utils.validate_and_correct(
            utils.populate_compile_config(llm_config)
        )
        self.sampling_params = sampling_params
        self.benchmark = benchmark
        self.kv_connector = kv_connector
        self.kv_buffer_size = kv_buffer_size
        self.role = role
        # When True, stream token chunks to the SUT (server scenario). Offline
        # use sets this False to emit a single full-output message instead.
        self.stream_output = stream_output

    @rpd_trace_range_non_timed("SUT:Worker")
    def start(self):
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join([str(d) for d in self.devices])
        os.environ["VLLM_CACHE_ROOT"] = utils.generate_vllm_cache_dir(
            str(self.devices[0])
        )
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = utils.generate_torch_inductor_cache_dir(
            str(self.devices[0])
        )
        # VLLM randomizes request IDs which are used to match requests from
        # prefill to decode instances, so we disable it.
        os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = str(1)
        self.process = mp.Process(target=self.launch)
        self.process.start()

    @rpd_trace_range_non_timed("SUT:Worker")
    def launch(self):
        self.context = zmq.Context()
        self.send_queue = queue.Queue()

        self.identity = f"{self.role}-{self.node_id}-gpu{'_'.join([str(d) for d in self.devices])}".encode()

        self.address, base = self.headnode_address.split(":")
        self.base_router_port = int(base)
        self.router_send_port = self.base_router_port + 1

        # Receiver: SUT -> worker (base + 0).
        self.receiver = self.context.socket(zmq.DEALER)
        self.receiver.setsockopt(zmq.IDENTITY, self.identity)

        # Block until the SUT's ROUTER has finished the ZMTP handshake with us.
        # Only after this is the SUT guaranteed to have our identity in its
        # routing table when it replies on `base` later, otherwise that reply
        # can be silently dropped and we deadlock on recv_pyobj() below.
        monitor = self.receiver.get_monitor_socket(zmq.EVENT_HANDSHAKE_SUCCEEDED)
        self.receiver.connect(f"tcp://{self.address}:{self.base_router_port}")
        while True:
            evt = recv_monitor_message(monitor)
            if evt["event"] == zmq.EVENT_HANDSHAKE_SUCCEEDED:
                break
        self.receiver.disable_monitor()
        monitor.close()

        nh.set_affinity_by_device(self.devices[0])

        # Address must come from the same resolver vLLM's connector uses, so the
        # KV markers the SUT embeds in request_ids match this worker's bind.
        local_address = detect_local_address()

        # Registration goes to base+1 (workers -> SUT). The sender_loop owns the
        # long-lived sender socket, but it does not exist yet, so use a
        # short-lived DEALER in this (main) thread for the handshake sends.
        # Default LINGER (block-until-delivered) is left in place so the final
        # KVADDR message is flushed when reg_sock is closed.
        reg_sock = self.context.socket(zmq.DEALER)
        reg_sock.setsockopt(zmq.IDENTITY, self.identity)
        reg_sock.connect(f"tcp://{self.address}:{self.router_send_port}")
        reg_sock.send_pyobj(
            make_register_msg(self.identity, self.role, local_address)
        )

        # SUT replies on `base` with [assigned_rank, N_p, N_d].
        reply = self.receiver.recv_pyobj()
        assigned_rank, n_p, n_d = reply
        self.kv_rank = assigned_rank
        self.total_prefills_count = n_p
        self.total_decodes_count = n_d
        self.own_kv_port = kv_port_for_rank(self.base_router_port, assigned_rank)
        self.kv_addr = f"{local_address}:{self.own_kv_port}"
        self.log(
            f"registered kv_rank={self.kv_rank} kv_addr={self.kv_addr} "
            f"(N_p={n_p}, N_d={n_d})"
        )

        self.log(f"llm_config={self.llm_config}")
        # TODO handle stop_seq_id_config properly
        self.sampling_params.pop("stop_seq_ids_config", None)
        self.configure_sampling_params()
        self.log(f"sampling_params={self.sampling_params}")
        self.sampling_params = SamplingParams(**self.sampling_params)

        # kv_rank / kv_port must be known before the engine is built.
        engine_args = AsyncEngineArgs(
            **self.llm_config,
            kv_transfer_config=setup_kv_transfer_config(
                kv_connector=self.kv_connector,
                kv_role=self.kv_role,
                kv_rank=self.kv_rank,
                kv_port=self.own_kv_port,
                kv_buffer_size=self.kv_buffer_size,
            ),
        )
        self.engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args,
            start_engine_loop=True,
        )

        # Report our KV listen address now that the engine is bound, then hand
        # off all subsequent worker -> SUT traffic to the sender_loop thread.
        reg_sock.send_pyobj(make_kvaddr_msg(self.identity, self.kv_addr))
        reg_sock.close()

        async_event_loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(
            target=run_async_event_loop, args=([async_event_loop]), daemon=True
        )
        self.async_thread.start()

        self.sender_thread = threading.Thread(target=self.sender_loop, daemon=True)
        self.sender_thread.start()

        # SUT echoes our identity once all workers are ready.
        self.log("Awaiting ack...")
        ack = self.receiver.recv_pyobj()
        self.log(f"Got {ack=}")
        if ack != self.identity:
            raise RuntimeError(f"Expected ack {self.identity}, got {ack}")

        self.run(async_event_loop)

    def configure_sampling_params(self):
        """Hook for role-specific sampling-param tweaks (applied to the dict
        before it is turned into a SamplingParams)."""
        pass

    def sender_loop(self):
        # ZMQ requires a dedicated context per thread.
        ctx = zmq.Context()
        sender = ctx.socket(zmq.DEALER)
        sender.setsockopt(zmq.IDENTITY, self.identity)
        sender.setsockopt(zmq.LINGER, 0)
        sender.connect(f"tcp://{self.address}:{self.router_send_port}")

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
        # Optionally take over GC: disable automatic collection and force one
        # every HARNESS_GC_LIMIT processed samples, so offline's large in-flight
        # counts don't thrash. No-op when HARNESS_GC_LIMIT is unset (server).
        sample_count = 0
        is_gc_limit_specified = HARNESS_GC_LIMIT > 0
        if is_gc_limit_specified:
            gc.collect()
            gc.disable()
        self.log("Processing started...")
        while True:
            try:
                batch = self.receiver.recv_pyobj()
                if batch is None:
                    del self.engine
                    self.error("recv got end signal...")
                    self.send_queue.put(None)
                    break
                sample_count += len(batch)
                if is_gc_limit_specified and sample_count >= HARNESS_GC_LIMIT:
                    gc.collect()
                    sample_count = 0
                asyncio.run_coroutine_threadsafe(
                    self.generate_batch(batch), async_event_loop
                )
            except Exception as e:
                self.error(f"{e=}")
                break
        self.close()

    async def generate_batch(self, batch):
        await asyncio.gather(*(self.generate(item) for item in batch))

    async def generate(self, item):
        raise NotImplementedError

    def log(self, message):
        log.info(f"Server {self.identity} - {message}")

    def error(self, message):
        log.error(f"Server {self.identity} - {message}")

    def debug(self, message):
        log.debug(f"Server {self.identity} - {message}")

    def close(self):
        self.sender_thread.join()
        self.receiver.close()
        self.context.term()


class PrefillServer(BaseDisaggServer):

    kv_role = "kv_producer"

    def configure_sampling_params(self):
        # Prefill only needs the KV produced by the first token; the generated
        # token itself is discarded.
        self.sampling_params["max_tokens"] = 1

    async def generate(self, item):
        try:
            self.debug("Calling generate...")
            formatted_request_id, original_id, prompt_token_ids, stop_ids = item
            results_generator = self.engine.generate(
                {"prompt_token_ids": prompt_token_ids},
                self.sampling_params,
                formatted_request_id,
            )
            # Break after the first output: the connector PUTs the KV cache.
            async for _ in results_generator:
                break
            self.send_queue.put(make_prefill_done_msg(formatted_request_id))
        except Exception as e:
            self.error(f"generate {e=}")


class DecodeServer(BaseDisaggServer):

    kv_role = "kv_consumer"

    async def generate(self, item):
        try:
            self.debug("Calling generate...")
            formatted_request_id, original_id, prompt_token_ids, stop_ids = item
            results_generator = self.engine.generate(
                {"prompt_token_ids": prompt_token_ids},
                self.sampling_params,
                formatted_request_id,
            )
            output_token_ids = []
            first_token_id_count = 0
            async for request_output in results_generator:
                output_token_ids = request_output.outputs[0].token_ids
                if self.stream_output and first_token_id_count == 0:
                    first_token_id_count = len(output_token_ids)
                    self.send_queue.put([original_id, output_token_ids])
            if self.stream_output:
                self.send_queue.put(
                    [original_id, output_token_ids[first_token_id_count:]]
                )
            else:
                self.send_queue.put([original_id, output_token_ids])
            self.send_queue.put([original_id, None])
        except Exception as e:
            self.error(f"generate {e=}")


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
        print("Provide node_id=<unique_name> via command line")
        sys.exit(1)
    if headnode_address is None:
        print("Provide headnode_address=<ip:port> via command line")
        sys.exit(1)

    set_mlperf_envs(conf["env_config"])
    assert conf.scenario.lower() in ("server", "offline")
    assert conf.backend.lower() == "vllm"
    assert conf.engine_version.lower() == "async"
    llm_config = conf["llm_config"]
    conf["harness_config"]["tensor_parallelism"] = llm_config["tensor_parallel_size"]
    conf["harness_config"]["pipeline_parallelism"] = llm_config[
        "pipeline_parallel_size"
    ]
    conf["harness_config"]["data_parallelism"] = llm_config["data_parallel_size"]
    sampling_params = conf["sampling_params"]
    harness_config = conf["harness_config"]

    # Server streams token chunks to the SUT; offline emits a single full
    # output per sample. Decode honors this; prefill ignores it.
    stream_output = conf.scenario.lower() == "server"

    # Per-role engine config: start from the shared base llm_config and overlay
    # any role-specific overrides supplied via vllm_engine_config_{prefill,decode}.
    prefill_overrides = conf.get_with_default("vllm_engine_config_prefill", None)
    decode_overrides = conf.get_with_default("vllm_engine_config_decode", None)
    prefill_llm_config = (
        OmegaConf.merge(llm_config, prefill_overrides)
        if prefill_overrides is not None
        else llm_config
    )
    decode_llm_config = (
        OmegaConf.merge(llm_config, decode_overrides)
        if decode_overrides is not None
        else llm_config
    )

    # Per-role KV buffer sizing: harness_config.kv_buffer_size_{prefill,decode}
    # take precedence over the shared harness_config.kv_buffer_size default.
    default_kv_buffer_size = harness_config.get("kv_buffer_size", None)
    prefill_kv_buffer_size = harness_config.get(
        "kv_buffer_size_prefill", default_kv_buffer_size
    )
    decode_kv_buffer_size = harness_config.get(
        "kv_buffer_size_decode", default_kv_buffer_size
    )

    tp = harness_config["tensor_parallelism"]
    pp = harness_config["pipeline_parallelism"]
    dp = harness_config["data_parallelism"]
    dc = harness_config.get("device_count", 8)
    visible_devices = get_visible_device_indices(dc)
    prefills_count = harness_config.get("prefills_count", None)
    decodes_count = harness_config.get("decodes_count", None)
    total_prefills_count = harness_config.get("total_prefills_count", prefills_count)
    total_decodes_count = harness_config.get("total_decodes_count", decodes_count)

    if prefills_count < 1 and decodes_count < 1:
        raise Exception(
            f"Need at least 1 local prefill or 1 local decode instance: "
            f"got {prefills_count=}, {decodes_count=}"
        )
    if total_prefills_count < 1 or total_decodes_count < 1:
        raise Exception(
            f"Need at least 1 global prefill and 1 global decode instance: "
            f"got {total_prefills_count=}, {total_decodes_count=}"
        )
    if prefills_count > total_prefills_count:
        raise Exception(
            f"local prefills_count={prefills_count} exceeds "
            f"total_prefills_count={total_prefills_count}"
        )
    if decodes_count > total_decodes_count:
        raise Exception(
            f"local decodes_count={decodes_count} exceeds "
            f"total_decodes_count={total_decodes_count}"
        )

    instance_count = prefills_count + decodes_count
    engine_device_size = dp * tp * pp
    expected_dc = instance_count * engine_device_size
    if expected_dc != dc:
        raise Exception(
            f"device_count mismatch: (prefills_count + decodes_count) * dp*tp*pp "
            f"= {instance_count} * {engine_device_size} = {expected_dc}, "
            f"but device_count={dc}"
        )
    check_parallelism_configuration(instance_count, dp, tp, pp, dc)

    def _spawn(server_cls, role, devices, role_llm_config, role_kv_buffer_size):
        # Ranks are SUT-assigned at registration time; workers do not compute
        # or pass any rank here.
        server = server_cls(
            node_id,
            headnode_address,
            devices,
            role_llm_config,
            sampling_params,
            conf["benchmark_name"],
            harness_config["kv_connector"],
            role_kv_buffer_size,
            role,
            stream_output,
        )
        server.start()

    device_cursor = 0

    for _ in range(prefills_count):
        devices = visible_devices[device_cursor : device_cursor + engine_device_size]
        _spawn(
            PrefillServer,
            ROLE_PREFILL,
            devices,
            prefill_llm_config,
            prefill_kv_buffer_size,
        )
        device_cursor += engine_device_size

    for _ in range(decodes_count):
        devices = visible_devices[device_cursor : device_cursor + engine_device_size]
        _spawn(
            DecodeServer,
            ROLE_DECODE,
            devices,
            decode_llm_config,
            decode_kv_buffer_size,
        )
        device_cursor += engine_device_size


def run_from_cli() -> None:
    harnessCfg = HarnessCfg().create_from_cli()
    create_workers(harnessCfg)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    try:
        run_from_cli()
    except Exception as e:
        raise e
