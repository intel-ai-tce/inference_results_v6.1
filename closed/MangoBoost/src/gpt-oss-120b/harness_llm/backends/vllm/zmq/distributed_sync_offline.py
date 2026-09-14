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
from harness_llm.backends.common.utils import check_parallelism_configuration, get_visible_device_indices
import harness_llm.backends.vllm.vllm_utils as utils

from vllm import LLM, SamplingParams
from vllm.inputs import TokenInputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

HARNESS_GC_LIMIT = int(os.getenv("HARNESS_GC_LIMIT", 0))
HARNESS_ZMQ_PERF_SAMPLE_COUNT = int(os.getenv("HARNESS_ZMQ_PERF_SAMPLE_COUNT", 3000))


class DistributedSyncOffline:

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
        self.llm_config = utils.validate_and_correct(
            utils.populate_compile_config(llm_config)
        )
        self.sampling_params = sampling_params
        self.benchmark = benchmark
        self.warmup_enabled = warmup_enabled

    @rpd_trace_range_non_timed("SUT:Worker")
    def start(self):
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join([str(d) for d in self.devices])
        os.environ["VLLM_CACHE_ROOT"] = utils.generate_vllm_cache_dir(
            str(self.devices[0])
        )
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = utils.generate_torch_inductor_cache_dir(
            str(self.devices[0])
        )
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
        self.sampling_params.pop("stop_seq_ids_config", None)
        self.log(f"sampling_params={self.sampling_params}")

        self.sampling_params = SamplingParams(**self.sampling_params)
        self.engine = LLM(**self.llm_config)

        self.run_warmup()

        perf = self.time_perf()

        self.sender = self.context.socket(zmq.DEALER)
        self.sender.setsockopt(zmq.IDENTITY, self.identity)
        self.sender.connect(f"tcp://{self.address}:{int(self.port) + 1}")
        self.sender.setsockopt(zmq.LINGER, 0)
        self.sender.send_pyobj([self.identity, perf])

        ack = self.receiver.recv_pyobj()
        self.log(f"Got {ack=}")
        assert ack == self.identity, f"Expected ack {self.identity}, got {ack}"

        self.run()

    def run_warmup(self):
        if self.warmup_enabled:
            self.log("Started warmup")
            self.generate_dummy(num_samples=10)
            self.log("Warmup completed")

    def time_perf(self):
        total_time = 1
        if HARNESS_ZMQ_PERF_SAMPLE_COUNT > 0:
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
        self.generate(prompt_token_ids)

    def generate(self, prompt_token_ids):
        return self.engine.generate(
            prompts=[
                TokenInputs(type="token", prompt_token_ids=token_ids)
                for token_ids in prompt_token_ids
            ],
            sampling_params=self.sampling_params,
            use_tqdm=(not (os.getenv("HARNESS_DISABLE_VLLM_LOGS", "0") == "1")),
        )

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
                    del self.engine
                    self.error(f"recv got end signal...")
                    self.sender.send_pyobj(None)
                    break
                start, end, prompt_token_ids, stop_token_ids = item
                sample_count += len(prompt_token_ids)
                if is_gc_limit_specified and sample_count >= HARNESS_GC_LIMIT:
                    gc.collect()
                    sample_count = 0
                pred_output_tokens = self.generate(prompt_token_ids)
                processed_output = [
                    output.outputs[0].token_ids for output in pred_output_tokens
                ]
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
    assert conf.backend.lower() == "vllm"
    assert conf.engine_version.lower() == "sync"
    llm_config = conf["llm_config"]
    conf["harness_config"]["tensor_parallelism"] = llm_config["tensor_parallel_size"]
    conf["harness_config"]["pipeline_parallelism"] = llm_config[
        "pipeline_parallel_size"
    ]
    conf["harness_config"]["data_parallelism"] = llm_config["data_parallel_size"]
    sampling_params = conf["sampling_params"]
    harness_config = conf["harness_config"]

    tp = harness_config["tensor_parallelism"]
    pp = harness_config["pipeline_parallelism"]
    dp = harness_config["data_parallelism"]
    dc = harness_config.get("device_count", 8)
    visible_devices = get_visible_device_indices(dc)
    instance_count = dc // (dp * tp * pp)
    check_parallelism_configuration(instance_count, dp, tp, pp, dc)

    for i in range(0, instance_count):
        engine_device_size = dp * tp * pp
        devices = visible_devices[engine_device_size * i: engine_device_size * (i + 1)]
        server = DistributedSyncOffline(
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
