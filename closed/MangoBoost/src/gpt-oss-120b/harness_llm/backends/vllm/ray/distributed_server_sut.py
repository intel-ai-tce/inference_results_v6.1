from ray.util.queue import Queue
from ray.util.placement_group import placement_group
import os
import gc
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray

from harness_llm.backends.vllm.ray.distributed_sync_server import DistributedSyncServer
from harness_llm.loadgen.sut import SUT, SUTConfig
import harness_llm.common.logging as logging
from harness_llm.backends.common.debug import DebugToolkit
from harness_llm.backends.common.server_utils import DeviceSelector, ServerInfo
from harness_llm.backends.common.utils import create_response_and_send_complete, create_response_and_send_first_token
import multiprocessing as mp

import harness_llm.backends.common.constants as constants
import time
from ray.runtime_env import RuntimeEnv
import threading
from typing import Dict

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


class DistributedServerSUT(SUT):

    def __init__(self, config: dict, engine):
        super().__init__(
            SUTConfig(
                model = config["llm_config"]["model"],
                dataset_path = config["harness_config"]["dataset_path"],
                total_sample_count = config["harness_config"]["total_sample_count"],
                model_max_length = None,
                debug=False
            )
        )

        self.env_config = {k: str(v) for k, v in config.env_config.items()}
        self.tp = config["harness_config"]["tensor_parallelism"]
        self.pp = config["harness_config"]["pipeline_parallelism"]
        self.dp = config["harness_config"]["data_parallelism"]
        self.dc = config["harness_config"]["device_count"]
        self.instance_count = self.dc // (self.dp * self.tp * self.pp)

        self.llm_config: dict = config["llm_config"]
        self.harness_config: dict = config["harness_config"]
        self.sampling_config: dict = config["sampling_params"]

        self.debug_toolkit: DebugToolkit = DebugToolkit(
            harness_config = self.harness_config,
            llm_config = self.llm_config
        )

        self.qdata_in = []
        self.qdata_out = []
        self.qstatus_out = Queue()

        self.engine = engine
        self.engine_actors = []
        self.output_collector_threads = []
        self.device_counter = 0

        self.servers: Dict[int, ServerInfo[Queue]] = {}

        self.warmup_sample = constants.WarmUp.ENCODED_SAMPLES.get(config["benchmark_name"], None)
        self.enable_warmup = self.harness_config["enable_warmup"] and (self.warmup_sample is not None)
        self.warm_up_done = []
        self.device_selector = DeviceSelector(self.servers)
        
        assert self.harness_config["schedule_algo"] in ["shortest_queue_with_tokens", "shortest_queue", "round_robin"], f'Unsupported schedule algo: {self.harness_config["schedule_algo"]}'

        if self.harness_config["schedule_algo"] == "shortest_queue_with_tokens":
            self.get_next_device = lambda: self.device_selector.next_best_device_id_with_tokens(
                self.instance_count, self.harness_config["load_balance_token_weight"]
            )
        elif  self.harness_config["schedule_algo"] == "shortest_queue":
            self.get_next_device = lambda: self.device_selector.next_best_device_id(self.instance_count)
        else:
            self.get_next_device = self.device_selector.next_device_id

        self.n_finished = 0
        self.n_finished_first = 0
        self.stopped = False
        self.response_buffer = {}

        # The GC is going to be called after certain number of samples
        self.HARNESS_GC_LIMIT = int(os.getenv('HARNESS_GC_LIMIT', 0))
        self.sample_count = 0
        self.is_gc_limit_specified = self.HARNESS_GC_LIMIT > 0
        if self.is_gc_limit_specified:
            gc.collect()
            gc.disable()


    def start(self):
        ray.init(address="auto", ignore_reinit_error=True)

        for i in range(0, self.instance_count):
            pg = placement_group([{"GPU": 1, "CPU" : 1}] * self.dp * self.tp * self.pp, strategy="STRICT_PACK")

            qdata_in = Queue()
            qdata_out = Queue()
            qstatus_out = Queue()

            ray.get(pg.ready())
            server = self.engine.options(
                                    runtime_env={"env_vars": self.env_config},
                                    num_gpus=self.dp * self.tp * self.pp,
                                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                                        placement_group=pg,
                                        placement_group_capture_child_tasks=True
                                        )
                                ).remote(
                                    devices = i,
                                    qdata_in = qdata_in,
                                    qdata_out = qdata_out,
                                    qstatus_out = qstatus_out,
                                    llm_config = self.llm_config,
                                    sampling_params = self.sampling_config
                                )



            self.servers[i] = ServerInfo(
                server=server,
                qdata_in=qdata_in,
                qdata_out=qdata_out,
                qstatus_out=qstatus_out
            )

            self.servers[i].start_remote()
            self.warm_up_done.append(threading.Event())
            self.output_collector_threads.append(threading.Thread(
                target=self.send_outputs, args=([qdata_out, i]), daemon=True
            ))
            self.output_collector_threads[-1].start()

        for index in self.servers:
            while True:
                log.info(f"i={index} | Polling server...")
                if self.servers[index].is_running_remote():
                    log.info(f"i={index} | Server is ready")
                    break
                else:
                    time.sleep(10)

        if self.enable_warmup:
            self.warm_up()


    def warm_up(self):
        log.info(f"Running warmup")
        for i in range(self.instance_count):
            items = [("0", self.warmup_sample, "WARMUP_QUERY_TYPE")]
            self.servers[i].qdata_in.put(items)
        for i in range(self.instance_count):
            log.info(f"Waiting for server[{i}] warmup to complete...")
            self.warm_up_done[i].wait()
        log.info("Running warmup finished")


    def issue_queries(self, query_samples):
        num_samples = len(query_samples)
        # log.info(f"[Server] Received {num_samples} samples")
        self.sample_count += num_samples
        if self.is_gc_limit_specified and self.sample_count >= self.HARNESS_GC_LIMIT:
            gc.collect()
            self.sample_count = 0
        for sample in query_samples:
            self.send_sample(sample)

        if self.harness_config['debug_record_sample_latencies']:
            for sample in query_samples:
                self.debug_toolkit.record_sample_latencies(sample.id, None, input_token_count=len(self.data_object.input_ids[sample.index]))

    def print_finished(self):
        self.debug_toolkit.print_server_progress(
            self.n_finished,
            self.n_finished_first,
            self.servers,
            self.instance_count
        )

    def post_proc(self, response, device_id):
        sample_id = int(response[0])
        token_ids = response[1]
        finished = token_ids is None
        if finished:
            if self.harness_config['debug_dump_model_output']:
                self.debug_toolkit.dump([self.response_buffer[sample_id]])

            create_response_and_send_complete(sample_id, self.response_buffer[sample_id])
            del self.response_buffer[sample_id]
            self.n_finished += 1
            self.servers[device_id].increment_finished()
        elif sample_id not in self.response_buffer:
            self.response_buffer[sample_id] = list(token_ids)
            create_response_and_send_first_token(sample_id, token_ids)
            self.n_finished_first += 1
        else:
            self.response_buffer[sample_id].extend(token_ids)

        if self.harness_config['debug_record_sample_latencies']:
            self.debug_toolkit.record_sample_latencies(sample_id, token_ids)

    def warmup_finished(self, token_ids, device_id):
        if token_ids is None:
            self.warm_up_done[device_id].set()
            return True
        return False

    def send_outputs(self, qdata_out, device_id):
        self.log("Collecting outputs started...")
        is_warmup_finished = False
        while True:
            response = qdata_out.get()
            if response is None:
                break
            if self.enable_warmup and not is_warmup_finished:
                is_warmup_finished = self.warmup_finished(response[1], device_id)
            else:
                self.post_proc(response, device_id)
            if not self.stopped and self.debug_toolkit.debug_print_finished:
                self.print_finished()

    def stop(self):
        for index in self.servers:
            self.servers[index].stop()
        self.stopped = True
        time.sleep(10)

    def send_sample(self, sample):
        prompt_token_ids = self.data_object.input_ids[sample.index]
        stop_ids = self.data_object.stop_ids[sample.index] if self.data_object.stop_ids else None
        device_id = self.get_next_device()
        if self.harness_config["schedule_algo"] == "shortest_queue_with_tokens":
            window_size = self.harness_config["load_balance_window_size"]
            if len(self.servers[device_id].tokens_in) > window_size:
                # 10 is used as the window_size for this algorithm. This can be tuned potentially for better perf
                self.servers[device_id].tokens_in.pop(0)
            self.servers[device_id].tokens_in.append(len(prompt_token_ids))
        self.servers[device_id].qdata_in.put(
            [(str(sample.id), prompt_token_ids, stop_ids)]
        )
        self.servers[device_id].increment_sent()

    def log(self, message: str):
        log.info(f"SUT - {message}")


class Sample:
    def __init__(self, index):
        self.index = index
        self.id = index

class DistributedSyncServerSUT(DistributedServerSUT):
    def __init__(self, config: dict):
        super().__init__(
            config=config,            
            engine=DistributedSyncServer
        )