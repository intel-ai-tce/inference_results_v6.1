import logging
import multiprocessing as mp
import time

from harness_llm.loadgen.sut import SUT
from harness_llm.common.rpd_trace_utils import rpd_trace_range, rpd_trace_range_non_timed
import threading
import gc
import os
from typing import Dict
from harness_llm.backends.common.debug import DebugToolkit
from harness_llm.backends.common.utils import check_parallelism_configuration, create_response_and_send_complete, create_response_and_send_first_token, get_visible_device_indices
from harness_llm.backends.common.server_utils import DeviceSelector, ServerInfo
from harness_llm.loadgen.sut import SUT, SUTConfig
import harness_llm.backends.common.constants as constants

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


class ServerBaseSUT(SUT):
    def __init__(
            self, config: dict,
            llm_config: dict,
            sampling_config: dict,
            engine
    ):
        log.info(f"Init SUTvLLMServer")

        self.llm_config: dict = llm_config
        self.harness_config: dict = config["harness_config"]
        self.debug_toolkit: DebugToolkit = DebugToolkit(
            harness_config = self.harness_config,
            llm_config = self.llm_config
        )

        super().__init__(
            SUTConfig(
                dataset_path=config["harness_config"]["dataset_path"],
                total_sample_count=(
                    config["harness_config"]["total_sample_count"]
                    if "total_sample_count" in config["harness_config"]
                    else 24576
                ),
                model_max_length=(
                    config["harness_config"]["model_max_length"]
                    if "model_max_length" in config["harness_config"]
                    else None
                ),
                debug=False,
                debug_toolkit=self.debug_toolkit
            )
        )
        self.engine_class = engine
        self.benchmark = config["benchmark_name"]
        self.sampling_params: dict = sampling_config

        self.tp = config["harness_config"]["tensor_parallelism"]
        self.pp = config["harness_config"]["pipeline_parallelism"]
        self.dp = config["harness_config"]["data_parallelism"]
        self.dc = self.harness_config.get("device_count", 8)
        self.visible_devices = get_visible_device_indices(self.dc)

        self.instance_count = self.dc // self.engine_device_size()
        self.check_parallelism_configuration()

        self.servers: Dict[int, ServerInfo[mp.Queue]] = {}
        self.output_collector_threads = []
        self.device_counter = 0
        self.device_selector = DeviceSelector(self.servers)

        self.total_sample_count = self.harness_config["total_sample_count"]

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


    @rpd_trace_range_non_timed("SUT:Main")
    def start(self):

        for i in range(0, self.instance_count):
            engine_device_size = self.engine_device_size()
            devices = self.visible_devices[engine_device_size * i: engine_device_size * (i + 1)]

            qdata_in = mp.Queue()
            qdata_out = mp.Queue()
            qstatus_out = mp.Queue()

            server = self.engine_class(
                devices,
                qdata_in,
                qdata_out,
                qstatus_out,
                self.llm_config,
                self.sampling_params,
                self.benchmark,
                self.harness_config["enable_warmup"]
            )

            self.servers[i] = ServerInfo(
                server=server,
                qdata_in=qdata_in,
                qdata_out=qdata_out,
                qstatus_out=qstatus_out
            )

            self.servers[i].start()
            self.output_collector_threads.append(threading.Thread(
                target=self.send_outputs, args=([qdata_out, i]), daemon=True
            ))
            self.output_collector_threads[-1].start()

        for index in self.servers:
            while True:
                log.info(f"i={index} | Polling server...")
                if self.servers[index].is_running():
                    log.info(f"i={index} | Server is ready")
                    break
                else:
                    time.sleep(10)

    @rpd_trace_range("SUT:Main")
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

    @rpd_trace_range("SUT:Main")
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

    def send_outputs(self, qdata_out, device_id):
        self.log("Collecting outputs started...")
        while True:
            response = qdata_out.get()
            if response is None:
                break
            self.post_proc(response, device_id)
            if not self.stopped and self.debug_toolkit.debug_print_finished:
                self.print_finished()

    @rpd_trace_range_non_timed("SUT:Main")
    def stop(self):
        for index in self.servers:
            self.servers[index].stop()
        self.stopped = True
        time.sleep(10)

    @rpd_trace_range("SUT:Main")
    def send_sample(self, sample):
        prompt_token_ids = self.data_object.input_ids[sample.index]
        device_id = self.get_next_device()
        if self.harness_config["schedule_algo"] == "shortest_queue_with_tokens":
            window_size = self.harness_config["load_balance_window_size"]
            if len(self.servers[device_id].tokens_in) > window_size:
                # 10 is used as the window_size for this algorithm. This can be tuned potentially for better perf
                self.servers[device_id].tokens_in.pop(0)
            self.servers[device_id].tokens_in.append(len(prompt_token_ids))
        self.servers[device_id].qdata_in.put_nowait(
            [(str(sample.id), prompt_token_ids, self.data_object.stop_ids[sample.index] if self.data_object.stop_ids else None)]
        )
        self.servers[device_id].increment_sent()

    def log(self, message: str):
        log.info(f"SUT - {message}")


class Sample:
    def __init__(self, index):
        self.index = index
        self.id = index
