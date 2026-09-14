import os
import gc
import zmq
import pickle

from harness_llm.loadgen.sut import SUT, SUTConfig
import harness_llm.common.logging as logging
from harness_llm.backends.common.debug import DebugToolkit
from harness_llm.backends.common.utils import (
    create_response_and_send_complete,
    create_response_and_send_first_token,
)
from harness_llm.backends.common.server_utils import DeviceSelector

import time
import threading
from typing import Dict, List
from dataclasses import dataclass

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


@dataclass
class QueryInfo:
    identity: str = ""
    sent: int = 0
    finished: int = 0
    tokens_in: List[int] = None

    def __post_init__(self):
        if self.tokens_in is None:
            self.tokens_in = []

    def increment_finished(self):
        self.finished += 1

    def increment_sent(self):
        self.sent += 1


class DistributedServerSUT(SUT):

    def __init__(self, config):
        super().__init__(
            SUTConfig(
                model=config["llm_config"]["model"],
                dataset_path=config["harness_config"]["dataset_path"],
                total_sample_count=config["harness_config"]["total_sample_count"],
                model_max_length=None,
                debug=False,
            )
        )

        self.env_config = {k: str(v) for k, v in config.env_config.items()}
        self.tensor_parallelism = config["harness_config"]["tensor_parallelism"]
        self.pipeline_parallelism = config["harness_config"]["pipeline_parallelism"]
        self.data_parallelism = config["harness_config"]["data_parallelism"]
        self.device_count = config["harness_config"]["device_count"]
        self.instance_count = self.device_count // (
            self.data_parallelism * self.tensor_parallelism * self.pipeline_parallelism
        )

        self.llm_config: dict = config["llm_config"]
        self.harness_config: dict = config["harness_config"]
        self.port = config["port"]

        self.debug_toolkit: DebugToolkit = DebugToolkit(
            harness_config=self.harness_config, llm_config=self.llm_config
        )

        self.output_collector_thread = None

        self.servers = {}
        self.server_mapping = {}
        self.device_selector = DeviceSelector(self.servers)

        assert self.harness_config["schedule_algo"] in [
            "shortest_queue_with_tokens",
            "shortest_queue",
            "round_robin",
        ], f'Unsupported schedule algo: {self.harness_config["schedule_algo"]}'

        if self.harness_config["schedule_algo"] == "shortest_queue_with_tokens":
            self.get_next_device = (
                lambda: self.device_selector.next_best_device_id_with_tokens(
                    self.instance_count, self.harness_config["load_balance_token_weight"]
                )
            )
        elif self.harness_config["schedule_algo"] == "shortest_queue":
            self.get_next_device = lambda: self.device_selector.next_best_device_id(
                self.instance_count
            )
        else:
            self.get_next_device = self.device_selector.next_device_id

        self.n_finished = 0
        self.n_finished_first = 0
        self.stopped = False
        self.response_buffer = {}

        # The GC is going to be called after certain number of samples
        self.HARNESS_GC_LIMIT = int(os.getenv("HARNESS_GC_LIMIT", 0))
        self.sample_count = 0
        self.is_gc_limit_specified = self.HARNESS_GC_LIMIT > 0
        if self.is_gc_limit_specified:
            gc.collect()
            gc.disable()

    def start(self):
        self.context = zmq.Context()
        self.context.set(zmq.MAX_SOCKETS, 1024)
        self.sender = self.context.socket(zmq.ROUTER)
        self.sender.bind(f"tcp://*:{self.port}")

        self.output_collector_thread = threading.Thread(
            target=self.recv_outputs, args=([self.instance_count]), daemon=True
        )
        self.output_collector_thread.start()
        self.wait_for_servers_ready()
        self.log(f"Server started with {len(self.servers)} workers")

    def wait_for_servers_ready(self):
        while len(self.servers) < self.instance_count:
            time.sleep(1)

        for i in range(0, self.instance_count):
            self.send_data(self.servers[i].identity, self.servers[i].identity)

    def send_data(self, identity, data):
        self.sender.send(identity, zmq.SNDMORE)
        self.sender.send_pyobj(data)

    def issue_queries(self, query_samples):
        num_samples = len(query_samples)
        self.sample_count += num_samples
        if self.is_gc_limit_specified and self.sample_count >= self.HARNESS_GC_LIMIT:
            gc.collect()
            self.sample_count = 0
        for sample in query_samples:
            self.send_sample(sample)

        if self.harness_config["debug_record_sample_latencies"]:
            for sample in query_samples:
                self.debug_toolkit.record_sample_latencies(
                    sample.id,
                    None,
                    input_token_count=len(self.data_object.input_ids[sample.index]),
                )

    def print_finished(self):
        self.debug_toolkit.print_server_progress(
            self.n_finished, self.n_finished_first, self.servers, self.instance_count
        )

    def post_proc(self, response, device_id):
        sample_id = int(response[0])
        token_ids = response[1]
        finished = token_ids is None
        if finished:
            if self.harness_config["debug_dump_model_output"]:
                self.debug_toolkit.dump([self.response_buffer[sample_id]])

            create_response_and_send_complete(
                sample_id, self.response_buffer[sample_id]
            )
            del self.response_buffer[sample_id]
            self.n_finished += 1
            self.servers[device_id].increment_finished()
        elif sample_id not in self.response_buffer:
            self.response_buffer[sample_id] = list(token_ids)
            create_response_and_send_first_token(sample_id, token_ids)
            self.n_finished_first += 1
        else:
            self.response_buffer[sample_id].extend(token_ids)

        if self.harness_config["debug_record_sample_latencies"]:
            self.debug_toolkit.record_sample_latencies(sample_id, token_ids)

    def register_servers(self, socket):
        for i in range(0, self.instance_count):
            self.log(f"{i=}/{self.instance_count} Wait for server register message")
            identity, data = socket.recv_multipart()
            data = pickle.loads(data)
            self.log(f"{i=}/{self.instance_count} Got new server with {identity=}")
            assert identity == data
            assert identity not in self.servers
            self.servers[i] = QueryInfo(identity=identity)
            self.server_mapping[identity] = i

    def recv_outputs(self, device_count):
        # ZMQ requires dedicated context per thread
        ctx = zmq.Context()
        receiver = ctx.socket(zmq.ROUTER)
        receiver.bind(f"tcp://*:{int(self.port) + 1}")
        receiver.setsockopt(zmq.LINGER, 0)
        self.register_servers(receiver)

        self.log("Collecting outputs started...")
        while True:
            identity, response = receiver.recv_multipart()
            response = pickle.loads(response)
            if response is None:
                self.log(f"{identity} exited")
                device_count -= 1
                if device_count <= 0:
                    break
                continue
            self.post_proc(response, self.server_mapping[identity])
            if not self.stopped and self.debug_toolkit.debug_print_finished:
                self.print_finished()
        self.log("Collecting outputs finished...")
        receiver.close()
        ctx.term()

    def stop(self):
        for server in self.servers.values():
            self.send_data(server.identity, None)
        self.stopped = True
        self.output_collector_thread.join()
        self.sender.close()
        self.context.term()
        time.sleep(10)

    def send_sample(self, sample):
        prompt_token_ids = self.data_object.input_ids[sample.index]
        stop_ids = (
            self.data_object.stop_ids[sample.index]
            if self.data_object.stop_ids
            else None
        )
        device_id = self.get_next_device()
        if self.harness_config["schedule_algo"] == "shortest_queue_with_tokens":
            window_size = self.harness_config["load_balance_window_size"]
            if len(self.servers[device_id].tokens_in) > window_size:
                # 10 is used as the window_size for this algorithm. This can be tuned potentially for better perf
                self.servers[device_id].tokens_in.pop(0)
            self.servers[device_id].tokens_in.append(len(prompt_token_ids))

        self.send_data(
            self.servers[device_id].identity,
            [(str(sample.id), prompt_token_ids, stop_ids)],
        )
        self.servers[device_id].increment_sent()

    def log(self, message: str):
        log.info(f"SUT - {message}")


class Sample:
    def __init__(self, index):
        self.index = index
        self.id = index
