import os
import gc
import zmq
import pickle
import array
import time
import threading
import numpy as np

from harness_llm.loadgen.sut import SUT, SUTConfig
import harness_llm.common.logging as logging
from harness_llm.backends.common.debug import DebugToolkit

import mlperf_loadgen as lg

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


class ServerInfo:
    def __init__(self, identity, perf):
        self.identity = identity
        self.perf = perf


class DistributedOfflineSUT(SUT):

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
        self.workload_distribution = []

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
        self.calculate_workload_distribution()
        self.log(f"Server started with {len(self.servers)} workers")

    def wait_for_servers_ready(self):
        while len(self.servers) < self.instance_count:
            time.sleep(1)

        for i in range(0, self.instance_count):
            self.send_data(self.servers[i].identity, self.servers[i].identity)

    def make_ranges(self, query_sample_count):
        ranges = []
        total_samples = query_sample_count
        current_start = 0

        for i, (identity, percentage) in enumerate(self.workload_distribution):
            size = int((percentage / 100.0) * total_samples)
            end = current_start + size
            if i == len(self.workload_distribution) - 1 or end > total_samples:
                end = None

            ranges.append((identity, current_start, end))
            if end is not None:
                current_start = end
            else:
                break
        self.log(f"Ranges: {ranges}")
        return ranges

    def send_data(self, identity, data):
        self.sender.send(identity, zmq.SNDMORE)
        self.sender.send_pyobj(data)

    def issue_queries(self, query_samples):
        self.log(f"Issue queries  |  number of queries = {len(query_samples)}")
        ranges = self.make_ranges(len(query_samples))
        self.sample_ids = [query_samples[i].id for i in range(len(query_samples))]
        prompt_token_ids = [
            self.data_object.input_ids[query_samples[i].index]
            for i in range(len(query_samples))
        ]

        self.log(
            f"Converted queries to prompt tokens  |  number of queries = {len(prompt_token_ids)}"
        )

        for identity, start, end in ranges:
            self.send_data(identity, (start, end, prompt_token_ids[start:end], None))
            self.log(f"Sent prompt tokens to {identity} {start=} {end=}")

    def post_proc(self, response):
        start, end, output_token_ids = response
        self.log(
            f"Got item  |  start, end = {start}, {end}  |  n outputs = {len(output_token_ids)}"
        )

        if self.harness_config["debug_dump_model_output"]:
            self.debug_toolkit.dump(output_token_ids)

        output_sample_ids = self.sample_ids[start:end]
        assert len(output_sample_ids) == len(output_token_ids)

        self.log(f"Signaling LoadGen output")

        try:
            for i in range(len(output_token_ids)):
                response_array = array.array(
                    "B", np.array(output_token_ids[i], np.int32).tobytes()
                )
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        output_sample_ids[i], bi[0], bi[1], len(output_token_ids[i])
                    )
                ]
                lg.QuerySamplesComplete(response)
        except:
            self.log(f"Error sending completed response to LoadGen")

    def register_servers(self, socket):
        for i in range(0, self.instance_count):
            self.log(f"{i=}/{self.instance_count} Wait for server register message")
            identity, data = socket.recv_multipart()
            data, perf = pickle.loads(data)
            self.log(f"{i=}/{self.instance_count} Got new server with {data=} {perf=}")
            assert identity == data
            assert identity not in self.server_mapping
            self.servers[i] = ServerInfo(identity, perf)
            self.server_mapping[identity] = i

    def calculate_workload_distribution(self):
        inverse_perfs = [
            1.0 / server_info.perf for server_info in self.servers.values()
        ]
        total_inverse = sum(inverse_perfs)
        percentages = [(inverse / total_inverse) * 100 for inverse in inverse_perfs]
        distribution = [
            (server_info.identity, percentages[i])
            for i, server_info in enumerate(self.servers.values())
        ]
        self.log(f"Workload distribution: {distribution}")
        self.workload_distribution = distribution

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
            self.post_proc(response)
        self.log("Collecting outputs finished...")
        receiver.close()
        ctx.term()

    def stop(self):
        for server_info in self.servers.values():
            self.send_data(server_info.identity, None)
        self.output_collector_thread.join()
        self.sender.close()
        self.context.term()
        time.sleep(10)

    def log(self, message: str):
        log.info(f"SUT - {message}")


class Sample:
    def __init__(self, index):
        self.index = index
        self.id = index
