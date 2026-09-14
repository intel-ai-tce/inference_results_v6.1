from typing import List, Dict, Generic, TypeVar
from dataclasses import dataclass
from harness_llm.common.rpd_trace_utils import rpd_trace_range

QueueT = TypeVar('QueueT')

@dataclass
class ServerInfo(Generic[QueueT]):
    server: object
    qdata_in: QueueT
    qdata_out: QueueT
    qstatus_out: QueueT
    sent: int = 0
    finished: int = 0
    tokens_in: List[int] = None
    
    def __post_init__(self):
        if self.tokens_in is None:
            self.tokens_in = []

    def start(self):
        self.server.start()

    def start_remote(self):
        self.server.start.remote()
    
    def is_running(self):
        return self.server.is_running()
    
    def is_running_remote(self):
        return self.server.is_running.remote()

    def increment_finished(self):
        self.finished += 1

    def increment_sent(self):
        self.sent += 1

    def stop(self):
        self.qdata_in.put(None)


class DeviceSelector:
    
    def __init__(self, servers: Dict[int, ServerInfo]):
        self.servers = servers
        self.device_counter = 0
    
    @rpd_trace_range("SUT:Main")
    def next_device_id(self):
        next_div_id = self.device_counter
        self.device_counter = (self.device_counter + 1) % len(self.servers)
        return next_div_id
    
    @rpd_trace_range("SUT:Main")
    def next_best_device_id(self, instance_count: int):
        next_div_id = 0
        min_queue = 1_000_000_000
        for d in range(instance_count):
            diff = self.servers[d].sent - self.servers[d].finished
            if diff < min_queue:
                min_queue = diff
                next_div_id = d
        return next_div_id
    
    @rpd_trace_range("SUT:Main")
    def next_best_device_id_with_tokens(self, instance_count: int, token_weight: float):
        next_div_id = 0
        min_queue = 1_000_000_000
        for d in range(instance_count):
            num_tokens = sum(self.servers[d].tokens_in)
            diff = (self.servers[d].sent - self.servers[d].finished) + token_weight * num_tokens
            # This is of the form y = theta_1*x_1 + theta_2*x_2, a linear combination of the two variables.
            # theta_1 = 1 is used but could be tuned for some better perf
            # theta_2 = 0.02 is a tuned value.
            
            if diff < min_queue:
                min_queue = diff
                next_div_id = d
        return next_div_id
