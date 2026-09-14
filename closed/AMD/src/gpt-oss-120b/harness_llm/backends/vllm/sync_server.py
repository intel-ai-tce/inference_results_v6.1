import logging
import multiprocessing as mp
import os
import time
import threading

import logging
import harness_llm.common.numa_helpers as nh
from harness_llm.common.rpd_trace_utils import rpd_trace_range_non_timed
from harness_llm.backends.common.constants import WarmUp
import harness_llm.backends.vllm.vllm_utils as utils

from harness_llm.backends.vllm.queue_llm import QueueLLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)


class SyncServer:

    SIG_RUN = 1
    SIG_STOP = 2

    def __init__(
        self,
        devices,
        qdata_in,
        qdata_out,
        qstatus_out: mp.Queue,
        llm_config: dict,
        sampling_params: dict,
        benchmark: str,
        warmup_enabled: bool,
    ):
        self.qdata_in = qdata_in
        self.qdata_out = qdata_out
        self.qstatus_out = qstatus_out
        self.devices = devices
        self.engine = None
        self.process = None
        self.llm_config = utils.validate_and_correct(utils.populate_compile_config(llm_config))
        self.sampling_params = sampling_params
        self.benchmark = benchmark
        self.warmup_enabled = warmup_enabled

    @rpd_trace_range_non_timed("SUT:Worker")
    def start(self):
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join([str(d) for d in self.devices])
        os.environ["VLLM_CACHE_ROOT"] = utils.generate_vllm_cache_dir(str(self.devices[0]))
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = utils.generate_torch_inductor_cache_dir(str(self.devices[0]))
        self.process = mp.Process(target=self.launch)
        self.process.start()

    @rpd_trace_range_non_timed("SUT:Worker")
    def launch(self):
        nh.set_affinity_by_device(self.devices[0])

        self.log(f"llm_config={self.llm_config}")
        self.log(f"sampling_params={self.sampling_params}")

        self.engine = QueueLLM(input_queue=self.qdata_in,
                               result_queue=self.qdata_out,
                               sampling_params_config=self.sampling_params,
                               **self.llm_config)

        
        signal_thread = threading.Thread(target=self.signal_running, daemon=True)
        signal_thread.start()

        use_tqdm = False if os.getenv("HARNESS_DISABLE_VLLM_LOGS", "0") == "1" else True
        self.engine.start(use_tqdm=use_tqdm)


    def run_warmup(self):
        log.info("Started warmup")
        self.engine.start_warmup()
        self.qdata_in.put_nowait(
            [(str(i), WarmUp.ENCODED_SAMPLES.get(self.benchmark, None), None) for i in range(10)]
        )
        time.sleep(5)
        while not self.engine.is_engine_empty():
            time.sleep(5)
        self.engine.stop_warmup()
        log.info("Warmup completed")


    def signal_running(self):
        if self.warmup_enabled:
            self.run_warmup()

        self.qstatus_out.put_nowait(SyncServer.SIG_RUN)


    def is_running(self):
        try:
            return self.qstatus_out.get_nowait() == SyncServer.SIG_RUN
        except:
            return False


    def log(self, message):
        log.info(f"Server {self.devices} - {message}")


    def error(self, message):
        log.error(f"Server {self.devices} - {message}")

