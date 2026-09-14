import logging
import multiprocessing as mp
import os
import asyncio
import logging
import harness_llm.common.numa_helpers as nh
import threading
from harness_llm.common.rpd_trace_utils import rpd_trace_range, rpd_trace_range_non_timed
import queue
from harness_llm.backends.common.constants import HarnessStates
from harness_llm.common.container_utils import remove_none_from_dict
import harness_llm.backends.sglang.engine_factory as engine_factory
import harness_llm.common.logging as logging

log = logging.get_logger(__file__)

class SGLangServerEngine:

    def __init__(
        self,
        devices,
        qdata_in: mp.Queue,
        qdata_out: mp.Queue,
        qstatus_out: mp.Queue,
        llm_config: dict,
        sampling_params: dict,
        benchmark: str,
        enable_warmup: bool = False,
    ):
        self.qdata_in = qdata_in
        self.qdata_out = qdata_out
        self.qstatus_out = qstatus_out
        self.devices = devices
        self.engine = None
        self.process = None
        self.llm_config = llm_config
        self.sampling_params = remove_none_from_dict(sampling_params)


    @rpd_trace_range_non_timed("SUT:Worker")
    def start(self):
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join([str(d) for d in self.devices])
        self.process = mp.Process(target=self.launch)
        self.process.start()


    @rpd_trace_range_non_timed("SUT:Worker")
    def launch(self):
        self.log(f"llm_config={self.llm_config}")        
        self.log(f"sampling_params={self.sampling_params}")

        self.engine = engine_factory.create_from(llm_config=self.llm_config)
        logging.set_level()

        self.signal_running()
        self.run()


    def signal_running(self):
        self.qstatus_out.put_nowait(HarnessStates.LLM_MODEL_LOAD_DONE)


    @rpd_trace_range("SUT:Worker")
    def run(self):
        async_event_loop = asyncio.new_event_loop()
        async_thread = threading.Thread(target=run_async_event_loop, args=([async_event_loop]), daemon=True)
        async_thread.start()
        self.log("Processing started...")
        while True:
            try:
                sample = self.qdata_in.get()
                if sample is None:
                    self.error("qdata_in got end signal...")
                    self.engine.shutdown()
                    break
                asyncio.run_coroutine_threadsafe(self.generate_v2(sample), async_event_loop)
            except queue.Empty:
                break


    def is_running(self):
        try:
            return self.qstatus_out.get_nowait() == HarnessStates.LLM_MODEL_LOAD_DONE
        except:
            return False


    async def generate_v2(self, samples):
        await asyncio.wait([asyncio.create_task(self.generate(sample)) for sample in samples])


    async def generate(self, sample):
        request_id = sample[0]
        prompt_token_ids = sample[1]
        results_generator = await self.engine.async_generate(input_ids=prompt_token_ids, sampling_params=self.sampling_params, stream=True)
        output_token_ids = []
        is_first_token = True
        async for request_output in results_generator:
            output_token_ids = request_output["output_ids"]
            if is_first_token:
                is_first_token = False
                self.qdata_out.put_nowait([request_id, output_token_ids])
        self.qdata_out.put_nowait([request_id, output_token_ids])
        self.qdata_out.put_nowait([request_id, None])


    def log(self, message):
        log.info(f"Server {self.devices} - {message}")


    def error(self, message):
        log.error(f"Server {self.devices} - {message}")


def run_async_event_loop(async_event_loop):
    asyncio.set_event_loop(async_event_loop)
    async_event_loop.run_forever()