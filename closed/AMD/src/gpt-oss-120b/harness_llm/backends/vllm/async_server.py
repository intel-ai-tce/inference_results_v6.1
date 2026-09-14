import logging
import multiprocessing as mp
import os
import asyncio
import logging
import harness_llm.common.numa_helpers as nh
import threading
from harness_llm.common.rpd_trace_utils import rpd_trace_range, rpd_trace_range_non_timed
import queue
from harness_llm.backends.common.constants import HarnessStates, WarmUp
from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs
import harness_llm.backends.vllm.vllm_utils as utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)

class AsyncServer:

    def __init__(
        self,
        devices,
        qdata_in: mp.Queue,
        qdata_out: mp.Queue,
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
        #TODO handle stop_seq_id_config properly
        self.sampling_params.pop("stop_seq_ids_config", None)
        self.log(f"sampling_params={self.sampling_params}")

        self.sampling_params = SamplingParams(**self.sampling_params)

        engine_args = AsyncEngineArgs(
            **self.llm_config
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args=engine_args, start_engine_loop=True)
        
        async_event_loop = asyncio.new_event_loop()
        asyncio.run_coroutine_threadsafe(self.signal_running(), async_event_loop)
        self.run(async_event_loop)


    async def signal_running(self):
        if self.warmup_enabled:
            await self.run_warmup()
        self.qstatus_out.put_nowait(HarnessStates.LLM_MODEL_LOAD_DONE)


    async def run_warmup(self):
        self.log("Started warmup")
        await self.warmup_generate()
        self.log("Warmup completed")


    async def warmup_generate(self):
        tasks = [self._warmup_generate(str(i)) for i in range(10)]    
        await asyncio.gather(*tasks)
    
    
    async def _warmup_generate(self, request_id: str):
        prompt_token_ids = WarmUp.ENCODED_SAMPLES.get(self.benchmark, None)
        results_generator = self.engine.generate({"prompt_token_ids": prompt_token_ids}, self.sampling_params, request_id)
        async for _ in results_generator:
            pass


    @rpd_trace_range("SUT:Worker")
    def run(self, async_event_loop):
        async_thread = threading.Thread(target=run_async_event_loop, args=([async_event_loop]), daemon=True)
        async_thread.start()
        self.log("Processing started...")
        while True:
            try:
                sample = self.qdata_in.get()
                if sample is None:
                    del self.engine
                    self.error("qdata_in got end signal...")
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
        results_generator = self.engine.generate({"prompt_token_ids": prompt_token_ids}, self.sampling_params, request_id)
        output_token_ids = []
        first_token_id_count = 0
        async for request_output in results_generator:
            output_token_ids = request_output.outputs[0].token_ids
            if 0 == first_token_id_count:
                first_token_id_count = len(output_token_ids)
                self.qdata_out.put_nowait([request_id, output_token_ids])
        self.qdata_out.put_nowait([request_id, output_token_ids[first_token_id_count:]])
        self.qdata_out.put_nowait([request_id, None])


    def log(self, message):
        log.info(f"Server {self.devices} - {message}")


    def error(self, message):
        log.error(f"Server {self.devices} - {message}")


def run_async_event_loop(async_event_loop):
    asyncio.set_event_loop(async_event_loop)
    async_event_loop.run_forever()
