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
        warmup_sample_count: int = 10,
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
        self.warmup_sample_count = warmup_sample_count

        # --- Torch profiler (vLLM native) ------------------------------------
        # Enabled via ENABLE_TORCH_PROFILE=1 (see run.sh), which sets
        # VLLM_TORCH_PROFILER_DIR. Profiling adds overhead, so only turn it on
        # for profiling runs, not perf-reproducibility runs.
        self._profile_dir = os.getenv("VLLM_TORCH_PROFILER_DIR", "")
        profile_devices = os.getenv("VLLM_PROFILE_DEVICES", "0")
        try:
            self._profile_device_set = {
                int(x) for x in profile_devices.replace(",", " ").split()
            }
        except ValueError:
            self._profile_device_set = {0}
        self._profile_delay_sec = float(os.getenv("VLLM_PROFILE_DELAY_SEC", "30"))
        self._profile_duration_sec = float(os.getenv("VLLM_PROFILE_DURATION_SEC", "10"))
        self._profile_enabled = bool(self._profile_dir) and (
            self.devices[0] in self._profile_device_set
        )

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

        async_engine_config = {k: v for k, v in self.llm_config.items() if k != "swap_space"}
        # --- profiler_config attach ------------------------------------------
        if self._profile_enabled:
            try:
                from vllm.config import ProfilerConfig
                async_engine_config["profiler_config"] = ProfilerConfig(
                    profiler="torch",
                    torch_profiler_dir=self._profile_dir,
                    torch_profiler_with_stack=False,
                    torch_profiler_record_shapes=False,
                    torch_profiler_use_gzip=True,
                )
                self.log(f"ProfilerConfig attached (torch -> {self._profile_dir})")
            except Exception as e:
                self.error(f"Could not build ProfilerConfig, disabling profiling: {e}")
                self._profile_enabled = False
        engine_args = AsyncEngineArgs(
            **async_engine_config
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
        self.log(f"Running {self.warmup_sample_count} warmup samples")
        base_tokens = WarmUp.ENCODED_SAMPLES.get(self.benchmark, None)
        if base_tokens is None:
            self.log("No warmup token IDs found for benchmark, skipping warmup")
            return

        # Submit ALL samples simultaneously (no batching).
        # Short uniform inputs maximize the number of concurrently decoding requests,
        # which drives high decode batch sizes and triggers the Triton JIT compilations
        # that reduce TPOT. Varied or longer inputs slow prefill, reduce concurrency,
        # and lower peak decode batch size — hurting warmup effectiveness.
        tasks = [self._warmup_generate(str(i), base_tokens)
                 for i in range(self.warmup_sample_count)]
        await asyncio.gather(*tasks)
        self.log(f"Warmup completed ({self.warmup_sample_count} samples, "
                 f"input_len={len(base_tokens)})")


    async def _warmup_generate(self, request_id: str, prompt_token_ids=None):
        if prompt_token_ids is None:
            prompt_token_ids = WarmUp.ENCODED_SAMPLES.get(self.benchmark, None)
        results_generator = self.engine.generate({"prompt_token_ids": prompt_token_ids}, self.sampling_params, request_id)
        async for _ in results_generator:
            pass


    @rpd_trace_range("SUT:Worker")
    def run(self, async_event_loop):
        async_thread = threading.Thread(target=run_async_event_loop, args=([async_event_loop]), daemon=True)
        async_thread.start()
        self.log("Processing started...")
        # --- torch profiler window kickoff -----------------------------------
        if self._profile_enabled:
            self.log(
                f"Torch profiler ENABLED on device {self.devices[0]}: "
                f"delay={self._profile_delay_sec}s duration={self._profile_duration_sec}s "
                f"dir={self._profile_dir}"
            )
            asyncio.run_coroutine_threadsafe(self._profile_window(), async_event_loop)
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


    # --- torch profiler capture window ---------------------------------------
    async def _profile_window(self):
        """Capture a fixed-duration vLLM torch trace once steady state is reached.

        engine.start_profile()/stop_profile() are coroutines on AsyncLLMEngine
        (v1) and fan out to the worker, which exports the trace to
        VLLM_TORCH_PROFILER_DIR on stop.
        """
        try:
            await asyncio.sleep(self._profile_delay_sec)
            self.log("Starting torch profiler capture window")
            await self.engine.start_profile()
            await asyncio.sleep(self._profile_duration_sec)
            await self.engine.stop_profile()
            self.log(
                f"Torch profiler capture complete; trace written under {self._profile_dir}"
            )
        except Exception as e:
            self.error(f"Torch profiler window failed: {e}")


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
