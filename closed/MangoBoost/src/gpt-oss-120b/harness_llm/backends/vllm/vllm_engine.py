from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
# vLLM-version compat: v0.14 fork exposes TokenInputs; upstream (>=0.24) uses TokensPrompt.
try:
    from vllm.inputs import TokenInputs
    def _make_token_prompt(token_ids):
        return TokenInputs(type="token", prompt_token_ids=token_ids)
except ImportError:
    from vllm.inputs import TokensPrompt
    def _make_token_prompt(token_ids):
        return TokensPrompt(prompt_token_ids=token_ids)

import logging
import multiprocessing as mp
from multiprocessing import connection as conn
import os, gc, asyncio

import harness_llm.common.numa_helpers as nh
from harness_llm.common.rpd_trace_utils import rpd_trace_range, rpd_trace_range_non_timed, ENABLE_TRACING_RPD
import harness_llm.backends.common.constants as constants
import harness_llm.backends.vllm.vllm_utils as utils

# ray is only needed for the multi-node path and is absent from stock vLLM images.
try:
    from ray.util.queue import Queue as RayQueue
except ImportError:
    class RayQueue:  # sentinel so isinstance(...) checks stay valid without ray
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

# flags to notify the processes.

HARNESS_GC_LIMIT = int(os.getenv('HARNESS_GC_LIMIT', 0))

def create_engine(llm_config: dict, async_engine=False):
    llm_config = utils.validate_and_correct(utils.populate_compile_config(llm_config))
    log.info(f"{llm_config=}")
    if async_engine:
        engine_args = AsyncEngineArgs(**llm_config)
        return AsyncLLMEngine.from_engine_args(
            engine_args = engine_args,
            start_engine_loop = True
        )
    return LLM(**llm_config)

@rpd_trace_range("SUT:Worker")
def _run_vllm(llm, prompt_token_ids, sampling_params):
    return llm.generate(
        prompts = [_make_token_prompt(token_ids) for token_ids in prompt_token_ids],
        sampling_params=sampling_params,
        use_tqdm=False if os.getenv("HARNESS_DISABLE_VLLM_LOGS", "0") == "1" else True,
    )

@rpd_trace_range_non_timed("SUT:Worker")
def initialize_engine_and_generate(
    device_ids: tuple[int, ...],
    qdata_in: conn.Connection,
    qdata_out: conn.Connection,
    qstatus_out: mp.Queue,
    llm_config: dict,
    sampling_params_config: dict = {"temperature": 0.0, "max_tokens": 1024},
    engine_version: str = "sync",
    benchmark: str = "",
):
    """
    Initialize the llm engine and generate the responses.
    """
    use_async_engine = (engine_version == "async")

    for id in device_ids:
        nh.set_affinity_by_device(int(id))

    # Initialize the vllm engine.
    os.environ["VLLM_CACHE_ROOT"] = utils.generate_vllm_cache_dir(str(device_ids[0]))
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = utils.generate_torch_inductor_cache_dir(str(device_ids[0]))
    llm = create_engine(llm_config = llm_config, async_engine = use_async_engine)

    qstatus_out.put(constants.HarnessStates.LLM_MODEL_LOAD_DONE)

    generate(device_ids, qdata_in, qdata_out, use_async_engine, llm, sampling_params_config)

@rpd_trace_range("SUT:Worker")
def generate(
        device_ids,
        data_in,
        data_output,
        use_async_engine,
        llm,
        sampling_params_config,
):
    if isinstance(data_in, conn.Connection):
        input_fn = data_in.recv
    elif isinstance(data_in, RayQueue):
        input_fn = data_in.get
    else:
        raise AttributeError(
            "The data_in object must be a known type (conn.Connection, RayQueue)"
        )

    event_loop = asyncio.new_event_loop() if use_async_engine else None

    # The GC is going to be called after certain number of steps
    sample_count = 0
    is_gc_limit_specified = HARNESS_GC_LIMIT > 0
    if is_gc_limit_specified:
        gc.collect()
        gc.disable()

    if ENABLE_TRACING_RPD:
        llm.start_profile()
    # Generates the completions for the input prompt tokens.
    while True:
        try:
            item = input_fn()
            if item is None:
                log.info(f"LLM is stopping")
                if use_async_engine:
                    del llm
                data_output.put(constants.HarnessStates.LLM_GENERATION_DONE)
                break

            start, end, prompt_token_ids, stop_token_ids = item
            sample_count += len(prompt_token_ids)
            if is_gc_limit_specified and sample_count >= HARNESS_GC_LIMIT:
                gc.collect()
                sample_count = 0

            sampling_params_list = []
            if stop_token_ids:
                for stop_seq_ids in stop_token_ids:
                    sampling_param = SamplingParams(**sampling_params_config)
                    sampling_param.stop_seq_ids = tuple(stop_seq_ids)
                    sampling_params_list.append(sampling_param)

            pred_output_tokens = None
            if use_async_engine:
                pred_output_tokens = _run_async_vllm(
                    engine=llm,
                    event_loop=event_loop,
                    sampling_params=sampling_params_list if sampling_params_list else SamplingParams(**sampling_params_config),
                    start=start,
                    prompt_token_ids=prompt_token_ids
                )
            else:
                pred_output_tokens = _run_vllm(llm, prompt_token_ids, sampling_params_list if sampling_params_list else SamplingParams(**sampling_params_config))

            if ENABLE_TRACING_RPD:
                llm.stop_profile()
            log.info(f"VLLM finished")

            processed_output = [
                output.outputs[0].token_ids for output in pred_output_tokens
            ]
            log.info(f"output tokens collected")

            data_output.put((start, end, processed_output))
            log.info(f"Processed output | start, end = {start}, {end}")
        except:
            logging.exception("Exception running vLLM")
            break
    log.info(f"vLLM engine thread finished for {device_ids=}")


def _run_async_vllm(engine, event_loop, sampling_params, start, prompt_token_ids):
    return event_loop.run_until_complete(
        _async_generate_batch(engine, sampling_params, start, prompt_token_ids)
    )


async def _async_generate_batch(engine, sampling_params, start, prompt_token_ids):
    tasks = []
    for i in range(len(prompt_token_ids)):
        tasks.append(_async_generate(engine,
                                     sampling_params,
                                     str(start + i),
                                     prompt_token_ids[i])
        )
    return await asyncio.gather(*tasks)


async def _async_generate(engine, sampling_params, sample_id, prompt_token_ids):
    results_generator = engine.generate(
        {"prompt_token_ids": prompt_token_ids},
        sampling_params,
        sample_id
    )
    final_request_output = None
    async for request_output in results_generator:
        final_request_output = request_output
    return final_request_output
