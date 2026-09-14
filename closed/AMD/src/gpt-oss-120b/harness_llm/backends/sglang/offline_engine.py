import harness_llm.common.logging as logging
import multiprocessing as mp
from multiprocessing import connection as conn
import os, gc
import harness_llm.common.numa_helpers as nh
from harness_llm.common.rpd_trace_utils import rpd_trace_range
import harness_llm.backends.common.constants as constants
from harness_llm.common.container_utils import remove_none_from_dict
import harness_llm.backends.sglang.engine_factory as engine_factory

log = logging.get_logger(__file__)

HARNESS_GC_LIMIT = int(os.getenv('HARNESS_GC_LIMIT', 0))

@rpd_trace_range("SUT:Worker")
def _generate(llm, prompt_token_ids, sampling_params):
    return llm.generate(
         input_ids=prompt_token_ids,
         sampling_params=sampling_params,
    )

@rpd_trace_range("SUT:Worker")
def run_engine(
    device_ids: tuple[int, ...],
    qdata_in: conn.Connection,
    qdata_out: conn.Connection,
    qstatus_out: mp.Queue,
    llm_config: dict,
    sampling_params_config: dict,
    engine_version: str, # only used in vllm engine
    benchmark: str, # only used in vllm engine
):
    """
    Initialize the llm engine and generate the responses.
    """

    # Initialize the SGLang engine.
    llm = engine_factory.create_from(llm_config=llm_config)

    # SGLang engine overwrites the logging configuration during its init
    # We have to reset it to have harness logging
    logging.set_level()
    sampling_params = remove_none_from_dict(sampling_params_config)
    qstatus_out.put(constants.HarnessStates.LLM_MODEL_LOAD_DONE)

    # The GC is going to be called after certain number of steps
    sample_count = 0
    is_gc_limit_specified = HARNESS_GC_LIMIT > 0
    if is_gc_limit_specified:
        gc.collect()
        gc.disable()

    # Generates the completions for the input prompt tokens.
    while True:
        try:
            item = qdata_in.recv()
            if item is None:
                log.info(f"LLM is stopping")
                qdata_out.put(constants.HarnessStates.LLM_GENERATION_DONE)
                llm.shutdown()
                break

            start, end, prompt_token_ids, query_types = item
            sample_count += len(prompt_token_ids)
            if is_gc_limit_specified and sample_count >= HARNESS_GC_LIMIT:
                gc.collect()
                sample_count = 0

            pred_output_tokens = _generate(llm, prompt_token_ids, sampling_params)
            log.info(f"SGLang finished")

            processed_output = [
                output["output_ids"] for output in pred_output_tokens
            ]

            log.info(f"output tokens collected")

            qdata_out.put((start, end, processed_output))

            log.info(f"Processed output | start, end = {start}, {end}")
        except Exception:
            log.exception()
            break
    log.info(f"SGLang engine thread finished for {device_ids=}")
