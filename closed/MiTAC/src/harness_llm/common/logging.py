import logging

from warnings import filterwarnings


def set_level(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

_logging_initialized = False

if not _logging_initialized:
    set_level()
    _logging_initialized = True

def get_logger(file: str):
    return logging.getLogger(file)

def set_library_loglevel(log_level: str):
    INFERENCE_LOG_LEVEL = logging._nameToLevel[log_level]

    # Aiter
    ## Create a handler to specify arbitrary log-level
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="[%(name)s] %(message)s")
    console_handler.setFormatter(formatter)
    console_handler.setLevel(INFERENCE_LOG_LEVEL)
    logging.getLogger("aiter").addHandler(console_handler)
    logging.getLogger("aiter").setLevel(INFERENCE_LOG_LEVEL)
    logging.getLogger("AITER_TRITON").setLevel(INFERENCE_LOG_LEVEL)

    # Sglang
    logging.getLogger("sglang.srt.managers.scheduler_metrics_mixin").setLevel(INFERENCE_LOG_LEVEL)

    # Vllm
    # TODO: Remove this filter when we use newer base image than rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250915
    filterwarnings("ignore", message="Logical operators 'and' and 'or' are deprecated for non-scalar tensors")
