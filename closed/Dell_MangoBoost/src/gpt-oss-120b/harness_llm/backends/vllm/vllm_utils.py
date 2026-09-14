from vllm.config import CompilationConfig
from omegaconf import OmegaConf, DictConfig
import harness_llm.common.logging as logger
import os


log = logger.get_logger(__file__)


def generate_cudagraph_capture_sizes(compilation_config: dict):
    capture_sizes = []
    for item in compilation_config["cudagraph_capture_range"]:
        if isinstance(item, list):
            if len(item) != 3:
                log.error(f"vllm['compilation_config']: malformed capture range {item}")
                os._exit(1)
            capture_sizes.extend(range(*item))
        else:
            capture_sizes.append(item)

    compilation_config["cudagraph_capture_sizes"] = capture_sizes
    # Delete range from config
    del compilation_config["cudagraph_capture_range"]


def populate_compile_config(vllm_engine_config: DictConfig):

    vllm_engine_config = OmegaConf.to_container(vllm_engine_config,
                                                resolve=True,
                                                enum_to_str=True)

    if "compilation_config" in vllm_engine_config:
        contra_dict = ["cudagraph_capture_sizes", "cudagraph_capture_range"]
        if all(key in vllm_engine_config["compilation_config"] for key in contra_dict):
            log.error(f"vllm['compilation_config']: {', '.join(contra_dict)} can't be defined together")
            os._exit(1)

        # Generate cudagraph_capture_sizes from the specified range
        if "cudagraph_capture_range" in vllm_engine_config["compilation_config"]:
            log.info("cudagraph_capture_sizes will be generated due to cudagraph_capture_range")
            generate_cudagraph_capture_sizes(vllm_engine_config["compilation_config"])

        vllm_engine_config["compilation_config"] = CompilationConfig(
            **(vllm_engine_config["compilation_config"])
        )

    return vllm_engine_config


def validate_and_correct(vllm_engine_config: dict):

    if (os.environ.get("VLLM_USE_V1", "0") == "1"
            and "num_scheduler_steps" in vllm_engine_config):

        log.warning("num_scheduler_steps removed from vllm config,"
                    + " it cannot be used with V1")
        del vllm_engine_config["num_scheduler_steps"]

    return vllm_engine_config


def generate_vllm_cache_dir(suffix: str):
    return os.path.expanduser("~/.cache/vllm_") + suffix

def generate_torch_inductor_cache_dir(suffix: str):
    return os.path.expanduser("~/.cache/torch_inductor_") + suffix
