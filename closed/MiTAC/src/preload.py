import os, sys, multiprocessing
from harness_llm.common.config_parser import HarnessCfg

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    command = " ".join(sys.argv)

    conf = HarnessCfg().create_from_cli()

    for env, val in conf["env_config"].items():
        if val is not None:
            os.environ[env] = str(val)

    llm_config = conf["llm_config"]

    # creating the engine triggers the compilation and caching of aiter kernels
    if "vllm" in command:
        from vllm import LLM
        import harness_llm.backends.vllm.vllm_utils as utils
        llm_config = utils.validate_and_correct(utils.populate_compile_config(llm_config))
        LLM(**llm_config)
    elif "sglang" in command:
        import harness_llm.backends.sglang.engine_factory as engine_factory
        engine_factory.create_from(llm_config)
