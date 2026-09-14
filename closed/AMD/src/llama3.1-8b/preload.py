from vllm import LLM

import os
import harness_llm.backends.vllm.vllm_utils as utils
from harness_llm.common.config_parser import HarnessCfg

if __name__ == "__main__":
    conf = HarnessCfg().create_from_cli()
    for env, val in conf["env_config"].items():
        if val is not None:
            os.environ[env] = str(val)
    llm_config = conf["llm_config"]
    llm_config = utils.validate_and_correct(utils.populate_compile_config(llm_config))

    #creating the engine triggers the compilation and caching of aiter kernels
    LLM(**llm_config)
