from harness_llm.backends.common.server_sut import ServerBaseSUT
from harness_llm.backends.sglang.server_engine import SGLangServerEngine

class AsyncServerSGLangSUT(ServerBaseSUT):

    def __init__(self, config: dict, llm_config: dict, sampling_config: dict):
        super().__init__(
            config=config,
            llm_config=llm_config,
            sampling_config=sampling_config, 
            engine=SGLangServerEngine
        )