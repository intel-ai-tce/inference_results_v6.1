from harness_llm.backends.common.server_sut import ServerBaseSUT

class ServerVLLMSUT(ServerBaseSUT):

    def __init__(self, 
                 config: dict, 
                 llm_config: dict, 
                 sampling_config: dict, 
                 engine):
        super().__init__(
            config=config,
            llm_config=llm_config,
            sampling_config=sampling_config, 
            engine=engine
        )
