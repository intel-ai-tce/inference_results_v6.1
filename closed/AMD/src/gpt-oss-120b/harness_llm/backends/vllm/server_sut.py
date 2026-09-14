from harness_llm.backends.common.server_sut import ServerBaseSUT
import harness_llm.backends.common.utils as utils

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


    def engine_device_size(self):
        return self.tp * self.pp * self.dp


    def check_parallelism_configuration(self):
        utils.check_parallelism_configuration(self.instance_count, self.dp, self.tp, self.pp, self.dc)
