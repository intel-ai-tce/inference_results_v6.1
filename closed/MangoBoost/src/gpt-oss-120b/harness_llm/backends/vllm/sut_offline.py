from harness_llm.backends.common.offline_sut import OfflineBaseSUT

from harness_llm.backends.vllm.vllm_engine import initialize_engine_and_generate
import harness_llm.backends.common.utils as utils

class OfflineVLLMSUT(OfflineBaseSUT):

    def __init__(self, config: dict, llm_config: dict, sampling_config: dict):
        super().__init__(
            config=config,
            llm_config=llm_config,
            sampling_config=sampling_config, 
            engine=initialize_engine_and_generate
        )


    def engine_device_size(self):
        return self.tp * self.pp * self.dp


    def check_parallelism_configuration(self):
        utils.check_parallelism_configuration(self.instance_count, self.dp, self.tp, self.pp, self.dc)
