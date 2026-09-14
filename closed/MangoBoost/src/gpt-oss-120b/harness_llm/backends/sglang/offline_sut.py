from harness_llm.backends.common.offline_sut import OfflineBaseSUT

from harness_llm.backends.sglang.offline_engine import run_engine
import harness_llm.backends.common.utils as utils

class OfflineSGLangSUT(OfflineBaseSUT):

    def __init__(self, config: dict, llm_config: dict, sampling_config: dict):
        super().__init__(
            config=config,
            llm_config=llm_config,
            sampling_config=sampling_config,
            engine=run_engine
        )
        
    
    def engine_device_size(self):
        return self.tp * self.pp
            

    def check_parallelism_configuration(self):
        utils.check_parallelism_configuration(self.instance_count, 1, self.tp, self.pp, self.dc)
