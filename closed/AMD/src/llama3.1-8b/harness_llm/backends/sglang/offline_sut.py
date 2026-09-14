from harness_llm.backends.common.offline_sut import OfflineBaseSUT

from harness_llm.backends.sglang.offline_engine import run_engine

class OfflineSGLangSUT(OfflineBaseSUT):

    def __init__(self, config: dict, llm_config: dict, sampling_config: dict):
        super().__init__(
            config=config,
            llm_config=llm_config,
            sampling_config=sampling_config,
            engine=run_engine
        )
