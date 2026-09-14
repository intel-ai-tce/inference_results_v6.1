import logging
from dataclasses import dataclass

import mlperf_loadgen as lg

from sut.dataset import Dataset

log = logging.getLogger(__name__)


@dataclass
class SUTConfig:
    model: str = None
    dataset_path: str = None
    total_sample_count: int = 24576
    model_max_length: int = None
    prompt_source: str = None
    tokenizer_path: str = None
    add_special_tokens: bool = True


class SUT:
    def __init__(self, config: SUTConfig):
        self.model_path = config.model
        self.dataset_path = config.dataset_path
        self.model_max_length = config.model_max_length
        self.total_sample_count = config.total_sample_count
        self.prompt_source = config.prompt_source
        self.tokenizer_path = config.tokenizer_path or config.model
        self.add_special_tokens = config.add_special_tokens

        self.tokenizer = None
        self.data_object = None
        self.qsl = None
        self.stop_test = False

        self._init_qsl()

    def _init_qsl(self):
        self.data_object = Dataset(
            dataset_path=self.dataset_path,
            total_sample_count=self.total_sample_count,
            prompt_source=self.prompt_source,
            tokenizer_path=self.tokenizer_path,
            add_special_tokens=self.add_special_tokens,
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

    def start(self):
        pass

    def stop(self):
        self.stop_test = True

    def issue_queries(self, query_samples):
        pass

    def flush_queries(self):
        pass
