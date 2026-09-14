import logging
from dataclasses import dataclass
import mlperf_loadgen as lg
from harness_llm.loadgen.dataset import Dataset
from harness_llm.backends.common.debug import DebugToolkit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

@dataclass
class SUTConfig:
    model: str = None
    dataset_path: str = None
    total_sample_count: int = 24576
    model_max_length: int = None
    debug: bool = False
    debug_toolkit: DebugToolkit = None


class SUT:
    """A SUT interface to integrate"""

    def __init__(
        self,
        config: SUTConfig
    ):
        log.info(f"Init SUT")
        self.debug = config.debug
        if self.debug:
            log.setLevel(logging.DEBUG)

        self.model_path = config.model
        self.dataset_path = config.dataset_path
        self.model_max_length = config.model_max_length
        self.total_sample_count = config.total_sample_count
        self.debug_toolkit = config.debug_toolkit

        self.tokenizer = None
        self.data_object = None
        self.qsl = None

        self.stop_test = False

        self.init_tokenizer()
        self.init_qsl()
        self.load()

    def init_tokenizer(self):
        pass

    def init_qsl(self):
        self.data_object = Dataset(
            dataset_path=self.dataset_path,
            total_sample_count=self.total_sample_count,
            debug_toolkit=self.debug_toolkit
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

    def start(self):
        """Start the SUT before LoadGen initiates the test."""
        pass

    def stop(self):
        """Stop the SUT when LoadGen signals that the test is done."""
        self.stop_test = True

    def load(self):
        pass

    def get_sut(self):
        pass

    def get_qsl(self):
        pass

    def predict(self):
        pass

    def issue_queries(self, query_samples):
        """LoadGen sends in queries here."""
        pass

    def flush_queries(self):
        pass

    def __del__(self):
        pass
