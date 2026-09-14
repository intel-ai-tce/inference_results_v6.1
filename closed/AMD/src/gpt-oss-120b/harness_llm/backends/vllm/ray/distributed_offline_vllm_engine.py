import ray
import os
import logging
from ray.util.queue import Queue
from vllm import LLM
import harness_llm.backends.vllm.vllm_utils as utils
import harness_llm.common.logging as logging
from harness_llm.backends.vllm.vllm_engine import generate

log = logging.get_logger(__file__)

@ray.remote
class DistributedOfflineVllmEngine:
    
    def __init__(
            self, 
            actor_id,
            input: Queue,
            output: Queue,
            engine_config: dict, 
            sampling_config: dict
            ):
        self.actor_id = actor_id
        self.input = input
        self.output = output
        self.engine_config = engine_config
        self.sampling_config = sampling_config
        self.use_async_engine = False

    def boot(self):
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join((str(i % 8) for i in ray.get_gpu_ids())) 
        os.environ["VLLM_CACHE_ROOT"] = utils.generate_vllm_cache_dir(str(ray.get_gpu_ids()[0] % 8))
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = utils.generate_torch_inductor_cache_dir(str(ray.get_gpu_ids()[0] % 8))
        self.engine_config = utils.validate_and_correct(utils.populate_compile_config(self.engine_config))
        self.engine = LLM(**self.engine_config)
        
        return ray.get_gpu_ids()
    
    def run_engine(self):
        generate(self.actor_id, self.input, self.output, self.use_async_engine, self.engine, self.sampling_config)
