from ray.util.queue import Queue
from ray.util.placement_group import placement_group
import os
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray
from harness_llm.backends.vllm.ray.distributed_offline_vllm_engine import DistributedOfflineVllmEngine
from harness_llm.loadgen.sut import SUT, SUTConfig
import harness_llm.common.logging as logging
import multiprocessing as mp
from harness_llm.common.sorting import SortingStrategy, validate_sorting_strategy
import harness_llm.backends.common.constants as constants
from threading import Thread, Event
import mlperf_loadgen as lg
import array
import numpy as np
import time
from ray.runtime_env import RuntimeEnv

log = logging.get_logger(__file__)

class DistributedOfflineSUT(SUT):
    
    def __init__(self, config: dict):
        super().__init__(
            SUTConfig(
                model = config["llm_config"]["model"],
                dataset_path = config["harness_config"]["dataset_path"],
                total_sample_count = config["harness_config"]["total_sample_count"],
                model_max_length = None,
                debug=False
            )
        )
        self.env_config = {k: str(v) for k, v in config.env_config.items()}  

        self.tensor_parallelism = config["harness_config"]["tensor_parallelism"]
        self.pipeline_parallelism = config["harness_config"]["pipeline_parallelism"]
        self.device_count = config["harness_config"]["device_count"]
        self.data_parallelism = self.device_count // (self.tensor_parallelism * self.pipeline_parallelism)

        self.engine_config: dict = config["llm_config"]
        self.harness_config: dict = config["harness_config"]
        self.sampling_config: dict = config["sampling_params"]

        self.input = []
        self.output = []
        self.status = Queue()
        self.engine_actors = []
        self.completion_threads = []

        self.warmup_sample = constants.WarmUp.ENCODED_SAMPLES.get(config["benchmark_name"], None)
        self.enable_warmup = self.harness_config["enable_warmup"] and (self.warmup_sample is not None)
        self.warm_up_done = []
        
        self.sorting_params = self.harness_config.get("sorting", {})
        validate_sorting_strategy(self.sorting_params)
        self.sorting = SortingStrategy(self.data_object,
                                       self.harness_config["max_num_batched_tokens"],
                                       config["benchmark_name"] == "llama3_1-405b")

        self.start_t = time.time()
        self.infer_start_t = time.time()


    def init_llms(self):
        ray.init(address="auto", ignore_reinit_error=True)
 
        for i in range(self.data_parallelism):
            pg = placement_group([{"GPU": 1, "CPU" : 1}] * self.tensor_parallelism, strategy="STRICT_PACK")
            input = Queue()
            self.input.append(input)
            output = Queue()
            self.output.append(output)
            self.warm_up_done.append(Event())

            ray.get(pg.ready())
            self.engine_actors.append(DistributedOfflineVllmEngine.options(
                runtime_env={"env_vars": self.env_config},
                num_gpus=self.tensor_parallelism,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True
                )
            )
            .remote(
                actor_id=i,
                input = input, 
                output = output,
                engine_config = self.engine_config, 
                sampling_config = self.sampling_config
            ))


    def start_completion_threads(self):
        for index in range(0, self.data_parallelism):
            thread = Thread(target=self.completion_reader, args=(index,), daemon=True)
            thread.start()
            self.completion_threads.append(thread)
            

    def warm_up(self):
        log.info("Running warmup")
        for i in range(self.data_parallelism):
            prompt_token_ids = [self.warmup_sample]
            self.input[i].put((0, None, prompt_token_ids, None))
        for i in range(self.data_parallelism):
            log.info(f"Waiting for server[{i}] warmup to complete...")
            self.warm_up_done[i].wait()
        log.info("Running warmup finished")


    def stop(self):
        for t in self.completion_threads:
            t.join()
        log.info(f"Total time spent with run: {time.time() - self.start_t}")


    def start(self):
        log.info("Start booting")
        self.init_llms()
        engines = [actor.boot.remote() for actor in self.engine_actors]
        for engine in ray.get(engines):
            log.info("engine booted: " + str(engine))
        for engine_actor in self.engine_actors:
            engine_actor.run_engine.remote()

        self.start_completion_threads()
        if self.enable_warmup:
            self.warm_up()
        log.info("End booting")

        self.infer_start_t = time.time()
        log.info(
            f"Time spent from start to inference start: {self.infer_start_t - self.start_t}"
        )


    def post_proc(self, response):
        start, end, output_token_ids = response
        log.info(
            f"Got item  |  start, end = {start}, {end}  |  n outputs = {len(output_token_ids)}"
        )

        if self.harness_config['debug_dump_model_output']:
            self.debug_toolkit.dump(output_token_ids)

        output_sample_ids = self.sample_ids[start:end]
        assert len(output_sample_ids) == len(output_token_ids)

        log.info(f"Signaling LoadGen output")

        try:
            for i in range(len(output_token_ids)):
                response_array = array.array(
                    "B", np.array(output_token_ids[i], np.int32).tobytes()
                )
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        output_sample_ids[i], bi[0], bi[1], len(output_token_ids[i])
                    )
                ]
                lg.QuerySamplesComplete(response)
        except:
            log.info(f"Error sending completed response to LoadGen")    


    def completion_reader(self, index):
        while True:
            try:
                response = self.output[index].get()
                if response == constants.HarnessStates.LLM_GENERATION_DONE:
                    log.info(f"Query chunk done. GPU index: {index}")
                    break
                else:
                    if self.enable_warmup and not self.warm_up_done[index].is_set():
                        self.warm_up_done[index].set()
                    else:
                        self.post_proc(response)
            except:
                logging.exception("Exception during completion")
                break


    def issue_queries(self, query_samples):
        log.info(f"Issue queries  |  number of queries = {len(query_samples)}")
        ranges, query_samples = self.sorting.sort_samples(query_samples, self.sorting_params, self.data_parallelism)
        self.sample_ids = [query_samples[i].id for i in range(len(query_samples))]
        prompt_token_ids = [
            self.data_object.input_ids[query_samples[i].index]
            for i in range(len(query_samples))
        ]

        log.info(
            f"Converted queries to prompt tokens  |  number of queries = {len(prompt_token_ids)}"
        )
      
        for i, (start, end) in enumerate(ranges):
            self.input[i].put((start, end, prompt_token_ids[start:end], None))
            log.info(f"Put prompt tokens in pipe #{i} {start=} {end=}")

        for i in range(self.data_parallelism):
            self.input[i].put(None)
