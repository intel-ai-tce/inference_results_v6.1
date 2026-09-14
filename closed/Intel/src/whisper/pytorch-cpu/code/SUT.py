# Copyright (c) 2020, Cerebras Systems, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard packages
import sys
import os
import array
import subprocess
import math
import queue
import time
import logging
import threading
import traceback

# Common math packages
import numpy as np
from tqdm import tqdm

# Framework packages
import torch
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams

# Local python packages
from QSL import AudioQSL, AudioQSLInMemory
import mlperf_loadgen as lg
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SUT")

WORKER_STARTUP_MAX_RETRIES = 3
WORKER_STARTUP_RETRY_DELAY_SECONDS = 1

def void(*args, **kwargs):
    pass

# Disable prints if progress bar is active
PBAR = int(os.environ.get("PBAR", "1"))
# Update frequency of the progress bar
PBAR_FREQ = int(os.environ.get("PBAR_FREQ", "10"))
if PBAR==1:
    print = void

# Only start workers in allowed NUMA nodes. Useful when node sharing
ALLOWED_NODES = os.environ.get("ALLOWED_NODES", "all")

def get_start_cores():
    start_cores = subprocess.check_output('lscpu | grep "NUMA node.* CPU.*" | awk "{print \\$4}" | cut -d "-" -f 1', shell=True)
    start_cores = start_cores.decode('ascii').rstrip().split('\n')
    start_cores = [int(_) for _ in start_cores]
    return start_cores

CORES_PER_INST = int(os.environ.get("CORES_PER_INST", "1"))
NUM_NUMA_NODES = int(os.environ.get("NUM_NUMA_NODES", "1"))
NODES_PER_INST = int(os.environ["NUM_NUMA_NODES"])/int(os.environ["NUM_INSTS"])
INSTS_PER_NODE = int(os.environ["INSTS_PER_NODE"])

SAMPLE_RATE = 16000

labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
labels_dict = {}
for i in range(len(labels)):
    labels_dict[labels[i]] = i

class Instance(mp.Process):
    def __init__(
        self,
        model_path=None,
        dataset_path=None,
        manifest_filepath=None,
        device="cpu",
        batch_size=-1,
        total_sample_count=-1,
        rank=-1,
        core_list=(),
        input_queue=None,
        output_queue=None,
        cond_var=None,
        alive_counter=None,
        sample_counter=None,
        current_counter=None,
        startup_queue=None,
    ):
        mp.Process.__init__(self)
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.manifest_filepath = manifest_filepath
        self.device = device
        self.batch_size = batch_size
        self.total_sample_count = total_sample_count
        self.rank = rank
        self.core_list = core_list
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.cond_var = cond_var
        self.alive_counter = alive_counter
        self.sample_counter = sample_counter
        self.num_samples = 0
        self.total_time = 0
        self.query_idx_mapping = []
        self.qid_mapping = []
        self.req_counter = 0
        self.finished = False
        self.current_counter = current_counter
        self.startup_queue = startup_queue

    def run(self):
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = (
            f"{self.core_list[0]}-{self.core_list[-1]}"
        )

        dataset_vocab = labels
        self.qsl = AudioQSLInMemory(
            self.dataset_path,
            self.manifest_filepath,
            dataset_vocab,
            SAMPLE_RATE,
            self.total_sample_count,
        )

        for attempt in range(1, WORKER_STARTUP_MAX_RETRIES + 1):
            try:
                self.model = LLM(
                    model=self.model_path,
                    limit_mm_per_prompt={"audio": 1},
                    kv_cache_dtype="fp8",
                )
                break
            except Exception as exc:
                log.warning(
                    "Worker %s LLM init attempt %s/%s failed: %s: %s",
                    self.rank, attempt, WORKER_STARTUP_MAX_RETRIES,
                    type(exc).__name__, exc,
                )
                if attempt == WORKER_STARTUP_MAX_RETRIES:
                    if self.startup_queue is not None:
                        self.startup_queue.put({
                            "rank": self.rank,
                            "success": False,
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        })
                    return
                time.sleep(WORKER_STARTUP_RETRY_DELAY_SECONDS)

        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=200,
        )
        if self.startup_queue is not None:
            self.startup_queue.put({"rank": self.rank, "success": True})
        with self.cond_var:
            self.alive_counter.value += 1
            self.cond_var.notify()

        keep_alive = True
        while keep_alive:
            keep_alive = self.process_queries()

        del self.model

    def process_queries(self):
        samples_to_fill = self.batch_size - self.model.llm_engine.get_num_unfinished_requests()
        # After receiving None, continue executing untill all requests are finished
        return_value = True
        if (samples_to_fill>0 and not self.finished):
            try:
                qitem_list = self.input_queue.get(False)
                print(f"Rank {self.rank} received one query")
            except queue.Empty:
                # When running multiple nodes/server, it's common for some workers to not receive a query in time
                # Under that scenario, the work shouldn't wait (hence get(False)), and should also not treat
                # the workload as finished
                qitem_list = 1

            if qitem_list is None:
                self.finished = True
            elif type(qitem_list)!=int:
                # TODO: use time_start_list in server
                qitem_list, time_start_list = qitem_list
                prompt_list = []
                for qitem in qitem_list:
                    prompt = self.qsl[qitem.index]
                    self.model.llm_engine.add_request(str(self.req_counter), prompt, self.sampling_params)
                    self.query_idx_mapping.append(qitem.index)
                    self.qid_mapping.append(qitem.id)
                    self.req_counter += 1
        results = []
        query_ids = []
        qid = []
        if self.model.llm_engine.has_unfinished_requests():
            print(f"Number of unfinished requests: {self.model.llm_engine.get_num_unfinished_requests()}")
 
            # Step once
            time_prestep = time.time()
            step_outputs = self.model.llm_engine.step()
            print(f"Step time {time.time()-time_prestep:.3f}s")
            for output in step_outputs:
                request_id = int(output.request_id)
                # Process finished outputs
                if output.finished:
                    vllm_text = output.outputs[0].text
                    token_ids = output.outputs[0].token_ids
                    results.append((vllm_text, len(token_ids)))
                    query_ids.append(self.query_idx_mapping[request_id])
                    qid.append(self.qid_mapping[request_id])
        elif self.finished:
            return_value = False
        else:
            # Avoid excessive mpty steps
            time.sleep(0.05)

        self.num_samples += len(results)
        
        for i,result_tuple in enumerate(results):
            # Whisper outputs space in the front and capitalizes things
            result, n_tokens = result_tuple
            result = result.lower().strip()
            transcript = []
            for s in result:
                if s in labels_dict:
                    transcript.append(labels_dict[s])
            transcript = [transcript]

            assert len(transcript) == 1
            response_array = array.array('q', transcript[0])

            self.output_queue.put((qid[i], n_tokens, response_array))
            # print(f"Finished {qid[i]}")
        with self.cond_var:
            print(f"Rank {self.rank} finished {len(results)} requests. Running counter value {self.current_counter.value}")
            self.current_counter.value -= len(results)
            self.cond_var.notify()

        return return_value

class vllmSUT:
    def __init__(self, dataset_dir, model_path,
                 manifest_filepath, perf_count, num_workers=1):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        self.model_path = model_path
        self.dataset_path = dataset_dir
        self.manifest_filepath = manifest_filepath
        self.device = "cpu"
        self.batch_size = 96
        self.prefill_batch_size = 1
        self.total_sample_count = perf_count
        self.num_workers = num_workers
        self.worker_threads = [None] * self.num_workers

        dataset_vocab = labels

        self.dev = torch.device("cuda:0") if torch.cuda.is_available() and os.environ.get("USE_GPU", "").lower() not in  [ "no", "false" ]  else torch.device("cpu")

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.qsl = AudioQSLInMemory(dataset_dir,
                                    manifest_filepath,
                                    dataset_vocab,
                                    SAMPLE_RATE,
                                    perf_count)

        self.query_queue_list = [mp.JoinableQueue() for _ in range(self.num_workers)]
        self.query_queue_int = mp.Queue()
        self.current_counter_list = [mp.Value("i", 0) for _ in range(self.num_workers)]
        self.output_queue = mp.Queue()
        self.alive_counter = mp.Value("i", 0)
        self.cond_var = mp.Condition(lock=mp.Lock())
        self.sample_counter = mp.Value("i", 0)
        self.progress = None
        self.query_thread = None
        self.response_thread = None

        # When using partial nodes, not all the workers will be active
        # Load balancing should be between active workers
        self.allowed_workers = []
        self.worker_startup_queues = {}

    def _get_active_worker_specs(self):
        node_start_cores = get_start_cores()
        core_lists = []
        if INSTS_PER_NODE > 0:
            for i in range(NUM_NUMA_NODES):
                for j in range(INSTS_PER_NODE):
                    core_lists.append(
                        list(
                            range(
                                node_start_cores[i] + j * CORES_PER_INST,
                                node_start_cores[i] + (j + 1) * CORES_PER_INST,
                            )
                        )
                    )

        node_list = list(range(len(node_start_cores)))
        allowed_nodes = [str(node) for node in node_list]
        if ALLOWED_NODES != "all":
            allowed_nodes = ALLOWED_NODES.split(",")

        worker_specs = []
        for rank in range(self.num_workers):
            cur_node = math.floor(rank * NODES_PER_INST)
            if str(cur_node) not in allowed_nodes:
                continue
            worker_specs.append(
                {
                    "rank": rank,
                    "core_list": tuple(core_lists[rank]),
                }
            )
        return worker_specs

    def _terminate_worker(self, rank):
        worker = self.worker_threads[rank]
        if worker is None:
            return
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=5)
        self.worker_threads[rank] = None

    def _terminate_started_workers(self):
        for rank in range(self.num_workers):
            try:
                self._terminate_worker(rank)
            except Exception:
                pass

    def start(self):
        worker_specs = self._get_active_worker_specs()
        self.allowed_workers = [spec["rank"] for spec in worker_specs]
        if not self.allowed_workers:
            raise RuntimeError(
                "No active Whisper workers were selected after applying ALLOWED_NODES"
            )

        try:
            for spec in worker_specs:
                rank = spec["rank"]
                startup_queue = mp.Queue()
                self.worker_startup_queues[rank] = startup_queue
                worker = Instance(
                    model_path=self.model_path,
                    dataset_path=self.dataset_path,
                    manifest_filepath=self.manifest_filepath,
                    device=self.device,
                    batch_size=self.batch_size,
                    total_sample_count=self.total_sample_count,
                    rank=rank,
                    core_list=spec["core_list"],
                    input_queue=self.query_queue_list[rank],
                    output_queue=self.output_queue,
                    cond_var=self.cond_var,
                    alive_counter=self.alive_counter,
                    sample_counter=self.sample_counter,
                    current_counter=self.current_counter_list[rank],
                    startup_queue=startup_queue,
                )
                worker.start()
                self.worker_threads[rank] = worker

            pending = set(self.allowed_workers)
            while pending:
                made_progress = False
                for rank in list(pending):
                    try:
                        status = self.worker_startup_queues[rank].get_nowait()
                    except queue.Empty:
                        continue
                    made_progress = True
                    if not status.get("success"):
                        raise RuntimeError(
                            f"Worker {rank} failed to start after "
                            f"{WORKER_STARTUP_MAX_RETRIES} attempt(s): "
                            f"{status.get('exception_type')}: {status.get('message')}"
                        )
                    log.info("Worker %s started successfully.", rank)
                    pending.remove(rank)
                if pending and not made_progress:
                    time.sleep(0.05)
        except Exception:
            self._terminate_started_workers()
            raise

        log.info("Starting internal issue query thread")
        self.query_thread = threading.Thread(target=self.issue_queries_int)
        self.query_thread.daemon = True
        self.query_thread.start()

        log.info("Starting Loadgen response thread")
        self.response_thread = threading.Thread(target=self.response_loadgen)
        self.response_thread.daemon = True
        self.response_thread.start()

    def get_best_rank(self, value_added):
        current_counters = np.array([(self.current_counter_list[i].value+value_added) for i in self.allowed_workers]) # Instances priority will be ordered by their respective in-flight queries
        target_rank = np.argmin(current_counters)
        if current_counters[target_rank]>self.batch_size:
            return -1
        else:
            return self.allowed_workers[target_rank]

    def issue_queries(self, query_samples):
        start_time = time.time()
        if PBAR and (self.progress is None):
            if len(query_samples)>1:
                self.progress = tqdm(total=len(query_samples), smoothing=0.0)
            else:
                self.progress = tqdm(smoothing=0.0)

        query_sample_list = []
        for query_sample in query_samples:
            # Continuous batching
            self.query_queue_int.put((query_sample, start_time))

    def try_dispatch(
        self,
        query_list,
        delete_list,
        bucket,
        input_len_bucket,
        time_start_bucket,
        server=False): # Inactive for now
        target_rank = -1
        wait_to_dispatch = True
        while wait_to_dispatch:
            with self.cond_var:
                target_rank = self.get_best_rank(1) # With prepacking, prefill batch size is always 1
                if target_rank!=-1:
                    self.current_counter_list[target_rank].value += 1
            # Always wait for an instance with an empty slot(for now)
            if target_rank==-1:
                time.sleep(0.01)
            else:
                wait_to_dispatch = False
        if target_rank!=-1:
            print(f"Sending to rank {target_rank}, num_queries {len(query_list)}, before add {self.current_counter_list[target_rank].value}")
            self.query_queue_list[target_rank].put((query_list, time_start_bucket))
            delete_list.append(bucket)
            return True
        return False

    # Generic load balancer. Watered-down version of bucketed prefill
    # Since there is not bucket, there is no point to use bucketed query list
    def issue_queries_int(self):
        keep_alive = True
        # TODO: add server and use real latency
        time_left = 9999
        time_compute = 0
        time_start_list = []
        query_list = []

        while keep_alive:
            print("Length of query_list", len(query_list))
            new_query = False
            try:
                query = self.query_queue_int.get(timeout=0.05)
            except:
                pass
            else:
                if query is None:
                    keep_alive = False
                else:
                    query, start_time = query
                    start_time_c = start_time
                    new_query = True
            
            if new_query:
                time_start_list.append(start_time_c)
                query_list.append(query)
            
            if len(query_list)>0:
                time_wait = time.time()-time_start_list[0]
                time_needed = time_wait + time_compute
                # When to dispatch
                # 1. Time limit passed
                # 2. Reached prefill batch size
                # 3. End of inference
                if (time_needed > time_left) or (len(query_list)==self.prefill_batch_size) or (not keep_alive):
                    target_rank = -1
                    # Continue trying to dispatch until there is an empty spot
                    while target_rank==-1:
                        with self.cond_var:
                            target_rank = self.get_best_rank(len(query_list))
                            if target_rank!=-1:
                                self.current_counter_list[target_rank].value += len(query_list)
                        if target_rank==-1:
                            time.sleep(0.01)
                    print(f"Sending to rank {target_rank}, length {len(query_list)}, wait_time {time_wait:.2f}s")
                    self.query_queue_list[target_rank].put((query_list, time_start_list))
                    query_list = []
                    time_start_list = []
        
        for i in range(self.num_workers):
            print("Putting none in", i)
            self.query_queue_list[i].put(None)

    def flush_queries(self):
        self.query_queue_int.put(None)
        pass

    def update_pbar(self, tok_count, last_count, update_value):
        postfix_str = f"{tok_count/self.progress.format_dict['elapsed']:.1f}toks/s"
        self.progress.set_postfix_str(postfix_str, refresh=False)
        self.progress.update(update_value)
        self.progress.refresh()
        last_count += update_value
        return last_count

    def response_loadgen(self):
        keep_alive = True
        processed_count = 0
        last_count = 0
        tok_count = 0
        while keep_alive:
            result = self.output_queue.get()
            if result is None:
                keep_alive = False
            else:
                qid, n_tokens, response_array = result
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(qid, bi[0],
                                                  bi[1] * response_array.itemsize, n_tokens)
                lg.QuerySamplesComplete([response])
                processed_count += 1
                tok_count += n_tokens
                if PBAR:
                    if processed_count - last_count >= PBAR_FREQ:
                        last_count = self.update_pbar(tok_count, last_count, PBAR_FREQ)
        if PBAR and processed_count>last_count:
            last_count = self.update_pbar(tok_count, last_count, processed_count-last_count)

    def stop(self):
        if self.query_thread is not None and self.query_thread.is_alive():
            self.query_queue_int.put(None)
        for worker in self.worker_threads:
            try:
                if worker is not None:
                    worker.join()
            except Exception:
                pass
        self.output_queue.put(None)
        if self.query_thread is not None:
            self.query_thread.join(timeout=5)
        if self.response_thread is not None:
            self.response_thread.join(timeout=5)


    def __del__(self):
        try:
            self._terminate_started_workers()
        except Exception:
            pass
        try:
            lg.DestroySUT(self.sut)
            print("Finished destroying SUT.")
        except Exception:
            pass
