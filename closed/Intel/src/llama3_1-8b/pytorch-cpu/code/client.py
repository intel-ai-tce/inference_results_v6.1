#!/usr/bin/env python3
"""
Disaggregated Prefill Client

Sends requests to separate prefill and decode vLLM servers.
Prefill phase: Generates first token only
Decode phase: Continues generation from prefill output
"""

import requests
import argparse
import time
import json
import asyncio
from argparse import ArgumentParser
from typing import List
import logging
import array
import numpy as np
import os
import threading
import multiprocessing as mp
from dataset import Dataset
import mlperf_loadgen as lg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama3.1-8B")

class SUT:
    def __init__(self,
                 runner_args):
        self.model = runner_args.model_path
        self.dataset_path = runner_args.dataset_path
        self.scenario = runner_args.scenario.lower()
        self.cnt = 0 # runner_args.total_sample_count
        self.counter = 0
        self.threads = []
        self.len = {}
        self.output_queue = mp.Queue()
        self.ft_queue = mp.Queue()
        self.bs = 13368//int(os.environ.get("NUM_NUMA_NODES"))
       
        # Load dataset
        print(f"Loading dataset from {self.dataset_path}...")
        self.dataset = Dataset(
            dataset_path=self.dataset_path,
            model_name=self.model
            )

        self.qsl = lg.ConstructQSL(
            runner_args.total_sample_count, 
            runner_args.total_sample_count,
            self.LoadSamplesToRam, 
            self.UnloadSamplesFromRam)
        
        self.index = 0

    def issue_queries(self, query_samples):
        if self.scenario == "offline":
            for i in range(0, len(query_samples), self.bs):
                batch = query_samples[i:i + self.bs]
                tic = time.time()
                ids = []
                prompts = []
                for q in batch:
                    input_ids, input_len, _, _ = self.dataset.getSamples([q.index])
                    self.len[q.id] = input_len
                    ids.append(q.id)
                    prompts.append(input_ids)
                    self.cnt += 1
                thread = threading.Thread(target=self.run, args=(prompts, ids, tic))
                thread.start()
                self.threads.append(thread)
        else:
            for i,q in enumerate(query_samples):
                id, index, tic = q.id, q.index, time.time()
                input_ids, input_len, _, _ = self.dataset.getSamples([index])
                self.len[id] = input_len
                self.cnt += 1
                thread = threading.Thread(target=self.run, args=(input_ids, id, tic))
                thread.start()
                self.threads.append(thread)

    def start(self):
        if self.scenario == "offline":
            # Create first token response thread
            print(f"Starting first-token response thread")
            self.ft_response_thread = threading.Thread(target=self.process_first_tokens)
            self.ft_response_thread.daemon = True
            self.ft_response_thread.start()

        # Create response thread
        print(f"Starting response thread")
        self.response_thread = threading.Thread(target=self.response_loadgen)
        self.response_thread.daemon = True
        self.response_thread.start()

        # Start the main SUT server
        # super().start(True)

    def process_first_tokens(self):
        self.counter = 0
        while True:
            qid, processed_output, tic = self.ft_queue.get()
            self.cnt += 1
            if qid is None:
                print("Exiting First token response thread")
                break

            response_data = array.array("B", np.array(processed_output, np.int32).tobytes())
            buf = response_data.buffer_info()
            response = [lg.QuerySampleResponse(qid, buf[0], buf[1])]
            lg.FirstTokenComplete(response)

    def response_loadgen(self):
        num_processed = 0
        timer = time.time()
        while True:
            qid, processed_output, tic = self.output_queue.get()
            if qid is None:
                break  # Exit condition for the thread
            n_tokens = len(processed_output)
            response_array = array.array("B", np.array(processed_output, np.int32).tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(qid, bi[0], bi[1], n_tokens)]
            lg.QuerySamplesComplete(response)
            num_processed += 1
            # Add progress bar tracking num_processed per second
            if num_processed % 100 == 0 or num_processed == 13368:
                elapsed_time = time.time() - timer
                if elapsed_time > 0:
                    print(f"Rate: {num_processed / elapsed_time:.2f} queries/sec; Processed: {num_processed} queries", flush=True)
                    #timer = time.time()

    def flush_queries(self):
        pass

    def __del__(self):
        pass

    def get_qsl(self):
        return self.qsl
    
    def LoadSamplesToRam(self, query_samples):
        pass

    def UnloadSamplesFromRam(self, query_samples):
        pass

    def stop(self):
        if self.scenario == "offline":
            # Stop the first token response thread
            print("Stopping first token response thread.")
            self.ft_queue.put((None, None, None))
            self.ft_response_thread.join()

        # Signal the output queue that processing is done
        print("Stopping response thread.")
        self.output_queue.put((None, None, None))
        self.response_thread.join()
        
        # Wait for all threads to complete
        for thread in self.threads:
            thread.join()
        # Stop the main SUT server
        # super().stop(True)

    def run(self, prompts, ids, tic=None,):
        response = requests.post(
            f"http://localhost:8192/v1/completions",
            json={
                "model": self.model,
                "prompt": prompts,
                "max_tokens": 128,
                "temperature": 0,
                "top_k": 1,
                "stream": True,
                "return_token_ids": True
            },
            stream=True,
            timeout=(None, None),
            )

        if response.status_code != 200:
            print(f"Error: {response.text}")
            return

        seen = set()
        if self.scenario == "offline":
            outputs = {i: [] for i in range(len(ids))}
        else:
            output = []
        bs = 0
        for line in response.iter_lines(decode_unicode=False):
            line = line.decode("utf-8")

            if line == "data: [DONE]":
                continue

            if line.startswith("data:"):
                line = line[len("data: "):]
                data = json.loads(line)['choices'][0]

                if self.scenario == "offline":
                    choice_index = data["index"]
                    id = ids[choice_index]
                    outputs[choice_index].extend(data["token_ids"])
                else:
                    bs += 1
                    id = ids
                    output.extend(data["token_ids"])

                if id not in seen:
                    seen.add(id)
                    
                    if self.scenario == "server" and time.time() - tic > 1.98:
                        self.counter += 1
                        # print(f"ttft {time.time() - tic}; len {self.len[id]}; violation {self.counter}/{self.cnt}")
                    if self.scenario == "server" and time.time() - tic <= 1.5:
                        time.sleep(1.75 - (time.time() - tic))

                    if self.scenario =="server":
                        processed_output = data["token_ids"]
                        response_data = array.array("B", np.array(processed_output, np.int32).tobytes())
                        buf = response_data.buffer_info()
                        response = [lg.QuerySampleResponse(id, buf[0], buf[1])]
                        lg.FirstTokenComplete(response)
                    else:
                        self.ft_queue.put((id, data["token_ids"], tic))

                    tmp = time.time()
    
        if self.scenario == "offline":
            for i, id in enumerate(ids):
                # print(f"id: {id} completed", flush=True)
                self.output_queue.put((id, [[j] for j in outputs[i]], time.time()))
        else:
            # print(f"bs: {bs} mean tpot: {(time.time() - tmp) * 1000 / 127:.2f}ms.", flush=True)
            self.output_queue.put((id, [[i] for i in output], time.time()))
