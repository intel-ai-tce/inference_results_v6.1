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

from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama3.1-8B")

def get_args():
    parser = argparse.ArgumentParser(description="Disaggregated prefill benchmark client")
    parser.add_argument('--model_path', type=str, required=False,
                       default="/model/Llama-3.1-8B-Instruct_calibrated-cpu",
                       help="Model path")
    parser.add_argument('--dataset_path', type=str, required=False,
                       default="/data/cnn_eval.json",
                       help="Dataset path")
    parser.add_argument('--cnt', type=int, required=False, default=128,
                       help="Number of requests to process")
    parser.add_argument('--input_len', type=int, required=False, default=1024,
                       help="Input length (not used directly, from dataset)")
    parser.add_argument('--output_len', type=int, required=False, default=128,
                       help="Total output length")
    parser.add_argument('--proxy_url', type=str, default="http://localhost:8192",
                       help="Proxy server URL")
    return parser.parse_args()

class Instance:
    def __init__(self,
                 model_path: str, 
                 dataset_path: str, 
                 proxy_url: bool = False,
                 cnt: bool = True,):
        self.model = model_path
        self.dataset_path = dataset_path
        self.proxy_url = proxy_url
        self.cnt = cnt

        self.seen = set()
        self.st = [0] * cnt
        self.ftl = [0] * cnt
        self.itl = [0] * cnt

    def start(self):
        # Load dataset
        print(f"Loading dataset from {self.dataset_path}...")
        self.dataset = Dataset(
            dataset_path=self.dataset_path,
            model_name=self.model
            )
        rng = np.random.default_rng(seed=42)
        self.wait = rng.exponential(scale=1/3.5, size=13368)

        print(f"Starting test...")
        st = time.time()
        self.add()

    def add(self):
        init = time.time()
        threads = []
        for i in range(self.cnt):
            st = time.time()
            input_ids, input_len, _, _ = self.dataset.getSamples([i])
            self.st[i] = st

            # Create and start independent thread
            thread = threading.Thread(target=self.run, args=(input_ids, i, input_len, st))
            thread.start()
            
            # Sleep before creating thread to respect arrival rate
            time.sleep(self.wait[i])
            
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        
        nd = time.time()
        print(f"fps: {self.cnt*128/(nd-init)}tok/s")

        ftl = np.percentile(self.ftl, [50, 90, 95, 97, 99, 99.5, 99.9])
        itl = np.percentile(self.itl, [50, 90, 95, 97, 99, 99.5, 99.9])

        print(f"FTL:",
              f"50%: {ftl[0]:.2f}s",
              f"90%: {ftl[1]:.2f}s",
              f"95%: {ftl[2]:.2f}s",
              f"97%: {ftl[3]:.2f}s",
              f"99%: {ftl[4]:.2f}s",
              f"99.5%: {ftl[5]:.2f}s",
              f"99.9%: {ftl[6]:.2f}s",)

        print(f"ITL:",
              f"50%: {itl[0]:.2f}ms",
              f"90%: {itl[1]:.2f}ms",
              f"95%: {itl[2]:.2f}ms",
              f"97%: {itl[3]:.2f}ms",
              f"99%: {itl[4]:.2f}ms",
              f"99.5%: {itl[5]:.2f}ms",
              f"99.9%: {itl[6]:.2f}ms",)

    def run(self, input_ids, id, leng, tic=None,):
            st = time.time()

            response = requests.post(
                f"{self.proxy_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": input_ids,
                    "max_tokens": 128,
                    "temperature": 0,
                    "top_k": 1,
                    "stream": True,
                },
                stream=True,
                # timeout=100
                )

            if response.status_code != 200:
                print(f"Error: {response.text}")
                return
            
            bs = 0
            output = "" # []
            for line in response.iter_lines(decode_unicode=False):
                line = line.decode("utf-8")

                if line == "data: [DONE]":
                    continue

                if line.startswith("data:"):
                    bs += 1

                    line = line[len("data: "):]
                    data = json.loads(line)['choices'][0]

                    text = data['text']
                    # print(f"text: {text}\n")
                    # output.append(data["token_ids"])
                    output += text

                    # Use the request id instead of response index (which is always 0)
                    req_id = id

                    if req_id not in self.seen:
                        nd = time.time()
                        # if nd - self.st[req_id] <= 1.5:
                        #     time.sleep(1.5 - (nd - self.st[req_id]))
                        self.seen.add(req_id)
                        self.ftl[req_id] = nd - self.st[req_id]
                        print(f"id:{req_id} len: {leng} ttft: {(nd - self.st[req_id]):.2f}s wait: {(time.time() - nd):.2f}s", flush=True)
                        self.st[req_id] = tmp = time.time()
                    # else:
                    #     nd = time.time()
                    #     self.itl.append(nd - self.st[req_id])
                    #     self.st[req_id] = nd
            
            nd = time.time()
            self.itl[req_id] = (nd - tmp) * 1000 / 127
            print(f"bs: {bs} mean tpot: {self.itl[req_id]:.2f}ms.", flush=True)
            print(f"output: {output}", flush=True)

def main():
    args = get_args()
    print(f"args: {args}")

    instance = Instance(args.model_path, 
                        args.dataset_path, 
                        args.proxy_url, 
                        args.cnt)
    instance.start()

if __name__ == "__main__":
    main()
