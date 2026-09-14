import numpy as np
from numa import schedule, memory
from vllm import LLM, SamplingParams
# from vllm.config import CompilationConfig, CompilationLevel
import os
import argparse
import time
from dataset import Dataset
import torch
import random

NODE_LIST = [3]
OMP_NUM_THREADS = 42
START_CORE = 86
OMP_THREADS_BIND = f"{START_CORE}-{START_CORE+OMP_NUM_THREADS-1}"
# OMP_THREADS_BIND = f"{START_CORE}-{START_CORE+OMP_NUM_THREADS-1}|{START_CORE+OMP_NUM_THREADS}-{START_CORE+2*OMP_NUM_THREADS-1}"

os.environ["VLLM_USE_V1"]="1"
os.environ["VLLM_LOGGING_LEVEL"]="INFO" # "DEBUG"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"]="2"
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
# os.environ["ONEDNN_VERBOSE"]="1"
os.environ["USE_PRIMITIVE_CACHE"]="ON"
os.environ["VLLM_CPU_OMP_THREADS_BIND"]=f"{OMP_THREADS_BIND}"
os.environ["OMP_NUM_THREADS"]=f"{OMP_NUM_THREADS}"
os.environ["VLLM_CPU_KVCACHE_SPACE"]="300"

def get_args():

    parser = argparse.ArgumentParser(description="Standalone vllm benchmark script")
    parser.add_argument('--model_path', type=str, required=False, default="Meta-Llama-3.1-8B-Instruct-quantized.w8a8", help="model weights")
    parser.add_argument('--dataset_path', type=str, required=False, default="/data/cnn_eval.json", help="dataset")
    parser.add_argument('--batch_size', type=int, required=False, default=21, help="default 1")
    parser.add_argument('--max_num_batched_tokens', type=int, required=False, default=16384, help="default 8192")
    parser.add_argument('--input_len', type=int, required=False, default=1024, help="1024(default), 2048, 4096, 8192, 16384")
    parser.add_argument('--output_len', type=int, required=False, default=1024, help="default 128")
    parser.add_argument('--cnt', type=int, required=False, default=21, help="default 10")
    parser.add_argument('--tensor_parallel_size', type=int, required=False, default=1, help="default 1")
    parser.add_argument('--max_model_len', type=int, required=False, default=9216, help="default input_len + output_len")
    parser.add_argument('--profile', action='store_true', required=False, help="enable torch profiler")
    parser.add_argument('--bf16', action='store_true', required=False, help="enable bf16")
    parser.add_argument('--accuracy', action='store_true', required=False, help="enable accuracy")
    
    # parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    return args

def setup_profiler(enabled):
    if not enabled:
        return None
    schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=0)
    DEVICE = 'cpu'
    activities = [torch.profiler.ProfilerActivity.CPU]

    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        #debug_activities=debug_activities,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler(
        #       'pytorch_profiler_internal',
        #       use_gzip=True),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=True)
    return profiler

def main():
    args = get_args()
    memory.set_membind_nodes(*NODE_LIST)

    model = LLM(
        model=args.model_path,
        dtype="bfloat16",
        skip_tokenizer_init=False,
        # trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.batch_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=0.95,
        kv_cache_dtype="auto",
        enforce_eager=False,
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_k=1,
        max_tokens=8192,
        min_tokens=8192
    )

    prompt = Dataset(
            dataset_path=args.dataset_path,
            model_name=args.model_path,
        )

    prompt.loadDataset()
    print(f"Dataset loaded.")

    st = {}
    seen = {}
    leng = {}

    rng = np.random.default_rng(seed=42)
    wait = rng.exponential(scale=1/0.4, size=13368)

    print(f"Warming up model...")
    # for i in range(8):
    #     token_ids,_,_,_ = prompt.getSamples([i])
    #     model.llm_engine.add_request(str(i), token_ids , sampling_params)
    # while model.llm_engine.has_unfinished_requests():
    #     step_outputs = model.llm_engine.step()
        

    # Prefill
    # if enabled:
    #     profiler_prefill.start()
    firs = time.time()
    for i in range(args.cnt):
        # token_ids,_,_,_ = prompt.getSamples([i])
        token_ids = random.sample(range(1, 127000), 1024)
        model.llm_engine.add_request(str(i), token_ids , sampling_params)
        st[i] = time.time()
        leng[i] = len(token_ids)
        # step_outputs = model.llm_engine.step()
        
        # for output in step_outputs:
            # if int(output.request_id) not in seen:
        #     id = output.request_id
        #     print(f"id: {id} ttft: {time.time() - st[int(id)]}s len: {leng[int(id)]}")
        # seen[i] = i
        
    #     if enabled:
    #         profiler_prefill.step()
        
    # if enabled:
    #     profiler_prefill.stop()

    # Decode
    results = []
    # if enabled:
    #     profiler_decode.start()
    # for i in range(10):
    while model.llm_engine.has_unfinished_requests():
        end = time.time()
        step_outputs = model.llm_engine.step()
        # if enabled:
        #     profiler_decode.step()
        kv_sum = 0
        breakpoint()
        for output in step_outputs:
            kv_sum += len(output.outputs[0].token_ids) + leng[int(output.request_id)]
            if output.finished and args.accuracy:
                id = int(output.request_id)
                # print(f"Finished id: {id}, {output.outputs[0].text}")
        print(f"bs: {len(step_outputs)} time: {(time.time() - end)*1000}ms kv: {kv_sum}")
    # if enabled:
    #     profiler_decode.stop()
    nd = time.time()
    print(f"e2e: {(nd-firs)}")

if __name__ == "__main__":
    main()
