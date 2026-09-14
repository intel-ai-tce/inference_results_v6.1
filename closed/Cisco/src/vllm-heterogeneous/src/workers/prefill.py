
"""
ZMQ-based prefill worker.

Usage:
  python3 -m src.workers.prefill --config <path.yaml>
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import uvloop  
    _HAS_UVLOOP = True
except ImportError:
    _HAS_UVLOOP = False

from workers.common import (
    nixl_kv_connector_extra_config, pack, unpack,
    load_yaml_config, build_capture_sizes, cfg, serialization_name,
    optional_bool_cfg, require_prefix_caching_disabled,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("prefill_worker")

DIAG_INTERVAL = int(os.environ.get("PREFILL_DIAG_INTERVAL", "5000"))


async def main():
    parser = argparse.ArgumentParser(description="ZMQ prefill worker")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()

    engine_cfg, net_cfg, _full_cfg = {}, {}, {}
    if args.config:
        engine_cfg, net_cfg, _full_cfg = load_yaml_config(args.config)
        log.info("Loaded config from %s", args.config)

    gc.collect()
    gc.disable()
    log.info("GC disabled (reduces prefill latency variance)")
    log.info("Serialization: %s", serialization_name())

    model_path = cfg(engine_cfg, "model", "MODEL_PATH", "")
    if not model_path:
        raise ValueError("model must be configured in the YAML or MODEL_PATH")
    served_model_name = cfg(engine_cfg, "served_model_name",
                            "SERVED_MODEL_NAME", "llama2-70b")
    tp_size = cfg(engine_cfg, "tensor_parallel_size", "TP_SIZE", "1", int)
    dp_size = cfg(engine_cfg, "data_parallel_size", "DP_SIZE", "8", int)
    max_model_len = cfg(engine_cfg, "max_model_len", "MAX_MODEL_LEN", "2048", int)
    max_num_seqs = cfg(engine_cfg, "max_num_seqs", "MAX_NUM_SEQS", "2048", int)
    max_batched_tokens = cfg(engine_cfg, "max_num_batched_tokens",
                             "MAX_BATCHED_TOKENS", "49152", int)
    gpu_mem_util = cfg(engine_cfg, "gpu_memory_utilization",
                       "GPU_MEM_UTIL", "0.92", float)
    quantization = cfg(engine_cfg, "quantization", "QUANTIZATION", "", str) or None
    model_seed = cfg(engine_cfg, "seed", "MODEL_SEED", "0", int)

    zmq_pull_port = cfg(net_cfg, "zmq_pull_port", "ZMQ_PULL_PORT", "5555", int)
    zmq_push_port = cfg(net_cfg, "zmq_push_port", "ZMQ_PUSH_PORT", "5556", int)
    num_workers = int(os.environ.get("NUM_WORKERS", str(max_num_seqs)))

    enable_prefix_caching = cfg(engine_cfg, "enable_prefix_caching",
                                "ENABLE_PREFIX_CACHING", "0",
                                lambda v: str(v).lower() in ("1", "true"))
    require_prefix_caching_disabled(enable_prefix_caching)
    enable_chunked_prefill = cfg(engine_cfg, "enable_chunked_prefill",
                                 "ENABLE_CHUNKED_PREFILL", "1",
                                 lambda v: str(v).lower() in ("1", "true"))
    enable_expert_parallel = cfg(engine_cfg, "enable_expert_parallel",
                                 "ENABLE_EXPERT_PARALLEL", "false",
                                 lambda v: str(v).lower() in ("1", "true"))
    enable_eplb = cfg(engine_cfg, "enable_eplb", "ENABLE_EPLB", "false",
                      lambda v: str(v).lower() in ("1", "true"))
    all2all_backend = cfg(engine_cfg, "all2all_backend", "ALL2ALL_BACKEND", "")
    moe_backend = str(cfg(engine_cfg, "moe_backend", "MOE_BACKEND", "")).strip()
    kv_role = cfg(engine_cfg, "kv_role", "KV_ROLE", "kv_producer")
    cudagraph_mode = cfg(engine_cfg, "cudagraph_mode",
                         "CUDAGRAPH_MODE", "PIECEWISE")

    default_compile = [2**i for i in range(1, 13) if 2**i <= max_batched_tokens]
    compile_sizes = engine_cfg.get("compile_sizes", None)
    if compile_sizes is None:
        env_cs = os.environ.get("COMPILE_SIZES", "")
        compile_sizes = json.loads(env_cs) if env_cs else default_compile
    compile_sizes = [s for s in compile_sizes if s <= max_batched_tokens]

    capture_range = engine_cfg.get("cudagraph_capture_range", None)
    if capture_range is None:
        env_cr = os.environ.get("CUDAGRAPH_CAPTURE_RANGE", "")
        capture_range = json.loads(env_cr) if env_cr else [4, 2, 1]

    log.info("Initializing vLLM engine: model=%s TP=%d DP=%d", model_path, tp_size, dp_size)

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.config import CompilationConfig

    capture_sizes = build_capture_sizes(capture_range)

    compilation_cfg = CompilationConfig(
        cudagraph_mode=cudagraph_mode,
        compile_sizes=compile_sizes,
        cudagraph_capture_sizes=capture_sizes,
    )

    block_size = cfg(engine_cfg, "block_size", "BLOCK_SIZE", "16", int)
    enforce_eager = cfg(engine_cfg, "enforce_eager", "ENFORCE_EAGER", "false",
                        lambda v: str(v).lower() in ("1", "true"))
    disable_sliding_window = cfg(engine_cfg, "disable_sliding_window",
                                 "DISABLE_SLIDING_WINDOW", "false",
                                 lambda v: str(v).lower() in ("1", "true"))
    disable_hybrid_kv_cache_manager = optional_bool_cfg(
        engine_cfg,
        "disable_hybrid_kv_cache_manager",
        "DISABLE_HYBRID_KV_CACHE_MANAGER",
    )
    async_scheduling = cfg(engine_cfg, "async_scheduling", "ASYNC_SCHEDULING",
                           "true",
                           lambda v: str(v).lower() in ("1", "true"))

    engine_args = AsyncEngineArgs(
        model=model_path,
        served_model_name=served_model_name,
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        seed=model_seed,
        dtype=cfg(engine_cfg, "dtype", "DTYPE", "bfloat16"),
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        block_size=block_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        enforce_eager=enforce_eager,
        disable_sliding_window=disable_sliding_window,
        **({"disable_hybrid_kv_cache_manager": disable_hybrid_kv_cache_manager}
           if disable_hybrid_kv_cache_manager is not None else {}),
        enable_expert_parallel=enable_expert_parallel,
        enable_eplb=enable_eplb,
        **({"all2all_backend": all2all_backend} if all2all_backend else {}),
        **({"moe_backend": moe_backend} if moe_backend else {}),
        kv_cache_dtype=cfg(engine_cfg, "kv_cache_dtype", "KV_CACHE_DTYPE", "fp8"),
        **({"quantization": quantization} if quantization else {}),
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_batched_tokens,
        disable_log_stats=True,
        enable_log_requests=False,
        async_scheduling=async_scheduling,
        compilation_config=compilation_cfg,
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_role": kv_role,
            "kv_buffer_device": "cuda",
            "kv_connector_extra_config": nixl_kv_connector_extra_config(),
        },
    )

    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER)

    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        disable_log_stats=engine_args.disable_log_stats,
        enable_log_requests=engine_args.enable_log_requests,
    )
    log.info("Engine initialized.")

    import zmq
    import zmq.asyncio

    ctx = zmq.asyncio.Context()

    pull_sock = ctx.socket(zmq.PULL)
    pull_sock.setsockopt(zmq.RCVHWM, 8192)
    pull_sock.bind(f"tcp://*:{zmq_pull_port}")

    push_sock = ctx.socket(zmq.PUSH)
    push_sock.setsockopt(zmq.SNDHWM, 4096)
    push_sock.bind(f"tcp://*:{zmq_push_port}")

    log.info("ZMQ ready: PULL=tcp://*:%d  PUSH=tcp://*:%d", zmq_pull_port, zmq_push_port)

    
    
    
    
    
    
    
    
    decode_forward_addrs = os.environ.get("DECODE_FORWARD_ADDRS", "")
    decode_forward_port = int(os.environ.get("DECODE_FORWARD_PORT", "5557"))
    decode_push = None
    if decode_forward_addrs:
        decode_push = ctx.socket(zmq.PUSH)
        decode_push.setsockopt(zmq.SNDHWM, 4096)
        connected = []
        for raw in decode_forward_addrs.split(","):
            addr = raw.strip()
            if not addr:
                continue
            if ":" in addr:
                host, port_str = addr.rsplit(":", 1)
                try:
                    port = int(port_str)
                except ValueError:
                    host, port = addr, decode_forward_port
            else:
                host, port = addr, decode_forward_port
            decode_push.connect(f"tcp://{host}:{port}")
            connected.append(f"{host}:{port}")
        log.info("Direct forwarding to %d decode endpoint(s): %s",
                 len(connected), ", ".join(connected))
    else:
        log.info("No DECODE_FORWARD_ADDRS set — SUT will relay to decode")

    n_completed = 0
    t0 = time.time()
    work_queue = asyncio.Queue()
    shutdown = asyncio.Event()
    append_first_mode = os.environ.get(
        "PD_APPEND_FIRST_TOKEN_TO_DECODE_PROMPT", "1").strip().lower()
    first_token_source = os.environ.get(
        "PD_FIRST_TOKEN_SOURCE", "prefill").strip().lower()
    if first_token_source not in ("prefill", "decode"):
        log.warning("Unknown PD_FIRST_TOKEN_SOURCE=%s; using prefill",
                    first_token_source)
        first_token_source = "prefill"
    report_prefill_first = first_token_source == "prefill"
    append_first_auto = append_first_mode == "auto"
    append_first_to_decode_prompt = (
        report_prefill_first
        and append_first_mode not in ("0", "false", "no", "off"))
    decode_block_size = int(os.environ.get("BLOCK_SIZE", "16"))

    log.info("Workers: %d", num_workers)
    log.info("First token source: %s", first_token_source)
    log.info("Decode prompt includes prefill token: %s",
             "auto" if append_first_auto else append_first_to_decode_prompt)

    async def reader():
        while True:
            raw = await pull_sock.recv()
            msg = unpack(raw)
            if msg.get("type") == "shutdown":
                log.info("Shutdown signal received.")
                shutdown.set()
                for _ in range(num_workers):
                    await work_queue.put(None)
                return
            await work_queue.put(msg)

    gen_times = []
    send_times = []

    async def worker(wid):
        nonlocal n_completed
        while True:
            msg = await work_queue.get()
            if msg is None:
                return
            try:
                request_id = msg["id"]
                prompt_token_ids = msg["prompt"]
                decode_max_tokens = msg.get("decode_max_tokens")

                prefill_max_tokens = 1
                if (append_first_to_decode_prompt and append_first_auto
                        and len(prompt_token_ids) % decode_block_size == 0):
                    
                    
                    
                    prefill_max_tokens = 2

                params = SamplingParams(
                    max_tokens=prefill_max_tokens,
                    temperature=0.0,
                    seed=model_seed,
                    extra_args={
                        "kv_transfer_params": {
                            "do_remote_decode": True,
                            "do_remote_prefill": False,
                            "remote_engine_id": None,
                            "remote_block_ids": None,
                            "remote_host": None,
                            "remote_port": None,
                        }
                    },
                )

                t_gen = time.time()
                final_output = None
                async for output in engine.generate(
                    {"prompt_token_ids": prompt_token_ids},
                    params,
                    request_id,
                ):
                    final_output = output
                gen_times.append(time.time() - t_gen)

                kv = {}
                first_token_ids = []
                if final_output is not None:
                    kv = getattr(final_output, "kv_transfer_params", None) or {}
                    if final_output.outputs:
                        generated_token_ids = list(final_output.outputs[0].token_ids)
                        first_token_ids = generated_token_ids[:1]

                t_send = time.time()
                if decode_push is not None:
                    
                    
                    decode_prompt_token_ids = prompt_token_ids
                    decode_token_budget = decode_max_tokens
                    if first_token_ids and append_first_to_decode_prompt:
                        should_append_first = True
                        if append_first_auto:
                            
                            
                            
                            should_append_first = True
                        if should_append_first:
                            decode_prompt_token_ids = (
                                list(prompt_token_ids) + list(first_token_ids))
                            if decode_max_tokens is not None:
                                decode_token_budget = max(
                                    1, int(decode_max_tokens) -
                                    len(first_token_ids))
                    decode_payload = {
                        "id": request_id,
                        "prompt": decode_prompt_token_ids,
                        "first_token_ids": (
                            first_token_ids if report_prefill_first else []),
                        "kv_transfer_params": kv,
                        "emit_first_token": first_token_source == "decode",
                    }
                    if decode_token_budget is not None:
                        decode_payload["decode_max_tokens"] = decode_token_budget
                    decode_msg = pack(decode_payload)
                    sut_msg = pack({
                        "id": request_id,
                        "first_token_ids": first_token_ids,
                    })
                    
                    
                    
                    if report_prefill_first:
                        await push_sock.send(sut_msg)
                    await decode_push.send(decode_msg)
                else:
                    await push_sock.send(pack({
                        "id": request_id,
                        "kv_transfer_params": kv,
                        "first_token_ids": first_token_ids,
                    }))
                send_times.append(time.time() - t_send)

                n_completed += 1
                if n_completed <= 5:
                    log.info(
                        "Prefilled %s engine=%s port=%s",
                        request_id,
                        str(kv.get("remote_engine_id", "?"))[:20],
                        kv.get("remote_port", "?"),
                    )
                if n_completed <= 3 or n_completed % DIAG_INTERVAL == 0:
                    elapsed = time.time() - t0
                    qd = work_queue.qsize()
                    if len(gen_times) >= 10:
                        gt = gen_times[-DIAG_INTERVAL:]
                        st = send_times[-DIAG_INTERVAL:]
                        log.info(
                            "Prefilled %d (%.1f pps, %.1fs) queue=%d "
                            "gen_ms: avg=%.1f p50=%.1f p99=%.1f  send_ms: avg=%.2f",
                            n_completed, n_completed / max(elapsed, 0.001),
                            elapsed, qd,
                            sum(gt) / len(gt) * 1000,
                            sorted(gt)[len(gt) // 2] * 1000,
                            sorted(gt)[int(len(gt) * 0.99)] * 1000,
                            sum(st) / len(st) * 1000,
                        )
                    else:
                        log.info(
                            "Prefilled %d (%.1f pps, %.1fs) queue=%d",
                            n_completed, n_completed / max(elapsed, 0.001),
                            elapsed, qd,
                        )
            except Exception:
                log.exception("Error handling prefill request %s",
                              msg.get("id", "?"))

    log.info("Prefill worker started. Waiting for requests...")

    try:
        tasks = [asyncio.create_task(reader())]
        tasks += [asyncio.create_task(worker(i)) for i in range(num_workers)]
        await asyncio.gather(*tasks)
    finally:
        log.info("Prefill worker shutting down. Total prefilled: %d", n_completed)
        engine.shutdown()
        pull_sock.close()
        push_sock.close()
        if decode_push is not None:
            decode_push.close()
        ctx.term()


def _run_main():
    if _HAS_UVLOOP:
        log.info("prefill worker: using uvloop event loop")
        uvloop.install()
    asyncio.run(main())


if __name__ == "__main__":
    _run_main()
