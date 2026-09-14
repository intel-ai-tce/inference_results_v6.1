
"""
ZMQ-based decode worker.

Usage:
  python3 -m src.workers.decode --config <path.yaml>
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
log = logging.getLogger("decode_worker")

GC_INTERVAL = int(os.environ.get("DECODE_GC_INTERVAL", "100000"))
DIAG_INTERVAL = int(os.environ.get("DECODE_DIAG_INTERVAL", "500"))


def optional_int_cfg(engine, key, env_var, default=None):
    if key in engine:
        val = engine[key]
    else:
        val = os.environ.get(env_var, default)
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in ("", "none", "null"):
        return None
    return int(val)


def assemble_pd_token_ids(token_ids, first_token_ids, max_tokens):
    """Merge the prefill token and enforce the request's output-token cap."""
    assembled = list(token_ids)
    if first_token_ids:
        
        
        
        first = list(first_token_ids)
        if assembled[:len(first)] != first:
            assembled = first + assembled

    
    
    
    if max_tokens is not None:
        assembled = assembled[:max(1, int(max_tokens))]
    return assembled

GPTOSS_EAGLE3_REFERENCE = "nvidia/gpt-oss-120b-Eagle3-long-context"


def get_speculative_config(engine_cfg):
    method = str(cfg(
        engine_cfg, "speculative_method", "SPECULATIVE_METHOD", "")).strip()
    if not method:
        return None

    scenario = str(cfg(
        engine_cfg, "mlperf_scenario", "MLPERF_SCENARIO", "")).strip().lower()
    if scenario != "interactive":
        raise ValueError(
            "GPT-OSS speculative decoding is restricted to the MLPerf "
            f"Interactive scenario; active scenario is {scenario or 'unset'}")

    model = str(cfg(
        engine_cfg, "speculative_model", "SPECULATIVE_MODEL", "")).strip()
    reference = str(cfg(
        engine_cfg,
        "speculative_model_reference",
        "SPECULATIVE_MODEL_REFERENCE",
        "",
    )).strip()
    num_tokens = cfg(
        engine_cfg,
        "speculative_num_tokens",
        "SPECULATIVE_NUM_TOKENS",
        "0",
        int,
    )
    eagle_topk = cfg(
        engine_cfg,
        "speculative_eagle_topk",
        "SPECULATIVE_EAGLE_TOPK",
        "0",
        float,
    )
    draft_sample_method = str(cfg(
        engine_cfg,
        "speculative_draft_sample_method",
        "SPECULATIVE_DRAFT_SAMPLE_METHOD",
        "",
    )).strip().lower()

    expected = {
        "method": "eagle3",
        "reference": GPTOSS_EAGLE3_REFERENCE,
        "num_tokens": 3,
        "eagle_topk": 1.0,
        "draft_sample_method": "greedy",
    }
    actual = {
        "method": method.lower(),
        "reference": reference,
        "num_tokens": num_tokens,
        "eagle_topk": eagle_topk,
        "draft_sample_method": draft_sample_method,
    }
    mismatches = [
        f"{name}={actual[name]!r} (expected {value!r})"
        for name, value in expected.items()
        if actual[name] != value
    ]
    if not model:
        mismatches.append("model path is empty")
    if mismatches:
        raise ValueError(
            "Non-compliant GPT-OSS EAGLE3 configuration: "
            + "; ".join(mismatches))

    
    
    
    log.info(
        "GPT-OSS EAGLE3 enabled: reference=%s steps=%d topk=%.1f "
        "draft_sample_method=%s model=%s",
        reference,
        num_tokens,
        eagle_topk,
        draft_sample_method,
        model,
    )
    return {
        "model": model,
        "method": "eagle3",
        "num_speculative_tokens": num_tokens,
        "draft_sample_method": draft_sample_method,
    }


async def main():
    parser = argparse.ArgumentParser(description="ZMQ decode worker")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()

    engine_cfg, net_cfg, _full_cfg = {}, {}, {}
    if args.config:
        engine_cfg, net_cfg, _full_cfg = load_yaml_config(args.config)
        log.info("Loaded config from %s", args.config)

    model_path = cfg(engine_cfg, "model", "MODEL_PATH", "")
    if not model_path:
        raise ValueError("model must be configured in the YAML or MODEL_PATH")
    served_model_name = cfg(engine_cfg, "served_model_name",
                            "SERVED_MODEL_NAME", "llama2-70b")
    tp_size = cfg(engine_cfg, "tensor_parallel_size", "TP_SIZE", "1", int)
    dp_size = cfg(engine_cfg, "data_parallel_size", "DP_SIZE", "8", int)
    max_model_len = cfg(engine_cfg, "max_model_len", "MAX_MODEL_LEN", "2048", int)
    max_num_seqs = cfg(engine_cfg, "max_num_seqs", "MAX_NUM_SEQS", "3392", int)
    max_batched_tokens = cfg(engine_cfg, "max_num_batched_tokens",
                             "MAX_BATCHED_TOKENS", "65536", int)
    gpu_mem_util = cfg(engine_cfg, "gpu_memory_utilization",
                       "GPU_MEM_UTIL", "0.95", float)
    quantization = cfg(engine_cfg, "quantization", "QUANTIZATION", "", str) or None
    model_seed = cfg(engine_cfg, "seed", "MODEL_SEED", "0", int)
    enable_chunked_prefill = cfg(engine_cfg, "enable_chunked_prefill",
                                 "ENABLE_CHUNKED_PREFILL", "1",
                                 lambda v: str(v).lower() in ("1", "true"))
    disable_custom_all_reduce = cfg(engine_cfg, "disable_custom_all_reduce",
                                    "DISABLE_CUSTOM_ALL_REDUCE", "0",
                                    lambda v: str(v).lower() in ("1", "true"))

    zmq_pull_port = cfg(net_cfg, "zmq_pull_port", "ZMQ_PULL_PORT", "5557", int)
    zmq_push_port = cfg(net_cfg, "zmq_push_port", "ZMQ_PUSH_PORT", "5558", int)

    decode_max_tokens = cfg(engine_cfg, "decode_max_tokens",
                            "DECODE_MAX_TOKENS", "1024", int)
    decode_min_tokens = optional_int_cfg(engine_cfg, "decode_min_tokens",
                                         "DECODE_MIN_TOKENS", "0")
    decode_temperature = cfg(engine_cfg, "decode_temperature",
                             "DECODE_TEMPERATURE", "0.0", float)
    decode_top_k = optional_int_cfg(engine_cfg, "decode_top_k",
                                    "DECODE_TOP_K", "1")
    decode_top_p = cfg(engine_cfg, "decode_top_p", "DECODE_TOP_P", "0.001", float)

    enable_prefix_caching = cfg(engine_cfg, "enable_prefix_caching",
                                "ENABLE_PREFIX_CACHING", "0",
                                lambda v: str(v).lower() in ("1", "true"))
    require_prefix_caching_disabled(enable_prefix_caching)
    enable_expert_parallel = cfg(engine_cfg, "enable_expert_parallel",
                                 "ENABLE_EXPERT_PARALLEL", "false",
                                 lambda v: str(v).lower() in ("1", "true"))
    enable_dbo = cfg(engine_cfg, "enable_dbo", "ENABLE_DBO", "false",
                     lambda v: str(v).lower() in ("1", "true"))
    dbo_token_threshold = cfg(engine_cfg, "dbo_decode_token_threshold",
                              "DBO_DECODE_TOKEN_THRESHOLD", "256", int)
    enable_eplb = cfg(engine_cfg, "enable_eplb", "ENABLE_EPLB", "false",
                      lambda v: str(v).lower() in ("1", "true"))
    all2all_backend = cfg(engine_cfg, "all2all_backend", "ALL2ALL_BACKEND", "")
    moe_backend = str(cfg(engine_cfg, "moe_backend", "MOE_BACKEND", "")).strip()
    kv_role = cfg(engine_cfg, "kv_role", "KV_ROLE", "kv_consumer")
    cudagraph_mode = cfg(engine_cfg, "cudagraph_mode",
                         "CUDAGRAPH_MODE", "FULL_DECODE_ONLY")

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

    gc.collect()
    gc.disable()
    log.info("GC disabled (manual collect every %d completions)", GC_INTERVAL)
    log.info("Serialization: %s", serialization_name())

    log.info("Initializing vLLM engine: model=%s TP=%d DP=%d", model_path, tp_size, dp_size)

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.config import CompilationConfig

    attention_backend = str(
        engine_cfg.get(
            "attention_backend",
            os.environ.get(
                "ATTENTION_BACKEND",
                os.environ.get("VLLM_ATTENTION_BACKEND", ""),
            ),
        )
    ).strip()
    if attention_backend:
        try:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum
            attention_backend = AttentionBackendEnum[attention_backend.upper()]
        except Exception:
            log.warning("Passing raw attention backend value: %s",
                        attention_backend)
        else:
            log.info("Forcing attention backend: %s", attention_backend.name)

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
    speculative_cfg = get_speculative_config(engine_cfg)


    engine_args = AsyncEngineArgs(
        model=model_path,
        served_model_name=served_model_name,
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        seed=model_seed,
        dtype=cfg(engine_cfg, "dtype", "DTYPE", "bfloat16"),
        **({"quantization": quantization} if quantization else {}),
        **({"speculative_config": speculative_cfg} if speculative_cfg else {}),
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        block_size=block_size,
        enable_prefix_caching=enable_prefix_caching,
        kv_cache_dtype=cfg(engine_cfg, "kv_cache_dtype", "KV_CACHE_DTYPE", "fp8"),
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        enforce_eager=enforce_eager,
        disable_sliding_window=disable_sliding_window,
        **({"disable_hybrid_kv_cache_manager": disable_hybrid_kv_cache_manager}
           if disable_hybrid_kv_cache_manager is not None else {}),
        enable_expert_parallel=enable_expert_parallel,
        enable_dbo=enable_dbo,
        dbo_decode_token_threshold=dbo_token_threshold,
        enable_eplb=enable_eplb,
        **({"all2all_backend": all2all_backend} if all2all_backend else {}),
        **({"moe_backend": moe_backend} if moe_backend else {}),
        disable_custom_all_reduce=disable_custom_all_reduce,
        **({"attention_backend": attention_backend} if attention_backend else {}),
        disable_log_stats=True,
        enable_log_requests=False,
        async_scheduling=cfg(engine_cfg, "async_scheduling", "ASYNC_SCHEDULING",
                             "true",
                             lambda v: str(v).lower() in ("1", "true")),
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
    push_sock.setsockopt(zmq.SNDHWM, 8192)
    push_sock.bind(f"tcp://*:{zmq_push_port}")

    log.info("ZMQ ready: PULL=tcp://*:%d  PUSH=tcp://*:%d", zmq_pull_port, zmq_push_port)

    num_workers = int(os.environ.get("NUM_WORKERS", str(max_num_seqs)))

    n_completed = 0
    n_output_tokens = 0
    n_received = 0
    gc_counter = 0
    t0 = time.time()

    work_queue = asyncio.Queue()

    base_decode_kwargs = {
        "max_tokens": decode_max_tokens,
        "temperature": decode_temperature,
        "top_p": decode_top_p,
        "seed": model_seed,
        "ignore_eos": False,
        "detokenize": False,
    }
    if decode_top_k is not None:
        base_decode_kwargs["top_k"] = decode_top_k
    if decode_min_tokens is not None:
        base_decode_kwargs["min_tokens"] = max(0, decode_min_tokens)
    base_decode_params = SamplingParams(**base_decode_kwargs)

    first_token_source = os.environ.get(
        "PD_FIRST_TOKEN_SOURCE", "prefill").strip().lower()
    emit_first_token_default = first_token_source == "decode"

    log.info("Workers: %d", num_workers)
    log.info("First token source: %s", first_token_source)

    async def reader():
        nonlocal n_received
        while True:
            raw = await pull_sock.recv()
            msg = unpack(raw)
            if msg.get("type") == "shutdown":
                log.info("Shutdown signal received.")
                for _ in range(num_workers):
                    await work_queue.put(None)
                return
            n_received += 1
            if n_received <= 5:
                kv_params = msg.get("kv_transfer_params", {})
                log.info(
                    "Decode req %s engine=%s port=%s blocks=%d",
                    msg["id"],
                    str(kv_params.get("remote_engine_id", "?"))[:20],
                    kv_params.get("remote_port", "?"),
                    len(kv_params.get("remote_block_ids", [])),
                )
            elif n_received % 1000 == 0:
                log.info("Received %d decode requests", n_received)
            await work_queue.put(msg)

    gen_times = []

    async def worker(wid):
        nonlocal n_completed, n_output_tokens, gc_counter
        while True:
            msg = await work_queue.get()
            if msg is None:
                return
            try:
                request_id = msg["id"]
                prompt_token_ids = msg["prompt"]
                kv_params = msg.get("kv_transfer_params", {})
                first_token_ids = msg.get("first_token_ids", [])
                emit_first_token = bool(
                    msg.get("emit_first_token", emit_first_token_default))
                first_token_sent = False

                decode_params = base_decode_params.clone()
                if msg.get("decode_max_tokens") is not None:
                    decode_params.max_tokens = max(
                        1, int(msg["decode_max_tokens"]))
                decode_params.extra_args = {"kv_transfer_params": kv_params}

                t_gen = time.time()
                final_output = None
                async for output in engine.generate(
                    {"prompt_token_ids": prompt_token_ids},
                    decode_params,
                    request_id,
                ):
                    final_output = output
                    if (emit_first_token and not first_token_sent
                            and output is not None and output.outputs):
                        token_ids_now = list(output.outputs[0].token_ids)
                        if token_ids_now:
                            await push_sock.send(pack({
                                "id": request_id,
                                "first_token_ids": token_ids_now[:1],
                            }))
                            first_token_sent = True
                gen_times.append(time.time() - t_gen)

                token_ids = []
                if final_output is not None and final_output.outputs:
                    token_ids = list(final_output.outputs[0].token_ids)

                token_ids = assemble_pd_token_ids(
                    token_ids, first_token_ids, decode_params.max_tokens)

                if len(token_ids) <= 1:
                    log.warning(
                        "Decode request %s produced %d total token(s) "
                        "after first-token assembly; first_token_len=%d",
                        request_id, len(token_ids), len(first_token_ids))

                await push_sock.send(pack({"id": request_id, "token_ids": token_ids}))

                n_completed += 1
                n_output_tokens += len(token_ids)
                gc_counter += 1
                if GC_INTERVAL > 0 and gc_counter >= GC_INTERVAL:
                    gc.collect()
                    gc_counter = 0

                if n_completed <= 3 or n_completed % DIAG_INTERVAL == 0:
                    elapsed = time.time() - t0
                    qd = work_queue.qsize()
                    if len(gen_times) >= 10:
                        gt = gen_times[-DIAG_INTERVAL:]
                        log.info(
                            "Decoded %d (%.1f qps, %.0f tok/s, %.1fs) "
                            "queue=%d avg_gen=%.2fs avg_toks=%.0f",
                            n_completed, n_completed / max(elapsed, 0.001),
                            n_output_tokens / max(elapsed, 0.001), elapsed,
                            qd, sum(gt) / len(gt),
                            n_output_tokens / max(n_completed, 1),
                        )
                    else:
                        log.info(
                            "Decoded %d (%.1f qps, %.0f tok/s, %.1fs) queue=%d",
                            n_completed, n_completed / max(elapsed, 0.001),
                            n_output_tokens / max(elapsed, 0.001), elapsed, qd,
                        )
            except Exception:
                log.exception("Error handling decode request %s",
                              msg.get("id", "?"))

    log.info("Decode worker started. Waiting for requests...")

    try:
        tasks = [asyncio.create_task(reader())]
        tasks += [asyncio.create_task(worker(i)) for i in range(num_workers)]
        await asyncio.gather(*tasks)
    finally:
        log.info(
            "Decode worker shutting down. Total decoded: %d, tokens: %d",
            n_completed, n_output_tokens,
        )
        engine.shutdown()
        pull_sock.close()
        push_sock.close()
        ctx.term()


def _run_main():
    if _HAS_UVLOOP:
        log.info("decode worker: using uvloop event loop")
        uvloop.install()
    asyncio.run(main())


if __name__ == "__main__":
    _run_main()
