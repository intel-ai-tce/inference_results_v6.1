
"""
ZMQ-based standalone worker — full prefill + decode in a single vLLM engine.

No kv_transfer_config; no PD disaggregation.  Used for baselines.

Ports:
  PULL (receive requests)  — default 5555
  PUSH (send results)      — default 5556

Message protocol:
  Request:  {"id": str, "prompt": [int, ...]}
  Response: {"id": str, "token_ids": [int, ...]}
            or, for server/interactive (streaming):
            {"id": str, "first_token_ids": [int, ...]}   (first token notification)
            {"id": str, "token_ids": [int, ...]}          (complete)
"""

import argparse
import asyncio
import faulthandler
import gc
import inspect
import json
import logging
import os
import sys
import time

try:
    import msgpack as _msgpack
    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False




try:
    import uvloop  
    _HAS_UVLOOP = True
except ImportError:
    _HAS_UVLOOP = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("standalone_worker")

GC_INTERVAL = int(os.environ.get("DECODE_GC_INTERVAL", "100000"))
WARMUP_COUNT = int(os.environ.get("WARMUP_COUNT", "10"))





WARMUP_LARGE_BATCH = os.environ.get("WARMUP_LARGE_BATCH")
WARMUP_LARGE_TOKENS = int(os.environ.get("WARMUP_LARGE_TOKENS", "8"))
PROFILE = os.environ.get("VLLM_PROFILE", "0") in ("1", "true")

MLPERF_PREFIX_CACHING_RULE = (
    "https://github.com/mlcommons<submission-root>_policies/blob/master/"
    "inference_rules.adoc#L894-L904"
)

_WARMUP_PROMPT = [
    1, 365, 3668, 23421, 22224, 7845, 27315, 29892, 5178, 312, 300,
    332, 594, 666, 275, 3277, 560, 277, 29889, 478, 342, 747, 352,
    398, 15937, 598, 2148, 2073, 398, 5065, 1056, 29892, 321, 657,
    9657, 398, 14172, 2497, 298, 355, 2872, 277, 263, 29889, 315,
    3417, 302, 747, 29882, 7866, 29877, 29892, 15937, 598, 7845,
    27315, 13081, 375, 7845, 27315, 29892, 782, 2801, 885, 7367,
    275, 802, 3737, 29889,
]


def _pack(obj):
    if _HAS_MSGPACK:
        return _msgpack.packb(obj, use_bin_type=True)
    return json.dumps(obj, separators=(',', ':')).encode()


def _unpack(data):
    if isinstance(data, (bytes, bytearray)) and data and data[0] not in (0x7b, 0x5b):
        if _HAS_MSGPACK:
            return _msgpack.unpackb(data, raw=False)
    return json.loads(data)


def _cfg(engine, key, env_var, default, typ=str):
    if key in engine:
        return typ(engine[key])
    return typ(os.environ.get(env_var, default))


def _optional_int_cfg(engine, key, env_var, default=None):
    if key in engine:
        val = engine[key]
    else:
        val = os.environ.get(env_var, default)
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in ("", "none", "null"):
        return None
    return int(val)


def _load_yaml_config(path):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("engine", {}), cfg.get("network", {}), cfg


def _build_capture_sizes(range_spec):
    sizes = []
    for item in range_spec:
        if isinstance(item, list):
            sizes.extend(range(*item))
        else:
            sizes.append(item)
    return sizes


def _apply_env_config(full_cfg):
    """Apply scalar model environment overrides before importing vLLM."""
    env_cfg = full_cfg.get("vllm_env_config", {})
    if not isinstance(env_cfg, dict):
        return
    for key, value in env_cfg.items():
        if isinstance(value, dict):
            continue
        if value is None or value == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)


def _load_generation_stop_token_ids(model_path):
    path = os.path.join(model_path, "generation_config.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    eos = cfg.get("eos_token_id")
    if eos is None:
        return []
    if not isinstance(eos, list):
        eos = [eos]

    stop_ids = []
    for token_id in eos:
        try:
            token_id = int(token_id)
        except (TypeError, ValueError):
            continue
        if token_id not in stop_ids:
            stop_ids.append(token_id)
    return stop_ids















DEEPSEEK_MTP_REQUIRED = {
    "num_tokens": 3,        
    "eagle_topk": 1.0,      
    "draft_sample_method": "greedy",
}





GPTOSS_EAGLE3_REFERENCE = "nvidia/gpt-oss-120b-Eagle3-long-context"
GPTOSS_EAGLE3_REQUIRED = {
    "num_tokens": 3,
    "eagle_topk": 1.0,
    "draft_sample_method": "greedy",
}


def _get_gptoss_eagle3_speculative_config(engine_cfg):
    """Build the policy-mandated GPT-OSS Interactive EAGLE3 configuration."""
    scenario = _cfg(
        engine_cfg, "mlperf_scenario", "MLPERF_SCENARIO", ""
    ).strip().lower()
    if scenario != "interactive":
        raise ValueError(
            "GPT-OSS-120B EAGLE3 speculative decoding is restricted to the "
            "MLPerf Interactive scenario (inference_policies appendix-"
            f"speculative-decoding); active scenario is {scenario or 'unset'}. "
            "Launch with --scenario interactive.")

    spec_model = _cfg(
        engine_cfg, "speculative_model", "SPECULATIVE_MODEL", "").strip()
    reference = _cfg(
        engine_cfg,
        "speculative_model_reference",
        "SPECULATIVE_MODEL_REFERENCE",
        "",
    ).strip()
    num_tokens = _cfg(
        engine_cfg, "speculative_num_tokens", "SPECULATIVE_NUM_TOKENS", "0", int)
    eagle_topk = _cfg(
        engine_cfg, "speculative_eagle_topk", "SPECULATIVE_EAGLE_TOPK", "0", float)
    draft_sample_method = _cfg(
        engine_cfg,
        "speculative_draft_sample_method",
        "SPECULATIVE_DRAFT_SAMPLE_METHOD",
        "",
    ).strip().lower()

    actual = {
        "num_tokens": num_tokens,
        "eagle_topk": eagle_topk,
        "draft_sample_method": draft_sample_method,
    }
    mismatches = [
        f"{name}={actual[name]!r} (MLCommons requires {value!r})"
        for name, value in GPTOSS_EAGLE3_REQUIRED.items()
        if actual[name] != value
    ]
    if not spec_model:
        mismatches.append("speculative_model must name the reference EAGLE3 checkpoint")
    if reference != GPTOSS_EAGLE3_REFERENCE:
        mismatches.append(
            "speculative_model_reference="
            f"{reference!r} (MLCommons requires {GPTOSS_EAGLE3_REFERENCE!r})")
    if mismatches:
        raise ValueError(
            "Non-compliant GPT-OSS-120B EAGLE3 speculative-decoding "
            "configuration (inference_policies appendix-speculative-decoding): "
            + "; ".join(mismatches))

    log.info(
        "GPT-OSS-120B EAGLE3 speculative decoding ENABLED (MLCommons "
        "Interactive): model=%s reference=%s num_speculative_tokens=%d "
        "eagle_topk=%.1f draft_sample_method=%s",
        spec_model, reference, num_tokens, eagle_topk, draft_sample_method,
    )
    return {
        "model": spec_model,
        "method": "eagle3",
        "num_speculative_tokens": num_tokens,
        "draft_sample_method": draft_sample_method,
    }


def get_speculative_config(engine_cfg):
    """Build a compliant standalone speculative_config, or None.

    Returns None when no speculative method is configured. Raises when the
    configuration deviates from a MLCommons-mandated Interactive profile so a
    non-compliant run can never start.
    """
    method = _cfg(
        engine_cfg, "speculative_method", "SPECULATIVE_METHOD", ""
    ).strip().lower()
    if not method:
        return None
    if method == "eagle3":
        return _get_gptoss_eagle3_speculative_config(engine_cfg)

    scenario = _cfg(
        engine_cfg, "mlperf_scenario", "MLPERF_SCENARIO", ""
    ).strip().lower()
    if scenario != "interactive":
        raise ValueError(
            "DeepSeek-R1 speculative decoding is restricted to the MLPerf "
            "Interactive scenario (inference_policies appendix-speculative-"
            f"decoding); active scenario is {scenario or 'unset'}. Launch with "
            "--scenario interactive.")

    if method not in ("deepseek_mtp", "mtp"):
        raise ValueError(
            "Standalone speculative decoding only supports the compliant "
            "DeepSeek-R1 MTP profile (method=deepseek_mtp); got "
            f"{method!r}.")

    num_tokens = _cfg(
        engine_cfg, "speculative_num_tokens", "SPECULATIVE_NUM_TOKENS", "0", int)
    eagle_topk = _cfg(
        engine_cfg, "speculative_eagle_topk", "SPECULATIVE_EAGLE_TOPK", "0", float)
    draft_sample_method = _cfg(
        engine_cfg,
        "speculative_draft_sample_method",
        "SPECULATIVE_DRAFT_SAMPLE_METHOD",
        "",
    ).strip().lower()
    
    
    
    spec_model = _cfg(
        engine_cfg, "speculative_model", "SPECULATIVE_MODEL", "").strip()

    actual = {
        "num_tokens": num_tokens,
        "eagle_topk": eagle_topk,
        "draft_sample_method": draft_sample_method,
    }
    mismatches = [
        f"{name}={actual[name]!r} (MLCommons requires {value!r})"
        for name, value in DEEPSEEK_MTP_REQUIRED.items()
        if actual[name] != value
    ]
    if spec_model:
        mismatches.append(
            f"speculative_model={spec_model!r} (must be empty: the reference "
            "in-checkpoint MTP head is required; a different head is disallowed)")
    if mismatches:
        raise ValueError(
            "Non-compliant DeepSeek-R1 MTP speculative-decoding configuration "
            "(inference_policies appendix-speculative-decoding): "
            + "; ".join(mismatches))

    log.info(
        "DeepSeek-R1 MTP speculative decoding ENABLED (MLCommons Interactive): "
        "method=deepseek_mtp num_speculative_tokens=%d eagle_topk=%.1f "
        "draft_sample_method=%s (reference in-checkpoint MTP head, "
        "target precision)",
        num_tokens, eagle_topk, draft_sample_method,
    )
    
    
    
    
    return {"method": "deepseek_mtp", "num_speculative_tokens": num_tokens}


async def main():
    parser = argparse.ArgumentParser(description="ZMQ standalone worker")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()

    engine_cfg, net_cfg, _full_cfg = {}, {}, {}
    if args.config:
        engine_cfg, net_cfg, _full_cfg = _load_yaml_config(args.config)
        log.info("Loaded config from %s", args.config)
        _apply_env_config(_full_cfg)

    model_path = _cfg(engine_cfg, "model", "MODEL_PATH", "")
    if not model_path:
        raise ValueError("model must be configured in the YAML or MODEL_PATH")
    served_model_name = _cfg(engine_cfg, "served_model_name",
                             "SERVED_MODEL_NAME", "llama2-70b")
    tp_size = _cfg(engine_cfg, "tensor_parallel_size", "TP_SIZE", "1", int)
    dp_size = _cfg(engine_cfg, "data_parallel_size", "DP_SIZE", "8", int)
    max_model_len = _cfg(engine_cfg, "max_model_len", "MAX_MODEL_LEN", "2048", int)
    max_num_seqs = _cfg(engine_cfg, "max_num_seqs", "MAX_NUM_SEQS", "2048", int)
    max_batched_tokens = _cfg(engine_cfg, "max_num_batched_tokens",
                              "MAX_BATCHED_TOKENS", "65536", int)
    gpu_mem_util = _cfg(engine_cfg, "gpu_memory_utilization",
                        "GPU_MEM_UTIL", "0.92", float)
    quantization = _cfg(engine_cfg, "quantization", "QUANTIZATION", "", str) or None
    model_seed = _cfg(engine_cfg, "seed", "MODEL_SEED", "0", int)

    zmq_pull_port = _cfg(net_cfg, "zmq_pull_port", "ZMQ_PULL_PORT", "5555", int)
    zmq_push_port = _cfg(net_cfg, "zmq_push_port", "ZMQ_PUSH_PORT", "5556", int)

    decode_max_tokens = _cfg(engine_cfg, "decode_max_tokens",
                             "DECODE_MAX_TOKENS", "1024", int)
    decode_min_tokens = _optional_int_cfg(engine_cfg, "decode_min_tokens",
                                          "DECODE_MIN_TOKENS", "0")
    decode_temperature = _cfg(engine_cfg, "decode_temperature",
                              "DECODE_TEMPERATURE", "0.0", float)
    decode_top_k = _optional_int_cfg(engine_cfg, "decode_top_k",
                                     "DECODE_TOP_K", "1")
    decode_top_p = _cfg(engine_cfg, "decode_top_p", "DECODE_TOP_P", "0.001", float)

    use_generation_stop_token_ids = _cfg(
        engine_cfg, "use_generation_stop_token_ids",
        "USE_GENERATION_STOP_TOKEN_IDS", "true",
        lambda v: str(v).lower() in ("1", "true", "yes"))
    generation_stop_token_ids = (
        _load_generation_stop_token_ids(model_path)
        if use_generation_stop_token_ids else [])
    if generation_stop_token_ids:
        log.info("Generation stop_token_ids: %s", generation_stop_token_ids)

    enable_prefix_caching = _cfg(engine_cfg, "enable_prefix_caching",
                                 "ENABLE_PREFIX_CACHING", "0",
                                 lambda v: str(v).lower() in ("1", "true"))
    if enable_prefix_caching:
        raise ValueError(
            "Automatic prefix caching is disallowed for MLPerf: every input "
            "query must be computed in its entirety. See "
            f"{MLPERF_PREFIX_CACHING_RULE}"
        )
    enable_chunked_prefill = _cfg(engine_cfg, "enable_chunked_prefill",
                                  "ENABLE_CHUNKED_PREFILL", "1",
                                  lambda v: str(v).lower() in ("1", "true"))
    enable_expert_parallel = _cfg(engine_cfg, "enable_expert_parallel",
                                  "ENABLE_EXPERT_PARALLEL", "false",
                                  lambda v: str(v).lower() in ("1", "true"))
    enable_eplb = _cfg(engine_cfg, "enable_eplb", "ENABLE_EPLB", "false",
                       lambda v: str(v).lower() in ("1", "true"))
    all2all_backend = _cfg(engine_cfg, "all2all_backend", "ALL2ALL_BACKEND", "")
    linear_backend = _cfg(engine_cfg, "linear_backend", "LINEAR_BACKEND", "auto", str).strip()
    cudagraph_mode = _cfg(engine_cfg, "cudagraph_mode",
                          "CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
    compilation_mode_raw = _cfg(engine_cfg, "compilation_mode",
                                "COMPILATION_MODE", "", str)
    block_size = _cfg(engine_cfg, "block_size", "BLOCK_SIZE", "16", int)
    enforce_eager = _cfg(engine_cfg, "enforce_eager", "ENFORCE_EAGER", "false",
                         lambda v: str(v).lower() in ("1", "true"))
    disable_sliding_window = _cfg(engine_cfg, "disable_sliding_window",
                                  "DISABLE_SLIDING_WINDOW", "false",
                                  lambda v: str(v).lower() in ("1", "true"))
    trust_remote_code = _cfg(engine_cfg, "trust_remote_code",
                             "TRUST_REMOTE_CODE", "false",
                             lambda v: str(v).lower() in ("1", "true"))
    async_scheduling = _cfg(engine_cfg, "async_scheduling", "ASYNC_SCHEDULING",
                            "true",
                            lambda v: str(v).lower() in ("1", "true"))
    disable_nccl_for_dp_synchronization = _cfg(
        engine_cfg,
        "disable_nccl_for_dp_synchronization",
        "DISABLE_NCCL_FOR_DP_SYNCHRONIZATION",
        "false",
        lambda v: str(v).lower() in ("1", "true"),
    )
    cudagraph_metrics = _cfg(
        engine_cfg,
        "cudagraph_metrics",
        "CUDAGRAPH_METRICS",
        "false",
        lambda v: str(v).lower() in ("1", "true"),
    )

    default_compile = [2**i for i in range(1, 13) if 2**i <= max_batched_tokens]
    compile_sizes = engine_cfg.get("compile_sizes", None)
    if compile_sizes is None and os.environ.get("COMPILE_SIZES"):
        import json as _json
        compile_sizes = _json.loads(os.environ["COMPILE_SIZES"])
    if compile_sizes is None:
        compile_sizes = default_compile
    compile_sizes = [s for s in compile_sizes if s <= max_batched_tokens]

    capture_range = engine_cfg.get("cudagraph_capture_range", None)
    if capture_range is None and os.environ.get("CUDAGRAPH_CAPTURE_RANGE"):
        import json as _json
        capture_range = _json.loads(os.environ["CUDAGRAPH_CAPTURE_RANGE"])
    if capture_range is None:
        capture_range = [4, 2, 1]

    if str(cudagraph_mode).upper() == "NONE":
        compile_sizes = []
        capture_range = []

    disable_custom_all_reduce = _cfg(
        engine_cfg, "disable_custom_all_reduce",
        "DISABLE_CUSTOM_ALL_REDUCE", "false",
        lambda v: str(v).lower() in ("1", "true"))
    calculate_kv_scales = _cfg(engine_cfg, "calculate_kv_scales",
                               "CALCULATE_KV_SCALES", "false",
                               lambda v: str(v).lower() in ("1", "true"))
    moe_backend = str(_cfg(engine_cfg, "moe_backend", "MOE_BACKEND", "")).strip()
    standalone_engine_backend = str(_cfg(
        engine_cfg,
        "standalone_engine_backend",
        "STANDALONE_ENGINE_BACKEND",
        "async_llm",
    )).strip().lower()
    if standalone_engine_backend in ("", "async"):
        standalone_engine_backend = "async_llm"
    if standalone_engine_backend not in ("async_llm", "sync_llm"):
        raise ValueError(
            "STANDALONE_ENGINE_BACKEND must be 'async_llm' or 'sync_llm', "
            f"got {standalone_engine_backend!r}"
        )

    gc.collect()
    gc.disable()
    log.info("GC disabled (manual collect every %d completions)", GC_INTERVAL)
    log.info("Serialization: %s", "msgpack" if _HAS_MSGPACK else "json")

    log.info("Initializing vLLM engine (standalone): model=%s TP=%d DP=%d",
             model_path, tp_size, dp_size)

    
    
    
    runtime_patch_names = _full_cfg.get("runtime_patches", [])
    is_moe = bool(_full_cfg.get("moe", False))
    if runtime_patch_names:
        try:
            from runtime_patches import prepare_runtime_for_standalone
        except ModuleNotFoundError:
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from runtime_patches import prepare_runtime_for_standalone
        applied = prepare_runtime_for_standalone(runtime_patch_names, do_aiter=is_moe)
        log.info("Applied standalone runtime patches: %s", applied)

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CompilationMode

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

    compilation_mode = None
    if compilation_mode_raw:
        try:
            compilation_mode = int(compilation_mode_raw)
        except ValueError:
            compilation_mode = CompilationMode[compilation_mode_raw]

    capture_sizes = _build_capture_sizes(capture_range)
    compilation_cfg = CompilationConfig(
        mode=compilation_mode,
        cudagraph_mode=cudagraph_mode,
        compile_sizes=compile_sizes,
        cudagraph_capture_sizes=capture_sizes,
    )

    
    
    
    speculative_cfg = get_speculative_config(engine_cfg)

    if standalone_engine_backend == "sync_llm":
        log.warning(
            "Using sync_llm standalone backend. This is a correctness/debug "
            "path and does not stream first-token timing for performance runs."
        )
        from vllm import LLM
        import zmq
        import zmq.asyncio

        compilation_config_arg = compilation_cfg
        if str(cudagraph_mode).upper() == "NONE":
            compilation_config_arg = {
                "cudagraph_mode": "NONE",
                "cudagraph_capture_sizes": [],
                "max_cudagraph_capture_size": 0,
            }

        
        
        
        llm_kwargs = {
            "model": model_path,
            "tokenizer": model_path,
            "tensor_parallel_size": tp_size,
            "seed": model_seed,
            "dtype": _cfg(engine_cfg, "dtype", "DTYPE", "auto"),
            "max_model_len": max_model_len,
            "kv_cache_dtype": _cfg(
                engine_cfg, "kv_cache_dtype", "KV_CACHE_DTYPE", "auto"
            ),
            "gpu_memory_utilization": gpu_mem_util,
            "block_size": block_size,
            "enable_prefix_caching": enable_prefix_caching,
            "enable_chunked_prefill": enable_chunked_prefill,
            "enforce_eager": enforce_eager,
            "disable_custom_all_reduce": disable_custom_all_reduce,
            "trust_remote_code": trust_remote_code,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_batched_tokens,
            "compilation_config": compilation_config_arg,
            "async_scheduling": async_scheduling,
            "disable_log_stats": not PROFILE,
        }
        if quantization:
            llm_kwargs["quantization"] = quantization
        if attention_backend:
            llm_kwargs["attention_backend"] = attention_backend
        if moe_backend:
            llm_kwargs["moe_backend"] = moe_backend
        if speculative_cfg:
            llm_kwargs["speculative_config"] = speculative_cfg

        t0 = time.monotonic()
        log.info("Starting vLLM LLM sync backend")
        engine = LLM(**llm_kwargs)
        log.info("LLM initialized in %.2fs.", time.monotonic() - t0)

        ctx = zmq.asyncio.Context()
        pull_sock = ctx.socket(zmq.PULL)
        pull_sock.setsockopt(zmq.RCVHWM, 8192)
        pull_sock.bind(f"tcp://*:{zmq_pull_port}")

        push_sock = ctx.socket(zmq.PUSH)
        push_sock.setsockopt(zmq.SNDHWM, 8192)
        push_sock.bind(f"tcp://*:{zmq_push_port}")

        log.info("ZMQ ready: PULL=tcp://*:%d  PUSH=tcp://*:%d",
                 zmq_pull_port, zmq_push_port)

        base_kwargs = {
            "max_tokens": decode_max_tokens,
            "temperature": decode_temperature,
            "top_p": decode_top_p,
            "seed": model_seed,
            "ignore_eos": False,
            "detokenize": False,
        }
        if decode_top_k is not None:
            base_kwargs["top_k"] = decode_top_k
        if decode_min_tokens is not None:
            base_kwargs["min_tokens"] = max(0, decode_min_tokens)
        if generation_stop_token_ids:
            base_kwargs["stop_token_ids"] = generation_stop_token_ids
        base_params = SamplingParams(**base_kwargs)

        n_completed = 0
        n_output_tokens = 0
        gc_counter = 0
        started = time.time()
        log.info("Standalone sync_llm worker started. Waiting for requests...")
        try:
            while True:
                raw = await pull_sock.recv()
                msg = _unpack(raw)
                if msg.get("type") == "shutdown":
                    log.info("Shutdown signal received.")
                    break

                if msg.get("type") == "offline_batch":
                    requests = msg.get("requests", [])
                    if not requests:
                        raise ValueError("offline_batch must contain requests")
                    prompt_inputs = []
                    request_ids = []
                    for request in requests:
                        request_ids.append(request["id"])
                        if "prompt_text" in request:
                            prompt_inputs.append(str(request["prompt_text"]))
                        else:
                            prompt_inputs.append(
                                {"prompt_token_ids": request["prompt"]})

                    params = base_params.clone()
                    if "max_tokens" in msg:
                        params.max_tokens = int(msg["max_tokens"])
                    if "min_tokens" in msg:
                        params.min_tokens = int(msg["min_tokens"])
                    if "ignore_eos" in msg:
                        params.ignore_eos = bool(msg["ignore_eos"])
                    outputs = engine.generate(
                        prompt_inputs, params, use_tqdm=False)
                    if len(outputs) != len(request_ids):
                        raise RuntimeError(
                            "offline_batch returned %d outputs for %d requests"
                            % (len(outputs), len(request_ids)))
                    results = []
                    for request_id, output in zip(request_ids, outputs):
                        token_ids = []
                        if output.outputs:
                            token_ids = list(output.outputs[0].token_ids)
                        results.append({"id": request_id, "token_ids": token_ids})

                    await push_sock.send(_pack({
                        "type": "offline_batch_result",
                        "results": results,
                    }))
                    n_completed += len(results)
                    n_output_tokens += sum(
                        len(result["token_ids"]) for result in results)
                    gc_counter += len(results)
                    if GC_INTERVAL > 0 and gc_counter >= GC_INTERVAL:
                        gc.collect()
                        gc_counter = 0
                    elapsed = time.time() - started
                    log.info(
                        "Completed Offline batch of %d (%d total, %.2f qps, "
                        "%.0f tok/s, %.1fs)",
                        len(results), n_completed,
                        n_completed / max(elapsed, 0.001),
                        n_output_tokens / max(elapsed, 0.001), elapsed,
                    )
                    continue

                request_id = msg["id"]
                if "prompt_text" in msg:
                    prompt_input = str(msg["prompt_text"])
                else:
                    prompt_input = {"prompt_token_ids": msg["prompt"]}

                params = base_params.clone()
                if "max_tokens" in msg:
                    params.max_tokens = int(msg["max_tokens"])
                if "min_tokens" in msg:
                    params.min_tokens = int(msg["min_tokens"])
                if "ignore_eos" in msg:
                    params.ignore_eos = bool(msg["ignore_eos"])

                outputs = engine.generate([prompt_input], params, use_tqdm=False)
                token_ids = []
                if outputs and outputs[0].outputs:
                    token_ids = list(outputs[0].outputs[0].token_ids)

                await push_sock.send(_pack({
                    "id": request_id,
                    "token_ids": token_ids,
                }))

                n_completed += 1
                n_output_tokens += len(token_ids)
                gc_counter += 1
                if GC_INTERVAL > 0 and gc_counter >= GC_INTERVAL:
                    gc.collect()
                    gc_counter = 0

                if n_completed <= 3 or n_completed % 50 == 0:
                    elapsed = time.time() - started
                    log.info(
                        "Completed %d (%.2f qps, %.0f tok/s, %.1fs)",
                        n_completed, n_completed / max(elapsed, 0.001),
                        n_output_tokens / max(elapsed, 0.001), elapsed,
                    )
        finally:
            log.info("Standalone sync_llm shutting down. Total: %d, tokens: %d",
                     n_completed, n_output_tokens)
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
            pull_sock.close()
            push_sock.close()
            ctx.term()
        return

    engine_kwargs = dict(
        model=model_path,
        served_model_name=served_model_name,
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        seed=model_seed,
        dtype=_cfg(engine_cfg, "dtype", "DTYPE", "bfloat16"),
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        block_size=block_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        enforce_eager=enforce_eager,
        disable_sliding_window=disable_sliding_window,
        trust_remote_code=trust_remote_code,
        enable_expert_parallel=enable_expert_parallel,
        enable_eplb=enable_eplb,
        disable_custom_all_reduce=disable_custom_all_reduce,
        kv_cache_dtype=_cfg(engine_cfg, "kv_cache_dtype", "KV_CACHE_DTYPE", "fp8"),
        calculate_kv_scales=calculate_kv_scales,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_batched_tokens,
        disable_log_stats=not PROFILE,
        enable_log_requests=False,
        cudagraph_metrics=cudagraph_metrics,
        async_scheduling=async_scheduling,
        disable_nccl_for_dp_synchronization=disable_nccl_for_dp_synchronization,
        compilation_config=compilation_cfg,
    )
    if all2all_backend:
        engine_kwargs["all2all_backend"] = all2all_backend
    if quantization:
        engine_kwargs["quantization"] = quantization
    if attention_backend:
        engine_kwargs["attention_backend"] = attention_backend
    if linear_backend:
        engine_kwargs["linear_backend"] = linear_backend
    if moe_backend:
        engine_kwargs["moe_backend"] = moe_backend
    if speculative_cfg:
        engine_kwargs["speculative_config"] = speculative_cfg

    
    
    
    
    _supported = set(inspect.signature(AsyncEngineArgs).parameters)
    _dropped = sorted(k for k in engine_kwargs if k not in _supported)
    if _dropped:
        log.warning(
            "Installed AsyncEngineArgs does not support %s; dropping for legacy "
            "vLLM compatibility (dropped values: %s)",
            _dropped, {k: engine_kwargs[k] for k in _dropped},
        )
        for k in _dropped:
            del engine_kwargs[k]
    engine_args = AsyncEngineArgs(**engine_kwargs)

    stack_dump_interval = int(os.environ.get("VLLM_CONFIG_STACK_DUMP_INTERVAL", "0"))
    if stack_dump_interval > 0:
        faulthandler.dump_traceback_later(stack_dump_interval, repeat=True)

    t0 = time.monotonic()
    log.info("Creating vLLM engine config")
    try:
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER)
    finally:
        if stack_dump_interval > 0:
            faulthandler.cancel_dump_traceback_later()
    log.info("Created vLLM engine config in %.2fs", time.monotonic() - t0)

    from vllm.v1.engine.async_llm import AsyncLLM

    t0 = time.monotonic()
    log.info("Starting AsyncLLM.from_vllm_config")
    engine = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        disable_log_stats=engine_args.disable_log_stats,
        enable_log_requests=engine_args.enable_log_requests,
    )
    log.info("Engine initialized (standalone — no KV transfer).")

    if WARMUP_COUNT > 0:
        log.info("Running warmup (exercising CUDA graphs + GEMM kernels) ...")
        warmup_kwargs = {
            "max_tokens": 4,
            "temperature": decode_temperature,
            "top_p": decode_top_p,
            "min_tokens": 1,
            "ignore_eos": False,
            "detokenize": False,
        }
        if decode_top_k is not None:
            warmup_kwargs["top_k"] = decode_top_k
        warmup_params = SamplingParams(**warmup_kwargs)
        warmup_lengths = [32, 64, 128, 256, 512, 1024, 2040]
        warmup_id = 0

        async def _warmup(rid, p):
            async for _ in engine.generate(
                {"prompt_token_ids": p},
                warmup_params,
                rid,
            ):
                pass

        for prompt_len in warmup_lengths:
            batch = min(WARMUP_COUNT, 8)
            tasks = []
            for b in range(batch):
                prompt = _WARMUP_PROMPT * ((prompt_len // len(_WARMUP_PROMPT)) + 1)
                prompt = prompt[:prompt_len]
                warmup_id += 1
                tasks.append(_warmup(f"_warmup_{warmup_id}", prompt))
            await asyncio.gather(*tasks)

        
        
        
        
        try:
            big_n = (int(WARMUP_LARGE_BATCH) if WARMUP_LARGE_BATCH is not None
                     else max(64, max_num_seqs // 2))
        except (TypeError, ValueError):
            big_n = max(64, max_num_seqs // 2)
        if big_n > 0:
            log.info("Large-batch warmup: %d concurrent x %d tokens "
                     "(prompt_len=256) ...", big_n, WARMUP_LARGE_TOKENS)
            big_kwargs = {
                "max_tokens": max(1, WARMUP_LARGE_TOKENS),
                "temperature": decode_temperature,
                "top_p": decode_top_p,
                "min_tokens": 1,
                "ignore_eos": True,  
                "detokenize": False,
            }
            if decode_top_k is not None:
                big_kwargs["top_k"] = decode_top_k
            big_params = SamplingParams(**big_kwargs)
            big_prompt = (_WARMUP_PROMPT * ((256 // len(_WARMUP_PROMPT)) + 1))[:256]
            big_tasks = []
            for b in range(big_n):
                warmup_id += 1

                async def _big(rid):
                    async for _ in engine.generate(
                        {"prompt_token_ids": big_prompt}, big_params, rid,
                    ):
                        pass
                big_tasks.append(_big(f"_warmup_big_{warmup_id}"))
            await asyncio.gather(*big_tasks)

        gc.collect()
        log.info("Warmup complete (%d prompt lengths, %d total requests).",
                 len(warmup_lengths), warmup_id)

    import zmq
    import zmq.asyncio

    ctx = zmq.asyncio.Context()

    pull_sock = ctx.socket(zmq.PULL)
    pull_sock.setsockopt(zmq.RCVHWM, 8192)
    pull_sock.bind(f"tcp://*:{zmq_pull_port}")

    push_sock = ctx.socket(zmq.PUSH)
    push_sock.setsockopt(zmq.SNDHWM, 8192)
    push_sock.bind(f"tcp://*:{zmq_push_port}")

    log.info("ZMQ ready: PULL=tcp://*:%d  PUSH=tcp://*:%d",
             zmq_pull_port, zmq_push_port)

    num_workers = int(os.environ.get("NUM_WORKERS", str(max_num_seqs)))

    n_completed = 0
    n_output_tokens = 0
    gc_counter = 0
    t0 = time.time()

    work_queue = asyncio.Queue()

    if PROFILE:
        _prof_generate_wait_ns = 0
        _prof_zmq_send_ns = 0
        _prof_n_generate_calls = 0
        _prof_n_yields = 0
        _prof_queue_depth_samples = 0
        _prof_queue_depth_sum = 0
        _prof_window_tokens = 0
        _prof_stream_tokens = 0
        _prof_window_t0 = time.monotonic()
        log.info("PROFILE MODE ON: collecting per-request timing")

    base_kwargs = {
        "max_tokens": decode_max_tokens,
        "temperature": decode_temperature,
        "top_p": decode_top_p,
        "seed": model_seed,
        "ignore_eos": False,
        "detokenize": False,
    }
    if decode_top_k is not None:
        base_kwargs["top_k"] = decode_top_k
    if decode_min_tokens is not None:
        base_kwargs["min_tokens"] = max(0, decode_min_tokens)
    if generation_stop_token_ids:
        base_kwargs["stop_token_ids"] = generation_stop_token_ids
    base_params = SamplingParams(**base_kwargs)

    log.info("Workers: %d", num_workers)

    async def reader():
        while True:
            raw = await pull_sock.recv()
            msg = _unpack(raw)
            if msg.get("type") == "shutdown":
                log.info("Shutdown signal received.")
                for _ in range(num_workers):
                    await work_queue.put(None)
                return
            if msg.get("type") == "offline_batch":
                
                
                
                
                requests = msg.get("requests", [])
                if not requests:
                    raise ValueError("offline_batch must contain requests")
                overrides = {
                    key: msg[key]
                    for key in ("max_tokens", "min_tokens", "ignore_eos")
                    if key in msg
                }
                for request in requests:
                    request_msg = dict(request)
                    for key, value in overrides.items():
                        request_msg.setdefault(key, value)
                    await work_queue.put(request_msg)
                continue

            await work_queue.put(msg)

    async def worker(wid):
        nonlocal n_completed, n_output_tokens, gc_counter
        if PROFILE:
            nonlocal _prof_generate_wait_ns, _prof_zmq_send_ns
            nonlocal _prof_n_generate_calls, _prof_n_yields
            nonlocal _prof_queue_depth_samples, _prof_queue_depth_sum
            nonlocal _prof_window_tokens, _prof_stream_tokens
        while True:
            msg = await work_queue.get()
            if msg is None:
                return
            try:
                request_id = msg["id"]
                if "prompt_text" in msg:
                    prompt_input = str(msg["prompt_text"])
                else:
                    prompt_input = {"prompt_token_ids": msg["prompt"]}

                params = base_params.clone()
                
                
                
                if "max_tokens" in msg:
                    params.max_tokens = int(msg["max_tokens"])
                if "min_tokens" in msg:
                    params.min_tokens = int(msg["min_tokens"])
                if "ignore_eos" in msg:
                    params.ignore_eos = bool(msg["ignore_eos"])
                final_output = None
                first_token_sent = False

                if PROFILE:
                    _prof_queue_depth_samples += 1
                    _prof_queue_depth_sum += work_queue.qsize()
                    _t_gen_start = time.monotonic_ns()

                last_stream_len = 0
                async for output in engine.generate(prompt_input, params, request_id):
                    if PROFILE:
                        _prof_n_yields += 1
                    final_output = output
                    if output is not None and output.outputs:
                        token_ids_now = output.outputs[0].token_ids
                        if PROFILE:
                            cur_len = len(token_ids_now)
                            if cur_len > last_stream_len:
                                _prof_stream_tokens += cur_len - last_stream_len
                                last_stream_len = cur_len
                    if (
                        not first_token_sent
                        and output is not None
                        and output.outputs
                    ):
                        first_token_ids = list(output.outputs[0].token_ids)
                        if first_token_ids:
                            await push_sock.send(_pack({
                                "id": request_id,
                                "first_token_ids": first_token_ids[:1],
                            }))
                            first_token_sent = True

                if PROFILE:
                    _prof_generate_wait_ns += time.monotonic_ns() - _t_gen_start
                    _prof_n_generate_calls += 1

                token_ids = []
                if final_output is not None and final_output.outputs:
                    token_ids = list(final_output.outputs[0].token_ids)

                if PROFILE:
                    _t_zmq = time.monotonic_ns()
                await push_sock.send(_pack({
                    "id": request_id,
                    "token_ids": token_ids,
                }))
                if PROFILE:
                    _prof_zmq_send_ns += time.monotonic_ns() - _t_zmq

                n_completed += 1
                n_output_tokens += len(token_ids)
                if PROFILE:
                    _prof_window_tokens += len(token_ids)
                gc_counter += 1
                if GC_INTERVAL > 0 and gc_counter >= GC_INTERVAL:
                    gc.collect()
                    gc_counter = 0

                if n_completed <= 3 or n_completed % 500 == 0:
                    elapsed = time.time() - t0
                    log.info(
                        "Completed %d (%.1f qps, %.0f tok/s, %.1fs)",
                        n_completed, n_completed / max(elapsed, 0.001),
                        n_output_tokens / max(elapsed, 0.001), elapsed,
                    )
            except Exception:
                log.exception("Error handling request %s", msg.get("id", "?"))

    async def profile_reporter():
        nonlocal _prof_generate_wait_ns, _prof_zmq_send_ns
        nonlocal _prof_n_generate_calls, _prof_n_yields
        nonlocal _prof_queue_depth_samples, _prof_queue_depth_sum
        nonlocal _prof_window_tokens, _prof_stream_tokens, _prof_window_t0
        while True:
            await asyncio.sleep(30)
            elapsed = time.time() - t0
            now = time.monotonic()
            window_dt = now - _prof_window_t0
            window_tps = _prof_window_tokens / max(window_dt, 0.001)
            stream_tps = _prof_stream_tokens / max(window_dt, 0.001)
            cumul_tps = n_output_tokens / max(elapsed, 0.001)
            if _prof_n_generate_calls == 0:
                log.info("PROFILE [%.0fs] — no completions yet  "
                         "stream=%.0f tok/s  complete=%.0f tok/s  "
                         "cumul=%.0f tok/s",
                         elapsed, stream_tps, window_tps, cumul_tps)
            else:
                avg_gen_ms = (_prof_generate_wait_ns / _prof_n_generate_calls) / 1e6
                avg_zmq_us = (_prof_zmq_send_ns / max(_prof_n_generate_calls, 1)) / 1e3
                avg_yields = _prof_n_yields / max(_prof_n_generate_calls, 1)
                avg_qd = (_prof_queue_depth_sum /
                           max(_prof_queue_depth_samples, 1))
                log.info(
                    "PROFILE [%.0fs] — completions=%d  stream=%.0f tok/s  "
                    "complete=%.0f tok/s  cumul=%.0f tok/s  avg_generate=%.1fms  "
                    "avg_zmq_send=%.0fus  avg_yields/req=%.0f  "
                    "avg_queue_depth=%.0f  active_tasks=%d",
                    elapsed, _prof_n_generate_calls, stream_tps, window_tps,
                    cumul_tps, avg_gen_ms,
                    avg_zmq_us, avg_yields, avg_qd,
                    sum(1 for t in asyncio.all_tasks() if not t.done()),
                )
            _prof_generate_wait_ns = 0
            _prof_zmq_send_ns = 0
            _prof_n_generate_calls = 0
            _prof_n_yields = 0
            _prof_queue_depth_samples = 0
            _prof_queue_depth_sum = 0
            _prof_window_tokens = 0
            _prof_stream_tokens = 0
            _prof_window_t0 = now

    log.info("Standalone worker started. Waiting for requests...")

    try:
        tasks = [asyncio.create_task(reader())]
        tasks += [asyncio.create_task(worker(i)) for i in range(num_workers)]
        if PROFILE:
            tasks.append(asyncio.create_task(profile_reporter()))
        await asyncio.gather(*tasks)
    finally:
        log.info("Standalone worker shutting down. Total: %d, tokens: %d",
                 n_completed, n_output_tokens)
        engine.shutdown()
        pull_sock.close()
        push_sock.close()
        ctx.term()


def _run_main():
    if _HAS_UVLOOP:
        log.info("standalone worker: using uvloop event loop")
        uvloop.install()
    asyncio.run(main())


if __name__ == "__main__":
    _run_main()
