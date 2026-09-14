"""Self-contained warm-up/preload for the fused MXFP4 Triton kernels.

The new docker image (rocm/mlperf-endpoints-private) adds two vLLM compilation
fusion passes for Quark MXFP4 models:

  * RocmAiterRMSNormMxfp4GroupQuantFusionPass  -> aiter ``fused_rms_mxfp4_quant``
    (fuses input/post-attn RMSNorm + MXFP4 dynamic-quant feeding qkv/gate_up)
  * RocmAiterSiluMulMxfp4GroupQuantFusionPass  -> aiter ``act_mul_and_mxfp4_quant``
    (fuses silu_and_mul + MXFP4 dynamic-quant feeding down_proj)

Both fused kernels are Triton @triton.heuristics kernels, so a *new* variant is
compiled for each distinct decode-batch shape (M). Nothing in the harness
preloads them for Llama-3.1-8B, so they JIT lazily at runtime, independently on
each of the 8 data-parallel engines, as new shapes appear. That desyncs the
engines and produces rotating-idle GPUs (0<->100% utilization sawtooth) plus
latency blowups.

This module compiles both fused kernels for every cudagraph capture size up
front (on the engine's own device), so the on-disk Triton cache is fully warm
before LoadGen timing starts and no compilation happens on the hot path.

Enabled by default. The engine worker processes do NOT inherit run.sh's shell
environment (they are given their env solely via the YAML ``env_config`` in
``set_mlperf_envs``), so gating on a shell-exported flag never reached them. To
explicitly DISABLE the warm-up, set HARNESS_WARMUP_FUSED_MXFP4_KERNELS=0 in the
YAML ``env_config``. It is always a safe no-op on the old/unfused image because
the fused aiter kernels are not importable there (see the import guard below).
"""

import json
import os

import harness_llm.common.logging as logger

log = logger.get_logger(__name__)

ENABLE_FLAG = "HARNESS_WARMUP_FUSED_MXFP4_KERNELS"


def _read_model_dims(model_path: str):
    """Return (hidden_size, intermediate_size, rms_eps) from the model config."""
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    # Some configs nest the language-model fields under "text_config".
    text = cfg.get("text_config", cfg)
    hidden = int(text["hidden_size"])
    inter = int(text["intermediate_size"])
    eps = float(text.get("rms_norm_eps", 1e-5))
    return hidden, inter, eps


def warmup_fused_mxfp4_kernels(capture_sizes, model_path, tp_size=1, device_label=""):
    """Compile the fused MXFP4 kernels for every capture size on the current device.

    Must be called from the engine/server process *after* HIP_VISIBLE_DEVICES and
    the per-device cache dirs are set, so the compiled kernels land in the same
    on-disk Triton cache the EngineCore worker reuses. ``cuda:0`` in this process
    maps to the engine's physical GPU.
    """
    # Default ON: engine workers don't inherit run.sh's shell env, so we cannot
    # rely on a shell-exported enable flag reaching this process. Only an explicit
    # "0" (e.g. via YAML env_config) disables it.
    if os.environ.get(ENABLE_FLAG, "1") == "0":
        return

    if not capture_sizes:
        log.warning(f"[fused-warmup{device_label}] no capture sizes provided; skipping")
        return

    if not model_path:
        log.warning(f"[fused-warmup{device_label}] no model path provided; skipping")
        return

    try:
        hidden, inter, eps = _read_model_dims(model_path)
    except Exception as e:
        log.warning(
            f"[fused-warmup{device_label}] could not read model dims from "
            f"{model_path} ({e}); skipping"
        )
        return

    try:
        import torch
        from aiter.ops.triton.activation import act_mul_and_mxfp4_quant
        from aiter.ops.triton.quant.fused_mxfp4_quant import fused_rms_mxfp4_quant
    except Exception as e:
        log.warning(
            f"[fused-warmup{device_label}] fused MXFP4 kernels not importable "
            f"({e}); skipping (old/unfused image?)"
        )
        return

    if not torch.cuda.is_available():
        log.warning(f"[fused-warmup{device_label}] CUDA not available; skipping")
        return

    tp_size = max(int(tp_size or 1), 1)
    inter_shard = inter // tp_size
    # gate_up_proj output (silu_and_mul input) is 2 * (sharded intermediate).
    act_in = 2 * inter_shard
    # RMSNorm input is the full (replicated) hidden dim, pre-projection.
    norm_in = hidden

    sizes = sorted({int(m) for m in capture_sizes if int(m) > 0}, reverse=True)
    dev = torch.device("cuda:0")

    log.info(
        f"[fused-warmup{device_label}] warming fused MXFP4 kernels for "
        f"{len(sizes)} capture sizes (hidden={hidden}, inter_shard={inter_shard}, "
        f"tp={tp_size}); range [{sizes[-1]}..{sizes[0]}]"
    )

    weight = torch.ones(norm_in, dtype=torch.bfloat16, device=dev)
    failures = 0
    for idx, m in enumerate(sizes):
        try:
            x_norm = torch.randn(m, norm_in, dtype=torch.bfloat16, device=dev)
            # RMSNorm + MXFP4 quant (no residual): input_layernorm of layer 0 etc.
            fused_rms_mxfp4_quant(x_norm, weight, eps, shuffle=True)
            # Fused add-RMSNorm + MXFP4 quant: the residual-add variant used by all
            # subsequent decoder layers.
            residual = torch.randn(m, norm_in, dtype=torch.bfloat16, device=dev)
            fused_rms_mxfp4_quant(x_norm, weight, eps, res1=residual, shuffle=True)
            # SiluMul + MXFP4 quant: down_proj input (gate_up output is 2*inter).
            x_act = torch.randn(m, act_in, dtype=torch.bfloat16, device=dev)
            act_mul_and_mxfp4_quant(x_act, activation="silu", shuffle=True)
        except Exception as e:
            failures += 1
            if failures <= 5:
                log.warning(f"[fused-warmup{device_label}] size {m} failed: {e}")
        finally:
            x_norm = residual = x_act = None
            if (idx % 16) == 0:
                torch.cuda.empty_cache()

    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    del weight
    torch.cuda.empty_cache()
    log.info(
        f"[fused-warmup{device_label}] done "
        f"({len(sizes) - failures}/{len(sizes)} sizes compiled, {failures} failures)"
    )
