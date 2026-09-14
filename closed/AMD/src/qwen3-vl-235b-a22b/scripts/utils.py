#!/usr/bin/env python3
"""Utilities for benchmark_mlperf6pt1.py"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache

log = logging.getLogger(__name__)

_FP8_KV_DTYPES = ("fp8", "fp8_e4m3", "fp8_e5m2")


def _empty_recipe() -> dict[str, Any]:
    """A blank recipe; mirrors the benchmark.yaml block shape parse_results flattens."""
    return {
        "method": None,
        "weight": {"element": None, "scale": None, "granularity": None},
        "activation": {"element": None, "scale": None, "granularity": None},
        "kv_cache": {"element": None, "scale": None, "granularity": None},
        "gemm": {"element": None, "accumulation": None},
    }


def effective_quantization_flag(server_cfg) -> str | None:
    """The ``--quantization`` value vLLM is actually launched with (None => flag omitted)."""
    q = server_cfg.get("quantization")
    return str(q) if q is not None and str(q).lower() != "none" else None


def effective_kv_cache_dtype(server_cfg) -> str | None:
    """The ``--kv-cache-dtype`` value actually launched (None => flag omitted, vLLM default)."""
    kv = str(server_cfg.get("kv_cache_dtype") or "").lower()
    return kv if kv in _FP8_KV_DTYPES else None


def effective_mm_encoder_attn_backend(server_cfg) -> str | None:
    """The ``--mm-encoder-attn-backend`` value to launch with (None => omit; vLLM auto-selects).

    An explicit backend name in the config is passed through; unset/``auto`` returns None so vLLM
    keeps its validated ROCm default."""
    val = server_cfg.get("mm_encoder_attn_backend")
    if val not in (None, "", "auto"):
        return str(val)
    return None


def _resolve_config_json(model: str) -> Path | None:
    """Locate the served model's config.json: a local dir, else the HF cache (or a tiny download)."""
    p = Path(model)
    if p.is_dir():
        cand = p / "config.json"
        return cand if cand.is_file() else None
    # Prefer the cache (no network); else fetch just config.json (~tiny) -- a repo id vLLM has not
    # fully downloaded yet, since detection runs before serve. Resolves the snapshot revision too.
    # A network failure here propagates to write_run_meta's guard (telemetry never aborts the run).
    hit = try_to_load_from_cache(repo_id=model, filename="config.json")
    if isinstance(hit, str) and Path(hit).is_file():
        return Path(hit)
    return Path(hf_hub_download(repo_id=model, filename="config.json"))


def _fp8_element(fmt: str | None) -> str:
    """Canonical fp8 weight/activation element from the checkpoint's ``fmt`` (e4m3/e5m2)."""
    f = (fmt or "").lower()
    return f"fp8_{f}" if f in ("e4m3", "e5m2") else "fp8"


def _canonical_dtype(dt: str | None) -> str | None:
    """Map a HF dtype string (e.g. 'bfloat16') to the recipe's element token (e.g. 'bf16')."""
    d = (dt or "").lower()
    if "bfloat16" in d or d == "bf16":
        return "bf16"
    if "float16" in d or d == "fp16":
        return "fp16"
    if "float32" in d or d == "fp32":
        return "fp32"
    return d or None


def _has_recipe(override: dict[str, Any]) -> bool:
    """True if the override declares any quantization value (i.e. is not the all-null default)."""
    if not override:
        return False
    if override.get("method") is not None:
        return True
    for tgt in ("weight", "activation", "kv_cache", "gemm"):
        if any(v is not None for v in (override.get(tgt) or {}).values()):
            return True
    return False


def _detect_from_checkpoint(model: str) -> dict[str, Any]:
    """Read the served checkpoint's config.json. Returns detected facts (best-effort, never raises)."""
    out: dict[str, Any] = {
        "revision": "",
        "model_path": "",
        "quant_config": None,
        "dtype": None,
    }
    config_path = _resolve_config_json(model)
    if config_path is None:
        return out
    out["model_path"] = str(config_path.parent)
    # HF cache layout is .../snapshots/<commit>/config.json -> the snapshot dir name is the revision.
    if config_path.parent.parent.name == "snapshots":
        out["revision"] = config_path.parent.name
    cfg = json.loads(config_path.read_text())
    out["quant_config"] = cfg.get("quantization_config")
    out["dtype"] = cfg.get("dtype") or (cfg.get("text_config") or {}).get("dtype")
    return out


def resolve_quantization(
    model: str, server_cfg, override: dict[str, Any] | None
) -> dict[str, Any]:
    """Resolve the quantization recipe actually in effect (detection wins; override fills the gap)."""
    override = override or {}
    detected = _detect_from_checkpoint(model)
    qc = detected.get("quant_config") or {}
    model_dtype = _canonical_dtype(detected.get("dtype"))
    launch_q = effective_quantization_flag(server_cfg)
    launch_kv = effective_kv_cache_dtype(server_cfg)

    recipe = _empty_recipe()
    # GEMM / kernel precision is never detectable -> override is its only source.
    if override.get("gemm"):
        recipe["gemm"].update(
            {k: v for k, v in dict(override["gemm"]).items() if v is not None}
        )
    # KV-cache precision is a serve-time choice: the launch flag, else override, else the model dtype
    # (vLLM's --kv-cache-dtype=auto keeps the KV cache at the model dtype).
    if launch_kv:
        recipe["kv_cache"]["element"] = launch_kv
    elif override.get("kv_cache") and any(
        v is not None for v in dict(override["kv_cache"]).values()
    ):
        recipe["kv_cache"].update(
            {k: v for k, v in dict(override["kv_cache"]).items() if v is not None}
        )
    else:
        recipe["kv_cache"]["element"] = model_dtype

    if qc:  # conclusive: pre-quantized checkpoint declares its recipe
        source = "checkpoint"
        method = qc.get("quant_method")
        fmt = qc.get("fmt")
        recipe["method"] = method
        recipe["weight"]["element"] = _fp8_element(fmt) if method == "fp8" else method
        wbs = qc.get("weight_block_size")
        if isinstance(wbs, (list, tuple)) and wbs:
            recipe["weight"]["granularity"] = f"block_{wbs[-1]}"
        scheme = str(qc.get("activation_scheme") or "").lower()
        if scheme:  # activations are quantized (dynamic/static)
            recipe["activation"]["element"] = (
                _fp8_element(fmt) if method == "fp8" else method
            )
            recipe["activation"]["granularity"] = (
                "per_token" if scheme == "dynamic" else "per_tensor"
            )
    elif launch_q:  # conclusive: runtime quantization requested via flag
        source = "launch_flag"
        recipe["method"] = launch_q
        recipe["weight"]["element"] = (
            "fp8_e4m3" if launch_q in ("fp8", "ptpc_fp8") else launch_q
        )
    elif _has_recipe(override):  # inconclusive (no config, no flag) + override supplied
        source = "override"
        if override.get("method") is not None:
            recipe["method"] = override["method"]
        for tgt in ("weight", "activation", "kv_cache"):
            if override.get(tgt):
                recipe[tgt].update(
                    {k: v for k, v in dict(override[tgt]).items() if v is not None}
                )
    else:  # inconclusive, no override -> unquantized baseline (the model dtype)
        source = "unquantized"
        recipe["weight"]["element"] = model_dtype

    # Detection wins: if an override is set but detection was conclusive and disagrees, keep the
    # detected value and warn (the operator asserted something the checkpoint/flag contradicts).
    if source in ("checkpoint", "launch_flag") and _has_recipe(override):
        ov_weight = (override.get("weight") or {}).get("element")
        if (
            ov_weight
            and str(ov_weight).lower() != str(recipe["weight"]["element"]).lower()
        ):
            log.warning(
                "quant_override weight=%s conflicts with detected %s (source=%s); keeping detected",
                ov_weight,
                recipe["weight"]["element"],
                source,
            )

    return {
        "model": str(model),
        "model_revision": detected.get("revision", ""),
        "model_path": detected.get("model_path", ""),
        "quant_source": source,
        "quantization": recipe,
    }


# Accelerator model pattern: AMD Instinct (MIxxx[X]). Matched against torch's reported device name.
_GPU_PATTERNS = (
    r"MI\d{3}[A-Z]?",  # MI300X, MI325X, MI350X, MI355X
)


def _match_gpu_model(raw: str) -> str:
    """Extract a short accelerator model (e.g. MI350X, MI355X) from a raw name; '' if none."""
    s = (raw or "").upper()
    for pat in _GPU_PATTERNS:
        m = re.search(pat, s)
        if m:
            return m.group(0)
    return ""


def detect_gpu() -> str:
    """Accelerator model the benchmark runs on (e.g. ``MI350X``, ``MI355X``).

    Prefers ``GPU_NAME``, which ``start_docker.sh`` sets from the host -- the exact AMD SKU is only
    reliably visible there (inside the container torch reports an empty name and rocm-smi reports
    N/A). Falls back to torch's device name for runs not launched via start_docker (usually empty on
    ROCm). Returns the normalized short model, else the raw name, else ``""``.
    """
    raw = os.environ.get("GPU_NAME", "").strip()
    if not raw and torch.cuda.is_available():
        raw = torch.cuda.get_device_name(0)
    return _match_gpu_model(raw) or raw


def write_run_meta(
    run_dir: Path, model: str, server_cfg, override: dict[str, Any] | None
) -> None:
    """Resolve the effective quantization + accelerator and write ``<run_dir>/run_meta.json``.

    Telemetry only, and called before the run begins, so it is wrapped to never abort the benchmark:
    on any failure (e.g. the checkpoint config is unreachable) it logs and writes nothing, and
    parse_results simply reports blank quant/gpu columns for the run.
    """
    try:
        meta = resolve_quantization(model, server_cfg, override)
        meta["gpu"] = detect_gpu()
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    except Exception as exc:  # noqa: BLE001 - telemetry must never sink a run
        log.warning("could not write run_meta.json: %s", exc)
