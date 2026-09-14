
"""Patch ROCm AITER unified attention for cross-vendor GPT-OSS PD.

GPT-OSS on MI350X selects rocm_aiter_unified_attn.py for fp8 KV decode.
The older shuffle_kv/kv_scale_override patches target rocm_aiter_fa.py, so
cross-vendor PD can silently use decoder-local KV scales in the active backend.
This patch mirrors the scale override behavior for the unified attention path.

It also advertises KV connector support on RocmAiterUnifiedAttentionBackend.
That class overrides the ROCM_ATTN cache shape with blocks-first layout
``(num_blocks, 2, block_size, num_kv_heads, head_size)`` and unbinds K/V along
dim 1, but inherits ``supports_kv_connector=False`` from RocmAttentionBackend,
whose comment refers to ROCM_ATTN's older K/V-first layout.
"""

from __future__ import annotations

import os
import shutil
import sys


def find_file(candidates, fallback_subpath=None):
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    if fallback_subpath:
        try:
            import vllm

            path = os.path.join(os.path.dirname(vllm.__file__), *fallback_subpath)
            if os.path.exists(path):
                return path
        except ImportError:
            pass
    return None


TARGET = find_file(
    [
    ],
    fallback_subpath=["v1", "attention", "backends", "rocm_aiter_unified_attn.py"],
)

if TARGET is None:
    print("ERROR: could not find rocm_aiter_unified_attn.py", file=sys.stderr)
    sys.exit(1)

print("=" * 60)
print("ROCm AITER Unified Attention KV Scale Override Patch")
print("=" * 60)
print(f"  Target: {TARGET}")

with open(TARGET, "r", encoding="utf-8") as f:
    text = f.read()

backup = TARGET + ".unified_kv_scale_bak"
if not os.path.exists(backup):
    shutil.copy2(TARGET, backup)
    print(f"  Backup: {backup}")

marker = "_unified_kv_scale_override"
connector_marker = "UNIFIED-KV-CONNECTOR"
if marker in text:
    print("  Unified KV scale override: already patched")
else:
    infra_anchor = "logger = init_logger(__name__)\n"
    if infra_anchor not in text:
        print("ERROR: logger anchor not found", file=sys.stderr)
        sys.exit(1)
    infra = r"""

# _unified_kv_scale_override: injected by distributed_inference.
_unified_kv_scale_override_cache = None
_unified_kv_scale_override_logged = False


def _load_unified_kv_scale_override():
    global _unified_kv_scale_override_cache
    if _unified_kv_scale_override_cache is not None:
        return _unified_kv_scale_override_cache
    import os as _os
    path = _os.environ.get("VLLM_KV_SCALE_OVERRIDE")
    scales = {}
    if path:
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                scales = json.load(f)
            print(f"[UNIFIED-KV-SCALE] loaded {len(scales)} layers from {path}")
        except Exception as exc:
            print(f"[UNIFIED-KV-SCALE] failed to load {path}: {exc}")
            scales = {}
    _unified_kv_scale_override_cache = scales
    return scales


def _scale_to_float(value):
    try:
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, (list, tuple)):
            return float(value[0])
        return float(value)
    except Exception:
        return None


def _maybe_override_unified_kv_scales(layer):
    global _unified_kv_scale_override_logged
    scales = _load_unified_kv_scale_override()
    if not scales:
        return
    layer_name = getattr(layer, "layer_name", None) or getattr(layer, "prefix", None)
    if not layer_name:
        return
    entry = scales.get(layer_name)
    if entry is None:
        entry = scales.get(layer_name.replace(".self_attn.attn", ".self_attn"))
    if entry is None:
        entry = scales.get(layer_name.replace(".self_attn", ".self_attn.attn"))
    if entry is None:
        try:
            import re
            match = re.search(r"\.layers\.(\d+)\.", layer_name)
            if match is not None:
                entry = scales.get(match.group(1))
        except Exception:
            entry = None
    if entry is None:
        return

    k_scale = entry.get("k_scale", entry.get("key_scale"))
    v_scale = entry.get("v_scale", entry.get("value_scale"))

    def _assign_layer_scale(attr_name, float_attr_name, value):
        value_float = _scale_to_float(value)
        if value_float is None:
            return None
        current = getattr(layer, attr_name, None)
        if hasattr(current, "fill_"):
            current.fill_(value_float)
        else:
            try:
                import torch
                setattr(layer, attr_name, torch.tensor(value_float))
            except Exception:
                setattr(layer, attr_name, value_float)
        if hasattr(layer, float_attr_name):
            setattr(layer, float_attr_name, value_float)
        return value_float

    k_float = _assign_layer_scale("_k_scale", "_k_scale_float", k_scale)
    v_float = _assign_layer_scale("_v_scale", "_v_scale_float", v_scale)

    if not _unified_kv_scale_override_logged:
        print(
            "[UNIFIED-KV-SCALE] active: "
            f"first layer {layer_name} k={_scale_to_float(k_scale)} "
            f"v={_scale_to_float(v_scale)}"
        )
        _unified_kv_scale_override_logged = True
"""
    text = text.replace(infra_anchor, infra_anchor + infra, 1)

    forward_anchors = (
        "        key_cache, value_cache = kv_cache.unbind(0)\n",
        "        key_cache, value_cache = self._split_kv_cache(kv_cache)\n",
    )
    for forward_anchor in forward_anchors:
        if forward_anchor in text:
            text = text.replace(
                forward_anchor,
                forward_anchor + "        _maybe_override_unified_kv_scales(layer)\n",
                1,
            )
            break
    else:
        print(
            "WARNING: unified KV scale forward anchor not found; "
            "continuing with connector support patch only",
            file=sys.stderr,
        )

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(text)
    print("  Unified KV scale override: applied")

if connector_marker in text:
    print("  Unified KV connector support: already patched")
else:
    class_anchor = """\
    @classmethod
    def supports_sink(cls) -> bool:
        return True

"""
    if class_anchor not in text:
        print("ERROR: supports_sink anchor not found", file=sys.stderr)
        sys.exit(1)
    connector_override = class_anchor + """\
    @classmethod
    def supports_kv_connector(cls) -> bool:
        # UNIFIED-KV-CONNECTOR: unlike ROCM_ATTN, this backend uses a
        # blocks-first KV cache shape compatible with vLLM's KV connector.
        return True

"""
    text = text.replace(class_anchor, connector_override, 1)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(text)
    print("  Unified KV connector support: applied")
