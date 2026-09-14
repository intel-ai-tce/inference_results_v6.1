
"""
Patch rocm_aiter_fa.py for NHD-mode cross-vendor PD.

When MI350X decodes with KV cache received from H200 in NHD layout,
two things must change in the AITER attention backend:

1. USING_SHUFFLE_LAYOUT must be env-controlled (original is hardcoded True).
   Setting VLLM_ROCM_SHUFFLE_LAYOUT=0 makes the backend use the NHD code path.

2. In NHD mode, the KV scales must match the H200 prefiller's scales (from JSON)
   instead of the local FP4 model's scales, so FP8 dequantization is correct.

This is a lightweight alternative to the full shuffle_kv + kv_scale_override
patches — no SHUFFLE conversion, no _fp8_kv_patch.py.

Environment variables:
    VLLM_ROCM_SHUFFLE_LAYOUT=0         Use NHD attention path
    VLLM_KV_SCALE_OVERRIDE=/path.json  Override KV scales from JSON

Run INSIDE the MI350X container.
"""

import os
import sys
import shutil
import re


def find_file(candidates, fallback_subpath=None):
    for c in candidates:
        if os.path.exists(c):
            return c
    if fallback_subpath:
        try:
            import vllm
            p = os.path.join(os.path.dirname(vllm.__file__), *fallback_subpath)
            if os.path.exists(p):
                return p
        except ImportError:
            pass
    return None


AITER_FILE = find_file(
    [
    ],
    fallback_subpath=["v1", "attention", "backends", "rocm_aiter_fa.py"],
)

if AITER_FILE is None:
    print("ERROR: Could not find rocm_aiter_fa.py")
    sys.exit(1)

print("=" * 60)
print("NHD Mode Patch (cross-vendor PD)")
print("=" * 60)
print(f"\n  Target: {AITER_FILE}")

with open(AITER_FILE, "r") as f:
    content = f.read()

backup = AITER_FILE + ".bak"
if not os.path.exists(backup):
    shutil.copy2(AITER_FILE, backup)
    print(f"  Backup: {backup}")

changes = 0





OLD_ATTENTION_IMPORT = "from vllm.attention.layer import Attention"
NEW_ATTENTION_IMPORT = (
    "from vllm.model_executor.layers.attention.attention import Attention"
    "  # AITER-FA-IMPORT-FIX"
)
if OLD_ATTENTION_IMPORT in content:
    content = content.replace(OLD_ATTENTION_IMPORT, NEW_ATTENTION_IMPORT, 1)
    changes += 1
    print("  Attention import: patched for this vLLM tree")
elif "AITER-FA-IMPORT-FIX" in content or NEW_ATTENTION_IMPORT in content:
    print("  Attention import: already patched")
else:
    print("  WARNING: Could not find stale Attention import")

OLD_CU_IMPORT = "from vllm.utils.platform_utils import get_cu_count"
NEW_CU_IMPORT = (
    "from vllm.utils.platform_utils import num_compute_units"
    "  # AITER-FA-IMPORT-FIX"
)
if OLD_CU_IMPORT in content:
    content = content.replace(OLD_CU_IMPORT, NEW_CU_IMPORT, 1)
    content = content.replace("get_cu_count()", "num_compute_units()")
    changes += 1
    print("  Compute-unit helper: patched for this vLLM tree")
elif "num_compute_units" in content:
    print("  Compute-unit helper: already patched")
else:
    print("  WARNING: Could not find stale compute-unit helper import")





if 'VLLM_ROCM_SHUFFLE_LAYOUT' in content:
    print("  USING_SHUFFLE_LAYOUT: already env-controlled")
elif 'USING_SHUFFLE_LAYOUT = True' in content:
    content = content.replace(
        'USING_SHUFFLE_LAYOUT = True',
        'USING_SHUFFLE_LAYOUT = os.environ.get("VLLM_ROCM_SHUFFLE_LAYOUT", "1") == "1"',
    )
    if "import os\n" not in content and "import os," not in content:
        content = content.replace(
            "import torch\n", "import os\nimport torch\n", 1)
        print("  Added 'import os'")
    changes += 1
    print("  USING_SHUFFLE_LAYOUT: patched -> env-var controlled")
elif "rocm_aiter_ops.is_shuffle_kv_cache_enabled()" in content:
    
    
    
    
    print("  USING_SHUFFLE_LAYOUT: direct ROCm AITER API in this vLLM tree")
else:
    print("  WARNING: Could not find 'USING_SHUFFLE_LAYOUT = True'")





NHD_MARKER = "_nhd_kv_scale_override"

if NHD_MARKER in content:
    print("  NHD KV scale override: already present")
else:
    
    
    
    scale_infra = r'''
# --- NHD KV scale override for cross-vendor PD ---
_nhd_kv_scale_override = None
_nhd_kv_scale_loaded = False
_nhd_layer_counter = [0]
_nhd_kv_scale_logged = False

def _load_nhd_kv_scales():
    global _nhd_kv_scale_override, _nhd_kv_scale_loaded
    if _nhd_kv_scale_loaded:
        return _nhd_kv_scale_override
    _nhd_kv_scale_loaded = True
    import json as _json
    _path = os.environ.get("VLLM_KV_SCALE_OVERRIDE", "")
    if _path and os.path.exists(_path):
        with open(_path) as _fh:
            _nhd_kv_scale_override = {str(k): v for k, v in _json.load(_fh).items()}
        import logging
        logging.getLogger(__name__).info(
            "NHD KV scale override: loaded %d layers from %s",
            len(_nhd_kv_scale_override), _path)
    else:
        _nhd_kv_scale_override = {}
    return _nhd_kv_scale_override

def _nhd_scale_to_float(value):
    try:
        if hasattr(value, "flatten"):
            flat = value.flatten()
            if flat.numel() >= 1:
                return float(flat[0].item())
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, (list, tuple)):
            return float(value[0])
        return float(value)
    except Exception:
        return None

def _nhd_layer_scale_entry(layer):
    scales = _load_nhd_kv_scales()
    if not scales:
        return None
    layer_name = getattr(layer, "layer_name", None) or getattr(layer, "prefix", None)
    candidates = []
    if layer_name:
        candidates.extend([
            layer_name,
            layer_name.replace(".self_attn.attn", ".self_attn"),
            layer_name.replace(".self_attn", ".self_attn.attn"),
        ])
        try:
            import re as _re
            match = _re.search(r"\.layers\.(\d+)\.", layer_name)
            if match:
                candidates.append(match.group(1))
        except Exception:
            pass
    if not candidates and not hasattr(layer, "_nhd_layer_idx"):
        layer._nhd_layer_idx = _nhd_layer_counter[0]
        _nhd_layer_counter[0] += 1
    if hasattr(layer, "_nhd_layer_idx"):
        candidates.append(str(layer._nhd_layer_idx))
    for key in candidates:
        if key in scales:
            return scales[key]
    return None

def _nhd_fill_layer_scale(layer, attr_name, value):
    value_float = _nhd_scale_to_float(value)
    if value_float is None:
        return False
    current = getattr(layer, attr_name, None)
    if hasattr(current, "fill_"):
        current.fill_(value_float)
        return True
    try:
        setattr(layer, attr_name, value_float)
        return True
    except Exception:
        return False

def _distributed_inference_apply_nhd_kv_scales(layer):
    """Use remote-prefill FP8 KV scales when ROCm reads NHD remote KV."""
    global _nhd_kv_scale_logged
    try:
        shuffle_enabled = rocm_aiter_ops.is_shuffle_kv_cache_enabled()
    except Exception:
        shuffle_enabled = bool(globals().get("USING_SHUFFLE_LAYOUT", True))
    if shuffle_enabled:
        return
    if getattr(layer, "_nhd_scale_set", False):
        return

    entry = _nhd_layer_scale_entry(layer)
    if not entry:
        return
    k_src = entry.get("k_scale", entry.get("key_scale"))
    v_src = entry.get("v_scale", entry.get("value_scale"))
    ok_k = _nhd_fill_layer_scale(layer, "_k_scale", k_src)
    ok_v = _nhd_fill_layer_scale(layer, "_v_scale", v_src)
    if ok_k and ok_v:
        layer._nhd_scale_set = True
        if not _nhd_kv_scale_logged:
            print(
                "[NHD-KV-SCALE] active: "
                f"k={_nhd_scale_to_float(k_src)} "
                f"v={_nhd_scale_to_float(v_src)}",
                flush=True,
            )
            _nhd_kv_scale_logged = True

'''

    insertion_targets = [
        ("logger = logging.getLogger(__name__)", "after"),
        ("# _distributed_inference_apply_shuffle_kv_scales", "before"),
        ('USING_SHUFFLE_LAYOUT = os.environ.get("VLLM_ROCM_SHUFFLE_LAYOUT", "1") == "1"', "after"),
        ("USING_SHUFFLE_LAYOUT", "after"),
    ]
    inserted = False
    for target, where in insertion_targets:
        idx = content.find(target)
        if idx < 0:
            continue
        if where == "after":
            eol = content.index("\n", idx)
            content = content[:eol + 1] + scale_infra + content[eol + 1:]
        else:
            line_start = content.rfind("\n", 0, idx) + 1
            content = content[:line_start] + scale_infra + content[line_start:]
        changes += 1
        inserted = True
        print("  NHD KV scale override: added infrastructure")
        break
    if not inserted:
        print("  WARNING: No insertion point found for scale override")









OVERRIDE_MARKER = "# NHD scale override"
if OVERRIDE_MARKER in content:
    print("  Forward-method scale override: already injected")
else:
    
    
    
    call_pattern = re.compile(
        r"^( {8,})_distributed_inference_apply_shuffle_kv_scales\(layer, attn_metadata\)",
        re.MULTILINE,
    )
    call_match = call_pattern.search(content)
    if call_match:
        line_start = call_match.start()
        indent = call_match.group(1)
        override_block = (
            f"{indent}# NHD scale override - use H200 prefill scales for "
            "FP8 KV dequantization when ROCm reads NHD remote KV\n"
            f"{indent}_distributed_inference_apply_nhd_kv_scales(layer)\n"
        )
        content = content[:line_start] + override_block + content[line_start:]
        changes += 1
        print("  Forward-method scale override: injected before shuffle scale helper")
    else:
        
        
        pattern = re.compile(r'^( {8,})if USING_SHUFFLE_LAYOUT:', re.MULTILINE)
        match = pattern.search(content)
        if match:
            indent = match.group(1)
            inject_pos = match.start()
            override_block = f"""{indent}# NHD scale override - use H200 prefill scales for FP8 KV dequantization
{indent}_distributed_inference_apply_nhd_kv_scales(layer)

"""
            content = content[:inject_pos] + override_block + content[inject_pos:]
            changes += 1
            print("  Forward-method scale override: injected before SHUFFLE check")
        else:
            print("  WARNING: Could not find a forward() scale-override anchor")





with open(AITER_FILE, "w") as f:
    f.write(content)

print(f"\n  Syntax check...")
try:
    compile(open(AITER_FILE).read(), AITER_FILE, "exec")
    print(f"  {os.path.basename(AITER_FILE)}: OK")
except SyntaxError as e:
    print(f"  SYNTAX ERROR: {e}")
    if os.path.exists(backup):
        shutil.copy2(backup, AITER_FILE)
        print("  Restored from backup.")
    sys.exit(1)


cache_dir = os.path.join(os.path.dirname(AITER_FILE), "__pycache__")
if os.path.isdir(cache_dir):
    for fn in os.listdir(cache_dir):
        if fn.startswith("rocm_aiter_fa"):
            os.remove(os.path.join(cache_dir, fn))
            print(f"  Cleared: __pycache__/{fn}")

print(f"""
{'=' * 60}
Done! {changes} change(s) applied.
{'=' * 60}

NHD mode is now env-controlled:
  VLLM_ROCM_SHUFFLE_LAYOUT=0  ->  NHD attention path (for cross-vendor PD)
  VLLM_ROCM_SHUFFLE_LAYOUT=1  ->  SHUFFLE attention path (default, standalone)

KV scale override (for correct FP8 dequantization in cross-vendor PD):
  VLLM_KV_SCALE_OVERRIDE=/tmp/fp8_kv_scales.json

Restart the MI350X decoder after applying.
""")
