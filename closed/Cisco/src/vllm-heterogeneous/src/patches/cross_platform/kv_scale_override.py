
"""
Patch rocm_aiter_fa.py to support KV scale override from JSON file.

When running FP4 weights on MI350X with FP8 KV cache from H200, the
k_scale/v_scale values may differ between the two model checkpoints.
This patch adds support for overriding the per-layer scales via a JSON
file, so MI350X uses the H200 prefiller's scales for KV cache encoding.

Requires: patch_shuffle_kv_transfer.py must be applied first (provides
the _orig_k_scale_val infrastructure this patch extends).

Environment variable:
    VLLM_KV_SCALE_OVERRIDE=/path/to/fp8_kv_scales.json

The JSON file format (produced by extract_kv_scales.py):
    {
        "0": {"k_scale": 0.0200, "v_scale": 0.0015},
        "1": {"k_scale": 0.0199, "v_scale": 0.0014},
        ...
    }

Run INSIDE the MI350X container:
    python3 <submission-root>/scripts/patch_kv_scale_override.py
"""

import os
import sys
import shutil


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
print("KV Scale Override Patch")
print("=" * 60)
print(f"\n  Target: {AITER_FILE}")

with open(AITER_FILE, "r") as f:
    content = f.read()


if "_distributed_inference_apply_shuffle_kv_scales" in content:
    print("\n  v0.23 metadata scale helper already provides KV scale override support.")
    print('  Set VLLM_KV_SCALE_OVERRIDE=/path/to/scales.json to use.')
    sys.exit(0)

if "_orig_k_scale_val" not in content:
    print("\n  ERROR: patch_shuffle_kv_transfer.py must be applied first.")
    print("  Run: python3 scripts/patch_shuffle_kv_transfer.py")
    sys.exit(1)

if "_kv_scale_override" in content:
    print("\n  Already patched! KV scale override support is active.")
    print('  Set VLLM_KV_SCALE_OVERRIDE=/path/to/scales.json to use.')
    sys.exit(0)


backup = AITER_FILE + ".bak_scale_override"
if not os.path.exists(backup):
    shutil.copy2(AITER_FILE, backup)
    print(f"  Backup: {backup}")



OLD_KVSCALE_HEADER = "_kvscale_tensors = None"
NEW_KVSCALE_HEADER = """\
_kvscale_tensors = None
_kv_scale_override = None
_kv_scale_override_idx = [0]

def _load_kv_scale_override():
    global _kv_scale_override
    if _kv_scale_override is not None:
        return
    import json
    path = os.environ.get("VLLM_KV_SCALE_OVERRIDE", "")
    if path and os.path.exists(path):
        with open(path) as _f:
            _kv_scale_override = {int(k): v for k, v in json.load(_f).items()}
        import logging
        logging.getLogger(__name__).info(
            "KV scale override: loaded %d layers from %s",
            len(_kv_scale_override), path)
    else:
        _kv_scale_override = {}"""

if OLD_KVSCALE_HEADER not in content:
    print("\n  ERROR: Could not find '_kvscale_tensors = None'")
    print("  The file may have been modified. Check manually.")
    sys.exit(1)

content = content.replace(OLD_KVSCALE_HEADER, NEW_KVSCALE_HEADER, 1)
print("  Added override loading infrastructure")



OLD_SCALE_SAVE = """\
                if not hasattr(layer, '_orig_k_scale_val'):
                    _ks = layer._k_scale
                    _vs = layer._v_scale
                    layer._orig_k_scale_val = (
                        float(_ks.flatten()[0])
                        if isinstance(_ks, torch.Tensor) and _ks.numel() >= 1
                        else float(_ks))
                    layer._orig_v_scale_val = (
                        float(_vs.flatten()[0])
                        if isinstance(_vs, torch.Tensor) and _vs.numel() >= 1
                        else float(_vs))"""

NEW_SCALE_SAVE = """\
                if not hasattr(layer, '_orig_k_scale_val'):
                    _load_kv_scale_override()
                    _ovr = _kv_scale_override.get(
                        _kv_scale_override_idx[0])
                    _kv_scale_override_idx[0] += 1
                    if _ovr:
                        layer._orig_k_scale_val = _ovr['k_scale']
                        layer._orig_v_scale_val = _ovr['v_scale']
                    else:
                        _ks = layer._k_scale
                        _vs = layer._v_scale
                        layer._orig_k_scale_val = (
                            float(_ks.flatten()[0])
                            if isinstance(_ks, torch.Tensor)
                            and _ks.numel() >= 1
                            else float(_ks))
                        layer._orig_v_scale_val = (
                            float(_vs.flatten()[0])
                            if isinstance(_vs, torch.Tensor)
                            and _vs.numel() >= 1
                            else float(_vs))"""

if OLD_SCALE_SAVE not in content:
    print("\n  ERROR: Could not find scale-saving block from shuffle transfer patch")
    print("  Expected exact text match. File may have been modified.")
    sys.exit(1)

content = content.replace(OLD_SCALE_SAVE, NEW_SCALE_SAVE, 1)
print("  Modified scale-saving to check JSON override first")


with open(AITER_FILE, "w") as f:
    f.write(content)


print(f"\n  Syntax check...")
try:
    compile(open(AITER_FILE).read(), AITER_FILE, "exec")
    print(f"  {os.path.basename(AITER_FILE)}: OK")
except SyntaxError as e:
    print(f"  SYNTAX ERROR: {e}")
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
Done! KV scale override support added.
{'=' * 60}

To use (for FP4 decoder with FP8 prefiller):

  1. Extract the H200 prefiller's KV scales (run on MI350X):
     python3 scripts/extract_kv_scales.py \\
         /model/llama2-70b-chat-hf/fp8_dynamic \\
         -o scripts/fp8_kv_scales.json

  2. Optionally compare scales between models:
     python3 scripts/extract_kv_scales.py \\
         /model/llama2-70b-chat-hf/fp8_dynamic \\
         --compare /model/llama2-70b-chat-hf/fp4_quantized_gptq

  3. Set env var in MI350X decoder start script:
     export VLLM_KV_SCALE_OVERRIDE=scripts/fp8_kv_scales.json
     export MODEL_PATH=/model/llama2-70b-chat-hf/fp4_quantized_gptq

  4. Restart the MI350X decoder.

When VLLM_KV_SCALE_OVERRIDE is set, the SHUFFLE path uses the JSON
scales instead of the loaded model's scales. When unset, behavior
is unchanged (uses the model's own scales).
""")
