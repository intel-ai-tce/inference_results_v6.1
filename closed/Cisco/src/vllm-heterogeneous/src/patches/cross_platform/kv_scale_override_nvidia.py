
"""
Patch flash_attn.py to support KV scale override from JSON file.

When running cross-platform PD with NVIDIA as decoder (MI350X prefill ->
NVIDIA decode), the decoder model's per-layer k_scale/v_scale may differ
from the prefiller's. This patch overrides the decoder's scales so it
correctly reads the KV cache that was encoded by the prefiller.

Environment variable:
    VLLM_KV_SCALE_OVERRIDE=/path/to/fp8_kv_scales.json

The JSON file format (produced by extract_kv_scales.py):
    {
        "0": {"k_scale": 0.0200, "v_scale": 0.0015},
        "1": {"k_scale": 0.0199, "v_scale": 0.0014},
        ...
    }

Run INSIDE the NVIDIA container:
    python3 src/patches/cross_platform/kv_scale_override_nvidia.py
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


FA_FILE = find_file(
    [
    ],
    fallback_subpath=["v1", "attention", "backends", "flash_attn.py"],
)

if FA_FILE is None:
    print("ERROR: Could not find flash_attn.py — is this an NVIDIA container?")
    sys.exit(1)

print("=" * 60)
print("NVIDIA KV Scale Override Patch (flash_attn.py)")
print("=" * 60)
print(f"\n  Target: {FA_FILE}")

with open(FA_FILE, "r") as f:
    content = f.read()

if "_kv_scale_override" in content:
    print("\n  Already patched! KV scale override support is active.")
    print('  Set VLLM_KV_SCALE_OVERRIDE=/path/to/scales.json to use.')
    sys.exit(0)

backup = FA_FILE + ".bak_scale_override"
if not os.path.exists(backup):
    shutil.copy2(FA_FILE, backup)
    print(f"  Backup: {backup}")



OLD_LOGGER = 'logger = init_logger(__name__)'
NEW_LOGGER = """\
logger = init_logger(__name__)

_kv_scale_override = None
_kv_scale_override_idx = [0]


def _load_kv_scale_override():
    global _kv_scale_override
    if _kv_scale_override is not None:
        return
    import json as _json
    path = os.environ.get("VLLM_KV_SCALE_OVERRIDE", "")
    if path and os.path.exists(path):
        with open(path) as _f:
            _kv_scale_override = {int(k): v for k, v in _json.load(_f).items()}
        logger.info("KV scale override: loaded %d layers from %s",
                     len(_kv_scale_override), path)
    else:
        _kv_scale_override = {}


def _maybe_override_kv_scale(layer):
    if hasattr(layer, '_kv_scale_overridden'):
        return
    _load_kv_scale_override()
    if not _kv_scale_override:
        layer._kv_scale_overridden = True
        return
    ovr = _kv_scale_override.get(_kv_scale_override_idx[0])
    _kv_scale_override_idx[0] += 1
    if ovr:
        layer._k_scale = torch.tensor(
            [ovr['k_scale']], dtype=layer._k_scale.dtype,
            device=layer._k_scale.device)
        layer._v_scale = torch.tensor(
            [ovr['v_scale']], dtype=layer._v_scale.dtype,
            device=layer._v_scale.device)
        logger.debug("KV scale override layer %d: k=%.6f v=%.6f",
                      _kv_scale_override_idx[0] - 1,
                      ovr['k_scale'], ovr['v_scale'])
    layer._kv_scale_overridden = True"""

if OLD_LOGGER not in content:
    print("\n  ERROR: Could not find logger line")
    sys.exit(1)


if "import os\n" not in content and "import os," not in content:
    content = content.replace("import copy\n", "import copy\nimport os\n", 1)
    print("  Added 'import os'")

content = content.replace(OLD_LOGGER, NEW_LOGGER, 1)
print("  Added override infrastructure")



OLD_FORWARD_ASSERTIONS = """\
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:"""

NEW_FORWARD_ASSERTIONS = """\
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        _maybe_override_kv_scale(layer)

        if output_scale is not None or output_block_scale is not None:"""

if OLD_FORWARD_ASSERTIONS not in content:
    print("\n  ERROR: Could not find forward() assertion block")
    print("  Expected exact text match. File may have been modified.")
    sys.exit(1)

content = content.replace(OLD_FORWARD_ASSERTIONS, NEW_FORWARD_ASSERTIONS, 1)
print("  Inserted scale override call in forward()")


with open(FA_FILE, "w") as f:
    f.write(content)


print(f"\n  Syntax check...")
try:
    compile(open(FA_FILE).read(), FA_FILE, "exec")
    print(f"  {os.path.basename(FA_FILE)}: OK")
except SyntaxError as e:
    print(f"  SYNTAX ERROR: {e}")
    shutil.copy2(backup, FA_FILE)
    print("  Restored from backup.")
    sys.exit(1)


cache_dir = os.path.join(os.path.dirname(FA_FILE), "__pycache__")
if os.path.isdir(cache_dir):
    for fn in os.listdir(cache_dir):
        if fn.startswith("flash_attn"):
            os.remove(os.path.join(cache_dir, fn))
            print(f"  Cleared: __pycache__/{fn}")

print(f"""
{'=' * 60}
Done! NVIDIA KV scale override support added.
{'=' * 60}

To use (for cross-platform PD with NVIDIA as decoder):

  1. Extract the prefiller's KV scales:
     python3 src/patches/cross_platform/extract_kv_scales.py \\
         /model/llama2-70b-chat-hf/fp8_dynamic \\
         -o /tmp/fp8_kv_scales.json

  2. Set env var before starting the NVIDIA decoder:
     export VLLM_KV_SCALE_OVERRIDE=/tmp/fp8_kv_scales.json

  3. Start the decoder (start_server.sh handles this automatically).

When VLLM_KV_SCALE_OVERRIDE is set, flash_attn uses the JSON
scales instead of the loaded model's scales. When unset, behavior
is unchanged.
""")
