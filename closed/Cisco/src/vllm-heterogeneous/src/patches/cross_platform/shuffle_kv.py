
"""
Patch MI350X for full-performance cross-platform KV cache transfer.

When H200 (prefiller) sends KV cache via NIXL to MI350X (decoder), the
raw bytes are in NHD layout and encoded with per-layer fp8 scale factors
(e.g. k_scale=0.020, v_scale=0.0015).

MI350X's AITER backend uses SHUFFLE memory layout for fast decode AND
hardcodes all scales to 1.0. Both mismatches cause garbled output.

This patch applies TWO fixes:

1. SCALE FIX (rocm_aiter_fa.py):
   Makes SHUFFLE path use the checkpoint's per-layer k_scale/v_scale
   instead of hardcoded 1.0. Uses a single shared tensor pair (~80 MB)
   that is re-filled per layer during each forward pass.

   This also improves standalone MI350X inference quality (the original
   scale=1.0 provides worse fp8 precision for small-magnitude values).

2. NHD-to-SHUFFLE CONVERSION (_fp8_kv_patch.py):
   After NIXL writes blocks in NHD format, rearranges bytes to SHUFFLE
   format in-place on GPU. Pure byte reordering, no value changes.

   K: NHD [block_size, heads, dim] -> SHUFFLE [heads, dim//x, block_size, x]
   V: NHD [block_size, heads, dim] -> SHUFFLE [heads, block_size//x, dim, x]
   where x = 16 // element_size (16 for fp8)

Result: full AITER SHUFFLE performance with correct cross-platform KV
transfer and zero precision loss.

Run INSIDE the MI350X container:
    python3 <submission-root>/scripts/patch_shuffle_kv_transfer.py
"""

import os
import sys
import shutil





def find_file(candidates, fallback_subpath=None, label="file"):
    for c in candidates:
        if os.path.exists(c):
            return c
    if fallback_subpath:
        try:
            import vllm
            vllm_dir = os.path.dirname(vllm.__file__)
            p = os.path.join(vllm_dir, *fallback_subpath)
            if os.path.exists(p):
                return p
        except ImportError:
            pass
    return None


AITER_FILE = find_file(
    [
    ],
    fallback_subpath=["v1", "attention", "backends", "rocm_aiter_fa.py"],
    label="rocm_aiter_fa.py",
)

NIXL_FILE = find_file(
    [
    ],
    fallback_subpath=["distributed", "kv_transfer", "kv_connector", "v1",
                       "nixl_connector.py"],
    label="nixl_connector.py",
)
NIXL_LAYOUT = "flat"
if NIXL_FILE is None:
    
    
    
    NIXL_FILE = find_file(
        [
        ],
        fallback_subpath=["distributed", "kv_transfer", "kv_connector", "v1",
                          "nixl", "__init__.py"],
        label="nixl/__init__.py",
    )
    NIXL_LAYOUT = "package"

if AITER_FILE is None:
    print("ERROR: Could not find rocm_aiter_fa.py")
    sys.exit(1)
if NIXL_FILE is None:
    print("ERROR: Could not find NIXL connector module/package")
    sys.exit(1)

print("=" * 60)
print("Full-Performance Cross-Platform KV Transfer Patch")
print("=" * 60)
print(f"\n  rocm_aiter_fa.py : {AITER_FILE}")
print(f"  NIXL patch target ({NIXL_LAYOUT}): {NIXL_FILE}")






print("\n--- Part 1: Scale fix (rocm_aiter_fa.py) ---")

with open(AITER_FILE, "r") as f:
    aiter_content = f.read()

backup = AITER_FILE + ".bak_scale_fix"
if not os.path.exists(backup):
    shutil.copy2(AITER_FILE, backup)
    print(f"  Backup: {backup}")

changes_made = 0



OLD_ATTENTION_IMPORT = "from vllm.attention.layer import Attention"
NEW_ATTENTION_IMPORT = (
    "from vllm.model_executor.layers.attention.attention import Attention"
    "  # AITER-FA-IMPORT-FIX"
)
if OLD_ATTENTION_IMPORT in aiter_content:
    aiter_content = aiter_content.replace(
        OLD_ATTENTION_IMPORT, NEW_ATTENTION_IMPORT, 1)
    changes_made += 1
    print("  Attention import: patched for this vLLM tree")
elif "AITER-FA-IMPORT-FIX" in aiter_content or NEW_ATTENTION_IMPORT in aiter_content:
    print("  Attention import: already patched")
else:
    print("  WARNING: Could not find stale Attention import")

OLD_CU_IMPORT = "from vllm.utils.platform_utils import get_cu_count"
NEW_CU_IMPORT = (
    "from vllm.utils.platform_utils import num_compute_units"
    "  # AITER-FA-IMPORT-FIX"
)
if OLD_CU_IMPORT in aiter_content:
    aiter_content = aiter_content.replace(OLD_CU_IMPORT, NEW_CU_IMPORT, 1)
    aiter_content = aiter_content.replace("get_cu_count()", "num_compute_units()")
    changes_made += 1
    print("  Compute-unit helper: patched for this vLLM tree")
elif "num_compute_units" in aiter_content:
    print("  Compute-unit helper: already patched")
else:
    print("  WARNING: Could not find stale compute-unit helper import")








BACKEND_CLASS_HEADER = """\
class AiterFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
"""
BACKEND_CLASS_HEADER_PATCHED = """\
class AiterFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_kv_cache_dtypes = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]  # AITER-FA-FP8-KV-FIX
"""
if "AITER-FA-FP8-KV-FIX" in aiter_content:
    print("  FP8 KV cache dtype support: already declared")
elif BACKEND_CLASS_HEADER in aiter_content:
    aiter_content = aiter_content.replace(
        BACKEND_CLASS_HEADER, BACKEND_CLASS_HEADER_PATCHED, 1)
    changes_made += 1
    print("  FP8 KV cache dtype support: declared")
else:
    print("  WARNING: Could not find AiterFlashAttentionBackend header")



OLD_KVSCALE = """\
@lru_cache(maxsize=1)
def get_static_kvscale(
        k_scale_float: float,
        v_scale_float: float,
        num_kv_heads: int,
        num_blocks: int,
        block_size: int,
        device,
    ):
    k_scale = torch.empty((num_kv_heads, num_blocks * block_size),
                        dtype=torch.float32,
                        device=device)
    v_scale = torch.empty((num_kv_heads, num_blocks * block_size),
                        dtype=torch.float32,
                        device=device)
    k_scale.fill_(k_scale_float)
    v_scale.fill_(v_scale_float)
    return k_scale, v_scale"""

NEW_KVSCALE = """\
_kvscale_tensors = None

def get_static_kvscale(
        k_scale_float: float,
        v_scale_float: float,
        num_kv_heads: int,
        num_blocks: int,
        block_size: int,
        device,
    ):
    global _kvscale_tensors
    if _kvscale_tensors is None:
        _kvscale_tensors = (
            torch.empty((num_kv_heads, num_blocks * block_size),
                        dtype=torch.float32, device=device),
            torch.empty((num_kv_heads, num_blocks * block_size),
                        dtype=torch.float32, device=device),
        )
    k_scale, v_scale = _kvscale_tensors
    k_scale.fill_(k_scale_float)
    v_scale.fill_(v_scale_float)
    return k_scale, v_scale"""

V023_MARKER = "_distributed_inference_apply_shuffle_kv_scales"

V023_HELPERS = r'''

# _distributed_inference_apply_shuffle_kv_scales: injected by distributed_inference.
_distributed_inference_kv_scale_override = None
_distributed_inference_kv_scale_logged = False
_distributed_inference_kv_scale_alias_logged = False
_distributed_inference_blockid_diag_prints = 0


def _distributed_inference_load_kv_scale_override():
    global _distributed_inference_kv_scale_override
    if _distributed_inference_kv_scale_override is not None:
        return _distributed_inference_kv_scale_override
    path = os.environ.get("VLLM_KV_SCALE_OVERRIDE", "")
    scales = {}
    if path and os.path.exists(path):
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                scales = json.load(f)
            print(f"[AITER-FA-KV-SCALE] loaded {len(scales)} layers from {path}",
                  flush=True)
        except Exception as exc:
            print(f"[AITER-FA-KV-SCALE] failed to load {path}: {exc}",
                  flush=True)
            scales = {}
    _distributed_inference_kv_scale_override = scales
    return scales


def _distributed_inference_scale_to_float(value):
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


def _distributed_inference_layer_scale_entry(layer):
    scales = _distributed_inference_load_kv_scale_override()
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
            import re
            match = re.search(r"\.layers\.(\d+)\.", layer_name)
            if match:
                candidates.append(match.group(1))
        except Exception:
            pass
    for key in candidates:
        if key in scales:
            return scales[key]
    return None


def _distributed_inference_fill_scale_tensor(dst, value):
    value_float = _distributed_inference_scale_to_float(value)
    if value_float is None or dst is None:
        return None
    if hasattr(dst, "fill_"):
        dst.fill_(value_float)
        return dst
    return value_float


def _distributed_inference_tensors_alias(left, right):
    try:
        return (
            left is not None
            and right is not None
            and hasattr(left, "data_ptr")
            and hasattr(right, "data_ptr")
            and left.data_ptr() == right.data_ptr()
        )
    except Exception:
        return False


def _distributed_inference_break_scale_alias(attn_metadata):
    global _distributed_inference_kv_scale_alias_logged
    if not _distributed_inference_tensors_alias(
        attn_metadata.k_scale, attn_metadata.v_scale
    ):
        return False
    attn_metadata.v_scale = torch.empty_like(attn_metadata.k_scale)
    if not _distributed_inference_kv_scale_alias_logged:
        print(
            "[AITER-FA-KV-SCALE] split aliased metadata k_scale/v_scale tensors",
            flush=True,
        )
        _distributed_inference_kv_scale_alias_logged = True
    return True


def _distributed_inference_apply_layer_kv_scale_override(layer):
    """Apply JSON/checkpoint scalar KV scales to layer tensors before cache update."""
    entry = _distributed_inference_layer_scale_entry(layer)
    if entry:
        k_src = entry.get("k_scale", entry.get("key_scale"))
        v_src = entry.get("v_scale", entry.get("value_scale"))
    else:
        k_src = getattr(layer, "_k_scale", None)
        v_src = getattr(layer, "_v_scale", None)

    k_float = _distributed_inference_scale_to_float(k_src)
    v_float = _distributed_inference_scale_to_float(v_src)
    if k_float is None or v_float is None:
        return

    _distributed_inference_fill_scale_tensor(getattr(layer, "_k_scale", None),
                                             k_float)
    _distributed_inference_fill_scale_tensor(getattr(layer, "_v_scale", None),
                                             v_float)


def _distributed_inference_apply_shuffle_kv_scales(layer, attn_metadata):
    """Fill vLLM 0.23 shuffle metadata scales from layer/checkpoint scales.

    vLLM 0.23 moved SHUFFLE KV scales into AiterFlashAttentionMetadataBuilder
    and initializes them to ones. Cross-platform PD needs the active per-layer
    scale tensor, otherwise the SHUFFLE attention path decodes fp8 KV with
    scale 1.0.
    """
    global _distributed_inference_kv_scale_logged
    if attn_metadata is None or not rocm_aiter_ops.is_shuffle_kv_cache_enabled():
        return
    if attn_metadata.k_scale is None or attn_metadata.v_scale is None:
        return

    entry = _distributed_inference_layer_scale_entry(layer)
    if entry:
        k_src = entry.get("k_scale", entry.get("key_scale"))
        v_src = entry.get("v_scale", entry.get("value_scale"))
    else:
        k_src = getattr(layer, "_k_scale", None)
        v_src = getattr(layer, "_v_scale", None)

    k_float = _distributed_inference_scale_to_float(k_src)
    v_float = _distributed_inference_scale_to_float(v_src)
    if k_float is None or v_float is None:
        return

    _distributed_inference_break_scale_alias(attn_metadata)
    _distributed_inference_fill_scale_tensor(attn_metadata.k_scale, k_float)
    _distributed_inference_fill_scale_tensor(attn_metadata.v_scale, v_float)

    # Keep local decode-side cache updates on the same scalar values when a JSON
    # override is active. If no override is active this is a no-op fill of the
    # existing layer tensors.
    _distributed_inference_fill_scale_tensor(getattr(layer, "_k_scale", None),
                                             k_float)
    _distributed_inference_fill_scale_tensor(getattr(layer, "_v_scale", None),
                                             v_float)

    _distributed_inference_maybe_log_block_table(attn_metadata)

    if not _distributed_inference_kv_scale_logged:
        print(
            "[AITER-FA-KV-SCALE] metadata scale active: "
            f"k={k_float} v={v_float} override={'yes' if entry else 'no'}",
            flush=True,
        )
        _distributed_inference_kv_scale_logged = True


def _distributed_inference_tensor_preview(value, limit=8):
    try:
        if value is None:
            return None
        tensor = value
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "flatten"):
            flat = tensor.flatten()
            n = min(int(flat.numel()), limit)
            return flat[:n].cpu().tolist()
    except Exception as exc:
        return f"error:{exc}"
    return None


def _distributed_inference_maybe_log_block_table(attn_metadata):
    global _distributed_inference_blockid_diag_prints
    if os.environ.get("VLLM_NHD_SHUFFLE_BLOCKID_DIAG", "0") != "1":
        return
    limit = int(os.environ.get("VLLM_NHD_SHUFFLE_BLOCKID_DIAG_LIMIT", "8"))
    if _distributed_inference_blockid_diag_prints >= limit:
        return
    block_table = getattr(attn_metadata, "block_table", None)
    if block_table is None:
        block_table = getattr(attn_metadata, "block_tables", None)
    slot_mapping = getattr(attn_metadata, "slot_mapping", None)
    _distributed_inference_blockid_diag_prints += 1
    print(
        "[NHD->SHUFFLE BLOCKID] aiter "
        f"num_decode={getattr(attn_metadata, 'num_decode_tokens', None)} "
        f"num_extend={getattr(attn_metadata, 'num_extend_tokens', None)} "
        f"block_table_shape={getattr(block_table, 'shape', None)} "
        f"block_table_head={_distributed_inference_tensor_preview(block_table)} "
        f"slot_mapping_head={_distributed_inference_tensor_preview(slot_mapping)}",
        flush=True,
    )
'''

V023_FORWARD_ANCHOR = """\
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_extend_tokens = attn_metadata.num_extend_tokens
        if not attn_metadata.use_cascade:"""

V023_FORWARD_PATCHED = """\
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_extend_tokens = attn_metadata.num_extend_tokens
        _distributed_inference_apply_shuffle_kv_scales(layer, attn_metadata)
        if not attn_metadata.use_cascade:"""

if V023_MARKER in aiter_content:
    print("  v0.23 metadata scale helper: already patched")
    if "_distributed_inference_break_scale_alias" not in aiter_content:
        aiter_content = aiter_content.replace(
            "_distributed_inference_kv_scale_logged = False\n"
            "_distributed_inference_blockid_diag_prints = 0\n",
            "_distributed_inference_kv_scale_logged = False\n"
            "_distributed_inference_kv_scale_alias_logged = False\n"
            "_distributed_inference_blockid_diag_prints = 0\n",
            1,
        )
        aiter_content = aiter_content.replace(
            "def _distributed_inference_apply_layer_kv_scale_override(layer):\n",
            "def _distributed_inference_tensors_alias(left, right):\n"
            "    try:\n"
            "        return (\n"
            "            left is not None\n"
            "            and right is not None\n"
            "            and hasattr(left, \"data_ptr\")\n"
            "            and hasattr(right, \"data_ptr\")\n"
            "            and left.data_ptr() == right.data_ptr()\n"
            "        )\n"
            "    except Exception:\n"
            "        return False\n\n\n"
            "def _distributed_inference_break_scale_alias(attn_metadata):\n"
            "    global _distributed_inference_kv_scale_alias_logged\n"
            "    if not _distributed_inference_tensors_alias(\n"
            "        attn_metadata.k_scale, attn_metadata.v_scale\n"
            "    ):\n"
            "        return False\n"
            "    attn_metadata.v_scale = torch.empty_like(attn_metadata.k_scale)\n"
            "    if not _distributed_inference_kv_scale_alias_logged:\n"
            "        print(\n"
            "            \"[AITER-FA-KV-SCALE] split aliased metadata k_scale/v_scale tensors\",\n"
            "            flush=True,\n"
            "        )\n"
            "        _distributed_inference_kv_scale_alias_logged = True\n"
            "    return True\n\n\n"
            "def _distributed_inference_apply_layer_kv_scale_override(layer):\n",
            1,
        )
        aiter_content = aiter_content.replace(
            "    _distributed_inference_fill_scale_tensor(attn_metadata.k_scale, k_float)\n"
            "    _distributed_inference_fill_scale_tensor(attn_metadata.v_scale, v_float)\n",
            "    _distributed_inference_break_scale_alias(attn_metadata)\n"
            "    _distributed_inference_fill_scale_tensor(attn_metadata.k_scale, k_float)\n"
            "    _distributed_inference_fill_scale_tensor(attn_metadata.v_scale, v_float)\n",
            1,
        )
        changes_made += 1
        print("  v0.23 metadata scale helper: patched aliased K/V scale tensors")
elif "_kvscale_tensors" in aiter_content:
    print("  get_static_kvscale: already patched")
elif OLD_KVSCALE in aiter_content:
    aiter_content = aiter_content.replace(OLD_KVSCALE, NEW_KVSCALE)
    changes_made += 1
    print("  get_static_kvscale: patched (removed @lru_cache, reusable tensors)")
elif V023_FORWARD_ANCHOR in aiter_content and "k_scale=self.scale" in aiter_content:
    if "import os\n" not in aiter_content and "import os," not in aiter_content:
        aiter_content = aiter_content.replace(
            "from typing import ClassVar\n\n",
            "from typing import ClassVar\n\nimport os\n",
            1,
        )
        changes_made += 1
        print("  Added 'import os' for v0.23 metadata scale helper")
    helper_anchor = "logger = init_logger(__name__)\n"
    if helper_anchor in aiter_content:
        aiter_content = aiter_content.replace(
            helper_anchor, helper_anchor + V023_HELPERS, 1)
    else:
        aiter_content = aiter_content.replace(
            "_CP_TOKENS_PER_ITER_ROCM = 32 * 1024\n",
            "_CP_TOKENS_PER_ITER_ROCM = 32 * 1024\n" + V023_HELPERS,
            1,
        )
    aiter_content = aiter_content.replace(
        V023_FORWARD_ANCHOR, V023_FORWARD_PATCHED, 1)
    changes_made += 1
    print("  v0.23 metadata scale helper: patched")
else:
    print("  ERROR: Could not find original get_static_kvscale")
    print("  Expected text not found and v0.23 metadata-scale anchor is absent.")
    sys.exit(1)



OLD_SCALE = """\
            if USING_SHUFFLE_LAYOUT:
                num_blocks, block_size, num_kv_heads, head_size = key_cache.shape

                k_scale, v_scale = get_static_kvscale(1.0, 1.0,  num_kv_heads,
                        num_blocks,
                        block_size,
                        kv_cache.device)
                
                layer._k_scale = k_scale
                layer._v_scale = v_scale"""

NEW_SCALE = """\
            if USING_SHUFFLE_LAYOUT:
                num_blocks, block_size, num_kv_heads, head_size = key_cache.shape

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
                        else float(_vs))
                k_scale, v_scale = get_static_kvscale(
                    layer._orig_k_scale_val, layer._orig_v_scale_val,
                    num_kv_heads, num_blocks, block_size, kv_cache.device)

                layer._k_scale = k_scale
                layer._v_scale = v_scale"""

if V023_MARKER in aiter_content:
    print("  Scale override: handled by v0.23 metadata scale helper")
elif "_orig_k_scale_val" in aiter_content:
    print("  Scale override: already patched")
elif OLD_SCALE in aiter_content:
    aiter_content = aiter_content.replace(OLD_SCALE, NEW_SCALE)
    changes_made += 1
    print("  Scale override: patched (uses checkpoint k_scale/v_scale per layer)")
else:
    print("  ERROR: Could not find scale override block in forward()")
    print("  Expected text not found. File may have been modified.")
    sys.exit(1)









PATCH_LOCAL_UPDATE_SCALE = (
    os.environ.get("VLLM_SHUFFLE_SCALE_LOCAL_UPDATE", "0") == "1"
)

OLD_CP_SHUFFLE_SCALE = """\
            if DEQUANT:
                k_scale = 1.0
                v_scale = 1.0
                k_reg = k_reg.to(tl.float32) * k_scale
                v_reg = v_reg.to(tl.float32) * v_scale"""

NEW_CP_SHUFFLE_SCALE = """\
            if DEQUANT:
                scale_offset = (
                    block_id * num_heads * PAGE_SIZE
                    + head_id * PAGE_SIZE
                    + slot_id
                )
                k_scale = tl.load(k_scale_ptr + scale_offset)
                v_scale = tl.load(v_scale_ptr + scale_offset)
                k_reg = k_reg.to(tl.float32) * k_scale
                v_reg = v_reg.to(tl.float32) * v_scale"""

OLD_CACHE_SHUFFLE_SCALE = """\
        if QUANT:
            k_scale = 1.0
            v_scale = 1.0
            k_dtype = key_cache_ptr.type.element_ty
            v_dtype = value_cache_ptr.type.element_ty
            k_val = (k_val.to(tl.float32) / k_scale).to(k_dtype)
            v_val = (v_val.to(tl.float32) / v_scale).to(v_dtype)"""

BAD_CACHE_SHUFFLE_SCALE = """\
        if QUANT:
            scale_offset = (
                block_id * num_kv_heads * block_size
                + head_id * block_size
                + block_offset
            )
            k_scale = tl.load(k_scale_ptr + scale_offset)
            v_scale = tl.load(v_scale_ptr + scale_offset)
            k_dtype = key_cache_ptr.type.element_ty
            v_dtype = value_cache_ptr.type.element_ty
            k_val = (k_val.to(tl.float32) / k_scale).to(k_dtype)
            v_val = (v_val.to(tl.float32) / v_scale).to(v_dtype)"""

NEW_CACHE_SHUFFLE_SCALE = """\
        if QUANT:
            k_scale = tl.load(k_scale_ptr)
            v_scale = tl.load(v_scale_ptr)
            k_dtype = key_cache_ptr.type.element_ty
            v_dtype = value_cache_ptr.type.element_ty
            k_val = (k_val.to(tl.float32) / k_scale).to(k_dtype)
            v_val = (v_val.to(tl.float32) / v_scale).to(v_dtype)"""

cp_scale_patched = "block_id * num_heads * PAGE_SIZE" in aiter_content
cache_scale_patched = (
    "k_scale = tl.load(k_scale_ptr)\n            v_scale = tl.load(v_scale_ptr)\n"
    "            k_dtype = key_cache_ptr.type.element_ty" in aiter_content
)

patched_helpers = 0
if cp_scale_patched:
    print("  cp_mha_gather_cache SHUFFLE scale: already patched")
elif OLD_CP_SHUFFLE_SCALE in aiter_content:
    aiter_content = aiter_content.replace(
        OLD_CP_SHUFFLE_SCALE, NEW_CP_SHUFFLE_SCALE, 1)
    patched_helpers += 1
    changes_made += 1
    print("  cp_mha_gather_cache SHUFFLE scale: patched")
else:
    print("  WARNING: Could not find cp_mha_gather_cache SHUFFLE scale block")

if PATCH_LOCAL_UPDATE_SCALE:
    if cache_scale_patched:
        print("  reshape_and_cache_shuffle scale: already patched")
    elif OLD_CACHE_SHUFFLE_SCALE in aiter_content:
        aiter_content = aiter_content.replace(
            OLD_CACHE_SHUFFLE_SCALE, NEW_CACHE_SHUFFLE_SCALE, 1)
        patched_helpers += 1
        changes_made += 1
        print("  reshape_and_cache_shuffle scale: patched")
    elif BAD_CACHE_SHUFFLE_SCALE in aiter_content:
        aiter_content = aiter_content.replace(
            BAD_CACHE_SHUFFLE_SCALE, NEW_CACHE_SHUFFLE_SCALE, 1)
        patched_helpers += 1
        changes_made += 1
        print("  reshape_and_cache_shuffle scale: repaired scalar pointer load")
    else:
        print("  WARNING: Could not find reshape_and_cache_shuffle scale block")
else:
    print("  reshape_and_cache_shuffle scale: left at AITER scale=1 convention")

KV_UPDATE_SCALE_ANCHOR = """\
        if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
            # We may calculate per token quant scale in
            # reshape_and_cache_shuffle_triton which might differ from
            # vllm's style when shuffle layout is used.
            k_scale = layer._k_scale
            v_scale = layer._v_scale"""

KV_UPDATE_SCALE_PATCHED = """\
        if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
            # Keep locally appended decode KV blocks on the same scalar scale
            # as the remote H200 fp8 prompt blocks before quantizing into cache.
            _distributed_inference_apply_layer_kv_scale_override(layer)
            # We may calculate per token quant scale in
            # reshape_and_cache_shuffle_triton which might differ from
            # vllm's style when shuffle layout is used.
            k_scale = layer._k_scale
            v_scale = layer._v_scale"""

if PATCH_LOCAL_UPDATE_SCALE:
    if "\n            _distributed_inference_apply_layer_kv_scale_override(layer)" in aiter_content:
        print("  SHUFFLE cache-update scale override: already patched")
    elif KV_UPDATE_SCALE_ANCHOR in aiter_content:
        aiter_content = aiter_content.replace(
            KV_UPDATE_SCALE_ANCHOR, KV_UPDATE_SCALE_PATCHED, 1)
        changes_made += 1
        print("  SHUFFLE cache-update scale override: patched")
    else:
        print("  WARNING: Could not find SHUFFLE cache-update scale override anchor")
else:
    print("  SHUFFLE cache-update scale override: disabled by default")

if not cp_scale_patched and patched_helpers == 0:
    print("  ERROR: Could not patch required SHUFFLE gather scale block")
    sys.exit(1)



if 'VLLM_ROCM_SHUFFLE_LAYOUT' in aiter_content:
    print("  USING_SHUFFLE_LAYOUT: already env-controlled")
elif 'USING_SHUFFLE_LAYOUT = True' in aiter_content:
    aiter_content = aiter_content.replace(
        'USING_SHUFFLE_LAYOUT = True',
        'USING_SHUFFLE_LAYOUT = os.environ.get("VLLM_ROCM_SHUFFLE_LAYOUT", "1") == "1"',
    )
    if "import os\n" not in aiter_content and "import os," not in aiter_content:
        aiter_content = aiter_content.replace(
            "import torch\n", "import os\nimport torch\n", 1)
        print("  Added 'import os'")
    changes_made += 1
    print("  USING_SHUFFLE_LAYOUT: patched to use VLLM_ROCM_SHUFFLE_LAYOUT env var")


with open(AITER_FILE, "w") as f:
    f.write(aiter_content)

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






print("\n--- Part 2: NHD→SHUFFLE conversion hook ---")

PATCH_DIR = os.path.dirname(NIXL_FILE)
PATCH_FILE = os.path.join(PATCH_DIR, "_fp8_kv_patch.py")

_EMBEDDED_ENV_DEFAULT_NAMES = (
    "VLLM_NHD_TO_SHUFFLE",
    "VLLM_NHD_SHUFFLE_FUSED",
    "VLLM_NHD_SHUFFLE_ASYNC",
    "VLLM_NHD_SHUFFLE_PERLAYER",
    "VLLM_FP8_KV_CROSS_PLATFORM",
    "VLLM_NHD_SHUFFLE_REQUANT",
    "VLLM_NHD_SHUFFLE_REQUANT_SCALE_PATH",
    "VLLM_NHD_SHUFFLE_REQUANT_TARGET_SCALE_PATH",
    "VLLM_NHD_SHUFFLE_SOURCE_FP8_DTYPE",
    "VLLM_NHD_SHUFFLE_TARGET_FP8_DTYPE",
    "VLLM_NHD_SHUFFLE_VERIFY",
    "VLLM_NHD_SHUFFLE_BLOCKID_DIAG",
    "VLLM_NHD_SHUFFLE_BLOCKID_DIAG_LIMIT",
    "VLLM_KV_SCALE_OVERRIDE",
)
_EMBEDDED_ENV_DEFAULTS = {
    name: os.environ.get(name, "") for name in _EMBEDDED_ENV_DEFAULT_NAMES
}

PATCH_CODE = r'''"""Post-transfer NHD-to-SHUFFLE KV cache layout conversion.

After H200 sends KV blocks via NIXL in NHD format, this hook rearranges
the bytes to SHUFFLE format so MI350X's AITER kernels read correct values.

SHUFFLE layout for FP8 (x = 16 // element_size = 16):
  K: NHD [block_size, heads, dim] -> [heads, dim//x, block_size, x]
  V: NHD [block_size, heads, dim] -> [heads, block_size//x, dim, x]

Pure byte reordering. Scale compatibility handled by rocm_aiter_fa.py fix.

Primary path: C++ fused HIP kernel (2 kernel launches for ALL layers).
Fallback: per-layer PyTorch ops (~480 kernel launches).

Environment variables:
  VLLM_NHD_TO_SHUFFLE=1  Enable conversion (for PD mode with SHUFFLE on)
  VLLM_NHD_TO_SHUFFLE=0  Disable (default; for standalone or SHUFFLE off)
  VLLM_NHD_SHUFFLE_FUSED=1  Try C++ fused kernel (default)
  VLLM_NHD_SHUFFLE_ASYNC=1  Run on separate HIP stream (default, all-at-once mode)
  VLLM_NHD_SHUFFLE_PERLAYER=1  Per-layer injection during forward pass (default)
"""
import os
import time
import logging
import torch

logger = logging.getLogger(__name__)

_GENERATED_ENV_DEFAULTS = __GENERATED_ENV_DEFAULTS__


def _env(name, default=""):
    value = os.environ.get(name, "")
    if value != "":
        return value
    value = _GENERATED_ENV_DEFAULTS.get(name, "")
    if value != "":
        return value
    return default


print(
    "[NHD->SHUFFLE ENV] "
    f"enabled={_env('VLLM_NHD_TO_SHUFFLE', '0')} "
    f"fused={_env('VLLM_NHD_SHUFFLE_FUSED', '1')} "
    f"async={_env('VLLM_NHD_SHUFFLE_ASYNC', '1')} "
    f"perlayer={_env('VLLM_NHD_SHUFFLE_PERLAYER', '1')} "
    f"requant={_env('VLLM_NHD_SHUFFLE_REQUANT', '0')} "
    f"source_scale={_env('VLLM_NHD_SHUFFLE_REQUANT_SCALE_PATH', '')} "
    f"target_scale={_env('VLLM_NHD_SHUFFLE_REQUANT_TARGET_SCALE_PATH', '')} "
    f"kv_override={_env('VLLM_KV_SCALE_OVERRIDE', '')}",
    flush=True,
)

_ENABLED = _env("VLLM_NHD_TO_SHUFFLE", "0") == "1"
_DIAG_DONE = False
_FUSED_MOD = None
_FUSED_TRIED = False
_CACHED_KV = None
_CONV_STREAM = None
_TOTAL_CONV_MS = 0.0
_TOTAL_CONV_CALLS = 0
_TOTAL_FAILED_XFERS = 0
_TOTAL_RECVING_SNAPSHOTS = 0
_TOTAL_SYNC_STALLS = 0

_BID_MAX = 2048
_BID_PINNED = None
_BID_GPU = None

_PERLAYER = _ENABLED and _env("VLLM_NHD_SHUFFLE_PERLAYER", "1") == "1"
_LAYER_PTR_CACHE = {}
_TOTAL_PERLAYER_CONV_CALLS = 0
_TOTAL_PERLAYER_CONV_MS = 0.0

_FN_TO_FNUZ = _env("VLLM_FP8_KV_CROSS_PLATFORM", "0") == "1"
_REQUANT_TO_SCALE1 = _env("VLLM_NHD_SHUFFLE_REQUANT", "0") == "1"
_SCALE_OVERRIDE = None
_TARGET_SCALE_OVERRIDE = None
_REQUANT_DIAG = False
_FN_TO_FNUZ_DIAG = False
_FP8_DTYPE_DIAG = False
_LEGACY_BLOCK_IDS_DIAG = False
_BLOCK_DUP_DIAG = False
_VERIFY_CONVERSION = _env("VLLM_NHD_SHUFFLE_VERIFY", "0") == "1"
_VERIFY_CONVERSION_LIMIT = int(_env("VLLM_NHD_SHUFFLE_VERIFY_LIMIT", "1"))
_VERIFY_CONVERSION_LAYER = _env("VLLM_NHD_SHUFFLE_VERIFY_LAYER", "")
_VERIFY_CONVERSION_COUNT = 0
_NHD_REF_DIAG = _env("VLLM_AITER_PA_NHD_REF_DIAG", "0") == "1"
_NHD_REF_LAYER = (_env("VLLM_NHD_REF_SNAPSHOT_LAYER", "")
                  or _env("VLLM_NHD_SHUFFLE_VERIFY_LAYER", ""))
_NHD_REF_LIMIT = int(_env("VLLM_NHD_REF_SNAPSHOT_LIMIT", "1"))
_NHD_REF_SNAPSHOT_COUNT = 0
_NHD_REF_SNAPSHOTS = {}
_BLOCKID_DIAG = _env("VLLM_NHD_SHUFFLE_BLOCKID_DIAG", "0") == "1"
_BLOCKID_DIAG_PRINTS = 0


def _container_len(value):
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        pass
    qsize = getattr(value, "qsize", None)
    if callable(qsize):
        try:
            return int(qsize())
        except Exception:
            return 0
    return 0


def _extend_flat_block_ids(out, bids):
    if not bids:
        return
    for bid in bids:
        if isinstance(bid, (list, tuple)):
            out.extend(int(x) for x in bid)
        else:
            out.append(int(bid))


def _dedupe_block_ids(block_ids):
    global _BLOCK_DUP_DIAG
    if not block_ids:
        return block_ids
    deduped = list(dict.fromkeys(block_ids))
    if len(deduped) != len(block_ids) and not _BLOCK_DUP_DIAG:
        _BLOCK_DUP_DIAG = True
        print(
            f"[NHD->SHUFFLE] duplicate block ids in conversion batch: "
            f"total={len(block_ids)} unique={len(deduped)} "
            f"dups={len(block_ids) - len(deduped)}",
            flush=True,
        )
    return deduped


def _flatten_block_ids(block_ids_list):
    out = []
    for bids in block_ids_list:
        _extend_flat_block_ids(out, bids)
    return _dedupe_block_ids(out)


def _short_block_ids(value, limit=8):
    out = []
    try:
        _extend_flat_block_ids(out, value)
    except Exception as exc:
        return f"error:{exc}"
    if len(out) > limit:
        return out[:limit] + [f"...(+{len(out) - limit})"]
    return out


def _maybe_log_completed_block_ids(req_id, logical_ids, physical_ids, convert_ids):
    global _BLOCKID_DIAG_PRINTS
    if not _BLOCKID_DIAG:
        return
    limit = int(_env("VLLM_NHD_SHUFFLE_BLOCKID_DIAG_LIMIT", "8"))
    if _BLOCKID_DIAG_PRINTS >= limit:
        return
    _BLOCKID_DIAG_PRINTS += 1
    print(
        "[NHD->SHUFFLE BLOCKID] complete "
        f"req={req_id} "
        f"logical={_short_block_ids(logical_ids)} "
        f"physical={_short_block_ids(physical_ids)} "
        f"convert={_short_block_ids(convert_ids)}",
        flush=True,
    )


def _load_scale_override():
    global _SCALE_OVERRIDE
    if _SCALE_OVERRIDE is not None:
        return _SCALE_OVERRIDE
    path = (_env("VLLM_NHD_SHUFFLE_REQUANT_SCALE_PATH", "")
            or _env("VLLM_KV_SCALE_OVERRIDE", ""))
    scales = {}
    if path and os.path.exists(path):
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                scales = json.load(f)
            print(f"[NHD->SHUFFLE REQUANT] loaded {len(scales)} scale entries from {path}", flush=True)
        except Exception as exc:
            print(f"[NHD->SHUFFLE REQUANT] failed to load {path}: {exc}", flush=True)
            scales = {}
    _SCALE_OVERRIDE = scales
    return scales


def _load_target_scale_override():
    global _TARGET_SCALE_OVERRIDE
    if _TARGET_SCALE_OVERRIDE is not None:
        return _TARGET_SCALE_OVERRIDE
    path = _env("VLLM_NHD_SHUFFLE_REQUANT_TARGET_SCALE_PATH", "")
    scales = {}
    if path and os.path.exists(path):
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                scales = json.load(f)
            print(f"[NHD->SHUFFLE REQUANT] loaded {len(scales)} target scale entries from {path}", flush=True)
        except Exception as exc:
            print(f"[NHD->SHUFFLE REQUANT] failed to load target scales {path}: {exc}", flush=True)
            scales = {}
    _TARGET_SCALE_OVERRIDE = scales
    return scales


def _layer_scale_entry(layer_name, layer_idx):
    scales = _load_scale_override()
    if not scales:
        return None
    candidates = [str(layer_idx), layer_name]
    try:
        import re
        match = re.search(r"\.layers\.(\d+)\.", layer_name)
        if match:
            candidates.insert(0, match.group(1))
    except Exception:
        pass
    if layer_name:
        candidates.extend([
            layer_name.replace(".self_attn.attn", ".self_attn"),
            layer_name.replace(".self_attn", ".self_attn.attn"),
        ])
    for key in candidates:
        if key in scales:
            return scales[key]
    return None


def _target_layer_scale_entry(layer_name, layer_idx):
    scales = _load_target_scale_override()
    if not scales:
        return None
    candidates = [str(layer_idx), layer_name]
    try:
        import re
        match = re.search(r"\.layers\.(\d+)\.", layer_name)
        if match:
            candidates.insert(0, match.group(1))
    except Exception:
        pass
    if layer_name:
        candidates.extend([
            layer_name.replace(".self_attn.attn", ".self_attn"),
            layer_name.replace(".self_attn", ".self_attn.attn"),
        ])
    for key in candidates:
        if key in scales:
            return scales[key]
    return None


def _scale_to_float(value):
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


def _fp8_dtype_from_name(name):
    value = (name or "").strip().lower()
    if value in ("e4m3fnuz", "fnuz", "float8_e4m3fnuz"):
        return getattr(torch, "float8_e4m3fnuz", None)
    if value in ("e4m3fn", "fn", "float8_e4m3fn"):
        return getattr(torch, "float8_e4m3fn", None)
    return None


def _source_fp8_dtype():
    return (_fp8_dtype_from_name(
        _env("VLLM_NHD_SHUFFLE_SOURCE_FP8_DTYPE", "float8_e4m3fn"))
        or getattr(torch, "float8_e4m3fn", None)
        or getattr(torch, "float8_e4m3fnuz", None))


def _target_fp8_dtype():
    explicit = _fp8_dtype_from_name(_env("VLLM_NHD_SHUFFLE_TARGET_FP8_DTYPE", ""))
    if explicit is not None:
        return explicit
    try:
        from vllm.platforms import current_platform
        dtype = current_platform.fp8_dtype()
        if dtype is not None:
            return dtype
    except Exception:
        pass
    return (_source_fp8_dtype()
            or getattr(torch, "float8_e4m3fn", None)
            or getattr(torch, "float8_e4m3fnuz", None))


def _fp8_dtype_label(dtype):
    if dtype is None:
        return "none"
    return str(dtype).replace("torch.", "")


def _requant_blocks_to_scale1(blocks, source_scale, target_scale=None):
    """Decode source fp8 bytes, then encode the platform fp8 cache dtype.

    If target_scale is omitted, the destination cache uses scale 1.0. If it is
    provided, values are divided by target_scale before conversion so the
    SHUFFLE attention metadata can multiply by that target scale.
    """
    global _FP8_DTYPE_DIAG
    if source_scale is None:
        return blocks
    src_dtype = _source_fp8_dtype()
    dst_dtype = _target_fp8_dtype()
    if src_dtype is None or dst_dtype is None:
        return blocks
    if not _FP8_DTYPE_DIAG:
        _FP8_DTYPE_DIAG = True
        print(
            "[NHD->SHUFFLE REQUANT] fp8 dtype "
            f"source={_fp8_dtype_label(src_dtype)} "
            f"target={_fp8_dtype_label(dst_dtype)}",
            flush=True,
        )

    orig_shape = blocks.shape
    raw = blocks.view(torch.uint8)
    vals = raw.reshape(-1).view(src_dtype).float()
    vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    vals.mul_(float(source_scale))
    if target_scale is not None and float(target_scale) > 0.0:
        vals.div_(float(target_scale))
    vals.clamp_(-240.0, 240.0)
    out = vals.to(dst_dtype)
    if blocks.dtype == torch.uint8:
        return out.view(torch.uint8).reshape(orig_shape)
    return out.reshape(orig_shape).to(blocks.dtype)




def _verify_tensor_values(label, before, after_reversed, source_scale, target_scale):
    src_dtype = _source_fp8_dtype()
    dst_dtype = _target_fp8_dtype()
    if src_dtype is None or dst_dtype is None or source_scale is None:
        return None
    if not (_REQUANT_TO_SCALE1 or _FN_TO_FNUZ):
        dst_dtype = src_dtype
    if target_scale is None:
        target_scale = 1.0
    before_vals = before.view(torch.uint8).reshape(-1).view(src_dtype).float()
    before_vals = torch.nan_to_num(before_vals, nan=0.0, posinf=0.0, neginf=0.0)
    before_vals.mul_(float(source_scale))
    after_vals = after_reversed.view(torch.uint8).reshape(-1).view(dst_dtype).float()
    after_vals = torch.nan_to_num(after_vals, nan=0.0, posinf=0.0, neginf=0.0)
    after_vals.mul_(float(target_scale))
    diff = (before_vals - after_vals).abs()
    return (
        label,
        float(diff.max().item()),
        float(diff.mean().item()),
        float(before_vals.abs().max().item()),
        float(after_vals.abs().max().item()),
    )


def _verify_tensor_bytes(label, before, after_reversed):
    try:
        before_bytes = before.contiguous().view(torch.uint8).reshape(-1)
        after_bytes = after_reversed.contiguous().view(torch.uint8).reshape(-1)
        if before_bytes.numel() != after_bytes.numel():
            return (
                label,
                False,
                -1,
                int(before_bytes.numel()),
                int(after_bytes.numel()),
            )
        neq = before_bytes != after_bytes
        mismatch_count = int(neq.sum().item())
        first_mismatch = -1
        if mismatch_count:
            first_mismatch = int(torch.nonzero(neq, as_tuple=False)[0].item())
        return (
            label,
            mismatch_count == 0,
            first_mismatch,
            mismatch_count,
            int(before_bytes.numel()),
        )
    except Exception as exc:
        return (label, False, f"error:{type(exc).__name__}:{exc}", -1, -1)


def _should_verify_conversion(name):
    if not _VERIFY_CONVERSION:
        return False
    if _VERIFY_CONVERSION_COUNT >= _VERIFY_CONVERSION_LIMIT:
        return False
    if _VERIFY_CONVERSION_LAYER and _VERIFY_CONVERSION_LAYER not in name:
        return False
    return True


def _should_capture_nhd_ref(name):
    if not _NHD_REF_DIAG:
        return False
    if _NHD_REF_SNAPSHOT_COUNT >= _NHD_REF_LIMIT:
        return False
    if _NHD_REF_LAYER and _NHD_REF_LAYER not in name:
        return False
    return True


def _scale_pair_for_layer(layer_name, layer_idx):
    entry = _layer_scale_entry(layer_name, layer_idx)
    if not entry:
        return 1.0, 1.0
    k_scale = _scale_to_float(entry.get("k_scale", entry.get("key_scale")))
    v_scale = _scale_to_float(entry.get("v_scale", entry.get("value_scale")))
    return (1.0 if k_scale is None else k_scale,
            1.0 if v_scale is None else v_scale)


def _record_nhd_ref_snapshot(name, layer_idx, indices, k_blocks, v_blocks,
                             block_size, num_kv_heads, head_dim):
    global _NHD_REF_SNAPSHOT_COUNT
    if not _should_capture_nhd_ref(name):
        return
    k_scale, v_scale = _scale_pair_for_layer(name, layer_idx)
    block_ids = [int(x) for x in indices.detach().cpu().tolist()]
    _NHD_REF_SNAPSHOTS[name] = {
        "block_ids": block_ids,
        "k": k_blocks.detach().clone(),
        "v": v_blocks.detach().clone(),
        "block_size": int(block_size),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "source_dtype": _fp8_dtype_label(_source_fp8_dtype()),
        "k_scale": float(k_scale),
        "v_scale": float(v_scale),
    }
    _NHD_REF_SNAPSHOT_COUNT += 1
    print(
        "[NHD->SHUFFLE NHD-REF] "
        f"captured layer={name} blocks={block_ids[:8]} "
        f"shape={list(k_blocks.shape)} dtype={_NHD_REF_SNAPSHOTS[name]['source_dtype']} "
        f"k_scale={float(k_scale)} v_scale={float(v_scale)}",
        flush=True,
    )


def get_nhd_ref_snapshot(layer_name):
    if not layer_name:
        return None
    candidates = [layer_name]
    candidates.extend([
        layer_name.replace(".self_attn.attn", ".self_attn"),
        layer_name.replace(".self_attn", ".self_attn.attn"),
    ])
    for key in candidates:
        snap = _NHD_REF_SNAPSHOTS.get(key)
        if snap is not None:
            return snap
    return None


def _maybe_verify_conversion(name, indices, k_before, v_before, k_view, v_view,
                             block_size, num_kv_heads, head_dim, x,
                             k_scale, v_scale, k_target_scale, v_target_scale):
    global _VERIFY_CONVERSION_COUNT
    if not _should_verify_conversion(name):
        return
    _VERIFY_CONVERSION_COUNT += 1
    n = min(int(indices.numel()), 2)
    if n <= 0:
        return
    idx = indices[:n]
    k_after = k_view.index_select(0, idx).clone()
    v_after = v_view.index_select(0, idx).clone()
    k_rev = (
        k_after.reshape(-1, num_kv_heads, head_dim // x, block_size, x)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(k_before[:n].shape)
    )
    v_rev = (
        v_after.reshape(-1, num_kv_heads, block_size // x, head_dim, x)
        .permute(0, 2, 4, 1, 3)
        .contiguous()
        .reshape(v_before[:n].shape)
    )
    stats = [
        _verify_tensor_values("K", k_before[:n], k_rev, k_scale, k_target_scale),
        _verify_tensor_values("V", v_before[:n], v_rev, v_scale, v_target_scale),
    ]
    parts = []
    byte_stats = [
        _verify_tensor_bytes("K", k_before[:n], k_rev),
        _verify_tensor_bytes("V", v_before[:n], v_rev),
    ]
    for label, ok, first, mismatches, total in byte_stats:
        parts.append(
            f"{label}:raw_equal={ok} first_mismatch={first} "
            f"mismatches={mismatches}/{total}"
        )
    for item in stats:
        if item is None:
            continue
        label, max_diff, mean_diff, before_abs, after_abs = item
        parts.append(
            f"{label}:max_diff={max_diff:.6g} mean_diff={mean_diff:.6g} "
            f"before_absmax={before_abs:.6g} after_absmax={after_abs:.6g}"
        )
    print(
        f"[NHD->SHUFFLE VERIFY] layer={name} blocks={idx.detach().cpu().tolist()} "
        + " ".join(parts),
        flush=True,
    )

def _is_flat_block_ids(block_ids):
    return (
        isinstance(block_ids, (list, tuple))
        and bool(block_ids)
        and not isinstance(block_ids[0], (list, tuple))
    )


def _normalize_legacy_block_ids(worker, metadata):
    """Wrap flat legacy NIXL block-id lists for newer grouped BlockIds API."""
    global _LEGACY_BLOCK_IDS_DIAG

    kv_cache_config = getattr(worker, "kv_cache_config", None)
    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not groups or len(groups) != 1:
        return

    changed = 0
    for meta in getattr(metadata, "reqs_to_recv", {}).values():
        remote = getattr(meta, "remote", None)
        if remote is not None and _is_flat_block_ids(getattr(remote, "block_ids", None)):
            remote.block_ids = [list(remote.block_ids)]
            changed += 1
        if _is_flat_block_ids(getattr(meta, "local_block_ids", None)):
            meta.local_block_ids = [list(meta.local_block_ids)]
            changed += 1
        if _is_flat_block_ids(getattr(meta, "local_physical_block_ids", None)):
            meta.local_physical_block_ids = [list(meta.local_physical_block_ids)]
            changed += 1

    if changed and not _LEGACY_BLOCK_IDS_DIAG:
        _LEGACY_BLOCK_IDS_DIAG = True
        print("[NIXL-COMPAT] Wrapped legacy flat block_ids for grouped NIXL API",
              flush=True)

# ---- C++ fused HIP kernel source (2 launches for ALL layers) ----

_KERNEL_CUDA_SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__global__ void nhd_to_shuffle_k(
    const int64_t* __restrict__ cache_ptrs,
    const int* __restrict__ block_ids,
    int num_blocks, int num_layers,
    int block_stride, int block_size, int num_heads, int head_dim, int x
) {
    int layer = blockIdx.y;
    int bidx  = blockIdx.x;
    if (layer >= num_layers || bidx >= num_blocks) return;
    unsigned char* base = (unsigned char*)cache_ptrs[layer];
    unsigned char* blk  = base + (long long)block_ids[bidx] * block_stride;
    extern __shared__ unsigned char smem[];
    int total = block_size * num_heads * head_dim;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        smem[i] = blk[i];
    __syncthreads();
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int bs = i / (num_heads * head_dim);
        int r  = i % (num_heads * head_dim);
        int h  = r / head_dim;
        int d  = r % head_dim;
        int dst = h * head_dim * block_size
                + (d / x) * block_size * x
                + bs * x + d % x;
        blk[dst] = smem[i];
    }
}

__global__ void nhd_to_shuffle_v(
    const int64_t* __restrict__ cache_ptrs,
    const int* __restrict__ block_ids,
    int num_blocks, int num_layers,
    int block_stride, int block_size, int num_heads, int head_dim, int x
) {
    int layer = blockIdx.y;
    int bidx  = blockIdx.x;
    if (layer >= num_layers || bidx >= num_blocks) return;
    unsigned char* base = (unsigned char*)cache_ptrs[layer];
    unsigned char* blk  = base + (long long)block_ids[bidx] * block_stride;
    extern __shared__ unsigned char smem[];
    int total = block_size * num_heads * head_dim;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        smem[i] = blk[i];
    __syncthreads();
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int bs = i / (num_heads * head_dim);
        int r  = i % (num_heads * head_dim);
        int h  = r / head_dim;
        int d  = r % head_dim;
        int dst = h * block_size * head_dim
                + (bs / x) * head_dim * x
                + d * x + bs % x;
        blk[dst] = smem[i];
    }
}

void launch_nhd_to_shuffle(
    torch::Tensor k_ptrs, torch::Tensor v_ptrs,
    torch::Tensor block_ids,
    int block_stride, int block_size,
    int num_heads, int head_dim, int x
) {
    int nb = block_ids.size(0);
    int nl = k_ptrs.size(0);
    if (nb == 0 || nl == 0) return;
    dim3 grid(nb, nl);
    int threads = 256;
    int smem = block_stride;
    auto stream = at::cuda::getCurrentCUDAStream();
    nhd_to_shuffle_k<<<grid, threads, smem, stream>>>(
        k_ptrs.data_ptr<int64_t>(), block_ids.data_ptr<int>(),
        nb, nl, block_stride, block_size, num_heads, head_dim, x);
    nhd_to_shuffle_v<<<grid, threads, smem, stream>>>(
        v_ptrs.data_ptr<int64_t>(), block_ids.data_ptr<int>(),
        nb, nl, block_stride, block_size, num_heads, head_dim, x);
}
"""

_KERNEL_CPP_SRC = """
#include <torch/extension.h>
void launch_nhd_to_shuffle(torch::Tensor, torch::Tensor, torch::Tensor,
                            int, int, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_nhd_to_shuffle, "NHD to SHUFFLE fused");
}
"""


def _load_fused_module():
    global _FUSED_MOD, _FUSED_TRIED
    if _FUSED_TRIED:
        return _FUSED_MOD
    _FUSED_TRIED = True
    try:
        from torch.utils.cpp_extension import load_inline
        print("[NHD->SHUFFLE] Compiling fused HIP kernel (this may take ~30s)...",
              flush=True)
        _FUSED_MOD = load_inline(
            name="nhd_shuffle_mi350x",
            cpp_sources=_KERNEL_CPP_SRC,
            cuda_sources=_KERNEL_CUDA_SRC,
            verbose=True,
        )
        print("[NHD->SHUFFLE] *** FUSED HIP KERNEL COMPILED OK *** "
              "(2 launches for all layers)", flush=True)
    except Exception as e:
        print(f"[NHD->SHUFFLE] *** FUSED KERNEL FAILED: {e} ***", flush=True)
        print("[NHD->SHUFFLE] Falling back to per-layer PyTorch ops "
              "(~480 kernel launches per batch)", flush=True)
        _FUSED_MOD = None
    return _FUSED_MOD


def _iter_cache_views(cache):
    """Yield (k_view, v_view, block_size, heads, dim, block_stride_bytes, layout)."""
    caches = cache if isinstance(cache, (list, tuple)) else [cache]
    for c in caches:
        if len(c.shape) != 5:
            continue
        es = c.element_size()
        # Older patched stacks exposed [2, num_blocks, block, heads, dim].
        if c.shape[0] == 2:
            _, _nb, block_size, num_kv_heads, head_dim = c.shape
            yield (
                c[0],
                c[1],
                block_size,
                num_kv_heads,
                head_dim,
                c[0].stride(0) * es,
                "kv_first",
            )
        # vLLM 0.24 AITER exposes [num_blocks, 2, block, heads, dim].
        elif c.shape[1] == 2:
            _nb, _, block_size, num_kv_heads, head_dim = c.shape
            yield (
                c[:, 0],
                c[:, 1],
                block_size,
                num_kv_heads,
                head_dim,
                c.stride(0) * es,
                "blocks_first",
            )


def _ensure_bid_buffers(n):
    """Pre-allocate pinned CPU + GPU tensors for block IDs (avoids device sync)."""
    global _BID_PINNED, _BID_GPU, _BID_MAX
    if _BID_PINNED is not None and _BID_MAX >= n:
        return
    _BID_MAX = max(n, 2048)
    _BID_PINNED = torch.empty(_BID_MAX, dtype=torch.int32).pin_memory()
    _BID_GPU = torch.empty(_BID_MAX, dtype=torch.int32, device="cuda")
    print(f"[NHD->SHUFFLE] Pre-allocated bid buffers: {_BID_MAX} slots", flush=True)



def _convert_fused(worker, block_ids_list):
    """2 kernel launches for ALL blocks x ALL layers x K+V.
    Uses pre-allocated pinned+GPU tensors to avoid implicit device sync."""
    global _DIAG_DONE, _CACHED_KV

    all_bids = _flatten_block_ids(block_ids_list)
    if not all_bids:
        return True

    n = len(all_bids)
    _ensure_bid_buffers(n)

    _BID_PINNED[:n].copy_(torch.tensor(all_bids, dtype=torch.int32))
    _BID_GPU[:n].copy_(_BID_PINNED[:n], non_blocking=True)
    bid_t = _BID_GPU[:n]

    if _CACHED_KV is None:
        if not hasattr(worker, 'device_kv_caches') or not worker.device_kv_caches:
            return False
        k_ptrs, v_ptrs = [], []
        bs_val = nkh_val = hd_val = x_val = stride_val = None
        layouts = {}
        for cache in worker.device_kv_caches.values():
            for k_view, v_view, bs, nkh, hd, stride, layout in _iter_cache_views(cache):
                es = k_view.element_size()
                x = 16 // es
                if hd % x != 0 or bs % x != 0:
                    continue
                if stride_val is None:
                    stride_val = stride
                    bs_val, nkh_val, hd_val = bs, nkh, hd
                    x_val = x
                elif (stride, bs, nkh, hd, x) != (stride_val, bs_val, nkh_val, hd_val, x_val):
                    print(
                        "[NHD->SHUFFLE] incompatible KV cache metadata: "
                        f"got stride={stride} bs={bs} heads={nkh} dim={hd} x={x}; "
                        f"expected stride={stride_val} bs={bs_val} heads={nkh_val} "
                        f"dim={hd_val} x={x_val}",
                        flush=True,
                    )
                    return False
                k_ptrs.append(k_view.data_ptr())
                v_ptrs.append(v_view.data_ptr())
                layouts[layout] = layouts.get(layout, 0) + 1
        if not k_ptrs:
            return False
        _CACHED_KV = (
            torch.tensor(k_ptrs, dtype=torch.int64, device="cuda"),
            torch.tensor(v_ptrs, dtype=torch.int64, device="cuda"),
            stride_val, bs_val, nkh_val, hd_val, x_val, layouts,
        )

    k_t, v_t, stride, bsz, nkh, hd, xv, layouts = _CACHED_KV
    _FUSED_MOD.launch(k_t, v_t, bid_t, stride, bsz, nkh, hd, xv)

    if not _DIAG_DONE:
        _DIAG_DONE = True
        print(f"[NHD->SHUFFLE] FUSED path active: layers={k_t.shape[0]} "
              f"blocks={n} stride={stride} bs={bsz} "
              f"heads={nkh} dim={hd} x={xv} layouts={layouts}", flush=True)
    return True


def _convert_python(worker, block_ids_list):
    """Fallback: per-layer PyTorch ops (~480 kernel launches)."""
    global _DIAG_DONE

    if not hasattr(worker, 'device_kv_caches') or not worker.device_kv_caches:
        return

    all_bids = _flatten_block_ids(block_ids_list)
    if not all_bids:
        return

    indices = torch.tensor(all_bids, device="cuda", dtype=torch.long)
    diag = not _DIAG_DONE
    converted = 0

    for layer_idx, (name, cache) in enumerate(worker.device_kv_caches.items()):
        scale_entry = _layer_scale_entry(name, layer_idx) if _REQUANT_TO_SCALE1 else None
        target_entry = _target_layer_scale_entry(name, layer_idx) if _REQUANT_TO_SCALE1 else None
        k_scale = _scale_to_float(scale_entry.get("k_scale", scale_entry.get("key_scale"))) if scale_entry else None
        v_scale = _scale_to_float(scale_entry.get("v_scale", scale_entry.get("value_scale"))) if scale_entry else None
        k_target_scale = _scale_to_float(target_entry.get("k_scale", target_entry.get("key_scale"))) if target_entry else None
        v_target_scale = _scale_to_float(target_entry.get("v_scale", target_entry.get("value_scale"))) if target_entry else None
        for k_view, v_view, block_size, num_kv_heads, head_dim, _stride, layout in _iter_cache_views(cache):
            x = 16 // k_view.element_size()

            if head_dim % x != 0 or block_size % x != 0:
                continue

            verify_layer = _should_verify_conversion(name)
            capture_nhd_ref = _should_capture_nhd_ref(name)
            k = k_view.index_select(0, indices).clone()
            k_before = k[:min(int(indices.numel()), 2)].clone() if verify_layer else None
            k_ref = k.clone() if capture_nhd_ref else None
            if _REQUANT_TO_SCALE1:
                k = _requant_blocks_to_scale1(k, k_scale, k_target_scale)
            ks = k.reshape(-1, block_size, num_kv_heads, head_dim // x, x)
            ks = ks.permute(0, 2, 3, 1, 4).contiguous()
            k_view.index_copy_(0, indices, ks.reshape(k.shape))

            v = v_view.index_select(0, indices).clone()
            v_before = v[:min(int(indices.numel()), 2)].clone() if verify_layer else None
            if capture_nhd_ref and k_ref is not None:
                _record_nhd_ref_snapshot(
                    name, layer_idx, indices, k_ref, v,
                    block_size, num_kv_heads, head_dim)
            if _REQUANT_TO_SCALE1:
                v = _requant_blocks_to_scale1(v, v_scale, v_target_scale)
            vs = v.reshape(-1, block_size // x, x, num_kv_heads, head_dim)
            vs = vs.permute(0, 3, 1, 4, 2).contiguous()
            v_view.index_copy_(0, indices, vs.reshape(v.shape))
            if k_before is not None and v_before is not None:
                _maybe_verify_conversion(
                    name, indices, k_before, v_before, k_view, v_view,
                    block_size, num_kv_heads, head_dim, x,
                    k_scale, v_scale, k_target_scale, v_target_scale)
            converted += 1

            if diag:
                print(
                    f"[NHD->SHUFFLE] PYTHON FALLBACK: {name} "
                    f"k_shape={list(k_view.shape)} v_shape={list(v_view.shape)} "
                    f"layout={layout} bs={block_size} heads={num_kv_heads} "
                    f"dim={head_dim} x={x} requant={_REQUANT_TO_SCALE1} "
                    f"k_scale={k_scale} v_scale={v_scale} "
                    f"k_target={k_target_scale} v_target={v_target_scale}",
                    flush=True,
                )

    if diag:
        _DIAG_DONE = True
        if converted == 0:
            print("[NHD->SHUFFLE] No compatible KV cache tensors found", flush=True)


def _convert_fn_to_fnuz(worker, block_ids_list):
    """Convert fp8 e4m3fn (NVIDIA, bias=7) to e4m3fnuz (AMD, bias=8) in-place.

    The exponent bias difference means every value is 2x off without conversion.
    Also, byte 0x80 is negative-zero in fn but NaN in fnuz."""
    global _FN_TO_FNUZ_DIAG
    if not _FN_TO_FNUZ:
        return

    fp8_fnuz = getattr(torch, "float8_e4m3fnuz", None)
    if fp8_fnuz is None:
        return
    if not hasattr(worker, 'device_kv_caches') or not worker.device_kv_caches:
        return

    all_bids = _flatten_block_ids(block_ids_list)
    if not all_bids:
        return

    indices = torch.tensor(all_bids, device="cuda", dtype=torch.long)
    diag = not _FN_TO_FNUZ_DIAG
    converted = 0

    for name, cache in worker.device_kv_caches.items():
        for k_view, v_view, *_meta in _iter_cache_views(cache):
            for kv_view in (k_view, v_view):
                blocks = kv_view.index_select(0, indices).clone()
                orig_shape = blocks.shape
                raw = blocks.view(torch.uint8).reshape(-1)
                raw[raw == 128] = 0
                f = raw.view(fp8_fnuz).float()
                f = torch.nan_to_num(f, nan=0.0)
                f.mul_(2.0)
                f.clamp_(-240.0, 240.0)
                result = f.to(fp8_fnuz)
                kv_view.index_copy_(
                    0, indices,
                    result.view(torch.uint8).reshape(orig_shape)
                    if kv_view.dtype == torch.uint8
                    else result.reshape(orig_shape))
                converted += 1

    if diag:
        _FN_TO_FNUZ_DIAG = True
        print(f"[FN->FNUZ] Converted {len(all_bids)} blocks across "
              f"{len(worker.device_kv_caches)} layers ({converted} K/V views)",
              flush=True)


def _convert_fn_to_fnuz_layer(worker, layer_name, block_ids_list):
    """Convert one layer's fp8 values from e4m3fn to e4m3fnuz in-place."""
    if not _FN_TO_FNUZ:
        return
    fp8_fnuz = getattr(torch, "float8_e4m3fnuz", None)
    if fp8_fnuz is None:
        return
    cache = worker.device_kv_caches.get(layer_name)
    if cache is None:
        return
    all_bids = _flatten_block_ids(block_ids_list)
    if not all_bids:
        return

    indices = torch.tensor(all_bids, device="cuda", dtype=torch.long)
    for k_view, v_view, *_meta in _iter_cache_views(cache):
        for kv_view in (k_view, v_view):
            blocks = kv_view.index_select(0, indices).clone()
            orig_shape = blocks.shape
            raw = blocks.view(torch.uint8).reshape(-1)
            raw[raw == 128] = 0
            f = raw.view(fp8_fnuz).float()
            f = torch.nan_to_num(f, nan=0.0)
            f.mul_(2.0)
            f.clamp_(-240.0, 240.0)
            result = f.to(fp8_fnuz)
            kv_view.index_copy_(
                0, indices,
                result.view(torch.uint8).reshape(orig_shape)
                if kv_view.dtype == torch.uint8
                else result.reshape(orig_shape))

def _get_conv_stream():
    global _CONV_STREAM
    if _CONV_STREAM is None:
        _CONV_STREAM = torch.cuda.Stream()
    return _CONV_STREAM


def _do_convert(worker, block_ids_list):
    use_fused = (_env("VLLM_NHD_SHUFFLE_FUSED", "1") != "0"
                 and not _REQUANT_TO_SCALE1)
    if use_fused:
        mod = _load_fused_module()
        if mod is not None:
            if _convert_fused(worker, block_ids_list):
                _convert_fn_to_fnuz(worker, block_ids_list)
                return
    _convert_python(worker, block_ids_list)
    _convert_fn_to_fnuz(worker, block_ids_list)


# ---- Per-layer conversion (one layer at a time during forward pass) ----

def _flatten_bids(block_ids_list):
    """Flatten list-of-lists of block IDs into a single list."""
    return _flatten_block_ids(block_ids_list)



def _ensure_layer_ptr(worker, layer_name):
    """Cache per-layer K/V GPU pointers and metadata for fused kernel."""
    global _LAYER_PTR_CACHE
    if layer_name in _LAYER_PTR_CACHE:
        return _LAYER_PTR_CACHE[layer_name]
    cache = worker.device_kv_caches.get(layer_name)
    if cache is None:
        return None
    for k_view, v_view, bs, nkh, hd, stride, layout in _iter_cache_views(cache):
        x = 16 // k_view.element_size()
        if hd % x != 0 or bs % x != 0:
            continue
        entry = (
            torch.tensor([k_view.data_ptr()], dtype=torch.int64, device="cuda"),
            torch.tensor([v_view.data_ptr()], dtype=torch.int64, device="cuda"),
            stride, bs, nkh, hd, x,
        )
        _LAYER_PTR_CACHE[layer_name] = entry
        print(f"[NHD->SHUFFLE PERLAYER] cached {layer_name} layout={layout} "
              f"stride={stride} bs={bs} heads={nkh} dim={hd} x={x}",
              flush=True)
        return entry
    return None

def _convert_single_layer_fused(worker, layer_name, block_ids_list):
    """Convert ONE layer's blocks using fused HIP kernel (2 launches)."""
    entry = _ensure_layer_ptr(worker, layer_name)
    if entry is None:
        return False
    all_bids = _flatten_bids(block_ids_list)
    if not all_bids:
        return True
    n = len(all_bids)
    _ensure_bid_buffers(n)
    _BID_PINNED[:n].copy_(torch.tensor(all_bids, dtype=torch.int32))
    _BID_GPU[:n].copy_(_BID_PINNED[:n], non_blocking=True)
    k_ptr, v_ptr, stride, bs, nkh, hd, x = entry
    _FUSED_MOD.launch(k_ptr, v_ptr, _BID_GPU[:n], stride, bs, nkh, hd, x)
    return True



def _convert_single_layer_python(worker, layer_name, block_ids_list):
    """Convert ONE layer's blocks using PyTorch ops (~6 kernel launches)."""
    cache = worker.device_kv_caches.get(layer_name)
    if cache is None:
        return
    all_bids = _flatten_bids(block_ids_list)
    if not all_bids:
        return
    layer_idx = 0
    try:
        import re
        match = re.search(r"\.layers\.(\d+)\.", layer_name)
        if match:
            layer_idx = int(match.group(1))
    except Exception:
        pass
    scale_entry = _layer_scale_entry(layer_name, layer_idx) if _REQUANT_TO_SCALE1 else None
    target_entry = _target_layer_scale_entry(layer_name, layer_idx) if _REQUANT_TO_SCALE1 else None
    k_scale = _scale_to_float(scale_entry.get("k_scale", scale_entry.get("key_scale"))) if scale_entry else None
    v_scale = _scale_to_float(scale_entry.get("v_scale", scale_entry.get("value_scale"))) if scale_entry else None
    k_target_scale = _scale_to_float(target_entry.get("k_scale", target_entry.get("key_scale"))) if target_entry else None
    v_target_scale = _scale_to_float(target_entry.get("v_scale", target_entry.get("value_scale"))) if target_entry else None
    indices = torch.tensor(all_bids, device="cuda", dtype=torch.long)
    for k_view, v_view, block_size, num_kv_heads, head_dim, _stride, _layout in _iter_cache_views(cache):
        x = 16 // k_view.element_size()
        if head_dim % x != 0 or block_size % x != 0:
            continue
        k = k_view.index_select(0, indices).clone()
        if _REQUANT_TO_SCALE1:
            k = _requant_blocks_to_scale1(k, k_scale, k_target_scale)
        ks = k.reshape(-1, block_size, num_kv_heads, head_dim // x, x)
        ks = ks.permute(0, 2, 3, 1, 4).contiguous()
        k_view.index_copy_(0, indices, ks.reshape(k.shape))
        v = v_view.index_select(0, indices).clone()
        if _REQUANT_TO_SCALE1:
            v = _requant_blocks_to_scale1(v, v_scale, v_target_scale)
        vs = v.reshape(-1, block_size // x, x, num_kv_heads, head_dim)
        vs = vs.permute(0, 3, 1, 4, 2).contiguous()
        v_view.index_copy_(0, indices, vs.reshape(v.shape))

def _do_convert_layer(worker, layer_name, block_ids_list):
    """Convert a single layer, choosing fused vs python."""
    use_fused = (_env("VLLM_NHD_SHUFFLE_FUSED", "1") != "0"
                 and not _REQUANT_TO_SCALE1)
    if use_fused:
        mod = _load_fused_module()
        if mod is not None:
            if _convert_single_layer_fused(worker, layer_name, block_ids_list):
                _convert_fn_to_fnuz_layer(worker, layer_name, block_ids_list)
                return
    _convert_single_layer_python(worker, layer_name, block_ids_list)
    _convert_fn_to_fnuz_layer(worker, layer_name, block_ids_list)


def _perlayer_wait_for_layer_load(worker, layer_name):
    """Per-layer NHD->SHUFFLE conversion called during forward pass.

    Converts only the specified layer's blocks on the conv stream,
    then ensures the default stream waits before that layer's attention runs.
    Each layer's conversion is ~0.18ms (14ms / 80 layers), which is
    much shorter than one layer's attention, avoiding contention."""
    global _TOTAL_PERLAYER_CONV_CALLS, _TOTAL_PERLAYER_CONV_MS
    if not hasattr(worker, '_pending_nhd_blocks') or not worker._pending_nhd_blocks:
        return
    if layer_name not in getattr(worker, 'device_kv_caches', {}):
        return

    t0 = time.perf_counter()
    stream = _get_conv_stream()
    with torch.cuda.stream(stream):
        _do_convert_layer(worker, layer_name, worker._pending_nhd_blocks)
    ev = torch.cuda.Event()
    ev.record(stream)
    torch.cuda.current_stream().wait_event(ev)

    _TOTAL_PERLAYER_CONV_CALLS += 1
    _TOTAL_PERLAYER_CONV_MS += (time.perf_counter() - t0) * 1000
    if _TOTAL_PERLAYER_CONV_CALLS in (1, 10, 100) or _TOTAL_PERLAYER_CONV_CALLS % 5000 == 0:
        avg = _TOTAL_PERLAYER_CONV_MS / _TOTAL_PERLAYER_CONV_CALLS
        print(f"[NHD->SHUFFLE PERLAYER] layer={layer_name} "
              f"calls={_TOTAL_PERLAYER_CONV_CALLS} avg={avg:.3f}ms",
              flush=True)


def _sync_prev_conversion(worker):
    """Sync previous async conversion. Returns True if had to stall."""
    if not hasattr(worker, '_nhd_conv_event'):
        return False
    stalled = not worker._nhd_conv_event.query()
    torch.cuda.current_stream().wait_event(worker._nhd_conv_event)
    return stalled


def _convert_nhd_to_shuffle(worker, block_ids_list):
    """Convert received KV blocks from NHD to SHUFFLE layout in-place.
    With ASYNC mode, launches on a separate HIP stream and returns
    immediately. The sync happens at the START of the next get_finished()
    call, overlapping conversion with the forward pass."""
    global _TOTAL_CONV_MS, _TOTAL_CONV_CALLS
    use_async = _env("VLLM_NHD_SHUFFLE_ASYNC", "1") != "0"

    t0 = time.perf_counter()

    if use_async:
        # Sync previous async conversion before reusing _BID_PINNED/_BID_GPU.
        if hasattr(worker, '_nhd_conv_event'):
            worker._nhd_conv_event.synchronize()
        stream = _get_conv_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _do_convert(worker, block_ids_list)
        if not hasattr(worker, '_nhd_conv_event'):
            worker._nhd_conv_event = torch.cuda.Event()
        worker._nhd_conv_event.record(stream)
    else:
        _do_convert(worker, block_ids_list)

    _TOTAL_CONV_CALLS += 1
    _TOTAL_CONV_MS += (time.perf_counter() - t0) * 1000
    if _TOTAL_CONV_CALLS in (1, 10, 100, 500) or _TOTAL_CONV_CALLS % 2000 == 0:
        avg = _TOTAL_CONV_MS / _TOTAL_CONV_CALLS
        print(f"[NHD->SHUFFLE] calls={_TOTAL_CONV_CALLS} "
              f"avg={avg:.2f}ms total={_TOTAL_CONV_MS:.0f}ms"
              f" stalls={_TOTAL_SYNC_STALLS}",
              flush=True)


def _install():
    """Wrap get_finished and wait_for_layer_load for NHD->SHUFFLE conversion."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import nixl_connector
        cls = nixl_connector.NixlConnectorWorker
        cls_connector = nixl_connector.NixlConnector
    except (ImportError, AttributeError):
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
                NixlConnectorWorker as cls,
            )
            from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
                NixlConnector as cls_connector,
            )
        except (ImportError, AttributeError):
            print("[NHD->SHUFFLE] NixlConnectorWorker not found - skipping",
                  flush=True)
            return

    if hasattr(cls.get_finished, '_nhd_shuffle_wrapped'):
        return

    _original = cls.get_finished

    def _wrapped_get_finished(self):
        global _TOTAL_FAILED_XFERS, _TOTAL_RECVING_SNAPSHOTS, _TOTAL_SYNC_STALLS

        # --- Short-circuit (Path 3): skip wrapper overhead when idle ---
        has_recving = bool(getattr(self, '_recving_transfers', {}))
        has_meta = bool(getattr(self, '_recving_metadata', {}))
        has_async_conv = hasattr(self, '_nhd_conv_event')
        if not has_recving and not has_meta and not has_async_conv and not _ENABLED:
            _TOTAL_RECVING_SNAPSHOTS += 1
            return _original(self)

        # Per-layer: clear consumed pending blocks from previous forward pass
        if _PERLAYER and hasattr(self, '_pending_nhd_blocks'):
            self._pending_nhd_blocks = []

        saved = {}
        saved_diag = {}
        if _ENABLED and has_meta:
            for req_id, meta in self._recving_metadata.items():
                if hasattr(meta, 'local_physical_block_ids'):
                    saved[req_id] = meta.local_physical_block_ids
                    if _BLOCKID_DIAG:
                        saved_diag[req_id] = (
                            getattr(meta, 'local_block_ids', None),
                            getattr(meta, 'local_physical_block_ids', None),
                        )

        n_invalid_before = _container_len(getattr(self, '_invalid_block_ids', None))

        result = _original(self)

        n_invalid_after = _container_len(getattr(self, '_invalid_block_ids', None))
        if n_invalid_after > n_invalid_before:
            _TOTAL_FAILED_XFERS += 1

        if _ENABLED and saved:
            completed = []
            for req_id, bids in saved.items():
                if req_id not in self._recving_metadata:
                    completed.append(bids)
                    logical_ids, physical_ids = saved_diag.get(req_id, (None, None))
                    _maybe_log_completed_block_ids(
                        req_id, logical_ids, physical_ids, bids)
            if completed:
                if _PERLAYER:
                    # Store for per-layer conversion during next forward pass
                    if not hasattr(self, '_pending_nhd_blocks'):
                        self._pending_nhd_blocks = []
                    self._pending_nhd_blocks.extend(completed)
                else:
                    _convert_nhd_to_shuffle(self, completed)
        _TOTAL_RECVING_SNAPSHOTS += 1
        if _TOTAL_RECVING_SNAPSHOTS in (100, 1000) or _TOTAL_RECVING_SNAPSHOTS % 5000 == 0:
            n_inflight = len(getattr(self, '_recving_metadata', {}))
            n_recving = len(getattr(self, '_recving_transfers', {}))
            perlayer_avg = (_TOTAL_PERLAYER_CONV_MS / max(1, _TOTAL_PERLAYER_CONV_CALLS))
            print(f"[NIXL-DIAG] steps={_TOTAL_RECVING_SNAPSHOTS} "
                  f"ok_xfers={_TOTAL_CONV_CALLS} "
                  f"failed_xfers={_TOTAL_FAILED_XFERS} "
                  f"sync_stalls={_TOTAL_SYNC_STALLS} "
                  f"perlayer={_TOTAL_PERLAYER_CONV_CALLS} "
                  f"perlayer_avg_ms={perlayer_avg:.2f} "
                  f"inflight_meta={n_inflight} "
                  f"inflight_rdma={n_recving}",
                  flush=True)

        return result

    _wrapped_get_finished._nhd_shuffle_wrapped = True
    cls.get_finished = _wrapped_get_finished

    # Wrap start_load_kv to sync previous async conversion BEFORE forward pass.
    # The forward pass runs between start_load_kv and get_finished, so the
    # sync must happen here (not in get_finished) to avoid reading KV blocks
    # that are still being converted on the side stream.
    _original_start_load = cls.start_load_kv

    def _wrapped_start_load_kv(self, metadata):
        global _TOTAL_SYNC_STALLS
        _normalize_legacy_block_ids(self, metadata)
        if not _PERLAYER and _sync_prev_conversion(self):
            _TOTAL_SYNC_STALLS += 1
        return _original_start_load(self, metadata)

    cls.start_load_kv = _wrapped_start_load_kv
    print("[NHD->SHUFFLE] Patched start_load_kv for async sync before forward",
          flush=True)

    # Wrap wait_for_layer_load for per-layer injection during forward pass
    if _PERLAYER:
        def _wrapped_wait_for_layer_load(self, layer_name):
            if self.connector_worker is not None:
                _perlayer_wait_for_layer_load(self.connector_worker, layer_name)
        _wrapped_wait_for_layer_load._nhd_shuffle_wrapped = True
        cls_connector.wait_for_layer_load = _wrapped_wait_for_layer_load
        print("[NHD->SHUFFLE] Patched wait_for_layer_load for per-layer "
              "injection (spreads conversion across forward pass)", flush=True)

        # On ROCm, opaque_attention_op() returns False -> use_direct_call=True
        # -> Attention.forward() calls impl.forward() directly, bypassing the
        # @maybe_transfer_kv_layer decorator. Patch the impl to call
        # wait_for_layer_load before the attention kernel reads KV cache.
        try:
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
                is_v1_kv_transfer_group,
            )
            from vllm.v1.attention.backends.rocm_aiter_fa import (
                AiterFlashAttentionImpl,
            )

            _original_impl_forward = AiterFlashAttentionImpl.forward

            def _wrapped_impl_forward(self, layer, query, key, value,
                                       kv_cache, attn_metadata, **kwargs):
                if (has_kv_transfer_group()
                        and is_v1_kv_transfer_group()
                        and attn_metadata is not None):
                    connector = get_kv_transfer_group()
                    if connector.has_connector_metadata():
                        connector.wait_for_layer_load(
                            getattr(layer, 'layer_name', ''))
                return _original_impl_forward(
                    self, layer, query, key, value, kv_cache, attn_metadata,
                    **kwargs)

            AiterFlashAttentionImpl.forward = _wrapped_impl_forward
            print("[NHD->SHUFFLE] Patched AiterFlashAttentionImpl.forward "
                  "for per-layer wait_for_layer_load", flush=True)
        except ImportError:
            print("[NHD->SHUFFLE] AiterFlashAttentionImpl not found, "
                  "skipping direct-call hook", flush=True)

    method = "UNKNOWN"
    if _ENABLED:
        use_fused = (_env("VLLM_NHD_SHUFFLE_FUSED", "1") != "0"
                     and not _REQUANT_TO_SCALE1)
        mod = None
        if use_fused:
            print("[NHD->SHUFFLE] Pre-compiling fused HIP kernel...", flush=True)
            mod = _load_fused_module()
        if _PERLAYER:
            method = ("PERLAYER_FUSED (2 launches/layer)" if mod
                      else "PERLAYER_PYTHON (~6 launches/layer)")
        else:
            method = ("FUSED_HIP (2 launches)" if mod
                      else "PYTHON_FALLBACK (~480 launches)")
    print(f"[NHD->SHUFFLE] Patched get_finished (enabled={_ENABLED}, "
          f"perlayer={_PERLAYER}, fn_to_fnuz={_FN_TO_FNUZ}, "
          f"requant={_REQUANT_TO_SCALE1}, method={method})", flush=True)


_install()
'''

PATCH_CODE = PATCH_CODE.replace(
    "__GENERATED_ENV_DEFAULTS__", repr(_EMBEDDED_ENV_DEFAULTS))

with open(PATCH_FILE, "w") as f:
    f.write(PATCH_CODE)

print(f"  Embedded NHD->SHUFFLE env defaults: {_EMBEDDED_ENV_DEFAULTS}")

print(f"  Created: {PATCH_FILE}")


with open(NIXL_FILE, "r") as f:
    nixl_content = f.read()

if "_fp8_kv_patch" in nixl_content:
    print("  Import already in NIXL patch target")
else:
    import_line = (
        "\ntry:\n"
        "    from . import _fp8_kv_patch  # NHD->SHUFFLE conversion\n"
        "except Exception:\n"
        "    pass\n"
    )
    nixl_content = nixl_content.rstrip() + "\n" + import_line
    with open(NIXL_FILE, "w") as f:
        f.write(nixl_content)
    print("  Added _fp8_kv_patch import to NIXL patch target")

nixl_cache = os.path.join(PATCH_DIR, "__pycache__")
if os.path.isdir(nixl_cache):
    for fn in os.listdir(nixl_cache):
        if "nixl_connector" in fn or "worker" in fn or "_fp8_kv_patch" in fn:
            os.remove(os.path.join(nixl_cache, fn))
            print(f"  Cleared: __pycache__/{fn}")


print(f"\n  Syntax check...")
try:
    compile(open(PATCH_FILE).read(), PATCH_FILE, "exec")
    print(f"  _fp8_kv_patch.py: OK")
except SyntaxError as e:
    print(f"  SYNTAX ERROR: {e}")
    sys.exit(1)






print(f"""
{'=' * 60}
Done! Summary of changes:
{'=' * 60}

1. rocm_aiter_fa.py:
   - get_static_kvscale: removed @lru_cache, uses shared tensor pair
   - SHUFFLE forward: saves per-layer checkpoint scale on first call,
     fills shared tensor with that scale each forward pass
   - USING_SHUFFLE_LAYOUT: env-controlled via VLLM_ROCM_SHUFFLE_LAYOUT

2. _fp8_kv_patch.py:
   - Hooks get_finished to intercept completed NIXL transfers
   - Converts received NHD-layout blocks to SHUFFLE byte order
   - Per-layer injection: spreads conversion across forward pass
     (one layer at a time, ~0.18ms each, overlaps with attention)
   - Fused C++ HIP kernel or PyTorch ops fallback
   - Short-circuit: skips wrapper overhead when no transfers active
   - Prints compilation status, timing, and diagnostics

For PD mode (H200 prefill -> MI350X decode), set in start script:
  export VLLM_ROCM_SHUFFLE_LAYOUT=1      # SHUFFLE on (full AITER perf)
  export VLLM_NHD_TO_SHUFFLE=1            # convert received NHD blocks
  export VLLM_KV_CACHE_LAYOUT=NHD         # vLLM-level layout hint
  export VLLM_NHD_SHUFFLE_PERLAYER=1      # per-layer injection (default)

For standalone MI350X (no cross-platform transfer):
  export VLLM_ROCM_SHUFFLE_LAYOUT=1   # (default, no other vars needed)

Restart the MI350X decoder after applying.
""")
