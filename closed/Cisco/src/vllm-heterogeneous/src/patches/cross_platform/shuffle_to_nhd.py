
"""
Patch NVIDIA vLLM for receiving SHUFFLE-layout KV cache from MI350X.

When MI350X (prefiller) sends KV cache via NIXL to an NVIDIA decoder,
the raw bytes are in SHUFFLE layout (AITER format). NVIDIA's FLASH_ATTN
backend expects NHD layout. This patch converts SHUFFLE→NHD in-place
on the NVIDIA GPU after each RDMA transfer completes.

The conversion is the exact inverse of the NHD→SHUFFLE conversion in
shuffle_kv.py:

  K: SHUFFLE [heads, dim//x, block_size, x] -> NHD [block_size, heads, dim]
  V: SHUFFLE [heads, block_size//x, dim, x] -> NHD [block_size, heads, dim]
  where x = 16 // element_size (16 for fp8)

Primary path: fused CUDA kernel (2 launches for ALL layers).
Fallback: per-layer PyTorch ops.

Environment variable:
  VLLM_SHUFFLE_TO_NHD=1  Enable conversion (set by start_server.sh)
  VLLM_SHUFFLE_TO_NHD=0  Disable (default)

Run INSIDE the NVIDIA container:
    python3 src/patches/cross_platform/shuffle_to_nhd.py
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

if NIXL_FILE is None:
    print("ERROR: Could not find NIXL connector module/package")
    sys.exit(1)

print("=" * 60)
print("SHUFFLE→NHD Conversion Patch (for NVIDIA decode from MI350X)")
print("=" * 60)
print(f"\n  NIXL patch target ({NIXL_LAYOUT}): {NIXL_FILE}")

PATCH_DIR = os.path.dirname(NIXL_FILE)
PATCH_FILE = os.path.join(PATCH_DIR, "_shuffle_to_nhd_patch.py")

PATCH_CODE = r'''"""Post-transfer SHUFFLE-to-NHD KV cache layout conversion.

After MI350X sends KV blocks via NIXL in SHUFFLE format, this hook
rearranges the bytes to NHD format so NVIDIA's FLASH_ATTN reads them
correctly.

This is the exact inverse of the NHD→SHUFFLE conversion in _fp8_kv_patch.py.

SHUFFLE→NHD for FP8 (x = 16 // element_size = 16):
  K: SHUFFLE [heads, dim//x, block_size, x] -> NHD [block_size, heads, dim]
  V: SHUFFLE [heads, block_size//x, dim, x] -> NHD [block_size, heads, dim]

Environment variables:
  VLLM_SHUFFLE_TO_NHD=1  Enable conversion
  VLLM_SHUFFLE_TO_NHD=0  Disable (default)
"""
import os
import time
import logging
import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("VLLM_SHUFFLE_TO_NHD", "0") == "1"
_DIAG_DONE = False
_FUSED_MOD = None
_FUSED_TRIED = False
_CACHED_KV = None
_CONV_STREAM = None
_TOTAL_CONV_MS = 0.0
_TOTAL_CONV_CALLS = 0
_BLOCK_DUP_DIAG = False

_BID_MAX = 2048
_BID_PINNED = None
_BID_GPU = None

# ---- Fused CUDA kernel: SHUFFLE→NHD (inverse of NHD→SHUFFLE) ----

_KERNEL_CUDA_SRC = """
#include <torch/extension.h>
#include <cstdint>

__global__ void shuffle_to_nhd_k(
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
    // Read from SHUFFLE position, write to NHD position
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int bs = i / (num_heads * head_dim);
        int r  = i % (num_heads * head_dim);
        int h  = r / head_dim;
        int d  = r % head_dim;
        int src = h * head_dim * block_size
                + (d / x) * block_size * x
                + bs * x + d % x;
        blk[i] = smem[src];
    }
}

__global__ void shuffle_to_nhd_v(
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
        int src = h * block_size * head_dim
                + (bs / x) * head_dim * x
                + d * x + bs % x;
        blk[i] = smem[src];
    }
}

void launch_shuffle_to_nhd(
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
    shuffle_to_nhd_k<<<grid, threads, smem>>>(
        k_ptrs.data_ptr<int64_t>(), block_ids.data_ptr<int>(),
        nb, nl, block_stride, block_size, num_heads, head_dim, x);
    shuffle_to_nhd_v<<<grid, threads, smem>>>(
        v_ptrs.data_ptr<int64_t>(), block_ids.data_ptr<int>(),
        nb, nl, block_stride, block_size, num_heads, head_dim, x);
}
"""

_KERNEL_CPP_SRC = """
#include <torch/extension.h>
void launch_shuffle_to_nhd(torch::Tensor, torch::Tensor, torch::Tensor,
                            int, int, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_shuffle_to_nhd, "SHUFFLE to NHD fused");
}
"""


def _load_fused_module():
    global _FUSED_MOD, _FUSED_TRIED
    if _FUSED_TRIED:
        return _FUSED_MOD
    _FUSED_TRIED = True
    try:
        from torch.utils.cpp_extension import load_inline
        print("[SHUFFLE->NHD] Compiling fused CUDA kernel ...", flush=True)
        _FUSED_MOD = load_inline(
            name="shuffle_to_nhd_nvidia",
            cpp_sources=_KERNEL_CPP_SRC,
            cuda_sources=_KERNEL_CUDA_SRC,
            verbose=True,
        )
        print("[SHUFFLE->NHD] *** FUSED CUDA KERNEL COMPILED OK ***", flush=True)
    except Exception as e:
        print(f"[SHUFFLE->NHD] *** FUSED KERNEL FAILED: {e} ***", flush=True)
        print("[SHUFFLE->NHD] Falling back to PyTorch ops", flush=True)
        _FUSED_MOD = None
    return _FUSED_MOD


def _ensure_bid_buffers(n):
    global _BID_PINNED, _BID_GPU, _BID_MAX
    if _BID_PINNED is not None and _BID_MAX >= n:
        return
    _BID_MAX = max(n, 2048)
    _BID_PINNED = torch.empty(_BID_MAX, dtype=torch.int32).pin_memory()
    _BID_GPU = torch.empty(_BID_MAX, dtype=torch.int32, device="cuda")


def _extend_flat_block_ids(out, bids):
    # vLLM 0.24 supplies nested physical block ids (one sublist per KV cache
    # group); older trees supplied a flat list. Flatten either shape to ints.
    if not bids:
        return
    for bid in bids:
        if isinstance(bid, (list, tuple)):
            out.extend(int(x) for x in bid)
        else:
            out.append(int(bid))


def _dedupe_block_ids(block_ids):
    # Converting the same physical block twice would double-permute (corrupt)
    # its KV bytes, so collapse duplicates before conversion.
    global _BLOCK_DUP_DIAG
    if not block_ids:
        return block_ids
    deduped = list(dict.fromkeys(block_ids))
    if len(deduped) != len(block_ids) and not _BLOCK_DUP_DIAG:
        _BLOCK_DUP_DIAG = True
        print(f"[SHUFFLE->NHD] duplicate block ids in conversion batch: "
              f"total={len(block_ids)} unique={len(deduped)} "
              f"dups={len(block_ids) - len(deduped)}", flush=True)
    return deduped


def _flatten_block_ids(block_ids_list):
    out = []
    for bids in block_ids_list:
        _extend_flat_block_ids(out, bids)
    return _dedupe_block_ids(out)


def _convert_fused(worker, block_ids_list):
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
        for cache in worker.device_kv_caches.values():
            cs = cache if isinstance(cache, (list, tuple)) else [cache]
            for c in cs:
                if c.shape[0] != 2:
                    continue
                _, _nb, bs, nkh, hd = c.shape
                es = c.element_size()
                if stride_val is None:
                    stride_val = bs * nkh * hd * es
                    bs_val, nkh_val, hd_val = bs, nkh, hd
                    x_val = 16 // es
                k_ptrs.append(c[0].data_ptr())
                v_ptrs.append(c[1].data_ptr())
        if not k_ptrs:
            return False
        _CACHED_KV = (
            torch.tensor(k_ptrs, dtype=torch.int64, device="cuda"),
            torch.tensor(v_ptrs, dtype=torch.int64, device="cuda"),
            stride_val, bs_val, nkh_val, hd_val, x_val,
        )

    k_t, v_t, stride, bsz, nkh, hd, xv = _CACHED_KV
    _FUSED_MOD.launch(k_t, v_t, bid_t, stride, bsz, nkh, hd, xv)

    if not _DIAG_DONE:
        _DIAG_DONE = True
        print(f"[SHUFFLE->NHD] FUSED path active: layers={k_t.shape[0]} "
              f"blocks={n} stride={stride} bs={bsz} "
              f"heads={nkh} dim={hd} x={xv}", flush=True)
    return True


def _convert_python(worker, block_ids_list):
    global _DIAG_DONE

    if not hasattr(worker, 'device_kv_caches') or not worker.device_kv_caches:
        return

    all_bids = _flatten_block_ids(block_ids_list)
    if not all_bids:
        return

    indices = torch.tensor(all_bids, device="cuda", dtype=torch.long)
    diag = not _DIAG_DONE

    for name, cache in worker.device_kv_caches.items():
        caches = cache if isinstance(cache, (list, tuple)) else [cache]
        for c in caches:
            if c.shape[0] != 2:
                continue

            _, _nb, block_size, num_kv_heads, head_dim = c.shape
            x = 16 // c.element_size()

            if head_dim % x != 0 or block_size % x != 0:
                continue

            # K: SHUFFLE [heads, dim//x, bs, x] -> NHD [bs, heads, dim]
            k = c[0].index_select(0, indices).clone()
            ks = k.reshape(-1, num_kv_heads, head_dim // x, block_size, x)
            ks = ks.permute(0, 3, 1, 2, 4).contiguous()
            c[0].index_copy_(0, indices, ks.reshape(k.shape))

            # V: SHUFFLE [heads, bs//x, dim, x] -> NHD [bs, heads, dim]
            v = c[1].index_select(0, indices).clone()
            vs = v.reshape(-1, num_kv_heads, block_size // x, head_dim, x)
            vs = vs.permute(0, 2, 4, 1, 3).contiguous()
            c[1].index_copy_(0, indices, vs.reshape(v.shape))

            if diag:
                print(f"[SHUFFLE->NHD] PYTHON FALLBACK: {name} shape={list(c.shape)} "
                      f"bs={block_size} heads={num_kv_heads} dim={head_dim} x={x}",
                      flush=True)

    if diag:
        _DIAG_DONE = True


def _do_convert(worker, block_ids_list):
    mod = _load_fused_module()
    if mod is not None:
        if _convert_fused(worker, block_ids_list):
            return
    _convert_python(worker, block_ids_list)


def _convert_shuffle_to_nhd(worker, block_ids_list):
    global _TOTAL_CONV_MS, _TOTAL_CONV_CALLS
    t0 = time.perf_counter()
    _do_convert(worker, block_ids_list)
    _TOTAL_CONV_CALLS += 1
    _TOTAL_CONV_MS += (time.perf_counter() - t0) * 1000
    if _TOTAL_CONV_CALLS in (1, 10, 100, 500) or _TOTAL_CONV_CALLS % 2000 == 0:
        avg = _TOTAL_CONV_MS / _TOTAL_CONV_CALLS
        print(f"[SHUFFLE->NHD] calls={_TOTAL_CONV_CALLS} "
              f"avg={avg:.2f}ms total={_TOTAL_CONV_MS:.0f}ms", flush=True)


def _install_hnd_handshake_compat(cls):
    """Allow the swap's HND metadata through while this patch owns conversion.

    vLLM 0.24 normally requires ``enable_permute_local_kv`` for a remote HND
    cache and local NHD cache. Enabling that persistently would make vLLM
    perform a second local permutation after this patch has converted the KV
    data, so enable it only while the stock handshake validates its remaining
    fields and restore the worker state before any transfer begins.
    """
    original = getattr(cls, "_validate_remote_agent_handshake", None)
    if original is None or hasattr(original, "_shuffle_to_nhd_wrapped"):
        return

    def _wrapped_validate(self, nixl_agent_meta, remote_tp_size):
        local_layout = getattr(self, "kv_cache_layout", None)
        remote_layout = getattr(nixl_agent_meta, "kv_cache_layout", None)
        use_custom_conversion = (
            _ENABLED and local_layout == "NHD" and remote_layout == "HND"
        )
        if not use_custom_conversion:
            return original(self, nixl_agent_meta, remote_tp_size)

        config = self.kv_transfer_config
        original_config_value = config.enable_permute_local_kv
        original_worker_value = getattr(self, "enable_permute_local_kv", False)
        try:
            # Let stock vLLM validate every other remote-agent field.
            config.enable_permute_local_kv = True
            result = original(self, nixl_agent_meta, remote_tp_size)
            print(
                "[SHUFFLE->NHD] Accepted remote HND metadata; "
                "custom conversion owns local permutation",
                flush=True,
            )
            return result
        finally:
            config.enable_permute_local_kv = original_config_value
            # Prevent vLLM built-in local permutation from double-converting.
            self.enable_permute_local_kv = original_worker_value

    _wrapped_validate._shuffle_to_nhd_wrapped = True
    cls._validate_remote_agent_handshake = _wrapped_validate
    print("[SHUFFLE->NHD] Patched HND->NHD handshake compatibility", flush=True)


def _install():
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import nixl_connector
        cls = nixl_connector.NixlConnectorWorker
    except (ImportError, AttributeError):
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
                NixlConnectorWorker as cls,
            )
        except (ImportError, AttributeError):
            print("[SHUFFLE->NHD] NixlConnectorWorker not found - skipping",
                  flush=True)
            return

    _install_hnd_handshake_compat(cls)

    if hasattr(cls.get_finished, '_shuffle_to_nhd_wrapped'):
        return

    _original = cls.get_finished

    def _wrapped_get_finished(self):
        saved = {}
        if _ENABLED:
            for req_id, meta in getattr(self, '_recving_metadata', {}).items():
                if hasattr(meta, 'local_physical_block_ids'):
                    saved[req_id] = meta.local_physical_block_ids

        result = _original(self)

        if _ENABLED and saved:
            completed = []
            for req_id, bids in saved.items():
                if req_id not in self._recving_metadata:
                    completed.append(bids)
            if completed:
                _convert_shuffle_to_nhd(self, completed)

        return result

    _wrapped_get_finished._shuffle_to_nhd_wrapped = True
    cls.get_finished = _wrapped_get_finished

    if _ENABLED:
        print("[SHUFFLE->NHD] Pre-compiling fused CUDA kernel...", flush=True)
        mod = _load_fused_module()
        method = "FUSED_CUDA" if mod else "PYTHON_FALLBACK"
    else:
        method = "DISABLED"
    print(f"[SHUFFLE->NHD] Patched get_finished (enabled={_ENABLED}, "
          f"method={method})", flush=True)


_install()
'''

with open(PATCH_FILE, "w") as f:
    f.write(PATCH_CODE)
print(f"  Created: {PATCH_FILE}")


with open(NIXL_FILE, "r") as f:
    nixl_content = f.read()

if "_shuffle_to_nhd_patch" in nixl_content:
    print("  Import already in NIXL patch target")
else:
    import_line = (
        "\ntry:\n"
        "    from . import _shuffle_to_nhd_patch  # SHUFFLE->NHD conversion\n"
        "except Exception:\n"
        "    pass\n"
    )
    nixl_content = nixl_content.rstrip() + "\n" + import_line
    with open(NIXL_FILE, "w") as f:
        f.write(nixl_content)
    print("  Added _shuffle_to_nhd_patch import to NIXL patch target")

nixl_cache = os.path.join(PATCH_DIR, "__pycache__")
if os.path.isdir(nixl_cache):
    for fn in os.listdir(nixl_cache):
        if ("nixl_connector" in fn or "worker" in fn or "__init__" in fn
                or "_shuffle_to_nhd" in fn):
            os.remove(os.path.join(nixl_cache, fn))
            print(f"  Cleared: __pycache__/{fn}")


print(f"\n  Syntax check...")
try:
    compile(open(PATCH_FILE).read(), PATCH_FILE, "exec")
    print(f"  _shuffle_to_nhd_patch.py: OK")
except SyntaxError as e:
    print(f"  SYNTAX ERROR: {e}")
    sys.exit(1)

print(f"""
{'=' * 60}
Done! SHUFFLE→NHD conversion patch installed.
{'=' * 60}

For cross-platform PD (MI350X prefill -> NVIDIA decode), set:
  export VLLM_SHUFFLE_TO_NHD=1

This is handled automatically by start_server.sh when using:
  ./start_server.sh --role decode --hardware h200 --remote-hardware mi350x
""")
