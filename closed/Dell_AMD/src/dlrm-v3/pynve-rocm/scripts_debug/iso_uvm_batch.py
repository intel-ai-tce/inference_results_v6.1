"""Isolation test: reproduce the 14.7 cache-fill fault on a SINGLE GPU.

Goal: determine whether the GPU memory access fault is
  (a) MPIMemBlock peer-access specific, or
  (b) a large-batch / large-table gather-kernel bug that 14.6 (n_keys=1024) never hit.

We use ManagedMemBlock (single-device, no peer access) at the same dims/batch as
the faulting prod-cap config, sweeping batch size and cache/table ratio.
"""
import os, sys, traceback
import torch
from pynve import nve
from pynve.torch.nve_layers import NVEmbedding, CacheType


def run_case(tag, cache_type, rows, dim, batch, cache_bytes, dtype=torch.float16, seed=0):
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    keys = torch.randint(0, rows, (batch,), dtype=torch.int64, generator=g).to(dev)
    try:
        if cache_type == CacheType.NoCache:
            emb = NVEmbedding(rows, dim, dtype, CacheType.NoCache, device=dev)
        else:
            mb = nve.ManagedMemBlock(dim, rows, _nve_dtype(dtype), [0])
            emb = NVEmbedding(rows, dim, dtype, CacheType.LinearUVM,
                              gpu_cache_size=cache_bytes, memblock=mb, device=dev)
        iters = int(os.environ.get("ITERS", "1"))
        for _ in range(iters):
            out = emb.forward(keys)
            torch.cuda.synchronize()
        ratio = (cache_bytes / (rows * dim * _dsize(dtype))) if cache_bytes else float("nan")
        print(f"[OK]   {tag}: rows={rows} dim={dim} batch={batch} "
              f"cache={cache_bytes/1e6:.0f}MB ratio={ratio:.3f} out={tuple(out.shape)}")
        return True
    except Exception as e:
        print(f"[FAIL] {tag}: rows={rows} dim={dim} batch={batch} cache={cache_bytes/1e6:.0f}MB :: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def _nve_dtype(t):
    return nve.DataType_t.Float16 if t == torch.float16 else nve.DataType_t.Float32


def _dsize(t):
    return 2 if t == torch.float16 else 4


if __name__ == "__main__":
    rows = int(os.environ.get("ROWS", 2_000_000))
    dim = int(os.environ.get("DIM", 512))
    cases = os.environ.get("CASES", "noc,big,small").split(",")
    batches = [int(x) for x in os.environ.get("BATCHES", "1024,50000,75000,100000,200000").split(",")]
    row_bytes = dim * 2
    table_bytes = rows * row_bytes
    big_cache = ((table_bytes + row_bytes) // row_bytes) * row_bytes  # >= whole table
    small_cache = 64 * 1024 * 1024  # 64MB, << table

    print(f"=== rows={rows} dim={dim} table={table_bytes/1e9:.2f}GB row_bytes={row_bytes} ===")
    for b in batches:
        if "noc" in cases:
            run_case("NoCache", CacheType.NoCache, rows, dim, b, 0)
        if "big" in cases:
            run_case("UVM cache>=table", CacheType.LinearUVM, rows, dim, b, big_cache)
        if "small" in cases:
            run_case("UVM cache<<table", CacheType.LinearUVM, rows, dim, b, small_cache)
        sys.stdout.flush()
