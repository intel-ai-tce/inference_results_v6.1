"""Minimal multi-rank MPIMemBlock LinearUVM repro for the 14.7 cache-fill fault.

Launch: mpirun -np 2 python3 repro_mpi_fault.py
Sweeps cache ratio and batch; prints per-rank OK/FAIL to localize the fault.
"""
import os, sys, traceback
import torch
from pynve import nve
from pynve.torch.nve_layers import NVEmbedding, CacheType

LR = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
WR = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
WS = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))


def log(msg):
    print(f"[rank {WR}/{WS} dev{LR}] {msg}", flush=True)


def run(rows, dim, batch, cache_frac, dtype=torch.float16, span="full"):
    torch.cuda.set_device(LR)
    dev = torch.device(f"cuda:{LR}")
    row_bytes = dim * (2 if dtype == torch.float16 else 4)
    table_bytes = rows * row_bytes
    cache_bytes = max(1, int(cache_frac * table_bytes))
    cache_bytes = (cache_bytes // row_bytes) * row_bytes
    ndt = nve.DataType_t.Float16 if dtype == torch.float16 else nve.DataType_t.Float32

    log(f"building MPIMemBlock rows={rows} dim={dim} table={table_bytes/1e9:.2f}GB cache_frac={cache_frac}")
    mb = nve.MPIMemBlock(dim, rows, ndt)  # collective across ranks
    try:
        va = mb.get_handle()
        log(f"MPIMemBlock built VA=0x{va:x} end=0x{va + table_bytes:x} size={table_bytes}")
    except Exception as _e:
        log(f"MPIMemBlock built (no VA: {_e})")
    opt_train = os.environ.get("OPT_TRAIN", "1") == "1"
    emb = NVEmbedding(rows, dim, dtype, CacheType.LinearUVM,
                      gpu_cache_size=cache_bytes, memblock=mb, device=dev,
                      optimize_for_training=opt_train)
    log(f"optimize_for_training={opt_train}")
    log("NVEmbedding built")

    g = torch.Generator(device="cpu").manual_seed(1234 + WR)
    if span == "local":
        # keys only within this rank's own shard => no peer reads
        shard = rows // WS
        keys = (torch.randint(0, shard, (batch,), dtype=torch.int64, generator=g) + WR * shard).to(dev)
    else:
        keys = torch.randint(0, rows, (batch,), dtype=torch.int64, generator=g).to(dev)

    iters = int(os.environ.get("ITERS", "1"))
    log(f"forward batch={batch} span={span} iters={iters} ...")
    sleep_ms = float(os.environ.get("SLEEP_MS", "0"))
    for it in range(iters):
        out = emb.forward(keys)
        torch.cuda.synchronize()
        if sleep_ms:
            import time as _t
            _t.sleep(sleep_ms / 1000.0)
        if it % 10 == 0:
            log(f"  iter {it} ok")
    log(f"forward OK out={tuple(out.shape)}")


if __name__ == "__main__":
    rows = int(os.environ.get("ROWS", 2_000_000))
    dim = int(os.environ.get("DIM", 512))
    batch = int(os.environ.get("BATCH", 100_000))
    cache_frac = float(os.environ.get("CACHE_FRAC", "0.05"))
    span = os.environ.get("SPAN", "full")
    try:
        run(rows, dim, batch, cache_frac, span=span)
        log("DONE OK")
    except Exception as e:
        log(f"EXCEPTION {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(3)
