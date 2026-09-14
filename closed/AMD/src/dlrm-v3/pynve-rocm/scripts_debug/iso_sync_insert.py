"""Synchronous explicit insert at a faulting batch size, bypassing the async
auto-insert threadpool path. If this is CLEAN while auto-insert CRASHES, the bug
is in the async handoff; if this ALSO crashes, the bug is in ComputeSetReplaceData."""
import os, time, traceback
import torch
from pynve import nve
from pynve.torch.nve_layers import NVEmbedding, CacheType

rows  = int(os.environ.get("ROWS", 2_000_000))
dim   = int(os.environ.get("DIM", 512))
batch = int(os.environ.get("BATCH", 400000))
iters = int(os.environ.get("ITERS", 1))
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)

row_bytes = dim * 2
table_bytes = rows * row_bytes
big_cache = ((table_bytes + row_bytes) // row_bytes) * row_bytes

g = torch.Generator(device="cpu").manual_seed(0)
keys = torch.randint(0, rows, (batch,), dtype=torch.int64, generator=g).to(dev)
values = torch.randn(batch, dim, dtype=torch.float16, device=dev)
try:
    mb = nve.ManagedMemBlock(dim, rows, nve.DataType_t.Float16, [0])
    emb = NVEmbedding(rows, dim, torch.float16, CacheType.LinearUVM,
                      gpu_cache_size=big_cache, memblock=mb, device=dev,
                      optimize_for_training=False)
    for i in range(iters):
        emb.insert(keys, values, 0)
        torch.cuda.synchronize()
        print(f"[OK-insert] iter={i} batch={batch}", flush=True)
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    traceback.print_exc()
print("[after-scope]", flush=True)
