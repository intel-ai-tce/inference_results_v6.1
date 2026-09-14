"""Locate the auto-insert race: forward at a faulting batch, sync, optionally
sleep, then tear down. If SLEEP makes the crash disappear, the async auto-insert
task is racing with teardown / buffer reuse."""
import os, sys, time, traceback
import torch
from pynve import nve
from pynve.torch.nve_layers import NVEmbedding, CacheType

rows  = int(os.environ.get("ROWS", 2_000_000))
dim   = int(os.environ.get("DIM", 512))
batch = int(os.environ.get("BATCH", 400000))
sleep = float(os.environ.get("SLEEP", "0"))
optim = os.environ.get("OPTIM", "0") == "1"
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)

row_bytes = dim * 2
table_bytes = rows * row_bytes
big_cache = ((table_bytes + row_bytes) // row_bytes) * row_bytes

g = torch.Generator(device="cpu").manual_seed(0)
keys = torch.randint(0, rows, (batch,), dtype=torch.int64, generator=g).to(dev)
try:
    mb = nve.ManagedMemBlock(dim, rows, nve.DataType_t.Float16, [0])
    emb = NVEmbedding(rows, dim, torch.float16, CacheType.LinearUVM,
                      gpu_cache_size=big_cache, memblock=mb, device=dev,
                      optimize_for_training=optim)
    out = emb.forward(keys)
    torch.cuda.synchronize()
    print(f"[OK-forward] batch={batch} optim={optim} sleep={sleep}", flush=True)
    if sleep > 0:
        time.sleep(sleep)
    print("[before-teardown]", flush=True)
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    traceback.print_exc()
print("[after-scope]", flush=True)
