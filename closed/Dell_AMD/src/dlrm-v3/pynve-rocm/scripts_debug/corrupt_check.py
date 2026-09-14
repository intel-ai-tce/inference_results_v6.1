"""Corroborate the ComputeSetReplaceData warp-size bug WITHOUT a rebuild.

Prediction: with optimize_for_training=True (auto-insert active), the buggy
ComputeSetKernel/ComputeSetReplaceData (blockDim=32 vs warpSize=64) leaves half
of sets[] uninitialized -> inserts wrong rows into the GPU cache. Subsequent
*cache hits* then return WRONG embedding values (silent corruption at W=1; the
out-of-bounds index only SIGSEGVs once the VMM peer reservation shifts layout).

We init the table with known values, repeatedly look up a FIXED key set (so they
get cached), and check forward(keys) == W[keys] every iter.
"""
import os, sys
import torch
from pynve import nve
from pynve.torch.nve_layers import NVEmbedding, CacheType

dev = torch.device("cuda:0")
torch.cuda.set_device(dev)

rows = int(os.environ.get("ROWS", 200_000))
dim = int(os.environ.get("DIM", 512))
batch = int(os.environ.get("BATCH", 100_000))
iters = int(os.environ.get("ITERS", "40"))
opt_train = os.environ.get("OPT_TRAIN", "1") == "1"
cache_frac = float(os.environ.get("CACHE_FRAC", "0.5"))

dtype = torch.float16
row_bytes = dim * 2
cache_bytes = (int(cache_frac * rows * row_bytes) // row_bytes) * row_bytes

# Known table: row i = (i mod 997) as fp16, broadcast across dim (exactly representable).
W = ((torch.arange(rows, dtype=torch.float32) % 997).to(torch.float16)
     .view(rows, 1).expand(rows, dim).contiguous())

ctype = os.environ.get("CTYPE", "uvm")
if ctype == "nocache":
    emb = NVEmbedding(rows, dim, dtype, CacheType.NoCache,
                      weight_init=W.to(dev), device=dev)
else:
    emb = NVEmbedding(rows, dim, dtype, CacheType.LinearUVM,
                      gpu_cache_size=cache_bytes, weight_init=W.to(dev),
                      optimize_for_training=opt_train, device=dev)

g = torch.Generator(device="cpu").manual_seed(7)
keys = torch.randint(0, rows, (batch,), dtype=torch.int64, generator=g).to(dev)
ref = W.to(dev)[keys]

print(f"rows={rows} dim={dim} batch={batch} cache={cache_bytes/1e6:.0f}MB "
      f"frac={cache_frac} opt_train={opt_train} iters={iters}", flush=True)

first_bad = None
for it in range(iters):
    out = emb.forward(keys)
    torch.cuda.synchronize()
    mism = (out != ref)
    nbad = int(mism.any(dim=1).sum().item())
    if nbad and first_bad is None:
        first_bad = it
        # show one example
        bi = int(mism.any(dim=1).nonzero()[0].item())
        print(f"  iter {it}: {nbad}/{batch} rows WRONG  e.g. key={int(keys[bi])} "
              f"got={float(out[bi,0])} want={float(ref[bi,0])}", flush=True)
    if it % 10 == 0:
        print(f"  iter {it}: wrong_rows={nbad}", flush=True)

if first_bad is None:
    print("RESULT: all lookups CORRECT across all iters (no corruption observed)")
else:
    print(f"RESULT: CORRUPTION starting at iter {first_bad} (cache returns wrong rows) "
          f"-> confirms buggy auto-insert path")
