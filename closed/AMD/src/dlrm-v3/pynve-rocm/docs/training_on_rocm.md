# Training with NVE on ROCm (gfx950)

This is the short version of how to *train* with the cached embedding tables on AMD
GPUs, and the one porting caveat (warp width) you need to know about. For lookup-only
inference see `overview.md` / `python_api.md`.

## How training works

NVE embedding layers are first-class `torch.autograd` participants. A lookup is a normal
differentiable op, and the optimizer update is intercepted and pushed straight into the
table — there is **no separate gradient buffer and no cache invalidation step**.

```python
import torch, pynve.torch.nve_layers as nve_layers

emb = nve_layers.NVEmbedding(
    num_embeddings, embedding_dim, torch.float32,
    nve_layers.CacheType.LinearUVM, gpu_cache_size=64 << 20,
)

keys   = torch.randint(0, num_embeddings, (B,), device="cuda")
out    = emb(keys)                 # forward  -> CacheEmbeddingOp -> native lookup
loss   = loss_fn(out, target)
loss.backward()                    # backward -> concat_backprop -> sparse grad on emb.weight
with torch.no_grad():
    emb.weight.add_(emb.weight.grad, alpha=-lr)   # step -> CachedTable.accumulate (write-through)
    emb.weight.grad = None
```

What happens under the hood:

* **Forward** — `NVEmbedding.forward` calls `CacheEmbeddingOp.apply(keys, weight)`, which
  runs the native `lookup` (cache hit/miss + auto-insert as usual).
* **Backward** — `CacheEmbeddingOp.backward` calls native `concat_backprop`, which
  **deduplicates** duplicate keys in the batch and sums their incoming gradients, returning
  a `torch.sparse_coo_tensor` over the *unique* keys. So `emb.weight.grad` is sparse — only
  the touched rows carry a gradient.
* **Step (write-through)** — `emb.weight` is a `CachedTable`; its `add_` is overridden to
  call native `accumulate`, a read-modify-write that adds the (scaled) gradient onto the
  existing row values **in both the cache and the backing store**. Because the update goes
  through the same coherent path as a lookup, a row that is currently cached stays correct —
  there is nothing to invalidate. `torch.optim.SGD` (with `foreach=False`) drives this for
  you; the manual `add_` above is the explicit form and is what the smoke test uses.

`optimize_for_training=True` (default for the trainable layers) pre-allocates the dedup /
gradient scratch buffers so the backward path doesn't allocate on the hot loop.

Storage dtype is **fp32 or fp16** (the embedding table itself); bf16/other dtypes are not a
valid NVE table format. Mixed-precision dense compute around the embedding is unaffected.

See `examples/training/train_nve_smoke.py` for a runnable end-to-end check (forward,
gradient-vs-`nn.Embedding`, optimizer step + cache coherence, multi-step loss decrease).

## Multi-GPU: the wedge fix (Plan 18)

At 1 TB-class sharded scale the legacy `cuMemSetAccess` granted the **full N×N device
mesh** over each rank's ~1 TB virtual reservation. On ROCm that exhausts the amdgpu
kernel's per-GPU peer-map budget and the grant fails with `hipErrorInvalidValue`
(`-ENOMEM` underneath), **wedging the driver**. The port grants only **this rank's own
device** (`cuMemSetAccess(..., &own_desc, /*count=*/1)`), which is all that is ever needed:
each `buffer_` is a process-local reservation only the owning GPU dereferences, and shards
are still read directly over xGMI (the lockstep-escape path is unchanged). See
`src/distributed.cpp` (search `own_desc`). `NVE_GRANT_FULL_MESH=1` restores the legacy
behaviour for debugging only (it reproduces the wedge at scale).

## Warp-width caveat (READ THIS before trusting the backward pass)

The gradient kernels were written for NVIDIA's **32-lane warp**. On gfx950 the wavefront is
**64 lanes**, and HIP's `__shfl_sync(mask, val, srcLane)` defaults to `width = warpSize = 64`.

* `NVE_SHFL_MASK` (`cuda_ops/kernels_common.cuh`) only widens the *mask* to 64 bits so HIP's
  `static_assert` passes — it does **not** change the shuffle *width*.
* `GradientDedup` (the **concat** backward used by `NVEmbedding`, in
  `cuda_ops/dedup_grads_kernel.cuh`) launches a `(SubwarpWidth=32, keys_per_sm)` block. With a
  64-lane wavefront that packs **two different keys per wavefront** (`threadIdx.y=0` →
  lanes 0–31, `threadIdx.y=1` → lanes 32–63), a `__shfl_sync(..., j)` with the default
  width=64 let the odd key read the even key's lanes — **wrong gradients for half the keys**.
  **FIXED**: the `LOOP`/`LOOP_SHORT` macros now pin the broadcast to the subwarp
  (`__shfl_sync(mask, val, j, SubwarpWidth)`). Verified by `train_nve_smoke.py`: the NVE
  gradient now matches `nn.Embedding` exactly in fp32 (max abs err `0.0`) on both NoCache and
  LinearUVM. (Before the fix the same check reported `grad_max_abs_err ≈ 12.6`.) On NVIDIA
  `SubwarpWidth == warpSize == 32`, so the change is a no-op there.
* **Still a caveat — `NVEmbeddingBag` mean/weighted-mean.** The **pooling** backward kernels in
  `cuda_ops/gradient_calculator.cuh`
  (`ComputeNormalizedWeights`, `ComputePoolingGradients`, used by `NVEmbeddingBag` mean/
  weighted-mean) carry the same 32-lane assumption (hardcoded 5-step `i<5` warp reduction and
  `count += warpSize` striding with only 32 active threads). These have the **same** hazard and
  are **not yet fixed** — they need the same width-pinning + a stride/active-lane audit before
  `NVEmbeddingBag` mean/weighted-mean training is trusted on gfx950. (`NVEmbedding` concat and
  `NVEmbeddingBag` *sum* do not go through `ComputeNormalizedWeights`.)

The runnable gate for the concat path is `examples/training/train_nve_smoke.py`: its
`*.backward` check compares the NVE sparse gradient against an identically-initialized
`torch.nn.Embedding` and will fail loudly if the warp-width hazard regresses.
