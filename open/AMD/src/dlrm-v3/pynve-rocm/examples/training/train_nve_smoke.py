#!/usr/bin/env python3
"""NVE training smoke test — exercises the *training* path of the ROCm pynve port and
verifies it against a plain torch.nn.Embedding reference.

Unlike inference (lookup only), this lights up the parts the certified harness never
runs:
  * forward  : CacheEmbeddingOp.apply  -> native lookup
  * backward : CacheEmbeddingOp.backward -> concat_backprop  (dedups duplicate keys into
               a sparse per-unique-key gradient — the GradientCalculator/GradientDedup
               kernels, which still carry warp-size-32 tuning; see TRAINING_ON_ROCM.md)
  * step     : weight.add_(grad, alpha=-lr) -> CachedTable.accumulate (write-through; the
               cache is updated in-place, no invalidation)

For each cache type we check, against an identically-initialized nn.Embedding:
  (A) forward lookup matches,
  (B) the NVE sparse gradient matches the reference dense gradient on touched rows
      (this is the warp-width-sensitive concat_backprop kernel — the key correctness gate),
  (C) after one SGD step, re-reading the updated rows from NVE matches the reference
      (accumulate write-through + cache coherence),
  (D) loss decreases monotonically over several steps.

Run after building (see ../../build_rocm.sh):
  PYTHONPATH=<repo>/python LD_LIBRARY_PATH=<repo>/build_rocm/lib python3 train_nve_smoke.py
(or just: bash run.sh)
"""
import json
import os

import torch
import torch.nn as nn
import pynve
import pynve.nve as nve
import pynve.torch.nve_layers as nve_layers

DEV = torch.device("cuda")
TRACE_OUT = os.environ.get("NVE_TRACE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_trace.jsonl"))
_trace_fh = open(TRACE_OUT, "w")
_ok = True


def trace(op, passed=None, **kw):
    global _ok
    if passed is not None:
        _ok &= bool(passed)
    rec = {"op": op, **({"passed": bool(passed)} if passed is not None else {}), **kw}
    _trace_fh.write(json.dumps(rec) + "\n")
    _trace_fh.flush()
    flag = "" if passed is None else ("  OK" if passed else "  *** FAIL ***")
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"  [{op:20s}] {extra}{flag}", flush=True)


def make_init(N, D, dtype):
    g = torch.Generator().manual_seed(123)
    return (torch.randn(N, D, generator=g) * 0.1).to(dtype)


def build_reference(init):
    """Plain nn.Embedding with identical weights, dense grad — the ground truth."""
    N, D = init.shape
    ref = nn.Embedding(N, D).to(DEV)
    with torch.no_grad():
        ref.weight.copy_(init.to(DEV).float())
    return ref


def run_case(name, cache_type, dtype, N, D, B, hot, lr, steps, atol, rtol, gpu_cache_mb=64):
    """One full fwd/bwd/step verification for a given cache type + dtype."""
    print(f"\n=== {name}: cache={cache_type.name} dtype={dtype} N={N} D={D} B={B} hot={hot} ===")
    init = make_init(N, D, dtype)

    kwargs = {}
    if cache_type != nve_layers.CacheType.NoCache:
        kwargs["gpu_cache_size"] = gpu_cache_mb * 1024 * 1024
    emb = nve_layers.NVEmbedding(N, D, dtype, cache_type, weight_init=init.to(dtype), **kwargs)
    ref = build_reference(init)

    torch.manual_seed(7)
    # keys drawn from a small hot range so there are MANY duplicates -> exercises the
    # gradient dedup/accumulate kernels (the warp-width-sensitive path).
    keys = torch.randint(0, hot, (B,), dtype=torch.int64, device=DEV)
    target = (torch.randn(B, D, generator=torch.Generator().manual_seed(9)) * 0.5).to(DEV)

    # ── (A) forward lookup correctness ──────────────────────────────────────────
    out = emb(keys)
    ref_out = ref(keys)
    fwd_max = (out.float() - ref_out.float()).abs().max().item()
    trace(f"{name}.forward", passed=torch.allclose(out.float(), ref_out.float(), atol=atol, rtol=rtol),
          max_abs_err=round(fwd_max, 6))

    # ── (B) gradient correctness (concat_backprop vs autograd reference) ────────
    loss = 0.5 * ((out.float() - target) ** 2).sum()
    ref_loss = 0.5 * ((ref_out.float() - target) ** 2).sum()
    emb.weight.grad = None
    ref.weight.grad = None
    loss.backward()
    ref_loss.backward()

    uniq = torch.unique(keys)
    nve_grad = emb.weight.grad.to_dense().float()         # sparse -> dense for comparison
    ref_grad = ref.weight.grad.to_dense().float() if ref.weight.grad.is_sparse else ref.weight.grad.float()
    grad_max = (nve_grad[uniq] - ref_grad[uniq]).abs().max().item()
    # untouched rows must have exactly zero grad
    touched_mask = torch.zeros(N, dtype=torch.bool, device=DEV); touched_mask[uniq] = True
    leak = nve_grad[~touched_mask].abs().max().item()
    trace(f"{name}.backward", passed=(grad_max <= max(atol, 1e-2) and leak == 0.0),
          n_unique=int(uniq.numel()), grad_max_abs_err=round(grad_max, 5), untouched_leak=round(leak, 8))

    # ── (C) optimizer step + cache coherence (accumulate write-through) ─────────
    with torch.no_grad():
        emb.weight.add_(emb.weight.grad, alpha=-lr)       # -> CachedTable.accumulate
        ref.weight.add_(ref.weight.grad.to_dense() if ref.weight.grad.is_sparse else ref.weight.grad, alpha=-lr)
    torch.cuda.synchronize()
    after = emb(uniq)                                     # re-read updated rows (may be cached)
    ref_after = ref(uniq)
    step_max = (after.float() - ref_after.float()).abs().max().item()
    trace(f"{name}.step+coherence", passed=torch.allclose(after.float(), ref_after.float(), atol=max(atol, 1e-3), rtol=rtol),
          updated_rows=int(uniq.numel()), max_abs_err=round(step_max, 5))

    # ── (D) loss decreases over several steps ───────────────────────────────────
    losses = []
    for _ in range(steps):
        o = emb(keys)
        l = 0.5 * ((o.float() - target) ** 2).mean()
        emb.weight.grad = None
        l.backward()
        with torch.no_grad():
            emb.weight.add_(emb.weight.grad, alpha=-lr)
        torch.cuda.synchronize()
        losses.append(l.item())
    decreased = losses[-1] < losses[0] and all(b <= a + 1e-6 for a, b in zip(losses, losses[1:]))
    trace(f"{name}.loss_decrease", passed=decreased,
          loss_first=round(losses[0], 5), loss_last=round(losses[-1], 5), steps=steps)


def main():
    assert torch.cuda.is_available(), "no GPU visible"
    print(f"pynve       : {pynve.__version__}")
    print(f"native ext  : {nve.__file__}")
    print(f"device      : {torch.cuda.get_device_name(0)}  |  torch {torch.__version__} hip={torch.version.hip}")
    print(f"trace file  : {os.path.abspath(TRACE_OUT)}")

    # NoCache fp32 — isolates the gradient/accumulate kernels (fully GPU-resident).
    run_case("nocache_fp32", nve_layers.CacheType.NoCache, torch.float32,
             N=20_000, D=128, B=8192, hot=4000, lr=0.05, steps=15, atol=1e-4, rtol=1e-4)

    # LinearUVM fp32 — same, but with a GPU cache present: verifies accumulate stays
    # coherent (write-through) when rows may be cached.
    run_case("linearuvm_fp32", nve_layers.CacheType.LinearUVM, torch.float32,
             N=200_000, D=128, B=8192, hot=4000, lr=0.05, steps=15, atol=1e-4, rtol=1e-4)

    # fp16 NoCache — lower precision; looser tolerance.
    run_case("nocache_fp16", nve_layers.CacheType.NoCache, torch.float16,
             N=20_000, D=128, B=8192, hot=4000, lr=0.05, steps=10, atol=5e-2, rtol=5e-2)

    _trace_fh.close()
    print(f"\nRESULT: {'NVE TRAINING OK — fwd/grad/step/coherence verified' if _ok else 'FAILED — see *** FAIL *** above'}")
    if not _ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
