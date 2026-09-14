#!/usr/bin/env python3
"""Minimal NVE inference example — proves the ROCm pynve port works and surfaces the
cache hit rate in a trace.

Exercises the two core layer types through the public `pynve.torch.nve_layers` API,
verifies correctness against a CPU reference, and captures the native NVE hit-rate
counter into a JSONL trace:

  Part 1  LinearUVM   — full table in CPU/UVM managed memory + a bounded GPU cache.
                        Drives a hot working set with batches >= the cache's auto-insert
                        threshold (min_insert_size_gpu = 1<<16 = 65536 keys) so the
                        DefaultInsertHeuristic fills the cache, and watches the per-iter
                        GPU hit rate climb 0 -> ~0.8 (the cache earning its keep).
  Part 2  NoCache     — whole table resident in GPU memory (GPUEmbedding); hit rate is
                        always 1.0 (everything on-GPU, nothing to miss).

The hit rate is NOT returned by forward(); the native C++ layer only logs it as a
"[NVE][P] Hit rates: gpu, host, remote" line (needs config={"logging_interval": N} and
NVE_LOG_LEVEL>=PERF). We redirect fd-1 around each lookup, parse that line, replay it to
the real log, and record (hit_gpu, hit_host) into the JSONL trace.

Run after building (see ../../build_rocm.sh):
  PYTHONPATH=<repo>/python LD_LIBRARY_PATH=<repo>/build_rocm/lib \
  NVE_LOG_LEVEL=VERBOSE python3 minimal_nve_example.py
(or just: bash run.sh)
"""
import ctypes
import json
import os
import re
import sys
import tempfile
import time

import torch
import pynve
import pynve.nve as nve
import pynve.torch.nve_layers as nve_layers

DEV = torch.device("cuda")
TRACE_OUT = os.environ.get("NVE_TRACE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nve_trace.jsonl"))
MIN_INSERT_SIZE = 1 << 16  # pynve LinearUVMEmbedding min_insert_size_gpu: cache auto-inserts only past this

_libc = ctypes.CDLL(None)
_HIT_RE = re.compile(r"\[NVE\]\[P\] Hit rates: ([-\d.eE+]+), ([-\d.eE+]+), ([-\d.eE+]+)")
_trace_fh = open(TRACE_OUT, "w")


def trace(op, **kw):
    rec = {"op": op, **kw}
    _trace_fh.write(json.dumps(rec) + "\n")
    _trace_fh.flush()
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"  [trace] {op:18s} {extra}", flush=True)


class CaptureNativeLog:
    """Redirect fd-1 (where NVE's C++ logger writes via std::cout) to a temp file for the
    duration of a block, then restore it. Captured text is replayed to the real stdout so
    the [NVE] lines still land in the log, and exposed for hit-rate parsing."""

    def __enter__(self):
        sys.stdout.flush()
        self._saved = os.dup(1)
        self._tmp = tempfile.TemporaryFile(mode="w+b")
        os.dup2(self._tmp.fileno(), 1)
        return self

    def __exit__(self, *exc):
        _libc.fflush(None)
        os.dup2(self._saved, 1)
        os.close(self._saved)
        self._tmp.seek(0)
        self.text = self._tmp.read().decode("utf-8", "replace")
        self._tmp.close()
        sys.stdout.write(self.text)
        sys.stdout.flush()
        return False

    @property
    def last_hitrate(self):
        m = _HIT_RE.findall(self.text)
        return tuple(round(float(x), 4) for x in m[-1]) if m else None


def make_reference(num_embeddings, dim):
    idx = torch.arange(num_embeddings, dtype=torch.float32).remainder(1024.0)
    col = torch.arange(dim, dtype=torch.float32) / 256.0
    return idx.unsqueeze(1) + col.unsqueeze(0)


def timed_lookup(layer, keys, capture=False):
    torch.cuda.synchronize()
    if capture:
        cap = CaptureNativeLog()
        with cap:
            t0 = time.perf_counter()
            out = layer(keys)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
        return out, ms, cap.last_hitrate
    t0 = time.perf_counter()
    out = layer(keys)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1e3, None


def main():
    assert torch.cuda.is_available(), "no GPU visible"
    print(f"pynve       : {pynve.__version__}")
    print(f"native ext  : {nve.__file__}")
    print(f"device      : {torch.cuda.get_device_name(0)}  |  torch {torch.__version__} hip={torch.version.hip}")
    print(f"trace file  : {os.path.abspath(TRACE_OUT)}")

    N, D = 1_000_000, 128
    ref = make_reference(N, D)
    full_bytes = N * D * 4
    ok = True

    print("\n[Part 1] LinearUVM: 1,000,000 x 128 fp32 table in UVM + 64MB GPU cache")
    gpu_cache = 64 * 1024 * 1024
    uvm = nve_layers.NVEmbedding(
        N, D, torch.float32, nve_layers.CacheType.LinearUVM,
        gpu_cache_size=gpu_cache, weight_init=ref,
        config={"logging_interval": 1},
    )
    trace("create_linearuvm", table_mb=full_bytes // 2**20, gpu_cache_mb=gpu_cache // 2**20,
          auto_insert_threshold=MIN_INSERT_SIZE)

    torch.manual_seed(0)
    keys = torch.randint(0, N, (MIN_INSERT_SIZE,), dtype=torch.int64, device=DEV)
    out, ms, hr = timed_lookup(uvm, keys, capture=True)
    match = torch.allclose(out.cpu(), ref[keys.cpu()])
    ok &= match
    trace("lookup_cold", n_keys=keys.numel(), ms=round(ms, 3),
          hit_gpu=hr[0] if hr else None, hit_host=hr[1] if hr else None, verified=match)

    HOT_ROWS = 40_000
    print(f"\n[Part 1b] warming the cache on a hot set of {HOT_ROWS:,} rows "
          f"(batch={MIN_INSERT_SIZE:,} >= auto-insert threshold) — watch hit_gpu climb:")
    first_hit = last_hit = None
    for i in range(24):
        hot = torch.randint(0, HOT_ROWS, (MIN_INSERT_SIZE,), dtype=torch.int64, device=DEV)
        o, t, hr = timed_lookup(uvm, hot, capture=True)
        if i in (0, 1, 2, 4, 8, 16, 23):
            verified = torch.allclose(o.cpu(), ref[hot.cpu()])
            ok &= verified
            thr = hot.numel() / (t / 1e3) / 1e6
            trace("lookup_warm", iter=i, n_keys=hot.numel(), ms=round(t, 3),
                  hit_gpu=hr[0] if hr else None, hit_host=hr[1] if hr else None,
                  Mlookups_per_s=round(thr, 2), verified=verified)
        if hr is not None:
            first_hit = first_hit if first_hit is not None else hr[0]
            last_hit = hr[0]
    print(f"  -> GPU hit rate moved from {first_hit} (cold) to {last_hit} (warm)")
    trace("hitrate_summary", cold=first_hit, warm=last_hit, climbed=bool(last_hit and last_hit > 0.5))
    ok &= bool(last_hit and last_hit > 0.5)

    ukeys = torch.tensor([7, 42, 1000, 999999], dtype=torch.int64, device=DEV)
    uvals = torch.full((ukeys.numel(), D), -7.0, dtype=torch.float32, device=DEV)
    uvm.update(ukeys, uvals)
    torch.cuda.synchronize()
    uout, _, _ = timed_lookup(uvm, ukeys)
    upd_match = torch.allclose(uout.cpu(), uvals.cpu())
    ok &= upd_match
    trace("update+readback", n_keys=ukeys.numel(), verified=upd_match, sample=uout[0, :3].tolist())

    print("\n[Part 2] NoCache: 200,000 x 128 fp32 table fully resident in GPU memory")
    N2 = 200_000
    ref2 = make_reference(N2, D)
    nocache = nve_layers.NVEmbedding(
        N2, D, torch.float32, nve_layers.CacheType.NoCache, weight_init=ref2,
        config={"logging_interval": 1},
    )
    trace("create_nocache", table_mb=(N2 * D * 4) // 2**20)
    keys2 = torch.randint(0, N2, (8192,), dtype=torch.int64, device=DEV)
    out2, ms2, hr2 = timed_lookup(nocache, keys2, capture=True)
    match2 = torch.allclose(out2.cpu(), ref2[keys2.cpu()])
    ok &= match2
    trace("lookup_gpu_resident", n_keys=keys2.numel(), ms=round(ms2, 3),
          hit_gpu=hr2[0] if hr2 else None, verified=match2)

    _trace_fh.close()
    print(f"\nRESULT: {'NVE OK — all lookups/updates verified, hit rate captured' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
