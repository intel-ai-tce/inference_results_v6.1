"""Plan 14 Phase 14.6 — NVE lookup correctness audit (G3).

Re-cert of the bit-exact lookup audit on the warp-size-fixed `.so` (Plan 14.8).
Mirrors Plan 12.4's matrix audit: NVE lookup vs a torch reference gather, asserting
bit-exact output on every cell of

    {NoCache, LinearUVM} x rows{4096, 65536, 1048576}
    x dim{64, 128, 512} x dtype{fp16, fp32}
    x distribution{uniform, zipf, sequential, hot_repeat, edges}   = 180 cells.

The LinearUVM cells run with optimize_for_training=True and a partial cache
(cache_frac<1) so the auto-insert cache-fill path — the site of the 14.8
warp-size bug — is actually exercised on every lookup.

Reference weights are exactly representable in both fp16/fp32 (row i = (i % 997)
broadcast across dim), so the NVE output must be *bit-identical* to W[keys].

Run (single GPU):
  PYTHONPATH=python HIP_VISIBLE_DEVICES=0 python3 scripts_debug/nve_lookup_audit.py
"""
import csv
import os
import sys
import traceback

import torch

from pynve import nve
from pynve.torch.nve_layers import NVEmbedding, CacheType

BASE_SEED = int(os.environ.get("BASE_SEED", "42"))
N_KEYS = int(os.environ.get("N_KEYS", "1024"))
ROWS = [int(x) for x in os.environ.get("ROWS", "4096,65536,1048576").split(",")]
DIMS = [int(x) for x in os.environ.get("DIMS", "64,128,512").split(",")]
DTYPES = [("float16", torch.float16), ("float32", torch.float32)]
DISTS = ["uniform", "zipf", "sequential", "hot_repeat", "edges"]
CACHE_FRAC = float(os.environ.get("CACHE_FRAC", "0.5"))
OPT_TRAIN = os.environ.get("OPT_TRAIN", "1") == "1"

OUT_DIR = os.environ.get("OUT_DIR", ".")


def _nve_dtype(t):
    return nve.DataType_t.Float16 if t == torch.float16 else nve.DataType_t.Float32


def _dsize(t):
    return 2 if t == torch.float16 else 4


def make_keys(dist, rows, n_keys, seed, dev):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dist == "uniform":
        k = torch.randint(0, rows, (n_keys,), dtype=torch.int64, generator=g)
    elif dist == "zipf":
        # power-law skew toward low ids, clamped into range
        u = torch.rand(n_keys, generator=g)
        k = (u.pow(4.0) * rows).to(torch.int64).clamp_(0, rows - 1)
    elif dist == "sequential":
        start = int(torch.randint(0, max(1, rows - n_keys), (1,), generator=g).item()) if rows > n_keys else 0
        k = (torch.arange(n_keys, dtype=torch.int64) + start) % rows
    elif dist == "hot_repeat":
        hot = torch.randint(0, rows, (8,), dtype=torch.int64, generator=g)
        idx = torch.randint(0, hot.numel(), (n_keys,), dtype=torch.int64, generator=g)
        k = hot[idx]
    elif dist == "edges":
        # boundary-heavy: 0 and rows-1 plus a few neighbours
        pool = torch.tensor([0, rows - 1, 1, rows - 2], dtype=torch.int64)
        idx = torch.randint(0, pool.numel(), (n_keys,), dtype=torch.int64, generator=g)
        k = pool[idx]
    else:
        raise ValueError(dist)
    return k.to(dev)


def build_emb(cache_type, rows, dim, dtype, dev, W):
    if cache_type == "NoCache":
        return NVEmbedding(rows, dim, dtype, CacheType.NoCache,
                           weight_init=W, device=dev)
    row_bytes = dim * _dsize(dtype)
    cache_bytes = max(row_bytes, (int(CACHE_FRAC * rows * row_bytes) // row_bytes) * row_bytes)
    return NVEmbedding(rows, dim, dtype, CacheType.LinearUVM,
                       gpu_cache_size=cache_bytes, weight_init=W,
                       optimize_for_training=OPT_TRAIN, device=dev)


def main():
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    print(f"device        : {torch.cuda.get_device_name(0)}")
    print(f"hip           : {torch.version.hip}")
    print(f"base_seed     : {BASE_SEED}")
    print(f"n_keys/cell   : {N_KEYS}")
    print(f"cache_frac    : {CACHE_FRAC}  opt_train={OPT_TRAIN}", flush=True)

    rows_csv = os.path.join(OUT_DIR, "audit.csv")
    f = open(rows_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["cell", "cache_type", "rows", "embed_dim", "dtype", "distribution",
                "n_keys", "out_shape", "bit_exact", "mismatches", "max_abs_diff", "note"])

    cell = 0
    npass = 0
    nfail = 0
    import time
    t0 = time.time()
    for cache_type in ["NoCache", "LinearUVM"]:
        for rows in ROWS:
            for dim in DIMS:
                for dname, dtype in DTYPES:
                    for dist in DISTS:
                        seed = BASE_SEED + cell
                        note = ""
                        try:
                            W = ((torch.arange(rows, dtype=torch.float32) % 997)
                                 .to(dtype).view(rows, 1).expand(rows, dim).contiguous())
                            Wd = W.to(dev)
                            keys = make_keys(dist, rows, N_KEYS, seed, dev)
                            ref = Wd[keys]
                            emb = build_emb(cache_type, rows, dim, dtype, dev, Wd)
                            # exercise the cache-fill path twice (fill, then hit)
                            out = emb.forward(keys)
                            torch.cuda.synchronize()
                            out = emb.forward(keys)
                            torch.cuda.synchronize()
                            mism = (out != ref)
                            n_mis = int(mism.any(dim=1).sum().item())
                            max_abs = float((out.float() - ref.float()).abs().max().item())
                            bit_exact = 1 if n_mis == 0 else 0
                            if bit_exact:
                                npass += 1
                            else:
                                nfail += 1
                                note = "MISMATCH"
                            w.writerow([cell, cache_type, rows, dim, dname, dist, N_KEYS,
                                        f"({N_KEYS}, {dim})", bit_exact, n_mis, max_abs, note])
                            del emb, W, Wd, ref, keys, out
                            torch.cuda.empty_cache()
                        except Exception as e:
                            nfail += 1
                            note = f"{type(e).__name__}: {e}"
                            w.writerow([cell, cache_type, rows, dim, dname, dist, N_KEYS,
                                        "-", 0, -1, -1, note])
                            traceback.print_exc()
                        if cell % 20 == 0:
                            print(f"  cell {cell}: {cache_type} rows={rows} dim={dim} "
                                  f"{dname} {dist} -> pass={npass} fail={nfail}", flush=True)
                        cell += 1
    f.close()
    elapsed = time.time() - t0

    summary = (
        "Plan 14 Phase 14.6 — NVE lookup correctness audit (G3)\n"
        "======================================================\n"
        f"device        : {torch.cuda.get_device_name(0)}\n"
        f"hip           : {torch.version.hip}\n"
        f"base_seed     : {BASE_SEED}\n"
        f"n_keys/cell   : {N_KEYS}\n"
        f"cells         : {cell}\n"
        f"PASS          : {npass}\n"
        f"FAIL          : {nfail}\n"
        f"elapsed_s     : {elapsed:.2f}\n\n"
        + ("OVERALL: PASS — NVE lookup is bit-exact vs the torch reference on every cell "
           "(cache_type × rows × dim × dtype × distribution).\n"
           if nfail == 0 else
           f"OVERALL: FAIL — {nfail} cell(s) not bit-exact. See audit.csv.\n")
    )
    with open(os.path.join(OUT_DIR, "SUMMARY.txt"), "w") as sf:
        sf.write(summary)
    print("\n" + summary, flush=True)
    sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
