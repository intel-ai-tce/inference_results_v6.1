"""Plan 14 Phase 14.7 — NVE production-cap lookup scaling bench (G4).

The headline gate. Measures whether NVE's data-parallel lookup (table-distributed
MPIMemBlock unified VA + per-rank GPU cache, *zero collectives on the lookup path*)
keeps per-rank throughput FLAT as W grows — i.e. inverts the routed-lockstep
anti-scaling measured in Plan 13 §13.6 (predict.sparse 320 -> 1768 -> 3683 ms at
W=2/4/8). NVE has no all-to-all on lookup, so per-rank keys/s should be ~constant
and aggregate keys/s should scale ~linearly with W.

Each rank builds the *same total table* (rows) distributed across W ranks via
MPIMemBlock, attaches a partial LinearUVM cache (cache_frac < 1, auto-insert on —
the path the 14.8 warp-size fix repaired), draws `batch` keys uniformly over the
WHOLE table (so a (W-1)/W fraction are peer reads over xGMI), and times `iters`
forwards. Per-rank dicts are gathered to rank 0 which writes nve_scaling_W{W}.json.

Each rank writes nve_scaling_W{W}_rank{R}.json; aggregate host-side with
--aggregate (or the launcher) into nve_scaling_W{W}.json.

Run:
  PYTHONPATH=python mpirun --allow-run-as-root -np 8 --bind-to none \
      python3 scripts_debug/nve_scaling_bench.py
Env: ROWS DIM BATCH CACHE_FRAC ITERS WARMUP OUT_DIR
"""
import glob
import json
import os
import statistics
import sys
import time
import traceback

import torch

from pynve import nve
from pynve.torch.nve_layers import NVEmbedding, CacheType

WR = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
WS = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
LR = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", str(WR)))


def log(msg):
    print(f"[rank {WR}/{WS} dev{LR}] {msg}", flush=True)


def main():
    rows = int(os.environ.get("ROWS", "2000000"))
    dim = int(os.environ.get("DIM", "512"))
    batch = int(os.environ.get("BATCH", "100000"))
    cache_frac = float(os.environ.get("CACHE_FRAC", "0.5"))
    iters = int(os.environ.get("ITERS", "50"))
    warmup = int(os.environ.get("WARMUP", "10"))
    out_dir = os.environ.get("OUT_DIR", ".")
    dtype = torch.float16
    ndt = nve.DataType_t.Float16

    torch.cuda.set_device(LR)
    dev = torch.device(f"cuda:{LR}")
    row_bytes = dim * 2
    table_bytes = rows * row_bytes
    cache_bytes = max(row_bytes, (int(cache_frac * table_bytes) // row_bytes) * row_bytes)
    gpu_cache_mb = cache_bytes / 1e6

    t_build = time.time()
    mb = nve.MPIMemBlock(dim, rows, ndt)  # collective: distributes table across ranks
    emb = NVEmbedding(rows, dim, dtype, CacheType.LinearUVM,
                      gpu_cache_size=cache_bytes, memblock=mb, device=dev,
                      optimize_for_training=True)
    build_s = round(time.time() - t_build, 3)

    g = torch.Generator(device="cpu").manual_seed(1234 + WR)
    keys = torch.randint(0, rows, (batch,), dtype=torch.int64, generator=g).to(dev)

    # warmup (fills cache; first iters pay cache-fill, excluded from timing)
    for _ in range(warmup):
        emb.forward(keys)
    torch.cuda.synchronize()

    lat_ms = []
    for _ in range(iters):
        t0 = time.time()
        emb.forward(keys)
        torch.cuda.synchronize()
        lat_ms.append((time.time() - t0) * 1e3)
    lat_ms.sort()

    p50 = lat_ms[len(lat_ms) // 2]
    p99 = lat_ms[min(len(lat_ms) - 1, int(len(lat_ms) * 0.99))]
    mean = statistics.fmean(lat_ms)
    keys_per_s = batch / (p50 / 1e3)

    rec = {
        "rank": WR, "world": WS, "device": LR, "rows": rows, "dim": dim,
        "batch": batch, "cache_frac": cache_frac, "gpu_cache_mb": round(gpu_cache_mb, 1),
        "iters": iters, "build_s": build_s,
        "p50_ms": round(p50, 4), "p99_ms": round(p99, 4), "mean_ms": round(mean, 4),
        "keys_per_s": round(keys_per_s, 1),
    }
    log(f"p50={p50:.3f}ms p99={p99:.3f}ms mean={mean:.3f}ms keys/s={keys_per_s/1e6:.1f}M build={build_s}s")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"nve_scaling_W{WS}_rank{WR}.json"), "w") as f:
        json.dump(rec, f)


def aggregate(out_dir, world):
    parts = sorted(glob.glob(os.path.join(out_dir, f"nve_scaling_W{world}_rank*.json")))
    per = sorted([json.load(open(p)) for p in parts], key=lambda r: r["rank"])
    if not per:
        print(f"no per-rank files for W={world} in {out_dir}", file=sys.stderr)
        return
    total_kps = sum(r["keys_per_s"] for r in per)
    r0 = per[0]
    out = {
        "world": world, "rows": r0["rows"], "dim": r0["dim"], "batch": r0["batch"],
        "cache_frac": r0["cache_frac"], "gpu_cache_mb": r0["gpu_cache_mb"],
        "max_p50_ms": round(max(r["p50_ms"] for r in per), 4),
        "mean_p50_ms": round(statistics.fmean([r["p50_ms"] for r in per]), 4),
        "max_p99_ms": round(max(r["p99_ms"] for r in per), 4),
        "per_rank_keys_per_s_mean": round(statistics.fmean([r["keys_per_s"] for r in per]), 1),
        "total_keys_per_s": round(total_kps, 1),
        "per_rank": per,
    }
    path = os.path.join(out_dir, f"nve_scaling_W{world}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[W={world}] batch={out['batch']} cache_frac={out['cache_frac']} "
          f"per-rank mean={out['per_rank_keys_per_s_mean']/1e6:.1f}M keys/s  "
          f"TOTAL={total_kps/1e6:.1f}M keys/s  max_p50={out['max_p50_ms']}ms max_p99={out['max_p99_ms']}ms -> {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--aggregate":
        aggregate(os.environ.get("OUT_DIR", "."), int(sys.argv[2]))
        sys.exit(0)
    try:
        main()
    except Exception as e:
        log(f"EXCEPTION {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(3)
