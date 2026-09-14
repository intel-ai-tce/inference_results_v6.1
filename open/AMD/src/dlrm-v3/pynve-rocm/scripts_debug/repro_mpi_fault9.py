"""9-rank MPIMemBlock repro mirroring the DLRM-v3 harness topology.

The harness launches ``mpirun -n 9`` = 8 NVE worker ranks (each owns one
``item_id`` shard on GPUs 0..7) + 1 LoadGen rank (rank 8) that joins the
``MPIMemBlock`` *collective* (get_backend_config runs on every rank) but owns no
shard and never reads embeddings. The standalone ``repro_mpi_fault.py`` only
ever ran ``np==WORLD`` with *every* rank a participant, so it never exercised the
non-participant rank that triggered the Plan 14.7c→d ``RuntimeError: invalid
argument`` in the peer-mapping phase.

This repro reproduces that exact topology with a TINY table (safe — no 128 GB/GPU
allocation) so the 14.7d fix (gate the non-participant rank out of all local VMM)
can be validated cheaply.

Launch: mpirun --allow-run-as-root -np 9 python3 repro_mpi_fault9.py
Expect (after the 14.7d fix): every participant prints ``forward OK`` and the
LoadGen rank prints ``LOADGEN-RANK no-shard OK``; all 9 print ``DONE OK``.
"""
import os, sys, traceback
import torch
from pynve import nve
from pynve.torch.nve_layers import NVEmbedding, CacheType

WR = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
WS = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
NGPU = torch.cuda.device_count()
DEV_ID = WR % NGPU  # matches harness: local_device_id = local_rank % device_count


def log(msg):
    print(f"[rank {WR}/{WS} dev{DEV_ID}] {msg}", flush=True)


def run(rows, dim, batch, cache_frac, dtype=torch.float16):
    # Participant set = all ranks except the last (the LoadGen rank), exactly as
    # the harness builds it: partial_ranks = list(range(world_size - 1)).
    worker_world = WS - 1
    ranks = list(range(worker_world))
    devices = [r % NGPU for r in ranks]
    is_participant = WR in ranks

    torch.cuda.set_device(DEV_ID)
    dev = torch.device(f"cuda:{DEV_ID}")
    ndt = nve.DataType_t.Float16 if dtype == torch.float16 else nve.DataType_t.Float32
    row_bytes = dim * (2 if dtype == torch.float16 else 4)
    table_bytes = rows * row_bytes

    log(f"building MPIMemBlock(5-arg) rows={rows} dim={dim} "
        f"table={table_bytes/1e9:.3f}GB participants={worker_world} "
        f"participant={is_participant}")
    # 5-arg collective form — ALL 9 ranks call it; rank WS-1 is a non-participant.
    mb = nve.MPIMemBlock(dim, rows, ndt, ranks, devices)
    log("MPIMemBlock built (collective returned)")

    if not is_participant:
        # LoadGen-like rank: no shard, no NVEmbedding, no lookups. It only needed
        # to drive the collective so the workers don't hang.
        log("LOADGEN-RANK no-shard OK (ptr is null by design)")
    else:
        cache_bytes = max(1, int(cache_frac * table_bytes))
        cache_bytes = (cache_bytes // row_bytes) * row_bytes
        emb = NVEmbedding(rows, dim, dtype, CacheType.LinearUVM,
                          gpu_cache_size=cache_bytes, memblock=mb, device=dev,
                          optimize_for_training=False)
        g = torch.Generator(device="cpu").manual_seed(1234 + WR)
        keys = torch.randint(0, rows, (batch,), dtype=torch.int64, generator=g).to(dev)
        out = emb.forward(keys)
        torch.cuda.synchronize()
        log(f"forward OK out={tuple(out.shape)}")
        del emb

    # Symmetric teardown: every rank drops the collective buffer together (the
    # C++ destructor barriers keep the 9 ranks in lockstep).
    del mb


if __name__ == "__main__":
    if WS < 2:
        log("need np>=2 (run with -np 9 to mirror the harness)")
        sys.exit(2)
    rows = int(os.environ.get("ROWS", 2_000_000))
    dim = int(os.environ.get("DIM", 512))
    batch = int(os.environ.get("BATCH", 50_000))
    cache_frac = float(os.environ.get("CACHE_FRAC", "0.1"))
    try:
        run(rows, dim, batch, cache_frac)
        log("DONE OK")
    except Exception as e:
        log(f"EXCEPTION {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(3)
