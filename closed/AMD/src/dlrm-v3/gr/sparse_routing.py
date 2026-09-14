"""
Phase 2b Step 3b — cross-rank consistency for sharded sparse inference.

After Step 3a each MPI worker could load a disjoint row-shard of the
capped sparse table. The remaining correctness gap was that each rank's
batch of query indices spans the whole capped global range, but rank
``r`` only owned rows ``[r*S, (r+1)*S)``, so any index outside that
range had to be either silently clamped (wrong) or fetched from a
peer (right).

For the current Phase 2b smoke scale (``DLRM_SPARSE_MAX_HASH_SIZE``
=100k * 2 workers = 200k rows * 512 * fp16 = 200 MB per rank) the
**replicated-cap** path is correct and trivially fits on every GPU:
every worker grows its live table to the full ``W*S`` global cap and
the ``SlicingLoadPlanner`` reads rows ``[0, W*S)`` on every rank from
the saved checkpoint. The rank-local lookup then returns the true
``embed[idx]`` for any ``idx < W*S`` without any cross-rank collective.

This module exposes the small helpers that drive that choice:

  * :func:`replicate_enabled`        — env-gated toggle (default ON when
                                       ``DLRM_SPARSE_WORLD > 1``)
  * :func:`effective_table_rows`     — per-rank live row count (with the
                                       ``*W`` bump when replicate is on)

The :func:`get_worker_process_group` + :func:`stub_all_to_all` helpers
below are scaffolding for the production Step 3c path. In Step 3c each
rank holds only its own ``S``-row shard and we exchange query indices
+ embedding vectors via ``all_to_all_single`` over a worker-only
``torch.distributed`` process group. The PG init pattern lives here
already so the future change is a forward-only PR (no env or wiring
churn). It is **not** invoked from the model forward in Step 3b — the
``ec_patched_forward_wo_embedding_copy`` rank-local clamp already
returns correct embeddings under the replicated cap.

Configuration
-------------
``DLRM_SPARSE_REPLICATE``       ``1`` (default when WORLD>1) → replicated
                                cap path described above. ``0`` keeps
                                the Step-3a per-rank shard storage with
                                a rank-local clamp (per-rank divergent
                                predictions; useful for ablation only).
``DLRM_SPARSE_PG_BACKEND``      Future Step 3c routing backend
                                (``gloo`` recommended; ``nccl``/``rccl``
                                requires resolving the per-rank
                                ``HIP_VISIBLE_DEVICES`` topology that
                                Step 3a installed).
``DLRM_SPARSE_PG_MASTER_ADDR``  Rendezvous host (default ``127.0.0.1``).
``DLRM_SPARSE_PG_MASTER_PORT``  TCP store port (default ``29501``).
``DLRM_SPARSE_PG_TIMEOUT_S``    PG init / collective timeout (default 300).
``DLRM_SPARSE_PG_DISABLE``      ``1`` to skip Step-3c PG init.
``DLRM_USE_MPI_LOOKUP``         ``1`` to swap ``route_lookup``'s three
                                ``dist.all_to_all_single`` calls for
                                ``mpi4py.MPI.Alltoallv`` over a dedicated
                                worker subcomm (Plan 12 §3.2). Requires
                                the harness to call
                                ``set_route_lookup_comm(worker_comm.Dup())``
                                during init; falls back to torch.distributed
                                with a one-shot warning if no comm is
                                registered. CPU-bounces every collective —
                                only intended for sharded W=8 where moving
                                rendezvous off RCCL's CUDA stream wins more
                                than the D2H/H2D cost (see ``plans/12_*``).
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
import time
from typing import Optional

import torch


logger: logging.Logger = logging.getLogger(__name__)


try:
    from timing_stats import record as _timing_record  # type: ignore
except Exception:  # noqa: BLE001 - optional instrumentation only

    def _timing_record(_name: str, _seconds: float) -> None:  # type: ignore
        return None


# --------------------------------------------------------------------------- #
# Plan 04 Phase 4.1 — memory-history snapshot around route_lookup
#
# Opt-in (`DLRM_MEM_HISTORY=1`): on the first route_lookup call per rank we
# enable torch's allocator history recording; on every Nth call (default 1, so
# every call) we dump a pickle snapshot to /tmp/dlrm_memhist_rank{R}_call{N}.pickle.
# The deterministic `Memory access fault by GPU node-N` on batch 2 kills the
# process before any atexit hook can run, so we *write the snapshot
# synchronously before AND after the collectives* — that way we always have the
# allocation map for the last successful call (N) and the would-be-successful
# call (N+1) bracketing the bad allocation.
#
# Cheap (no rebuild). Post-process with:
#
#     python3 -c "
#     import pickle, sys
#     snap = pickle.load(open(sys.argv[1], 'rb'))
#     for seg in snap['segments']:
#         for blk in seg['blocks']:
#             addr = seg['address'] + blk['offset']
#             if (addr & 0xfffff) == 0x9f000:
#                 print(addr, blk.get('frames', [])[:5])
#     "
# --------------------------------------------------------------------------- #

_MEM_HIST_ENABLED: bool = os.environ.get("DLRM_MEM_HISTORY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Plan 04 — when enabled, start recording allocator history at *module import*
# (not first route_lookup) so the early/pre-warmup tensors (model weights, RCCL
# comm buffers, torchrec/fbgemm intermediates) ALSO have Python frames in the
# snapshots. The recording is per-process, so each rank's import wires its own
# tape.
if _MEM_HIST_ENABLED:
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", max_entries=200_000
        )
        logger.warning(
            "[plan04] mem-history recording ENABLED at sparse_routing import "
            "(captures pre-warmup allocations)"
        )
    except Exception as _exc:  # noqa: BLE001
        logger.error("[plan04] early _record_memory_history failed: %r", _exc)
_MEM_HIST_DIR: str = os.environ.get("DLRM_MEM_HISTORY_DIR", "/tmp")
_MEM_HIST_EVERY: int = max(1, int(os.environ.get("DLRM_MEM_HISTORY_EVERY", "1")))
_MEM_HIST_MAX_CALLS: int = int(os.environ.get("DLRM_MEM_HISTORY_MAX_CALLS", "20"))
_MEM_HIST_STARTED: bool = False
_MEM_HIST_CALL_IDX: int = 0
_MEM_HIST_FAULT_PAGE_MASK: int = int(
    os.environ.get("DLRM_MEM_HISTORY_PAGE_MASK", "0xfffff"), 0
)
_MEM_HIST_FAULT_PAGE: int = int(
    os.environ.get("DLRM_MEM_HISTORY_PAGE", "0x9f000"), 0
)


def _mem_history_dump(stage: str) -> None:
    """Dump a torch allocator snapshot tagged with the current call/stage."""
    global _MEM_HIST_STARTED, _MEM_HIST_CALL_IDX
    if not _MEM_HIST_ENABLED:
        return
    if _MEM_HIST_CALL_IDX > _MEM_HIST_MAX_CALLS:
        return
    if _MEM_HIST_CALL_IDX % _MEM_HIST_EVERY != 0:
        return
    try:
        import pickle  # noqa: WPS433

        _MEM_HIST_STARTED = True  # recording is started at module import now
        rank = _sparse_rank()
        path = (
            f"{_MEM_HIST_DIR}/dlrm_memhist_rank{rank}"
            f"_call{_MEM_HIST_CALL_IDX:04d}_{stage}.pickle"
        )
        snap = torch.cuda.memory._snapshot()
        # Eagerly look for any allocation that lives at the suspected fault
        # page offset, so even if the pickle is unreadable we still log a
        # smoking-gun line in stderr.
        hits = []
        for seg in snap.get("segments", []):
            base = int(seg.get("address", 0))
            for blk in seg.get("blocks", []):
                if blk.get("state") not in {"active_allocated", "active_pending_free"}:
                    continue
                addr = base + int(blk.get("offset", 0))
                if (addr & _MEM_HIST_FAULT_PAGE_MASK) == _MEM_HIST_FAULT_PAGE:
                    frames = blk.get("frames") or []
                    frame_summary = [
                        (
                            f.get("filename", "?").rsplit("/", 1)[-1]
                            + ":"
                            + str(f.get("line", "?"))
                            + " "
                            + f.get("name", "?")
                        )
                        for f in frames[:6]
                    ]
                    hits.append((hex(addr), int(blk.get("size", 0)), frame_summary))
        if hits:
            logger.warning(
                "[plan04] rank=%d call=%d stage=%s ALLOC AT FAULT PAGE: %s",
                rank,
                _MEM_HIST_CALL_IDX,
                stage,
                hits,
            )
        with open(path, "wb") as fh:
            pickle.dump(snap, fh)
        logger.warning(
            "[plan04] rank=%d call=%d stage=%s mem-snapshot -> %s (%d segments)",
            rank,
            _MEM_HIST_CALL_IDX,
            stage,
            path,
            len(snap.get("segments", [])),
        )
    except Exception as exc:  # noqa: BLE001 — instrumentation only, must not crash run
        logger.error("[plan04] mem-history dump failed (%s): %r", stage, exc)


def _truthy(val: str) -> bool:
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _falsy(val: str) -> bool:
    return val.strip().lower() in {"0", "false", "no", "off"}


def _sparse_world() -> int:
    try:
        return max(1, int(os.environ.get("DLRM_SPARSE_WORLD", "1")))
    except ValueError:
        return 1


def _sparse_rank() -> int:
    try:
        return max(0, int(os.environ.get("DLRM_SPARSE_RANK", "0")))
    except ValueError:
        return 0


def replicate_enabled() -> bool:
    """Return ``True`` when each rank should hold the full ``W*S`` global cap.

    Defaults to ``True`` when ``DLRM_SPARSE_WORLD > 1`` so the multi-worker
    smoke produces correct predictions out of the box. Set
    ``DLRM_SPARSE_REPLICATE=0`` to opt back into the Step-3a per-rank
    shard storage (rank-local divergent predictions; intended only for
    ablating the load planner without the cap bump).
    """
    raw = os.environ.get("DLRM_SPARSE_REPLICATE", "").strip().lower()
    if raw:
        if _truthy(raw):
            return True
        if _falsy(raw):
            return False
    return _sparse_world() > 1


def effective_table_rows(per_rank_rows: int) -> int:
    """Per-rank live row count for a sharded FQN.

    ``per_rank_rows`` is ``DLRM_SPARSE_MAX_HASH_SIZE`` (or the saved
    table size if smaller). In replicated mode each rank grows its live
    table to the full global cap so any global index can be answered
    by a rank-local lookup; otherwise the per-rank cap stands and the
    legacy Step-3a sharded path applies.
    """
    if per_rank_rows <= 0:
        return per_rank_rows
    if replicate_enabled():
        return per_rank_rows * _sparse_world()
    return per_rank_rows


# ---------------------------------------------------------------------------
# Step 3c scaffolding (NOT INVOKED by the Step 3b replicated path)
# ---------------------------------------------------------------------------


_PG_LOCK = threading.Lock()
_PG: Optional["torch.distributed.ProcessGroup"] = None
_PG_INIT_FAILED: bool = False
_PG_BACKEND: str = ""


def _resolve_pg_backend() -> str:
    explicit = os.environ.get("DLRM_SPARSE_PG_BACKEND", "").strip().lower()
    if explicit:
        return explicit
    # Default to gloo for the scaffolding: NCCL/RCCL on ROCm requires
    # every rank to see every peer's GPU (no per-rank HIP_VISIBLE_DEVICES
    # pin) which conflicts with the Step 3a stability fix. gloo runs
    # over CPU sockets and is topology-agnostic.
    return "gloo"


def get_worker_process_group() -> Optional["torch.distributed.ProcessGroup"]:
    """Lazy-init the worker-only PG used by the future Step 3c routing.

    Returns ``None`` when ``WORLD<=1`` or ``DLRM_SPARSE_PG_DISABLE=1``.
    Thread-safe; first caller takes the cost of init.

    **Not called by the Step 3b replicated-cap forward path.** It exists
    so the Step 3c all-to-all path can be wired in without touching env
    contracts or the launcher.
    """

    global _PG, _PG_INIT_FAILED, _PG_BACKEND

    if _PG is not None:
        return _PG
    if _PG_INIT_FAILED:
        return None
    if _truthy(os.environ.get("DLRM_SPARSE_PG_DISABLE", "")):
        return None

    world = _sparse_world()
    if world <= 1:
        return None

    with _PG_LOCK:
        if _PG is not None:
            return _PG
        if _PG_INIT_FAILED:
            return None

        try:
            import torch.distributed as dist  # noqa: WPS433
        except Exception as exc:
            logger.error("[phase2b-3c] torch.distributed unavailable: %r", exc)
            _PG_INIT_FAILED = True
            return None

        rank = _sparse_rank()
        backend = _resolve_pg_backend()
        master_addr = os.environ.get("DLRM_SPARSE_PG_MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("DLRM_SPARSE_PG_MASTER_PORT", "29501")
        try:
            timeout_s = float(os.environ.get("DLRM_SPARSE_PG_TIMEOUT_S", "300"))
        except ValueError:
            timeout_s = 300.0

        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", master_port)

        if dist.is_initialized():
            _PG = dist.group.WORLD
            _PG_BACKEND = dist.get_backend(_PG)
            return _PG

        try:
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world,
                rank=rank,
                timeout=datetime.timedelta(seconds=timeout_s),
            )
        except Exception as exc:
            logger.error(
                "[phase2b-3c] process group init failed (backend=%s): %r; "
                "Step 3c routing will be unavailable",
                backend,
                exc,
            )
            _PG_INIT_FAILED = True
            return None

        _PG = dist.group.WORLD
        _PG_BACKEND = backend
        logger.warning(
            "[phase2b-3c] worker process group ready: backend=%s rank=%d/%d "
            "master=%s:%s (scaffolding only — not invoked by 3b replicated path)",
            backend,
            rank,
            world,
            master_addr,
            master_port,
        )
        return _PG


def routing_backend() -> str:
    """Resolved Step-3c PG backend name (``""`` until first init)."""
    return _PG_BACKEND


# ---------------------------------------------------------------------------
# Plan 12 §3.2 — MPI-lookup substitute helpers (registration API + pinned
# pool + the three Alltoall* helpers). The harness should call
# ``set_route_lookup_comm(worker_comm.Dup())`` once during init; route_lookup
# then branches on ``DLRM_USE_MPI_LOOKUP`` (env) or its ``use_mpi_lookup``
# kwarg to choose the MPI vs torch.distributed transport.
#
# The helpers shape-match the Phase 12.1 microbench
# (`scripts/debug/plan12_route_lookup_microbench.py`) so the cost model
# carries over: at smoke cap W=8 the 3 helpers together cost ~1.23 ms p99
# per batch vs RCCL's ~0.7 ms × 3 = 2 ms. The bet is stream decoupling
# under sharded dispatch (see ``plans/12_MPI_Lookup_Substitute.md`` §1.1).
# ---------------------------------------------------------------------------

_ROUTE_LOOKUP_COMM_LOCK = threading.Lock()
_ROUTE_LOOKUP_COMM = None  # mpi4py.MPI.Comm or None
_ROUTE_LOOKUP_COMM_BANNER_SHOWN = False
_ROUTE_LOOKUP_PATH_BANNER_SHOWN = False

# Pinned-CPU staging buffers, keyed on (torch.dtype, capacity). Grow on demand,
# reused across route_lookup invocations. Soft cap 64 MB / hard cap 256 MB
# per buffer protects against runaway outlier payloads (Plan 12 §2.3 spec).
_PINNED_POOL: dict = {}
_PINNED_SOFT_CAP_BYTES = 64 * 1024 * 1024
_PINNED_HARD_CAP_BYTES = 256 * 1024 * 1024


def set_route_lookup_comm(comm) -> None:
    """Register an mpi4py subcomm for the MPI-lookup substitute path.

    Plan 12 §2.3 / §2.4: pass ``worker_comm.Dup()`` from the harness so the
    route_lookup collectives run on a dedicated communicator isolated from
    the lockstep Allreduce/Bcast subcomm (MPI_THREAD_FUNNELED safety). The
    setter is idempotent on the same comm; replacing with a different comm
    warns; passing ``None`` unregisters (used by Phase 12.4 correctness
    audit to switch back to the torch path mid-run).
    """
    global _ROUTE_LOOKUP_COMM
    with _ROUTE_LOOKUP_COMM_LOCK:
        if _ROUTE_LOOKUP_COMM is comm:
            return
        if _ROUTE_LOOKUP_COMM is not None and comm is not None:
            logger.warning(
                "[plan12] set_route_lookup_comm overwriting existing comm "
                "(was %r → %r)",
                _ROUTE_LOOKUP_COMM,
                comm,
            )
        _ROUTE_LOOKUP_COMM = comm
        if comm is not None:
            try:
                logger.warning(
                    "[plan12] route_lookup_comm registered: rank=%d/%d",
                    comm.Get_rank(),
                    comm.Get_size(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[plan12] registered comm rejects Get_rank/Get_size: %r",
                    exc,
                )


def get_route_lookup_comm():
    """Return the registered mpi4py subcomm, or ``None``."""
    return _ROUTE_LOOKUP_COMM


def _use_mpi_lookup_env() -> bool:
    """Read ``DLRM_USE_MPI_LOOKUP`` env; default off."""
    return _truthy(os.environ.get("DLRM_USE_MPI_LOOKUP", ""))


def _pinned(dtype: torch.dtype, n_elem: int, purpose: str = "default") -> torch.Tensor:
    """Return a pinned-CPU torch tensor of at least ``n_elem`` of ``dtype``.

    Grow-on-demand, keyed on (dtype, purpose); reused across calls. The
    ``purpose`` tag separates send / recv pools so a single MPI call can
    safely pass distinct send and recv buffers — without it, two
    back-to-back ``_pinned(int64, W)`` calls would alias the same storage
    and the MPI collective would silently do an in-place exchange (which
    is undefined behaviour without explicit ``MPI_IN_PLACE``; observed as
    ``MPI_ERR_TRUNCATE`` under asymmetric counts on the route_lookup
    indices collective during Phase 12.3 harness smoke). Buffers are
    sliced on return when over-sized.
    """
    if n_elem == 0:
        # Zero-length pinned buffers occasionally trip torch's allocator on
        # ROCm; return a tiny placeholder. The caller must guard with
        # numel()==0 checks before using it.
        n_elem = 1
    key = (dtype, purpose)
    cached = _PINNED_POOL.get(key)
    if cached is not None and cached.numel() >= n_elem:
        return cached[:n_elem]
    bytes_needed = n_elem * torch.empty(0, dtype=dtype).element_size()
    if bytes_needed > _PINNED_HARD_CAP_BYTES:
        raise RuntimeError(
            f"[plan12] _pinned() refused {bytes_needed:,} byte alloc "
            f"(> {_PINNED_HARD_CAP_BYTES:,} hard cap); n_elem={n_elem} "
            f"dtype={dtype} purpose={purpose!r}. Outlier payload — "
            f"investigate before retry."
        )
    if bytes_needed > _PINNED_SOFT_CAP_BYTES:
        logger.warning(
            "[plan12] _pinned() growing past %d MB soft cap: %.1f MB "
            "(dtype=%s n_elem=%d purpose=%r)",
            _PINNED_SOFT_CAP_BYTES // (1024 * 1024),
            bytes_needed / (1024 * 1024),
            dtype,
            n_elem,
            purpose,
        )
    buf = torch.empty(n_elem, dtype=dtype, pin_memory=True)
    _PINNED_POOL[key] = buf
    return buf


@torch.no_grad()
def _mpi_alltoall_int64(send_gpu: torch.Tensor,
                         recv_gpu: torch.Tensor,
                         comm) -> None:
    """Symmetric Alltoall on int64 — route_lookup step 4 (counts collective).

    Both tensors must be 1-D int64 of length W. Stages D2H → mpi4py.Alltoall
    → H2D using pinned CPU buffers. Synchronous on the current CUDA stream.
    """
    from mpi4py import MPI  # noqa: WPS433 - lazy import

    world = comm.Get_size()
    assert send_gpu.dtype == torch.int64 and recv_gpu.dtype == torch.int64
    assert send_gpu.numel() == world and recv_gpu.numel() == world, (
        f"counts shape mismatch: send={send_gpu.numel()} "
        f"recv={recv_gpu.numel()} W={world}"
    )
    p_send = _pinned(torch.int64, world, purpose="counts_send")
    p_recv = _pinned(torch.int64, world, purpose="counts_recv")
    p_send.copy_(send_gpu.detach(), non_blocking=True)
    torch.cuda.synchronize()
    comm.Alltoall(p_send.detach().numpy(), p_recv.detach().numpy())
    recv_gpu.copy_(p_recv, non_blocking=True)
    torch.cuda.synchronize()


@torch.no_grad()
def _mpi_alltoallv_int64(send_gpu: torch.Tensor,
                          send_counts: list,
                          recv_gpu: torch.Tensor,
                          recv_counts: list,
                          comm) -> None:
    """Variable-size Alltoallv on int64 — route_lookup step 5 (indices)."""
    import numpy as np  # noqa: WPS433
    from mpi4py import MPI  # noqa: WPS433

    world = comm.Get_size()
    assert len(send_counts) == world and len(recv_counts) == world
    assert send_gpu.dtype == torch.int64 and recv_gpu.dtype == torch.int64
    assert send_gpu.numel() == sum(send_counts), (
        f"send buf size {send_gpu.numel()} != sum(send_counts) {sum(send_counts)}"
    )
    assert recv_gpu.numel() == sum(recv_counts), (
        f"recv buf size {recv_gpu.numel()} != sum(recv_counts) {sum(recv_counts)}"
    )
    sc = np.asarray(send_counts, dtype=np.int64)
    rc = np.asarray(recv_counts, dtype=np.int64)
    sd = np.zeros(world, dtype=np.int64)
    sd[1:] = np.cumsum(sc[:-1])
    rd = np.zeros(world, dtype=np.int64)
    rd[1:] = np.cumsum(rc[:-1])

    n_send = int(send_gpu.numel())
    n_recv = int(recv_gpu.numel())
    p_send = _pinned(torch.int64, max(1, n_send), purpose="indices_send")
    p_recv = _pinned(torch.int64, max(1, n_recv), purpose="indices_recv")
    if n_send > 0:
        p_send[:n_send].copy_(send_gpu.detach(), non_blocking=True)
        torch.cuda.synchronize()
    send_np = p_send[:n_send].detach().numpy() if n_send > 0 else np.empty(0, dtype=np.int64)
    recv_np = p_recv[:n_recv].detach().numpy() if n_recv > 0 else np.empty(0, dtype=np.int64)
    comm.Alltoallv(
        (send_np, (sc, sd), MPI.INT64_T),
        (recv_np, (rc, rd), MPI.INT64_T),
    )
    if n_recv > 0:
        recv_gpu.copy_(p_recv[:n_recv], non_blocking=True)
        torch.cuda.synchronize()


@torch.no_grad()
def _mpi_alltoallv_typed_bytes(send_gpu: torch.Tensor,
                                 send_counts_elem: list,
                                 recv_gpu: torch.Tensor,
                                 recv_counts_elem: list,
                                 comm) -> None:
    """Variable-size Alltoallv via MPI.BYTE view — route_lookup step 7 (vectors).

    ``counts_elem`` are in SCALAR elements (for fp16 vectors with
    embed_dim=512, a row count of 100 means 100*512 = 51 200 scalar
    elements). We send/receive a BYTE view of the pinned buffer so the
    same helper works for fp16 / bf16 / fp32 without depending on MPI
    having a native dtype.
    """
    import numpy as np  # noqa: WPS433
    from mpi4py import MPI  # noqa: WPS433

    world = comm.Get_size()
    assert len(send_counts_elem) == world and len(recv_counts_elem) == world
    assert send_gpu.dtype == recv_gpu.dtype
    bytes_per = send_gpu.element_size()
    assert send_gpu.numel() == sum(send_counts_elem)
    assert recv_gpu.numel() == sum(recv_counts_elem)

    sc_e = np.asarray(send_counts_elem, dtype=np.int64)
    rc_e = np.asarray(recv_counts_elem, dtype=np.int64)
    sc_b = sc_e * bytes_per
    rc_b = rc_e * bytes_per
    sd_b = np.zeros(world, dtype=np.int64)
    sd_b[1:] = np.cumsum(sc_b[:-1])
    rd_b = np.zeros(world, dtype=np.int64)
    rd_b[1:] = np.cumsum(rc_b[:-1])

    n_send = int(send_gpu.numel())
    n_recv = int(recv_gpu.numel())
    p_send = _pinned(send_gpu.dtype, max(1, n_send), purpose="vectors_send")
    p_recv = _pinned(recv_gpu.dtype, max(1, n_recv), purpose="vectors_recv")
    # ``send_gpu`` / ``recv_gpu`` may be 2-D (e.g. (n_rows, embed_dim)) but the
    # pinned buffer is 1-D — flatten via .view(-1) so .copy_() works without
    # broadcasting (requires the caller to assert contiguity, which the
    # route_lookup vectors site does).
    if n_send > 0:
        p_send[:n_send].copy_(send_gpu.detach().reshape(-1), non_blocking=True)
        torch.cuda.synchronize()
    if n_send > 0:
        send_view = p_send[:n_send].detach().numpy().view(np.uint8)
    else:
        send_view = np.empty(0, dtype=np.uint8)
    if n_recv > 0:
        recv_view = p_recv[:n_recv].detach().numpy().view(np.uint8)
    else:
        recv_view = np.empty(0, dtype=np.uint8)
    comm.Alltoallv(
        (send_view, (sc_b, sd_b), MPI.BYTE),
        (recv_view, (rc_b, rd_b), MPI.BYTE),
    )
    if n_recv > 0:
        recv_gpu.detach().reshape(-1).copy_(p_recv[:n_recv], non_blocking=True)
        torch.cuda.synchronize()


def _resolve_route_lookup_path(use_mpi_lookup: Optional[bool]) -> tuple:
    """Pick the route_lookup transport — torch.distributed or mpi4py.

    Returns ``(use_mpi: bool, mpi_comm or None)``. Logs a one-shot warning
    if MPI was requested but no comm is registered (and falls back to torch).
    """
    global _ROUTE_LOOKUP_COMM_BANNER_SHOWN, _ROUTE_LOOKUP_PATH_BANNER_SHOWN

    if use_mpi_lookup is None:
        use_mpi_lookup = _use_mpi_lookup_env()
    if not use_mpi_lookup:
        return False, None

    mpi_comm = get_route_lookup_comm()
    if mpi_comm is None:
        if not _ROUTE_LOOKUP_COMM_BANNER_SHOWN:
            logger.warning(
                "[plan12] DLRM_USE_MPI_LOOKUP=1 but no route_lookup_comm "
                "registered (harness did not call set_route_lookup_comm); "
                "falling back to torch.distributed transport"
            )
            _ROUTE_LOOKUP_COMM_BANNER_SHOWN = True
        return False, None

    try:
        import mpi4py.MPI  # noqa: F401, WPS433 - probe importability
    except Exception as exc:
        if not _ROUTE_LOOKUP_COMM_BANNER_SHOWN:
            logger.warning(
                "[plan12] DLRM_USE_MPI_LOOKUP=1 but mpi4py import failed (%r); "
                "falling back to torch.distributed transport",
                exc,
            )
            _ROUTE_LOOKUP_COMM_BANNER_SHOWN = True
        return False, None

    if not _ROUTE_LOOKUP_PATH_BANNER_SHOWN:
        logger.warning(
            "[plan12] route_lookup using MPI-substitute transport "
            "(DLRM_USE_MPI_LOOKUP=1, comm rank=%d/%d)",
            mpi_comm.Get_rank(),
            mpi_comm.Get_size(),
        )
        _ROUTE_LOOKUP_PATH_BANNER_SHOWN = True
    return True, mpi_comm


def route_lookup(
    emb_module: "torch.nn.Embedding",
    global_idx: torch.Tensor,
    shard_rows: int,
    *,
    use_mpi_lookup: Optional[bool] = None,
) -> torch.Tensor:
    """Cross-rank sharded embedding lookup via ``all_to_all_single``.

    Each rank holds rows ``[r*S, (r+1)*S)`` of a global ``W*S``-row
    capped embedding (Step 3a per-rank shard storage). This helper
    routes each query index to its owning rank, performs the local
    lookup there, and ships the resulting embedding back, so every
    rank ends up with a globally-correct ``(N, D)`` output even though
    it only holds its own shard.

    Algorithm
    ---------
    1. Bucket each local query index by ``target_rank = global_idx //
       S``. Indices outside ``[0, W*S)`` are flagged as OOV.
    2. Stable-sort the indices by ``target_rank`` and trim OOV (they
       are sorted to the end and dropped).
    3. ``all_to_all_single(recv_counts, send_counts)`` so each rank
       learns how many indices it will receive from every peer.
    4. ``all_to_all_single`` the **local** row IDs (``global_idx -
       target_rank * S``) using the matched per-rank splits.
    5. Local lookup on the received row IDs in ``[0, S)`` row space.
    6. Reverse ``all_to_all_single`` of the resulting embedding
       vectors with the splits swapped.
    7. Inverse-permute the vectors back into the caller's original
       index order. OOV positions stay at zero.

    Parameters
    ----------
    emb_module:
        Local ``torch.nn.Embedding`` (or equivalent) holding this
        rank's ``S``-row shard. Its ``weight.shape[0]`` must equal
        ``shard_rows``.
    global_idx:
        1-D ``LongTensor`` of length ``N`` containing **global**
        (un-clamped) row indices in ``[0, W*S)``. May live on CPU or
        the same device as ``emb_module.weight``; the helper bounces
        through CPU for gloo and stays on-device for nccl/rccl.
    shard_rows:
        ``S`` — per-rank row count. ``W*S`` is the implied global cap.

    Returns
    -------
    ``Tensor`` of shape ``(N, D)`` on the same device as
    ``global_idx``, dtype matching ``emb_module.weight.dtype``.

    Fallback
    --------
    If ``get_worker_process_group()`` returns ``None`` (single rank,
    PG init failed, or ``DLRM_SPARSE_PG_DISABLE=1``) the helper does
    a local clamped lookup and returns immediately — caller doesn't
    have to special-case the no-PG path.
    """

    # Plan 12 §3.2 — resolve transport (torch.distributed vs mpi4py) BEFORE
    # any pg work. ``use_mpi`` only goes True if (a) requested AND (b) a
    # route_lookup_comm is registered AND (c) mpi4py is importable.
    use_mpi, mpi_comm = _resolve_route_lookup_path(use_mpi_lookup)

    # We still need ``pg`` for world_size + the torch.distributed fallback,
    # so resolve it lazily. Under the MPI path ``world`` comes from
    # ``mpi_comm.Get_size()`` to avoid forcing PG init when not needed.
    pg = None if use_mpi else get_worker_process_group()

    weight_device = emb_module.weight.device
    embed_dim = int(emb_module.weight.shape[1])
    embed_dtype = emb_module.weight.dtype
    src_device = global_idx.device

    # Plan 04 Phase 4.1 — bump the per-rank call counter on every entry; the
    # snapshot helper is a no-op when DLRM_MEM_HISTORY!=1.
    global _MEM_HIST_CALL_IDX
    _MEM_HIST_CALL_IDX += 1
    _mem_history_dump("entry")

    if not use_mpi and pg is None:
        # No PG and no MPI → fall back to the Step-3a rank-local clamp behavior
        # so callers get a tensor of the right shape/dtype either way.
        local = torch.clamp(global_idx, 0, shard_rows - 1)
        if local.device != weight_device:
            local = local.to(weight_device, non_blocking=True)
        out = emb_module(input=local)
        if out.device != src_device:
            out = out.to(src_device, non_blocking=True)
        return out

    import torch.distributed as dist  # noqa: WPS433

    if use_mpi:
        world = int(mpi_comm.Get_size())
        backend = "mpi"
        # MPI helpers stage through their own pinned-CPU buffers. The
        # in/out tensors should stay on the GPU; helpers handle D2H/H2D.
        collective_on_cpu = False
        coll_device = weight_device
    else:
        world = dist.get_world_size(pg)
        backend = dist.get_backend(pg)
        # gloo does not support GPU collective tensors reliably across
        # ROCm builds — route everything through CPU and copy back at the
        # end. nccl/rccl stays on-device.
        collective_on_cpu = backend == "gloo"
        coll_device = torch.device("cpu") if collective_on_cpu else weight_device

    route_t0 = time.perf_counter()

    # ---- step 1: bucket by owning rank ------------------------------
    # Keep the index tensor's original dtype (int64 from KJT); cast the
    # bucket math to int64 explicitly so torch.bincount is happy.
    # ``non_blocking=True`` is unsafe for D→H copies whose result is read
    # synchronously by the very next CPU op (in_range/argsort): on ROCm
    # the destination is not pinned, so the copy enqueues asynchronously
    # against pending HIP work and the CPU-side read may race the copy.
    # Use a blocking copy when moving off-GPU; keep non_blocking for the
    # GPU→GPU or H→H cases where it has no effect.
    _idx_non_blocking = not (coll_device.type == "cpu"
                              and global_idx.device.type == "cuda")
    idx = global_idx.to(coll_device, dtype=torch.int64,
                        non_blocking=_idx_non_blocking)
    N = int(idx.numel())
    global_cap = shard_rows * world
    in_range = (idx >= 0) & (idx < global_cap)
    # OOV indices get sentinel target = world so they sort to the end
    # and are trimmed by the [:world] bincount slice below.
    target = torch.where(
        in_range,
        idx // shard_rows,
        torch.full_like(idx, world),
    )

    # ---- step 2: stable-sort by target rank ------------------------
    sort_perm = torch.argsort(target, stable=True)
    idx_sorted = idx[sort_perm]
    target_sorted = target[sort_perm]

    # ---- step 3: build send_counts; trim OOV tail ------------------
    # bincount with minlength=world+1 returns one extra bin for the
    # OOV sentinel; [:world] drops it.
    send_counts = torch.bincount(target_sorted, minlength=world + 1)[:world]
    send_list = send_counts.tolist()
    n_valid = int(sum(send_list))
    # All-valid prefix of the sort-sorted indices.
    idx_to_send = idx_sorted[:n_valid]
    target_to_send = target_sorted[:n_valid]
    # Convert to local row IDs in the owner's row space.
    local_rows_to_send = (idx_to_send - target_to_send * shard_rows).to(
        torch.int64
    )
    t_bucket = time.perf_counter()
    _timing_record("route.bucket_sort", t_bucket - route_t0)

    # ---- step 4: exchange counts so peers know what they'll receive --
    recv_counts = torch.empty(world, dtype=torch.int64, device=coll_device)
    send_counts_dev = send_counts.to(coll_device)
    if use_mpi:
        _mpi_alltoall_int64(send_counts_dev, recv_counts, mpi_comm)
    else:
        dist.all_to_all_single(recv_counts, send_counts_dev, group=pg)
    recv_list = recv_counts.tolist()
    n_recv = int(sum(recv_list))
    t_a2a_counts = time.perf_counter()
    _timing_record(
        "route.mpi_a2a_counts" if use_mpi else "route.a2a_counts",
        t_a2a_counts - t_bucket,
    )

    # ---- step 5: exchange local row IDs ----------------------------
    recv_local_rows = torch.empty(n_recv, dtype=torch.int64, device=coll_device)
    # Plan 05 (W=8 lockstep deadlock fix): every rank MUST participate in
    # this collective. The previous `if n_valid == 0 and n_recv == 0: pass`
    # was a per-rank early-exit written for gloo edge cases; under W=8
    # lockstep dispatch only the ZMQ-receiver rank holds the batch indices,
    # so most ranks have n_valid==0. If the receiver's per-peer bucket
    # happens to be zero for one specific peer p, peer p has BOTH
    # n_valid==0 AND n_recv==0 and skips, while the other 7 ranks call
    # into the collective with NumelIn=0 / NumelOut=non-zero. Asymmetric
    # participation deadlocks RCCL's alltoallv decomposition. RCCL handles
    # an all-zero alltoall_single as a no-op barrier correctly, so the
    # guard is unnecessary; gloo's edge cases the comment cited were a
    # different stack. Trigger probability is non-zero on any batch where
    # bucket distribution leaves one peer empty.
    local_rows_dev = local_rows_to_send.to(coll_device)
    if use_mpi:
        _mpi_alltoallv_int64(
            local_rows_dev, send_list, recv_local_rows, recv_list, mpi_comm
        )
    else:
        dist.all_to_all_single(
            recv_local_rows,
            local_rows_dev,
            output_split_sizes=recv_list,
            input_split_sizes=send_list,
            group=pg,
        )
    t_a2a_indices = time.perf_counter()
    _timing_record(
        "route.mpi_a2a_indices" if use_mpi else "route.a2a_indices",
        t_a2a_indices - t_a2a_counts,
    )
    _timing_record("route.indices_bytes", float(n_valid * 8))
    _mem_history_dump("after_a2a_indices")

    # ---- step 6: local lookup on received rows ---------------------
    if n_recv > 0:
        rows_for_lookup = recv_local_rows
        if rows_for_lookup.device != weight_device:
            rows_for_lookup = rows_for_lookup.to(
                weight_device, non_blocking=True
            )
        # Defensive clamp — peers should already have sent local-space
        # rows in [0, shard_rows), but a buggy sender shouldn't take
        # down the receiver with an OOB index → HSA fault.
        rows_for_lookup = torch.clamp(rows_for_lookup, 0, shard_rows - 1)
        recv_embeddings = emb_module(input=rows_for_lookup)
    else:
        recv_embeddings = torch.empty(
            (0, embed_dim), dtype=embed_dtype, device=weight_device
        )

    if collective_on_cpu and recv_embeddings.device.type != "cpu":
        recv_embeddings = recv_embeddings.cpu()
    t_local_lookup = time.perf_counter()
    _timing_record("route.local_lookup", t_local_lookup - t_a2a_indices)

    # ---- step 7: reverse exchange (vectors back to requesters) -----
    send_embeddings = torch.empty(
        (n_valid, embed_dim), dtype=embed_dtype, device=recv_embeddings.device
    )
    # Plan 05 (W=8 lockstep deadlock fix): symmetric to step 5, every rank
    # must participate. See the comment at step 5 above for the rationale.
    if use_mpi:
        # Step 7 is the REVERSE exchange of step 5 — we send the per-peer
        # locally-looked-up vectors (``recv_embeddings`` shape (n_recv,
        # embed_dim), counts = ``recv_list``) and receive the per-peer
        # results for our own queries (``send_embeddings`` shape (n_valid,
        # embed_dim), counts = ``send_list``). Convert row counts → scalar
        # element counts (× embed_dim) for the BYTE-view helper.
        send_elem = [c * embed_dim for c in send_list]
        recv_elem = [c * embed_dim for c in recv_list]
        assert send_embeddings.is_contiguous() and recv_embeddings.is_contiguous(), \
            "MPI vectors path requires contiguous send/recv embedding buffers"
        _mpi_alltoallv_typed_bytes(
            recv_embeddings, recv_elem,    # OUT to peers: this rank's local-lookup results
            send_embeddings, send_elem,    # IN  from peers: vectors for this rank's queries
            mpi_comm,
        )
    else:
        dist.all_to_all_single(
            send_embeddings,
            recv_embeddings,
            output_split_sizes=send_list,
            input_split_sizes=recv_list,
            group=pg,
        )
    t_a2a_vectors = time.perf_counter()
    _timing_record(
        "route.mpi_a2a_vectors" if use_mpi else "route.a2a_vectors",
        t_a2a_vectors - t_local_lookup,
    )
    # bytes/side of the vec exchange (fp16=2, fp32=4) for bandwidth maths.
    _timing_record(
        "route.vectors_bytes",
        float(n_valid * embed_dim * embed_dtype.itemsize),
    )

    # ---- step 8: scatter back into original order ------------------
    output = torch.zeros(
        (N, embed_dim),
        dtype=embed_dtype,
        device=send_embeddings.device,
    )
    if n_valid > 0:
        # First n_valid entries of sort_perm point to the original
        # positions of the in-range indices (OOV is sorted past them).
        valid_positions = sort_perm[:n_valid].to(output.device)
        output.index_copy_(0, valid_positions, send_embeddings)

    if output.device != src_device:
        output = output.to(src_device, non_blocking=True)
    t_scatter = time.perf_counter()
    _timing_record("route.scatter", t_scatter - t_a2a_vectors)
    _timing_record("route.total", t_scatter - route_t0)
    _mem_history_dump("exit")
    return output
