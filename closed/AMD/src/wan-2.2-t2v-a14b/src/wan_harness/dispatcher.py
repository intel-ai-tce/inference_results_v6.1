"""Dispatcher: routes batches of queries from the SUT to a Backend.

The dispatcher abstracts the topology that connects the LoadGen-owning
process (always rank 0) to its workers. The SUT is dispatcher-agnostic:
it calls :meth:`generate(prompts, indices)` and consumes
``GeneratedVideo`` instances.

Four concrete implementations cover all current use cases:

* :class:`SingleProcessDispatcher` – Mock backend / 1-GPU runs. No
  ``torch.distributed`` involvement.
* :class:`UlyssesDispatcher` – Wan backend in SingleStream. Rank 0
  broadcasts each prompt to all ranks; everyone calls
  ``backend.run_unit`` in lockstep so xfuser's Ulysses SP collective ops
  stay aligned. Only rank 0's output is used.
* :class:`WaveDispatcher` – Wan backend in Offline (default DP).
  Synchronous waves of up to ``world_size`` samples processed by all
  ranks in lockstep using world-group ``broadcast`` + ``gather_object``
  collectives only. No per-pair NCCL sub-communicators, so no lazy-init
  tax during the measured LoadGen window. Wave latency is
  ``max(per-rank latency)``: a slow rank stalls the whole wave.
* :class:`AsyncDPDispatcher` – Wan backend in Offline (opt-in via
  ``parallelism.dispatch: async``). Pull-style scheduler: rank 0 hands
  the next pending prompt to whichever rank just returned a result, so
  fast ranks do not idle behind slow ones. Uses a dedicated Gloo
  subgroup for control + result transfer so the world PG
  (typically NCCL) is left untouched and the wave path remains
  bit-identical.

The non-rank-0 entry point is :meth:`Dispatcher.run_worker_loop`. The
runner branches to it on workers and to LoadGen on rank 0.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Sequence

from .wire import (
    WAVE_CMD_EXIT,
    WAVE_CMD_RUN,
    Result,
    Shutdown,
    SlotRelease,
    WorkUnit,
    WorkerFailure,
    broadcast_object,
    broadcast_str_list,
    configure_result_transport,
    decode_wave_command,
    encode_wave_command,
    gather_results_to_rank0,
    recv_object_any_src,
    recv_object_from_src,
    release_worker_slot,
    send_object_pt2pt,
    use_shm_for_results,
)
from .shm_pool import ShmResultPool, compute_slot_layout, init_shm_result_pool
from .post_run_overhead import (
    PHASE_RESULT_PACK,
    PHASE_RUN_UNIT,
    PHASE_WIRE_TRANSFER,
    get_collector,
)

if TYPE_CHECKING:
    import torch.distributed as torch_dist  # noqa: F401  (typing only)

    from .backends.base import Backend, GeneratedVideo
    from .backends.wan22 import WanBackend

_log = logging.getLogger(__name__)

__all__ = [
    "Dispatcher",
    "DispatcherInfo",
    "SingleProcessDispatcher",
    "UlyssesDispatcher",
    "WaveDispatcher",
    "AsyncDPDispatcher",
    "build_dispatcher",
]


@dataclass(frozen=True)
class DispatcherInfo:
    """Lightweight description of the topology surfaced in logs / metadata."""

    name: str
    rank: int
    world_size: int


class Dispatcher:
    """Abstract dispatcher.

    See module docstring for the three concrete implementations.
    """

    info: DispatcherInfo

    def generate(
        self,
        prompts: Sequence[str],
        indices: Sequence[int],
    ) -> Iterator["GeneratedVideo"]:
        raise NotImplementedError

    def flush(self) -> None:
        """Block until all in-flight work is complete."""

    def is_response_owner(self) -> bool:
        """Return True if this process should call ``QuerySamplesComplete``."""
        return True

    def run_worker_loop(self) -> None:
        """Entry point for non-rank-0 ranks. Returns when rank 0 broadcasts
        :class:`~wan_harness.wire.Shutdown`.
        """
        return

    def shutdown(self) -> None:
        """Tear down whatever the dispatcher is holding (workers, comms)."""

    # ------------------------------------------------------------------
    # Warmup. Rank-0-only entry point; workers are already in
    # ``run_worker_loop`` and process warmup units indistinguishably
    # from real LoadGen prompts.
    # ------------------------------------------------------------------
    def _warmup_multiplier(self) -> int:
        """How many wire-level prompts must be dispatched to give every
        rank exactly one warmup run.

        * Ulysses / single-process: ``1`` (every rank participates in
          every dispatched prompt).
        * Data parallel: ``world_size`` (each prompt only touches one
          worker, so we need one per rank).
        """
        return 1

    def warmup(self, num_prompts_per_rank: int, prompt: str) -> None:
        """Dispatch ``_warmup_multiplier() * num_prompts_per_rank`` units
        through :meth:`generate` and discard the results.

        Negative ``sample_index`` values are used so warmup completions
        cannot collide with real QSL indices when greping logs.
        """
        if num_prompts_per_rank <= 0:
            return
        total = self._warmup_multiplier() * int(num_prompts_per_rank)
        _log.info(
            "%s.warmup: dispatching %d prompts (n_per_rank=%d, mult=%d)",
            type(self).__name__,
            total,
            num_prompts_per_rank,
            self._warmup_multiplier(),
        )
        t0 = time.perf_counter()
        prompts = [prompt] * total
        indices = list(range(-total, 0))
        consumed = 0
        for _video in self.generate(prompts, indices):
            consumed += 1
        _log.info(
            "%s.warmup: done %d/%d in %.2fs",
            type(self).__name__, consumed, total, time.perf_counter() - t0,
        )


# ----------------------------------------------------------------------
# Single-process (Mock backend).
# ----------------------------------------------------------------------


class SingleProcessDispatcher(Dispatcher):
    """Pass-through dispatcher used by the Mock backend and single-GPU runs."""

    def __init__(self, backend: "Backend") -> None:
        self._backend = backend
        self.info = DispatcherInfo(name="single-process", rank=0, world_size=1)

    def generate(
        self,
        prompts: Sequence[str],
        indices: Sequence[int],
    ) -> Iterator["GeneratedVideo"]:
        n = len(prompts)
        _log.info("SingleProcessDispatcher.generate: n=%d starting", n)
        t_batch = time.perf_counter()
        completed = 0
        for video in self._backend.generate(prompts=prompts, indices=indices):
            completed += 1
            _log.info(
                "SingleProcessDispatcher.complete sample=%d (%d/%d)",
                int(video.sample_index), completed, n,
            )
            yield video
        _log_batch_throughput("SingleProcessDispatcher", completed, n, t_batch)

    def flush(self) -> None:
        return

    def shutdown(self) -> None:
        return


# ----------------------------------------------------------------------
# Helpers shared by the two multi-rank dispatchers.
# ----------------------------------------------------------------------


def _generated_from_result(result: Result) -> "GeneratedVideo":
    """Convert a wire :class:`Result` back into the dataclass the SUT consumes."""
    from .backends.base import GeneratedVideo

    return GeneratedVideo(
        sample_index=int(result.sample_index),
        frames_bytes=result.frames_bytes,
        frame_count=int(result.frame_count),
        height=int(result.height),
        width=int(result.width),
        mp4_bytes=result.mp4_bytes,
    )


def _resolve_result_transport_mode(backend: "Backend") -> str:
    """Return ``shm`` or ``gloo`` for bulk Result transfer."""
    cfg = backend.config
    if getattr(cfg, "result_transport", None) is not None:
        return str(cfg.result_transport)
    if backend.name == "wan22":
        return str(backend.backend_config.parallelism.result_transport)
    return "shm"


def _init_result_transport(
    backend: "Backend",
    *,
    rank: int,
    world_size: int,
    group: "torch_dist.ProcessGroup | None",
) -> ShmResultPool | None:
    mode = _resolve_result_transport_mode(backend)
    if mode != "shm":
        configure_result_transport(None, use_shm=False)
        _log.info(
            "Result transport: gloo bulk transfer (result_transport=%s rank=%d)",
            mode,
            rank,
        )
        return None
    layout = compute_slot_layout(
        height=int(backend.config.height),
        width=int(backend.config.width),
        num_frames=int(backend.config.num_frames),
    )
    pool = init_shm_result_pool(
        rank=int(rank),
        world_size=int(world_size),
        layout=layout,
        group=group,
    )
    configure_result_transport(pool, use_shm=pool is not None)
    if pool is not None:
        _log.info(
            "Result transport: SHM data plane (rank=%d slot=%.1f MiB)",
            rank,
            pool.slot_bytes / (1024 * 1024),
        )
    return pool


def _teardown_result_transport(pool: ShmResultPool | None, *, rank: int) -> None:
    configure_result_transport(None, use_shm=False)
    if pool is not None:
        pool.close(unlink=True)


def _result_from_generated(video: "GeneratedVideo") -> Result:
    """Convert a backend-produced :class:`GeneratedVideo` into a wire-friendly
    :class:`Result` for cross-rank send.
    """
    return Result(
        sample_index=int(video.sample_index),
        frames_bytes=bytes(video.frames_bytes),
        frame_count=int(video.frame_count),
        height=int(video.height),
        width=int(video.width),
        mp4_bytes=video.mp4_bytes,
    )


def _run_local_slot_timed(
    backend: "Backend",
    *,
    idx: int,
    prompt: str,
    rank: int,
) -> "Result | WorkerFailure":
    """Execute one active slot and record run_unit / result_pack timings."""
    oc = get_collector()
    try:
        unit = backend.build_work_unit(prompt=prompt, sample_index=int(idx))
        t_run = time.perf_counter()
        video = backend.run_unit(unit)
        oc.record(
            PHASE_RUN_UNIT,
            time.perf_counter() - t_run,
            sample_index=int(idx),
            rank=rank,
        )
        t_pack = time.perf_counter()
        result = _result_from_generated(video)
        oc.record(
            PHASE_RESULT_PACK,
            time.perf_counter() - t_pack,
            sample_index=int(idx),
            rank=rank,
            nbytes=len(result.frames_bytes),
        )
        return result
    except Exception as exc:  # noqa: BLE001 – surface ANY failure
        _log.exception(
            "Dispatcher: rank %d failed on sample %d",
            rank, int(idx),
        )
        return WorkerFailure(
            rank=rank,
            sample_index=int(idx),
            error_repr=repr(exc),
        )


def _gather_result_timed(
    local: "Result | WorkerFailure | None",
    *,
    world_size: int,
    rank: int,
    sample_index: int,
) -> list["Result | WorkerFailure | None"]:
    """Gather one slot's payload to rank 0 and record wire-transfer time."""
    oc = get_collector()
    nbytes = 0
    if isinstance(local, Result):
        nbytes = len(local.frames_bytes)
    t_wire = time.perf_counter()
    bucket = gather_results_to_rank0(
        local, world_size=world_size, rank=rank
    )
    oc.record(
        PHASE_WIRE_TRANSFER,
        time.perf_counter() - t_wire,
        sample_index=int(sample_index),
        rank=rank,
        nbytes=nbytes,
    )
    return bucket


def _send_result_timed(
    payload: "Result | WorkerFailure",
    *,
    dst: int,
    group: "torch_dist.ProcessGroup | None",
    rank: int,
    sample_index: int,
) -> None:
    """Point-to-point result send with wire-transfer timing."""
    oc = get_collector()
    nbytes = len(payload.frames_bytes) if isinstance(payload, Result) else 0
    t_wire = time.perf_counter()
    send_object_pt2pt(payload, dst=dst, group=group, src_rank=rank)
    if use_shm_for_results() and isinstance(payload, Result):
        ack = recv_object_from_src(src=0, group=group)
        if not isinstance(ack, SlotRelease) or int(ack.rank) != int(rank):
            raise RuntimeError(
                f"rank {rank}: expected SlotRelease for rank {rank}, got {ack!r}"
            )
    oc.record(
        PHASE_WIRE_TRANSFER,
        time.perf_counter() - t_wire,
        sample_index=int(sample_index),
        rank=rank,
        nbytes=nbytes,
    )


def _log_batch_throughput(
    name: str, completed: int, total: int, t_start: float
) -> None:
    """Log the rank-0-visible aggregate throughput for a ``generate`` call."""
    elapsed = time.perf_counter() - t_start
    rate = (completed / elapsed) if elapsed > 0 else 0.0
    _log.info(
        "%s.generate done: %d/%d in %.2fs (%.3f prompts/sec, %.2fs/prompt avg)",
        name, completed, total, elapsed, rate, (elapsed / completed) if completed else 0.0,
    )


# ----------------------------------------------------------------------
# Ulysses (SingleStream).
# ----------------------------------------------------------------------


class UlyssesDispatcher(Dispatcher):
    """All ranks call ``backend.run_unit`` in lockstep.

    The xfuser-wrapped transformer issues Ulysses-SP collective ops inside
    ``_run_pipe``, so every rank must enter ``run_unit`` at the same time
    with identical arguments. The result is the same on every rank (the
    sequence-parallel gather is internal to xfuser), so we just use
    rank 0's copy.
    """

    def __init__(
        self,
        backend: "WanBackend",
        *,
        rank: int,
        world_size: int,
        device: str | None = None,
    ) -> None:
        self._backend = backend
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self.info = DispatcherInfo(name="ulysses", rank=rank, world_size=world_size)

    def is_response_owner(self) -> bool:
        return self._rank == 0

    # ------------------------------------------------------------------
    # Rank-0 path.
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: Sequence[str],
        indices: Sequence[int],
    ) -> Iterator["GeneratedVideo"]:
        if self._rank != 0:
            raise RuntimeError("UlyssesDispatcher.generate is rank-0 only")
        if len(prompts) != len(indices):
            raise ValueError(
                f"len(prompts)={len(prompts)} != len(indices)={len(indices)}"
            )

        n = len(prompts)
        _log.info(
            "UlyssesDispatcher.generate: n=%d world_size=%d starting",
            n, self._world_size,
        )
        t_batch = time.perf_counter()
        completed = 0
        for prompt, idx in zip(prompts, indices):
            unit = self._backend.build_work_unit(prompt=prompt, sample_index=int(idx))
            broadcast_object(unit, src=0, rank=self._rank, device=self._device)
            t_unit = time.perf_counter()
            video = self._backend.run_unit(unit)
            completed += 1
            _log.info(
                "UlyssesDispatcher.complete sample=%d (%d/%d) latency=%.3fs",
                int(idx), completed, n, time.perf_counter() - t_unit,
            )
            yield video
        _log_batch_throughput("UlyssesDispatcher", completed, n, t_batch)

    def shutdown(self) -> None:
        if self._rank == 0:
            broadcast_object(Shutdown(), src=0, rank=self._rank, device=self._device)

    # ------------------------------------------------------------------
    # Worker path (ranks 1..N-1).
    # ------------------------------------------------------------------
    def run_worker_loop(self) -> None:
        if self._rank == 0:
            return
        _log.info("UlyssesDispatcher: rank %d entering worker loop", self._rank)
        while True:
            msg = broadcast_object(None, src=0, rank=self._rank, device=self._device)
            if isinstance(msg, Shutdown):
                _log.info("UlyssesDispatcher: rank %d received Shutdown", self._rank)
                break
            if isinstance(msg, WorkUnit):
                # Run the same unit; output is discarded (rank 0 produces the canonical result).
                self._backend.run_unit(msg)
                continue
            raise RuntimeError(
                f"UlyssesDispatcher rank {self._rank} got unexpected message {type(msg).__name__}"
            )


# ----------------------------------------------------------------------
# Wave-based data parallel (Offline – default).
# ----------------------------------------------------------------------


class WaveDispatcher(Dispatcher):
    """Collective DP dispatcher: world-group broadcasts + gather.

    Selected when ``parallelism.mode == 'data_parallel'``. Mirrors the
    topology used in the upstream xfuser ``feat/reimplement-xdit-dp``
    reference.

    Wire protocol per wave (all collectives on the world process group):

      1. ``dist.broadcast`` of the fixed-shape int64 command tensor
         (``[cmd, n, idx_0, ..., idx_{W-1}]``).
      2. ``broadcast_object_list`` of the per-rank prompts (``""`` for
         inactive tail slots).
      3. Each rank runs ``backend.run_unit`` for its slot (active slots
         only; inactive ranks contribute ``None`` to step 4).
      4. ``gather_object`` of the per-rank :class:`Result` /
         :class:`~wan_harness.wire.WorkerFailure` / ``None`` back to
         rank 0.

    Properties:

      * Only the world PG is used, so NCCL never has to lazy-init a
        per-pair sub-communicator during the measured LoadGen window –
        the world PG is hot the moment ``dist.init_process_group``
        returns, and xfuser's own DP groups are warmed inside
        ``WanBackend.setup`` via ``_model.initialize(seed_input_args)``.
      * Every rank works on every wave, so the per-rank GPU activity
        histogram is flat by construction.

    Rank 0 also processes its own slot in line (no self-worker thread):
    every rank is doing the same model call and they all finish in
    roughly the same wall time, so there is no overlap to win.
    """

    def __init__(
        self,
        backend: "WanBackend",
        *,
        rank: int,
        world_size: int,
        device: str | None = None,
    ) -> None:
        if world_size < 2:
            raise ValueError(
                f"WaveDispatcher requires world_size>=2, got {world_size}; the "
                f"factory should have selected SingleProcessDispatcher instead."
            )
        self._backend = backend
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self.info = DispatcherInfo(name="wave", rank=rank, world_size=world_size)
        self._shm_pool: ShmResultPool | None = None
        self._transport_ready = False

    # ------------------------------------------------------------------
    # Warmup. Multiplier == world_size so a single ``num_prompts=1``
    # warmup pass dispatches exactly ``world_size`` synthetic prompts,
    # all live in one full wave, and every rank compiles + warms its
    # kernels.
    # ------------------------------------------------------------------
    def _warmup_multiplier(self) -> int:
        return self._world_size

    def is_response_owner(self) -> bool:
        return self._rank == 0

    def _ensure_transport(self) -> None:
        if self._transport_ready:
            return
        self._shm_pool = _init_result_transport(
            self._backend,
            rank=self._rank,
            world_size=self._world_size,
            group=None,
        )
        self._transport_ready = True

    # ------------------------------------------------------------------
    # Rank-0 driver.
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: Sequence[str],
        indices: Sequence[int],
    ) -> Iterator["GeneratedVideo"]:
        if self._rank != 0:
            raise RuntimeError("WaveDispatcher.generate is rank-0 only")
        if len(prompts) != len(indices):
            raise ValueError(
                f"len(prompts)={len(prompts)} != len(indices)={len(indices)}"
            )

        n_total = len(prompts)
        if n_total == 0:
            return

        self._ensure_transport()

        _log.info(
            "WaveDispatcher.generate: n=%d world_size=%d starting",
            n_total, self._world_size,
        )
        t_batch = time.perf_counter()
        completed = 0

        for start in range(0, n_total, self._world_size):
            wave_pmts = list(prompts[start:start + self._world_size])
            wave_idx = [int(i) for i in indices[start:start + self._world_size]]
            n = len(wave_idx)
            t_wave = time.perf_counter()

            self._broadcast_wave(wave_idx, wave_pmts)

            # Rank 0's slot is active iff this wave has any work at all.
            # The active count ``n`` is the only gate; we deliberately do
            # not sniff the slot ints for ``WAVE_INACTIVE_INDEX`` since
            # warmup uses negative sample indices that would collide with
            # the padding sentinel.
            if n > 0:
                local: "Result | WorkerFailure | None" = _run_local_slot_timed(
                    self._backend,
                    idx=wave_idx[0],
                    prompt=wave_pmts[0],
                    rank=self._rank,
                )
            else:
                local = None

            bucket = _gather_result_timed(
                local,
                world_size=self._world_size,
                rank=self._rank,
                sample_index=wave_idx[0] if n > 0 else -1,
            )

            wave_elapsed = time.perf_counter() - t_wave
            for slot in range(n):
                cell = bucket[slot]
                idx = wave_idx[slot]
                if isinstance(cell, WorkerFailure):
                    raise RuntimeError(
                        f"WaveDispatcher: rank {cell.rank} failed on sample "
                        f"{cell.sample_index}: {cell.error_repr}"
                    )
                if cell is None:
                    raise RuntimeError(
                        f"WaveDispatcher: rank {slot} returned no Result for "
                        f"active sample {idx}"
                    )
                if not isinstance(cell, Result):
                    raise RuntimeError(
                        f"WaveDispatcher: rank {slot} returned unexpected "
                        f"payload {type(cell).__name__} for sample {idx}"
                    )
                if int(cell.sample_index) != idx:
                    raise RuntimeError(
                        f"WaveDispatcher: rank {slot} returned sample_index="
                        f"{cell.sample_index} but expected {idx}"
                    )
                completed += 1
                _log.info(
                    "WaveDispatcher.complete sample=%d slot=%d (%d/%d) "
                    "wave_latency=%.3fs",
                    idx, slot, completed, n_total, wave_elapsed,
                )
                yield _generated_from_result(cell)

        _log_batch_throughput("WaveDispatcher", completed, n_total, t_batch)

    def _broadcast_wave(
        self, indices: Sequence[int], prompts: Sequence[str]
    ) -> None:
        """Steps 1 + 2: push the command + prompts to every rank."""
        import torch.distributed as dist  # noqa: WPS433

        cmd_t = encode_wave_command(
            WAVE_CMD_RUN, indices, self._world_size, device=self._device
        )
        dist.broadcast(cmd_t, src=0)

        n = len(indices)
        padded: list[str] = list(prompts) + [""] * (self._world_size - n)
        broadcast_str_list(
            padded, src=0, rank=0, world_size=self._world_size
        )

    def shutdown(self) -> None:
        _teardown_result_transport(self._shm_pool, rank=self._rank)
        if self._rank != 0:
            return
        _log.info(
            "WaveDispatcher.shutdown: broadcasting EXIT to %d workers",
            self._world_size - 1,
        )
        import torch.distributed as dist  # noqa: WPS433

        cmd_t = encode_wave_command(
            WAVE_CMD_EXIT, [], self._world_size, device=self._device
        )
        dist.broadcast(cmd_t, src=0)

    # ------------------------------------------------------------------
    # Worker path (ranks 1..N-1).
    # ------------------------------------------------------------------
    def run_worker_loop(self) -> None:
        if self._rank == 0:
            return
        self._ensure_transport()
        _log.info("WaveDispatcher: rank %d entering worker loop", self._rank)
        import torch  # noqa: WPS433
        import torch.distributed as dist  # noqa: WPS433

        cmd_tensor_size = 2 + self._world_size
        while True:
            cmd_t = torch.zeros(
                cmd_tensor_size, dtype=torch.long, device=self._device
            )
            dist.broadcast(cmd_t, src=0)
            cmd, wave_idx = decode_wave_command(cmd_t)

            if cmd == WAVE_CMD_EXIT:
                _log.info(
                    "WaveDispatcher: rank %d received EXIT", self._rank
                )
                _teardown_result_transport(self._shm_pool, rank=self._rank)
                return
            if cmd != WAVE_CMD_RUN:
                raise RuntimeError(
                    f"WaveDispatcher rank {self._rank}: unknown wave cmd {cmd}"
                )

            wave_pmts = broadcast_str_list(
                None, src=0, rank=self._rank, world_size=self._world_size
            )
            n = len(wave_idx)
            # Active iff this rank's slot is within the active prefix of
            # the wave. ``n`` is the only gate – see the matching comment
            # in ``generate``.
            if self._rank < n:
                local: "Result | WorkerFailure | None" = _run_local_slot_timed(
                    self._backend,
                    idx=wave_idx[self._rank],
                    prompt=wave_pmts[self._rank],
                    rank=self._rank,
                )
            else:
                local = None
            _gather_result_timed(
                local,
                world_size=self._world_size,
                rank=self._rank,
                sample_index=wave_idx[self._rank] if self._rank < n else -1,
            )

    # ------------------------------------------------------------------
    # Active slot execution. The caller is responsible for only invoking
    # the timed helpers on active slots; the active/inactive split is
    # determined by the wave's ``n`` (active-count) field, never by
    # sniffing the slot index for a sentinel value (warmup uses negative
    # sample indices that would collide with ``WAVE_INACTIVE_INDEX``).
    # ------------------------------------------------------------------
    def _run_local_slot(
        self, *, idx: int, prompt: str
    ) -> "Result | WorkerFailure":
        return _run_local_slot_timed(
            self._backend,
            idx=idx,
            prompt=prompt,
            rank=self._rank,
        )


# ----------------------------------------------------------------------
# Async data-parallel (Offline – opt-in via parallelism.dispatch=async).
# ----------------------------------------------------------------------


class _SelfWorkerShutdown:
    """Sentinel pushed into the rank-0 self-worker inbox to ask it to exit."""


class AsyncDPDispatcher(Dispatcher):
    """Pull-style DP dispatcher: rank 0 hands the next prompt to whichever
    rank just returned a result.

    Selected by ``parallelism.mode == 'data_parallel'`` together with
    ``parallelism.dispatch == 'async'``. Designed for workloads where
    the per-rank ``run_unit`` latency varies wave-to-wave (e.g.
    prompt-dependent attention reorder costs) so the
    :class:`WaveDispatcher` straggler tax (``wave_latency = max
    rank_latency``) eats more throughput than the cost of an
    asynchronous schedule.

    Wire model
    ----------
    A dedicated Gloo subgroup (``ctrl_pg``) is used for ALL of the
    rank-0 ↔ worker traffic. The world process group is left alone, so
    the wave dispatcher and any backend-internal collectives keep
    running on the same hot world PG (typically NCCL). Choosing Gloo
    here lets rank 0 use ``recv_object_list(src=None)`` to consume from
    *any* worker (NCCL does not support ANY-source recv), which is the
    primitive that powers the pull-style schedule.

    Per work unit:

      1. Rank 0 picks an idle worker and sends a :class:`WorkUnit` to
         it via ``send_object_list`` on the ctrl PG.
      2. The worker runs ``backend.run_unit`` and sends a
         :class:`Result` (or :class:`WorkerFailure` on exception) back
         to rank 0.
      3. Rank 0 drains the next completion via
         :func:`recv_object_any_src` and yields the corresponding
         :class:`GeneratedVideo` to the SUT.

    Shutdown is a single :class:`Shutdown` object sent to each worker.

    Rank 0 also runs work
    ---------------------
    Just like :class:`WaveDispatcher`, rank 0 contributes its GPU to
    the throughput pool. We do this with a dedicated *self-worker*
    thread driven by an in-process queue:

      * The dispatch loop pushes :class:`WorkUnit`-like jobs onto the
        rank-0 inbox.
      * The self-worker thread pops one, runs ``backend.run_unit``, and
        pushes the result onto a shared completion queue.
      * A second thread (the *reader*) blocks on
        :func:`recv_object_any_src` and pushes incoming Gloo results
        onto the same completion queue.

    The dispatch loop blocks on the completion queue, multiplexing
    the local + remote streams without polling.

    Determinism
    -----------
    ``WanBackend.run_unit`` seeds ``torch.Generator`` from the per-call
    ``input_args["seed"]`` and uses the rank-independent fixed latent
    loaded at backend setup, so the produced ``frames_bytes`` for a
    given ``(prompt, seed, fixed_latent)`` is identical regardless of
    which rank ran it. Async dispatch is bit-identical with wave
    dispatch for accuracy mode.
    """

    # Default join timeout for the self-worker thread on shutdown.
    # 30s is comfortably longer than a real ``run_unit`` call (~100s)
    # would take, but we expect shutdown to land while the thread is
    # idle (parked on its inbox), so this is just a backstop.
    _SELF_WORKER_JOIN_TIMEOUT_S = 30.0

    def __init__(
        self,
        backend: "WanBackend",
        *,
        rank: int,
        world_size: int,
        device: str | None = None,
    ) -> None:
        if world_size < 2:
            raise ValueError(
                f"AsyncDPDispatcher requires world_size>=2, got {world_size}; "
                f"the factory should have selected SingleProcessDispatcher instead."
            )
        self._backend = backend
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self.info = DispatcherInfo(
            name="async-dp", rank=rank, world_size=world_size
        )

        # Lazily created on the first ``generate`` / ``run_worker_loop``
        # call so cheap construction (e.g. for selection unit tests) does
        # not require ``torch.distributed`` to be initialised.
        self._ctrl_pg: "torch_dist.ProcessGroup | None" = None
        self._shm_pool: ShmResultPool | None = None
        self._transport_ready = False
        # Rank-0-only plumbing.
        self._self_worker_thread: threading.Thread | None = None
        self._self_worker_inbox: "queue.Queue[WorkUnit | _SelfWorkerShutdown] | None" = None
        self._reader_thread: threading.Thread | None = None
        self._results_queue: "queue.Queue[tuple[int, Result | WorkerFailure]] | None" = None
        self._stopped = False

    # ------------------------------------------------------------------
    # Warmup. Multiplier == world_size so a single ``num_prompts=1``
    # warmup call dispatches exactly ``world_size`` synthetic prompts;
    # because all workers start idle the first ``world_size`` prompts
    # naturally fan out one-to-one, giving every rank a warmup pass.
    # ------------------------------------------------------------------
    def _warmup_multiplier(self) -> int:
        return self._world_size

    def is_response_owner(self) -> bool:
        return self._rank == 0

    # ------------------------------------------------------------------
    # Lazy bringup of the Gloo subgroup + rank-0 helper threads.
    # ------------------------------------------------------------------
    def _ensure_started(self) -> None:
        if self._ctrl_pg is not None:
            return

        import torch.distributed as dist  # noqa: WPS433

        # ``new_group`` is collective: every rank must call it, in the
        # same order, with the same ``ranks`` list. ``build_dispatcher``
        # is called on every rank (via ``_run_worker`` on workers and
        # the runner's main path on rank 0), so this contract holds.
        self._ctrl_pg = dist.new_group(
            ranks=list(range(self._world_size)),
            backend="gloo",
        )
        self._shm_pool = _init_result_transport(
            self._backend,
            rank=self._rank,
            world_size=self._world_size,
            group=self._ctrl_pg,
        )
        self._transport_ready = True
        _log.info(
            "AsyncDPDispatcher: rank %d created Gloo ctrl subgroup "
            "(world_size=%d)",
            self._rank, self._world_size,
        )

        if self._rank == 0:
            self._results_queue = queue.Queue()
            self._self_worker_inbox = queue.Queue()
            self._self_worker_thread = threading.Thread(
                target=self._self_worker_loop,
                name="async-dp-self-worker",
                daemon=True,
            )
            self._self_worker_thread.start()
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name="async-dp-reader",
                daemon=True,
            )
            self._reader_thread.start()

    # ------------------------------------------------------------------
    # Rank-0 helper threads.
    # ------------------------------------------------------------------
    def _self_worker_loop(self) -> None:
        """Run rank-0's own ``backend.run_unit`` calls off the dispatch
        thread. Pulls jobs from the inbox, pushes outcomes onto the
        shared completion queue.
        """
        assert self._self_worker_inbox is not None
        assert self._results_queue is not None
        while True:
            msg = self._self_worker_inbox.get()
            if isinstance(msg, _SelfWorkerShutdown):
                return
            unit = msg
            payload = _run_local_slot_timed(
                self._backend,
                idx=int(unit.sample_index),
                prompt=unit.prompt,
                rank=0,
            )
            self._results_queue.put((0, payload))

    def _reader_loop(self) -> None:
        """Block on the ctrl PG draining results from any worker rank.

        Exits when ``recv_object_any_src`` raises (e.g. because the
        Gloo subgroup was destroyed on shutdown). The thread is a
        daemon, so a stray exit at process teardown is harmless.
        """
        assert self._results_queue is not None
        while True:
            try:
                t_wire = time.perf_counter()
                src, obj = recv_object_any_src(group=self._ctrl_pg)
                elapsed = time.perf_counter() - t_wire
            except Exception:  # noqa: BLE001 – PG destroy / process exit
                return
            if isinstance(obj, Result):
                get_collector().record(
                    PHASE_WIRE_TRANSFER,
                    elapsed,
                    sample_index=int(obj.sample_index),
                    rank=int(src),
                    nbytes=len(obj.frames_bytes),
                )
                if use_shm_for_results() and int(src) != 0:
                    release_worker_slot(
                        worker_rank=int(src),
                        slot_id=0,
                        dst=int(src),
                        group=self._ctrl_pg,
                    )
            self._results_queue.put((src, obj))

    # ------------------------------------------------------------------
    # Rank-0 driver.
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: Sequence[str],
        indices: Sequence[int],
    ) -> Iterator["GeneratedVideo"]:
        if self._rank != 0:
            raise RuntimeError("AsyncDPDispatcher.generate is rank-0 only")
        if len(prompts) != len(indices):
            raise ValueError(
                f"len(prompts)={len(prompts)} != len(indices)={len(indices)}"
            )

        n_total = len(prompts)
        if n_total == 0:
            return

        self._ensure_started()
        assert self._results_queue is not None
        assert self._self_worker_inbox is not None

        _log.info(
            "AsyncDPDispatcher.generate: n=%d world_size=%d starting",
            n_total, self._world_size,
        )
        t_batch = time.perf_counter()

        pending: deque[tuple[str, int]] = deque(
            (str(p), int(i)) for p, i in zip(prompts, indices)
        )
        # ``in_flight[r]`` -> sample_index currently being processed by
        # rank ``r``. Used both to gate redispatch (a rank is idle iff
        # not in this dict) and to validate the sample_index that comes
        # back from the worker.
        in_flight: dict[int, int] = {}
        # ``dispatch_time[r]`` -> ``perf_counter`` reading captured the
        # moment we handed rank ``r`` its current unit. Used solely to
        # compute the per-sample dispatch->complete latency we log
        # below; this is the *real* per-rank wall time (model run +
        # encode + ctrl-PG round-trip), comparable to
        # :class:`WaveDispatcher`'s ``wave_latency``.
        dispatch_time: dict[int, float] = {}
        completed = 0

        try:
            while pending or in_flight:
                # 1. Dispatch to every idle rank that still has work
                #    waiting. This keeps the GPU pool saturated on
                #    every loop iteration, including after a completion
                #    frees up a worker.
                self._dispatch_idle(pending, in_flight, dispatch_time)

                # 2. Block for the next completion. The reader thread +
                #    self-worker thread feed this queue, so this single
                #    ``get`` multiplexes local + remote completions.
                src, payload = self._results_queue.get()

                if isinstance(payload, WorkerFailure):
                    raise RuntimeError(
                        f"AsyncDPDispatcher: rank {payload.rank} failed on "
                        f"sample {payload.sample_index}: {payload.error_repr}"
                    )
                if not isinstance(payload, Result):
                    raise RuntimeError(
                        f"AsyncDPDispatcher: rank {src} returned unexpected "
                        f"payload {type(payload).__name__}"
                    )

                expected = in_flight.pop(src, None)
                if expected is None:
                    raise RuntimeError(
                        f"AsyncDPDispatcher: got Result from rank {src} but "
                        f"that rank was not marked in-flight (sample_index="
                        f"{payload.sample_index})"
                    )
                if int(payload.sample_index) != expected:
                    raise RuntimeError(
                        f"AsyncDPDispatcher: rank {src} returned sample_index="
                        f"{payload.sample_index} but expected {expected}"
                    )

                t_dispatch = dispatch_time.pop(src, None)
                latency = (
                    time.perf_counter() - t_dispatch
                    if t_dispatch is not None
                    else float("nan")
                )
                completed += 1
                _log.info(
                    "AsyncDPDispatcher.complete sample=%d rank=%d (%d/%d) "
                    "latency=%.3fs",
                    int(payload.sample_index), src, completed, n_total,
                    latency,
                )
                yield _generated_from_result(payload)
        finally:
            _log_batch_throughput(
                "AsyncDPDispatcher", completed, n_total, t_batch
            )

    def _dispatch_idle(
        self,
        pending: deque,
        in_flight: dict[int, int],
        dispatch_time: dict[int, float],
    ) -> None:
        """Hand one pending prompt to each currently-idle rank."""
        assert self._self_worker_inbox is not None
        for r in range(self._world_size):
            if not pending:
                return
            if r in in_flight:
                continue
            prompt, idx = pending.popleft()
            unit = self._backend.build_work_unit(
                prompt=prompt, sample_index=int(idx)
            )
            # Stamp the dispatch time *before* the (potentially
            # blocking) ctrl-PG send so the ``latency`` we log on
            # completion is an upper-bound on the dispatch->run->return
            # round trip rather than under-counting the wire time.
            dispatch_time[r] = time.perf_counter()
            if r == 0:
                self._self_worker_inbox.put(unit)
            else:
                send_object_pt2pt(unit, dst=r, group=self._ctrl_pg)
            in_flight[r] = int(idx)

    # ------------------------------------------------------------------
    # Worker path (ranks 1..N-1).
    # ------------------------------------------------------------------
    def run_worker_loop(self) -> None:
        if self._rank == 0:
            return
        self._ensure_started()
        _log.info(
            "AsyncDPDispatcher: rank %d entering worker loop", self._rank
        )
        while True:
            src, msg = recv_object_any_src(group=self._ctrl_pg)
            if src != 0:
                raise RuntimeError(
                    f"AsyncDPDispatcher rank {self._rank}: unexpected message "
                    f"from rank {src} (only rank 0 should be sending)"
                )
            if isinstance(msg, Shutdown):
                _log.info(
                    "AsyncDPDispatcher: rank %d received Shutdown",
                    self._rank,
                )
                _teardown_result_transport(self._shm_pool, rank=self._rank)
                return
            if not isinstance(msg, WorkUnit):
                raise RuntimeError(
                    f"AsyncDPDispatcher rank {self._rank}: unexpected payload "
                    f"{type(msg).__name__} from rank 0"
                )
            payload = _run_local_slot_timed(
                self._backend,
                idx=int(msg.sample_index),
                prompt=msg.prompt,
                rank=self._rank,
            )
            _send_result_timed(
                payload,
                dst=0,
                group=self._ctrl_pg,
                rank=self._rank,
                sample_index=int(msg.sample_index),
            )

    # ------------------------------------------------------------------
    # Shutdown. Idempotent.
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if self._rank != 0:
            return
        if self._stopped:
            return
        self._stopped = True
        if self._ctrl_pg is None:
            # ``generate`` never ran; nothing to tear down.
            return

        if self._shm_pool is not None:
            self._shm_pool.close_foreign_attachments()

        _log.info(
            "AsyncDPDispatcher.shutdown: sending Shutdown to %d workers",
            self._world_size - 1,
        )
        for r in range(1, self._world_size):
            try:
                send_object_pt2pt(Shutdown(), dst=r, group=self._ctrl_pg)
            except Exception:  # noqa: BLE001 – best-effort tear-down
                _log.exception(
                    "AsyncDPDispatcher.shutdown: send to rank %d failed", r
                )

        if self._self_worker_inbox is not None:
            self._self_worker_inbox.put(_SelfWorkerShutdown())
        if self._self_worker_thread is not None:
            self._self_worker_thread.join(
                timeout=self._SELF_WORKER_JOIN_TIMEOUT_S
            )

        _teardown_result_transport(self._shm_pool, rank=self._rank)

        # The reader thread is a daemon parked on ``recv_object_list``;
        # we leave it to be cleaned up when the process exits or when
        # the underlying PG is destroyed by the runner. Trying to
        # destroy it from here races with in-flight messages from the
        # workers' final ``send`` (during their natural exit) and is
        # not worth the complexity for a once-per-test teardown.


# ----------------------------------------------------------------------
# Factory.
# ----------------------------------------------------------------------


def build_dispatcher(
    backend: "Backend",
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Dispatcher:
    """Pick the right dispatcher for ``backend`` and the current topology.

    Selection rules:
        * Mock backend with ``world_size == 1`` -> :class:`SingleProcessDispatcher`.
        * Mock backend with ``world_size > 1`` and ``config.mock_dispatch``
          set -> :class:`WaveDispatcher` or :class:`AsyncDPDispatcher`.
        * Wan backend -> :class:`UlyssesDispatcher` or
          :class:`WaveDispatcher`, picked from
          ``backend.backend_config.parallelism.mode``.
    """
    if backend.name == "mock":
        if world_size == 1:
            return SingleProcessDispatcher(backend)
        dispatch = backend.config.mock_dispatch
        if dispatch is None:
            _log.warning(
                "Mock backend with world_size=%d but mock_dispatch unset; "
                "falling back to SingleProcessDispatcher on rank 0 only",
                world_size,
            )
            return SingleProcessDispatcher(backend)
        if dispatch == "wave":
            return WaveDispatcher(
                backend, rank=rank, world_size=world_size, device="cpu"
            )
        if dispatch == "async":
            return AsyncDPDispatcher(
                backend, rank=rank, world_size=world_size, device="cpu"
            )
        raise ValueError(
            f"Unknown mock_dispatch {dispatch!r}; expected 'wave' or 'async'"
        )

    if world_size == 1:
        return SingleProcessDispatcher(backend)

    # Wan backend specific: introspect the parallelism mode.
    from .backends.wan22 import WanBackend  # local import to avoid xfuser at module load

    if not isinstance(backend, WanBackend):
        raise TypeError(
            f"build_dispatcher: world_size>1 with non-Wan backend {backend.name!r} "
            f"is not supported"
        )
    bcfg = backend.backend_config
    mode = bcfg.parallelism.mode
    device = backend.distributed_device

    if mode == "ulysses":
        return UlyssesDispatcher(
            backend, rank=rank, world_size=world_size, device=device
        )
    if mode == "data_parallel":
        dispatch = bcfg.parallelism.dispatch
        if dispatch == "wave":
            return WaveDispatcher(
                backend, rank=rank, world_size=world_size, device=device
            )
        if dispatch == "async":
            return AsyncDPDispatcher(
                backend, rank=rank, world_size=world_size, device=device
            )
        raise ValueError(
            f"Unknown parallelism.dispatch {dispatch!r}; "
            f"expected 'wave' or 'async'"
        )
    raise ValueError(f"Unknown parallelism mode {mode!r}")
