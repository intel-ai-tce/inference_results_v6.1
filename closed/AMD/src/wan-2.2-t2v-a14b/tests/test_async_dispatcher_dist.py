"""End-to-end tests for :class:`AsyncDPDispatcher` against real ``torch.distributed``.

Mirrors :mod:`tests.test_wave_dispatcher_dist`: spawns ``world_size``
forked Linux child processes, rendezvouses them over a Gloo TCP
process group, and drives the dispatcher through scripted scenarios.
The async dispatcher creates its own Gloo subgroup *inside* the
already-initialised world PG, so each child must enter
``init_process_group`` first, then construct the dispatcher.

The contract pinned here is:

  * Pull-style schedule: every issued sample comes back on rank 0
    exactly once even when work is generated faster than any single
    rank can process it.
  * A genuinely slow rank does NOT delay the rest: while a slow rank
    is running its single unit, the remaining ranks must drain the
    queue concurrently. ``test_async_dispatches_to_idle_rank_first``
    asserts the resulting per-rank work histogram.
  * Backend failures surface as :class:`RuntimeError` on rank 0 with a
    fail-fast message that names the offending rank + sample.
  * Warmup (fed through the same ``generate`` path as real prompts)
    visits every rank at least once.

The tests skip cleanly when ``torch`` or the Gloo backend are not
installed.
"""

from __future__ import annotations

import json
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

if not dist.is_available():  # pragma: no cover - env dependent
    pytest.skip("torch.distributed not available", allow_module_level=True)
if not dist.is_gloo_available():  # pragma: no cover - env dependent
    pytest.skip("Gloo backend not available", allow_module_level=True)

import torch.multiprocessing as torch_mp  # noqa: E402

from wan_harness.backends.base import GeneratedVideo  # noqa: E402
from wan_harness.config import HarnessConfig  # noqa: E402
from wan_harness.dispatcher import AsyncDPDispatcher  # noqa: E402
from wan_harness.wire import WorkUnit  # noqa: E402

_CTX = torch_mp.get_context("fork")
_PROCESS_TIMEOUT_S = 90.0


# ----------------------------------------------------------------------
# Helpers (mirrored from tests/test_wave_dispatcher_dist.py).
# ----------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _setup_dist(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _teardown_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _record_outcome(outcome_dir: str, rank: int, body: dict[str, Any]) -> None:
    Path(outcome_dir, f"rank-{rank}.json").write_text(
        json.dumps(body, default=str), encoding="utf-8"
    )


def _read_outcomes(outcome_dir: Path, world_size: int) -> list[dict[str, Any]]:
    return [
        json.loads((outcome_dir / f"rank-{r}.json").read_text(encoding="utf-8"))
        for r in range(world_size)
    ]


def _run_world(target, *, world_size: int, args: tuple) -> list[dict[str, Any]]:
    procs = [
        _CTX.Process(target=target, args=(rank, *args), daemon=False)
        for rank in range(world_size)
    ]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=_PROCESS_TIMEOUT_S)
        hung = [p for p in procs if p.is_alive()]
        if hung:
            for p in hung:
                p.terminate()
                p.join(timeout=5.0)
            raise AssertionError(
                f"{len(hung)} child process(es) hung beyond "
                f"{_PROCESS_TIMEOUT_S:.0f}s and were terminated"
            )
        nonzero = [p for p in procs if p.exitcode != 0]
        if nonzero:
            ranks = [i for i, p in enumerate(procs) if p.exitcode != 0]
            raise AssertionError(
                f"child process(es) exited with non-zero status; ranks={ranks!r}, "
                f"exit codes={[p.exitcode for p in procs]!r}"
            )
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
    return _read_outcomes(Path(args[-1]), world_size)


# ----------------------------------------------------------------------
# In-process FakeBackend.
# ----------------------------------------------------------------------


class _FakeBackend:
    """Deterministic minimal backend.

    ``run_unit`` produces a tiny "video" whose first byte is the
    *producer rank*, so rank 0 can decode which rank ran each sample
    just by looking at ``frames_bytes[0]``. Per-rank latency can be
    tuned via ``slow_rank`` / ``slow_seconds`` for the straggler-bypass
    test, and any rank can be forced to raise via ``fail_on_rank``.

    A uniform ``min_seconds`` floor latency can be applied to all
    ranks (independent of ``slow_rank``) to keep the test
    deterministic on hosts where rank 0's in-process self-worker would
    otherwise run away with the pending queue before the remote
    workers' Gloo round-trip even completes their first send.
    """

    name = "fake-wan22"

    @property
    def config(self) -> HarnessConfig:
        return HarnessConfig(height=8, width=8, num_frames=1, result_transport="shm")

    def __init__(
        self,
        *,
        slow_rank: int | None = None,
        slow_seconds: float = 0.0,
        min_seconds: float = 0.0,
        fail_on_rank: int | None = None,
        fail_on_sample: int | None = None,
    ) -> None:
        self._slow_rank = slow_rank
        self._slow_seconds = float(slow_seconds)
        self._min_seconds = float(min_seconds)
        self._fail_on_rank = fail_on_rank
        self._fail_on_sample = fail_on_sample

    def build_work_unit(self, *, prompt: str, sample_index: int) -> WorkUnit:
        return WorkUnit(
            sample_index=int(sample_index),
            prompt=prompt,
            input_args={"prompt": prompt},
        )

    def run_unit(self, unit: WorkUnit) -> GeneratedVideo:
        my_rank = int(os.environ.get("RANK", 0))
        if self._slow_rank is not None and my_rank == self._slow_rank:
            time.sleep(self._slow_seconds)
        elif self._min_seconds > 0:
            time.sleep(self._min_seconds)
        if (
            self._fail_on_rank is not None
            and my_rank == self._fail_on_rank
            and (
                self._fail_on_sample is None
                or int(unit.sample_index) == self._fail_on_sample
            )
        ):
            raise RuntimeError(
                f"_FakeBackend: forced failure on rank {my_rank} "
                f"sample {unit.sample_index}"
            )
        # 1 frame, 4x4 RGB. ``frames_bytes[0]`` carries the producer
        # rank (0..255) verbatim so the test can recover the per-rank
        # work histogram. The rest is filler.
        h, w, t = 4, 4, 1
        frames = bytes([my_rank & 0xFF]) + b"\x00" * (t * h * w * 3 - 1)
        return GeneratedVideo(
            sample_index=int(unit.sample_index),
            frames_bytes=frames,
            frame_count=t,
            height=h,
            width=w,
            mp4_bytes=None,
        )


# ----------------------------------------------------------------------
# Per-rank entry points.
# ----------------------------------------------------------------------


def _entry_normal(
    rank: int,
    world_size: int,
    port: int,
    prompts: list[str],
    indices: list[int],
    slow_rank: int | None,
    slow_seconds: float,
    min_seconds: float,
    outcome_dir: str,
) -> None:
    """Driver: rank 0 runs ``generate``; everyone else runs the worker loop."""
    outcome: dict[str, Any] = {"rank": rank, "ok": False}
    try:
        _setup_dist(rank, world_size, port)
        backend = _FakeBackend(
            slow_rank=slow_rank,
            slow_seconds=slow_seconds,
            min_seconds=min_seconds,
        )
        disp = AsyncDPDispatcher(
            backend, rank=rank, world_size=world_size, device="cpu"
        )
        if rank == 0:
            try:
                results = list(disp.generate(prompts, indices))
                outcome["results"] = [
                    {
                        "sample_index": r.sample_index,
                        # _FakeBackend stamps the producer rank in the
                        # leading frame byte; see _FakeBackend.run_unit.
                        "producer_rank": int(r.frames_bytes[0]),
                    }
                    for r in results
                ]
            finally:
                disp.shutdown()
        else:
            disp.run_worker_loop()
        outcome["ok"] = True
    except Exception as exc:  # noqa: BLE001 – surface to the parent
        outcome["error"] = repr(exc)
        outcome["traceback"] = traceback.format_exc()
    finally:
        _teardown_dist()
        _record_outcome(outcome_dir, rank, outcome)


def _entry_with_failure(
    rank: int,
    world_size: int,
    port: int,
    prompts: list[str],
    indices: list[int],
    fail_on_rank: int,
    outcome_dir: str,
) -> None:
    """Same as ``_entry_normal`` but the backend on ``fail_on_rank`` raises."""
    outcome: dict[str, Any] = {"rank": rank, "ok": False}
    try:
        _setup_dist(rank, world_size, port)
        backend = _FakeBackend(fail_on_rank=fail_on_rank)
        disp = AsyncDPDispatcher(
            backend, rank=rank, world_size=world_size, device="cpu"
        )
        if rank == 0:
            try:
                with _ExpectRuntimeError(outcome):
                    list(disp.generate(prompts, indices))
            finally:
                disp.shutdown()
        else:
            disp.run_worker_loop()
        outcome["ok"] = True
    except Exception as exc:  # noqa: BLE001
        outcome["error"] = repr(exc)
        outcome["traceback"] = traceback.format_exc()
    finally:
        _teardown_dist()
        _record_outcome(outcome_dir, rank, outcome)


class _ExpectRuntimeError:
    """Mirror of the helper in test_wave_dispatcher_dist.py."""

    def __init__(self, outcome: dict[str, Any]) -> None:
        self._outcome = outcome
        self._outcome.setdefault("expected_failure", None)

    def __enter__(self) -> "_ExpectRuntimeError":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            return False
        if not issubclass(exc_type, RuntimeError):
            return False
        self._outcome["expected_failure"] = repr(exc)
        return True


# ----------------------------------------------------------------------
# Tests.
# ----------------------------------------------------------------------


def test_async_full_run_returns_all_samples(tmp_path: Path) -> None:
    """world_size=4, n=12 samples. Every issued sample must come back
    exactly once. Order is not guaranteed (async schedule)."""
    world_size = 4
    port = _free_port()
    outcome_dir = tmp_path / "out"
    outcome_dir.mkdir()
    n_total = 12
    prompts = [f"p-{i}" for i in range(n_total)]
    indices = list(range(100, 100 + n_total))

    # Apply a small uniform floor latency so rank 0's in-process
    # self-worker doesn't run away with the queue before the remote
    # workers' Gloo round-trip even completes their first send. Without
    # this, the rank-0 self-worker (no socket, no pickle round-trip)
    # consistently outruns the others on a fast _FakeBackend and the
    # producer-coverage assertion below becomes timing-dependent.
    min_seconds = 0.05

    outcomes = _run_world(
        _entry_normal,
        world_size=world_size,
        args=(
            world_size, port, prompts, indices,
            None, 0.0, min_seconds, str(outcome_dir),
        ),
    )

    for rank, out in enumerate(outcomes):
        assert out["ok"], (
            f"rank {rank} failed: {out.get('error')}\n{out.get('traceback', '')}"
        )

    rank0 = outcomes[0]
    returned = sorted(int(r["sample_index"]) for r in rank0["results"])
    assert returned == sorted(indices), (
        f"every sample must come back exactly once; got {returned}"
    )

    # Every rank participated in producing results. With n_total=12,
    # W=4 and a uniform 50ms floor latency the work distributes evenly
    # enough that every rank is guaranteed to land at least one sample.
    producers = sorted({int(r["producer_rank"]) for r in rank0["results"]})
    assert producers == [0, 1, 2, 3], (
        f"every rank must have produced at least one result; got producers={producers}"
    )


def test_async_dispatches_to_idle_rank_first(tmp_path: Path) -> None:
    """Pull-style scheduler must NOT block the fast ranks behind the slow
    rank. With world_size=4 and one slow rank, while the slow rank
    processes its single unit, the other 3 ranks should drain the
    queue. We assert the resulting work histogram skews the slow rank
    down to ~1 unit.
    """
    world_size = 4
    port = _free_port()
    outcome_dir = tmp_path / "out"
    outcome_dir.mkdir()
    n_total = 16  # plenty of work to expose the schedule.
    prompts = [f"p-{i}" for i in range(n_total)]
    indices = list(range(200, 200 + n_total))
    slow_rank = 2
    # The slow rank sleeps slow_seconds; every other rank applies a
    # uniform min_seconds floor latency. The latter keeps rank 0's
    # in-process self-worker from outpacing the remote ranks' Gloo
    # round-trip and consuming the entire queue. The 30x ratio is
    # plenty to demonstrate that the slow rank does NOT block the
    # others (its expected count is 1; each fast rank should land ~5
    # samples on average).
    slow_seconds = 1.5
    min_seconds = 0.05

    outcomes = _run_world(
        _entry_normal,
        world_size=world_size,
        args=(
            world_size, port, prompts, indices,
            slow_rank, slow_seconds, min_seconds, str(outcome_dir),
        ),
    )

    for rank, out in enumerate(outcomes):
        assert out["ok"], (
            f"rank {rank} failed: {out.get('error')}\n{out.get('traceback', '')}"
        )

    rank0 = outcomes[0]
    # Coverage: every sample comes back exactly once.
    returned = sorted(int(r["sample_index"]) for r in rank0["results"])
    assert returned == sorted(indices)

    # Per-rank work counts. With slow_seconds >> per-fast-rank time, the
    # slow rank should have processed exactly 1 unit (the one that was
    # in flight when fast ranks drained the rest), and the fast ranks
    # should have processed >1 each.
    counts: dict[int, int] = {r: 0 for r in range(world_size)}
    for r in rank0["results"]:
        counts[int(r["producer_rank"])] += 1
    assert sum(counts.values()) == n_total
    assert counts[slow_rank] == 1, (
        f"slow rank should have processed exactly 1 unit (the one it picked up "
        f"before the fast ranks drained the queue), got {counts[slow_rank]}; "
        f"counts={counts}"
    )
    for r in range(world_size):
        if r == slow_rank:
            continue
        assert counts[r] > 1, (
            f"fast rank {r} should have processed >1 unit (proves it did not "
            f"sit idle behind the slow rank); got {counts[r]}; counts={counts}"
        )


def test_async_warmup_visits_every_rank(tmp_path: Path) -> None:
    """Warmup uses negative sample indices and dispatches
    ``world_size * num_prompts_per_rank`` units. With all workers idle
    at the start, the first ``world_size`` units fan out one-to-one,
    so every rank must produce at least one warmup completion.
    """
    world_size = 4
    port = _free_port()
    outcome_dir = tmp_path / "out"
    outcome_dir.mkdir()
    # Match what dispatcher.warmup() would issue for num_prompts_per_rank=1.
    indices = [-4, -3, -2, -1]
    prompts = ["warmup-prompt"] * world_size
    # n_total == world_size, so every rank gets exactly one unit on the
    # initial fanout regardless of rank-0 self-worker speed; no floor
    # latency needed for this test.

    outcomes = _run_world(
        _entry_normal,
        world_size=world_size,
        args=(
            world_size, port, prompts, indices,
            None, 0.0, 0.0, str(outcome_dir),
        ),
    )

    for rank, out in enumerate(outcomes):
        assert out["ok"], (
            f"rank {rank} failed: {out.get('error')}\n{out.get('traceback', '')}"
        )
    rank0 = outcomes[0]
    returned = sorted(int(r["sample_index"]) for r in rank0["results"])
    assert returned == sorted(indices)
    producers = sorted({int(r["producer_rank"]) for r in rank0["results"]})
    assert producers == [0, 1, 2, 3], (
        f"warmup must visit every rank; got producers={producers}"
    )


def test_async_worker_failure_propagates_to_rank0(tmp_path: Path) -> None:
    """One rank's backend raises mid-run. Rank 0 must observe a
    :class:`RuntimeError` naming the offending rank + sample, and all
    workers must shut down cleanly afterwards (no hung process)."""
    world_size = 4
    port = _free_port()
    outcome_dir = tmp_path / "out"
    outcome_dir.mkdir()
    fail_on_rank = 2
    n_total = 8
    prompts = [f"p-{i}" for i in range(n_total)]
    indices = list(range(400, 400 + n_total))

    outcomes = _run_world(
        _entry_with_failure,
        world_size=world_size,
        args=(
            world_size, port, prompts, indices, fail_on_rank, str(outcome_dir),
        ),
    )

    for rank, out in enumerate(outcomes):
        assert out["ok"], (
            f"rank {rank} did not finish cleanly: {out.get('error')}\n"
            f"{out.get('traceback', '')}"
        )

    rank0 = outcomes[0]
    assert rank0.get("expected_failure") is not None, (
        "rank 0 should have observed a RuntimeError from the failing worker"
    )
    msg = rank0["expected_failure"]
    assert "rank 2" in msg, msg
    # The first thing rank 2 picks up is index 400 + 2 = 402 (initial
    # one-to-one fanout); every later assignment goes to whichever rank
    # finished first. With deterministic timing in the test it should be
    # 402, but we just check that *some* sample number from the input
    # appears in the message.
    assert any(str(i) in msg for i in indices), msg
