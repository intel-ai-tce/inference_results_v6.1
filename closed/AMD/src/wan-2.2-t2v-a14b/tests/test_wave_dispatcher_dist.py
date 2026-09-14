"""End-to-end tests for :class:`WaveDispatcher` against real ``torch.distributed``.

These tests spawn ``world_size`` Linux child processes (``fork`` start
method), rendezvous them over a Gloo TCP process group, and drive the
dispatcher through a scripted set of waves. They are the only place in
the repo that exercises actual ``dist.broadcast`` / ``dist.gather_object``
traffic, and they pin the wave protocol contract:

  * Full waves round-trip every sample.
  * Short last waves work (n < world_size).
  * EXIT broadcast unwinds all worker loops cleanly.
  * Warmup negative-index waves are indistinguishable from real waves.
  * A backend exception on rank K becomes a fail-fast ``RuntimeError`` on
    rank 0 (the rest of the gather still completes, so workers do not
    hang).

The tests skip cleanly when ``torch`` or the Gloo backend are not
installed (e.g. on CPU dev hosts without the AMD pytorch-xdit image
underneath them).
"""

from __future__ import annotations

import json
import os
import socket
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
from wan_harness.dispatcher import WaveDispatcher  # noqa: E402
from wan_harness.wire import WorkUnit  # noqa: E402

# Use fork so child processes inherit the parent's imports + sys.path
# (in particular the tests/conftest.py path bootstrap). Spawn would
# require re-importing wan_harness from sys.path which is brittle under
# pytest discovery.
_CTX = torch_mp.get_context("fork")

# Test scenarios should never take more than a few seconds on CPU; a
# generous timeout still catches a hung gather without trapping CI.
_PROCESS_TIMEOUT_S = 60.0


# ----------------------------------------------------------------------
# Helpers shared across scenarios.
# ----------------------------------------------------------------------


def _free_port() -> int:
    """Pick an unused localhost TCP port for the Gloo rendezvous."""
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
    """Spawn ``world_size`` fork processes targeting ``target(rank, *args)``.

    Returns the per-rank outcome dicts the children wrote out.
    """
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
    return _read_outcomes(Path(args[-1]), world_size)  # outcome_dir is last arg


# ----------------------------------------------------------------------
# In-process FakeBackend (importable from forked children).
# ----------------------------------------------------------------------


class _FakeBackend:
    """Deterministic minimal backend.

    ``run_unit`` produces a tiny "video" whose first byte encodes
    ``(sample_index + rank * 1000)`` so the test can prove a given slot
    was actually executed by its intended rank.
    """

    name = "fake-wan22"

    @property
    def config(self) -> HarnessConfig:
        return HarnessConfig(height=8, width=8, num_frames=1, result_transport="shm")

    def __init__(
        self,
        *,
        fail_on_rank: int | None = None,
        fail_on_sample: int | None = None,
    ) -> None:
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
        # 1 frame, 4x4 RGB. The first byte is rank+sample-keyed so the
        # test can verify slot routing.
        h, w, t = 4, 4, 1
        marker = (int(unit.sample_index) + my_rank * 1000) & 0xFFFF
        first = (marker >> 8) & 0xFF
        body = (marker & 0xFF).to_bytes(1, "little") * (t * h * w * 3 - 1)
        frames = first.to_bytes(1, "little") + body
        return GeneratedVideo(
            sample_index=int(unit.sample_index),
            frames_bytes=frames,
            frame_count=t,
            height=h,
            width=w,
            mp4_bytes=None,
        )


# ----------------------------------------------------------------------
# Per-rank entry points (must be module-level for multiprocessing).
# ----------------------------------------------------------------------


def _entry_normal(
    rank: int,
    world_size: int,
    port: int,
    prompts: list[str],
    indices: list[int],
    outcome_dir: str,
) -> None:
    """Driver: rank 0 runs ``generate``; everyone else runs the worker loop."""
    outcome: dict[str, Any] = {"rank": rank, "ok": False}
    try:
        _setup_dist(rank, world_size, port)
        backend = _FakeBackend()
        disp = WaveDispatcher(
            backend, rank=rank, world_size=world_size, device="cpu"
        )
        if rank == 0:
            try:
                results = list(disp.generate(prompts, indices))
                outcome["results"] = [
                    {
                        "sample_index": r.sample_index,
                        "first_byte": r.frames_bytes[0],
                        "frame_count": r.frame_count,
                        "height": r.height,
                        "width": r.width,
                    }
                    for r in results
                ]
            finally:
                disp.shutdown()
        else:
            disp.run_worker_loop()
        outcome["ok"] = True
    except Exception as exc:  # noqa: BLE001 – surface everything to the parent
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
    """Same as ``_entry_normal`` but the backend on ``fail_on_rank`` raises.

    Rank 0 must propagate that as a :class:`RuntimeError` instead of
    hanging the gather.
    """
    outcome: dict[str, Any] = {"rank": rank, "ok": False}
    try:
        _setup_dist(rank, world_size, port)
        backend = _FakeBackend(fail_on_rank=fail_on_rank)
        disp = WaveDispatcher(
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
    """Tiny inline 'assert raises RuntimeError' helper that records the
    expected failure in the per-rank outcome instead of letting it abort
    the child process. The parent verifies the outcome's recorded error.

    On no-exception: ``expected_failure`` stays ``None`` and the outcome
    is reported as such so the parent assertion can flag a missed
    failure. On a non-RuntimeError exception: re-raised (child exits
    non-zero, the parent flags that). On a RuntimeError: caught and
    recorded.
    """

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


def test_wave_full_wave(tmp_path: Path) -> None:
    """world_size == n. Every rank produces one Result; rank 0 sees them in order."""
    world_size = 4
    port = _free_port()
    outcome_dir = tmp_path / "out"
    outcome_dir.mkdir()
    prompts = [f"p-{i}" for i in range(world_size)]
    indices = [100, 101, 102, 103]

    outcomes = _run_world(
        _entry_normal,
        world_size=world_size,
        args=(world_size, port, prompts, indices, str(outcome_dir)),
    )

    for rank, out in enumerate(outcomes):
        assert out["ok"], (
            f"rank {rank} failed: {out.get('error')}\n{out.get('traceback', '')}"
        )

    rank0 = outcomes[0]
    assert [r["sample_index"] for r in rank0["results"]] == indices
    for slot, r in enumerate(rank0["results"]):
        marker = (indices[slot] + slot * 1000) & 0xFFFF
        expected_first = (marker >> 8) & 0xFF
        assert r["first_byte"] == expected_first, (
            f"slot {slot}: expected first_byte={expected_first}, got {r['first_byte']}; "
            f"this is the per-rank routing assertion (rank K must have produced slot K's Result)"
        )


def test_wave_short_last_wave(tmp_path: Path) -> None:
    """n_total not divisible by world_size: 7 samples across 4 ranks → 2 waves, last with 3 active slots."""
    world_size = 4
    port = _free_port()
    outcome_dir = tmp_path / "out"
    outcome_dir.mkdir()
    n_total = 7
    prompts = [f"p-{i}" for i in range(n_total)]
    indices = list(range(200, 200 + n_total))

    outcomes = _run_world(
        _entry_normal,
        world_size=world_size,
        args=(world_size, port, prompts, indices, str(outcome_dir)),
    )

    for rank, out in enumerate(outcomes):
        assert out["ok"], (
            f"rank {rank} failed: {out.get('error')}\n{out.get('traceback', '')}"
        )

    rank0 = outcomes[0]
    assert [r["sample_index"] for r in rank0["results"]] == indices

    # Slot routing per wave: wave 0 covers slots 0..3, wave 1 covers slots 0..2 only.
    wave0 = rank0["results"][:4]
    wave1 = rank0["results"][4:]
    for slot, r in enumerate(wave0):
        marker = (indices[slot] + slot * 1000) & 0xFFFF
        assert r["first_byte"] == (marker >> 8) & 0xFF
    for slot, r in enumerate(wave1):
        # In wave 1 the slot index is the rank that produced it (0..2).
        marker = (indices[4 + slot] + slot * 1000) & 0xFFFF
        assert r["first_byte"] == (marker >> 8) & 0xFF


def test_wave_two_back_to_back_waves(tmp_path: Path) -> None:
    """Two full waves in one generate() call. Verifies the worker loop's
    wave→wave transition doesn't drop or reorder samples."""
    world_size = 4
    port = _free_port()
    outcome_dir = tmp_path / "out"
    outcome_dir.mkdir()
    indices = list(range(300, 308))  # 8 samples → exactly 2 waves of 4
    prompts = [f"p-{i}" for i in indices]

    outcomes = _run_world(
        _entry_normal,
        world_size=world_size,
        args=(world_size, port, prompts, indices, str(outcome_dir)),
    )
    for rank, out in enumerate(outcomes):
        assert out["ok"], (
            f"rank {rank} failed: {out.get('error')}\n{out.get('traceback', '')}"
        )
    assert [r["sample_index"] for r in outcomes[0]["results"]] == indices


def test_wave_warmup_with_negative_indices(tmp_path: Path) -> None:
    """Warmup uses negative sample indices. The wave protocol must not
    confuse those with the WAVE_INACTIVE_INDEX sentinel because the
    number of active slots is conveyed by ``n``, not by sniffing for
    -1 in the slot ints.
    """
    world_size = 4
    port = _free_port()
    outcome_dir = tmp_path / "out"
    outcome_dir.mkdir()
    # The base-class warmup dispatches indices = range(-total, 0).
    # For a full wave with world_size==4: indices [-4, -3, -2, -1].
    prompts = ["warmup-prompt"] * 4
    indices = [-4, -3, -2, -1]

    outcomes = _run_world(
        _entry_normal,
        world_size=world_size,
        args=(world_size, port, prompts, indices, str(outcome_dir)),
    )

    for rank, out in enumerate(outcomes):
        assert out["ok"], (
            f"rank {rank} failed: {out.get('error')}\n{out.get('traceback', '')}"
        )
    assert [r["sample_index"] for r in outcomes[0]["results"]] == indices


def test_wave_worker_failure_propagates_to_rank0(tmp_path: Path) -> None:
    """One rank's backend raises mid-wave. Rank 0 must see RuntimeError;
    the other ranks must still complete the wave's gather and accept
    the subsequent EXIT instead of hanging on the collective.
    """
    world_size = 4
    port = _free_port()
    outcome_dir = tmp_path / "out"
    outcome_dir.mkdir()
    fail_on_rank = 2
    prompts = [f"p-{i}" for i in range(world_size)]
    indices = [400, 401, 402, 403]

    outcomes = _run_world(
        _entry_with_failure,
        world_size=world_size,
        args=(world_size, port, prompts, indices, fail_on_rank, str(outcome_dir)),
    )

    # All processes exit cleanly.
    for rank, out in enumerate(outcomes):
        assert out["ok"], (
            f"rank {rank} did not finish cleanly: {out.get('error')}\n"
            f"{out.get('traceback', '')}"
        )

    rank0 = outcomes[0]
    assert rank0.get("expected_failure") is not None, (
        "rank 0 should have observed a RuntimeError from the failing wave"
    )
    msg = rank0["expected_failure"]
    assert "rank 2" in msg, msg
    assert "402" in msg, msg
