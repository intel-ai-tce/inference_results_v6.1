"""Integration tests for SHM-backed Result transfer over Gloo."""

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

if not dist.is_available():  # pragma: no cover
    pytest.skip("torch.distributed not available", allow_module_level=True)
if not dist.is_gloo_available():  # pragma: no cover
    pytest.skip("Gloo backend not available", allow_module_level=True)

import torch.multiprocessing as torch_mp  # noqa: E402

from wan_harness.backends.base import GeneratedVideo  # noqa: E402
from wan_harness.config import HarnessConfig  # noqa: E402
from wan_harness.dispatcher import AsyncDPDispatcher  # noqa: E402
from wan_harness.wire import (  # noqa: E402
    Result,
    configure_result_transport,
    reset_result_transport,
    use_shm_for_results,
)
from wan_harness.wire import WorkUnit  # noqa: E402

_CTX = torch_mp.get_context("fork")
_PROCESS_TIMEOUT_S = 60.0


class _TinyBackend:
    name = "mock"

    @property
    def config(self) -> HarnessConfig:
        return HarnessConfig(height=8, width=8, num_frames=1, result_transport="shm")

    def build_work_unit(self, *, prompt: str, sample_index: int) -> WorkUnit:
        return WorkUnit(sample_index=int(sample_index), prompt=prompt, input_args={})

    def run_unit(self, unit: WorkUnit) -> GeneratedVideo:
        rank = int(os.environ.get("RANK", 0))
        frames = bytes([rank & 0xFF]) + b"\x00" * 47
        return GeneratedVideo(
            sample_index=int(unit.sample_index),
            frames_bytes=frames,
            frame_count=1,
            height=8,
            width=8,
            mp4_bytes=None,
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _setup_dist(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _teardown_dist() -> None:
    reset_result_transport()
    if dist.is_initialized():
        dist.destroy_process_group()


def _record_outcome(outcome_dir: str, rank: int, body: dict[str, Any]) -> None:
    Path(outcome_dir, f"rank-{rank}.json").write_text(
        json.dumps(body, default=str), encoding="utf-8"
    )


def _entry_shm_async(
    rank: int,
    world_size: int,
    port: int,
    outcome_dir: str,
) -> None:
    outcome: dict[str, Any] = {"rank": rank, "ok": False}
    try:
        _setup_dist(rank, world_size, port)
        backend = _TinyBackend()
        disp = AsyncDPDispatcher(
            backend, rank=rank, world_size=world_size, device="cpu"
        )
        if rank == 0:
            try:
                results = list(
                    disp.generate(["p0", "p1"], [10, 11])
                )
                outcome["used_shm"] = use_shm_for_results()
                outcome["results"] = [
                    {
                        "sample_index": r.sample_index,
                        "producer_rank": int(r.frames_bytes[0]),
                    }
                    for r in results
                ]
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


def test_async_shm_transport_two_rank_smoke(tmp_path: Path) -> None:
    outcome_dir = str(tmp_path / "outcomes")
    Path(outcome_dir).mkdir()
    port = _free_port()
    procs = [
        _CTX.Process(
            target=_entry_shm_async,
            args=(rank, 2, port, outcome_dir),
            daemon=False,
        )
        for rank in range(2)
    ]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=_PROCESS_TIMEOUT_S)
        assert all(p.exitcode == 0 for p in procs)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()

    outcomes = [
        json.loads((Path(outcome_dir) / f"rank-{r}.json").read_text())
        for r in range(2)
    ]
    rank0 = outcomes[0]
    assert rank0["ok"] is True
    assert rank0["used_shm"] is True
    assert {r["sample_index"] for r in rank0["results"]} == {10, 11}


def test_configure_result_transport_gloo_disables_shm() -> None:
    configure_result_transport(None, use_shm=False)
    assert use_shm_for_results() is False
    reset_result_transport()
