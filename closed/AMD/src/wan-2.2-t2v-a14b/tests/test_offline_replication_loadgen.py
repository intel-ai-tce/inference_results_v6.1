"""End-to-end test for LoadGen Offline replication of QSL indices.

In Offline, LoadGen draws ``samples_per_query`` ``(query_id, sample_index)``
pairs from the loaded QSL *with replacement* whenever ``samples_per_query``
exceeds the loaded sample count. The same ``sample_index`` therefore
appears multiple times in the issued batch with distinct ``query_id``
values. The SUT must call ``QuerySamplesComplete`` once per ``query_id``
or LoadGen logs ``error_runtime: "Attempted to complete a sample twice."``
and stalls in its post-test phase – the v6.1 baseline crash mode that
tripped the NCCL watchdog after 10 minutes of idle.

This test forces that exact LoadGen behavior with the Mock backend and
asserts the LoadGen detail log is clean.

The actual LoadGen ``StartTestWithLogSettings`` call runs in a child
process so a regression that causes the v6.1 post-test hang fails the
test in seconds rather than wedging the pytest runner.

Skipped automatically (via the ``loadgen`` marker registered in
``conftest.py``) when ``mlperf_loadgen`` isn't importable.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import pytest

from wan_harness.config import HarnessConfig

pytestmark = pytest.mark.loadgen

# Generous cap; the actual run is far under a second on a CPU laptop.
_RUN_TIMEOUT_S = 30.0


def _detail_records(out_dir: Path) -> list[dict]:
    """Parse ``mlperf_log_detail.txt`` MLLOG-prefixed JSON records."""
    text = (out_dir / "mlperf_log_detail.txt").read_text()
    records: list[dict] = []
    for line in text.splitlines():
        if not line.startswith(":::MLLOG "):
            continue
        try:
            records.append(json.loads(line[len(":::MLLOG "):]))
        except json.JSONDecodeError:
            continue
    return records


def _run_offline_in_child(cfg_kwargs: dict[str, Any], result_path: str) -> None:
    """Subprocess entry point: run one Offline test and dump the (qid, idx)
    list LoadGen issued plus the harness ``RunResult`` to ``result_path``.

    Runs in a fresh interpreter so a hang here can't wedge the pytest runner –
    the parent enforces a wallclock timeout via ``Process.join``.
    """
    from wan_harness import sut as sut_mod
    from wan_harness.config import HarnessConfig as _HarnessConfig
    from wan_harness.loadgen_runner import run

    issued_indices: list[int] = []
    issued_qids: list[int] = []
    orig_issue = sut_mod.WanSUT.issue_queries

    def spy(self, query_samples):  # type: ignore[no-untyped-def]
        samples = list(query_samples)
        issued_indices.extend(int(s.index) for s in samples)
        issued_qids.extend(int(s.id) for s in samples)
        return orig_issue(self, samples)

    sut_mod.WanSUT.issue_queries = spy

    # Re-hydrate Path objects after going through pickle.
    cfg_kwargs = {
        k: (Path(v) if k.endswith("_path") or k == "output_dir" else v)
        for k, v in cfg_kwargs.items()
    }
    cfg = _HarnessConfig(**cfg_kwargs)
    result = run(cfg, rank=0, world_size=1)

    Path(result_path).write_text(
        json.dumps({
            "issued_indices": issued_indices,
            "issued_qids": issued_qids,
            "issued": result.issued,
            "completed": result.completed,
            "output_dir": str(result.output_dir),
        })
    )


def test_offline_replication_does_not_double_complete(tmp_path: Path) -> None:
    """Offline run with ``min_query_count`` > ``total_sample_count`` forces
    LoadGen to draw the same QSL index multiple times (it samples with
    replacement from ``[0, total_sample_count)``). Verify:

      * The harness run completes within ``_RUN_TIMEOUT_S`` (the v6.1 bug
        manifested as a post-test hang of >10 minutes inside
        ``StartTestWithLogSettings``).
      * ``mlperf_log_summary.txt`` is non-empty (LoadGen finalised).
      * ``mlperf_log_detail.txt`` contains zero
        ``"Attempted to complete a sample twice."`` errors.
      * LoadGen actually issued duplicate ``sample_index`` values (so this
        test is genuinely exercising the replication path).
      * Every issued ``query_id`` was completed exactly once.
    """
    # Tiny prompts file so the QSL has only 4 unique indices.
    prompts_path = tmp_path / "tiny_prompts.txt"
    prompts_path.write_text(
        "\n".join(f"prompt-{i}" for i in range(4)) + "\n", encoding="utf-8"
    )
    out_dir = tmp_path / "Offline-performance-replication"
    cfg_kwargs = dict(
        backend="mock",
        scenario="Offline",
        mode="performance",
        height=8,
        width=16,
        num_frames=2,
        prompts_path=str(prompts_path),
        output_dir=str(out_dir),
        performance_sample_count=4,
        # Force samples_per_query >> total_sample_count so LoadGen draws
        # each of the 4 QSL indices roughly 8× with distinct query_ids.
        min_query_count=32,
        min_duration_ms=10,
    )

    result_path = tmp_path / "child_result.json"
    # Use ``spawn`` so the child does not inherit any pytest fixtures /
    # threads / loadgen state from the parent.
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_run_offline_in_child,
        args=(cfg_kwargs, str(result_path)),
        name="loadgen-offline-replication",
    )
    proc.start()
    proc.join(timeout=_RUN_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)
        pytest.fail(
            f"Offline LoadGen run did not return within {_RUN_TIMEOUT_S:.0f}s; "
            "this is the v6.1 baseline failure mode (LoadGen stalled in "
            "post-test waiting on un-completed query_ids because the SUT "
            "collapsed duplicate sample_index occurrences)."
        )
    assert proc.exitcode == 0, (
        f"child process exited with code {proc.exitcode}; "
        "the harness run failed before the assertion phase."
    )

    payload = json.loads(result_path.read_text())
    issued_indices: list[int] = payload["issued_indices"]
    issued_qids: list[int] = payload["issued_qids"]
    issued = payload["issued"]
    completed = payload["completed"]

    summary_path = out_dir / "mlperf_log_summary.txt"
    detail_path = out_dir / "mlperf_log_detail.txt"
    assert summary_path.exists() and summary_path.stat().st_size > 0, (
        f"LoadGen summary is empty at {summary_path}; LoadGen did not "
        "finalise the test."
    )
    assert detail_path.exists() and detail_path.stat().st_size > 0

    # Sanity: LoadGen really did replicate. Without this guard the test
    # would silently no-op if a future LoadGen stops drawing with
    # replacement.
    n_issued = len(issued_indices)
    n_unique_indices = len(set(issued_indices))
    n_unique_qids = len(set(issued_qids))
    assert n_unique_qids == n_issued, (
        "test invariant violated: LoadGen issued duplicate query_ids"
    )
    assert n_unique_indices < n_issued, (
        f"test misconfigured: LoadGen issued {n_issued} samples but they "
        f"resolve to {n_unique_indices} unique sample_indices; configure "
        f"samples_per_query > total_sample_count so the replication path "
        f"is actually exercised."
    )

    # Regression assertion: zero double-complete errors.
    records = _detail_records(out_dir)
    double_complete = [
        r for r in records
        if r.get("key") == "error_runtime"
        and "complete a sample twice" in str(r.get("value", ""))
    ]
    assert not double_complete, (
        f"LoadGen reported {len(double_complete)} "
        f"'Attempted to complete a sample twice.' errors; "
        f"the SUT is collapsing duplicate sample_index occurrences."
    )

    # Every issued query_id was completed exactly once.
    assert issued == n_issued
    assert completed == n_issued


# Make sure the subprocess can ``import wan_harness`` even when pytest is
# launched without an editable install (mirrors ``conftest.py``).
os.environ.setdefault(
    "PYTHONPATH",
    str(Path(__file__).resolve().parents[1] / "src")
    + os.pathsep
    + os.environ.get("PYTHONPATH", ""),
)
