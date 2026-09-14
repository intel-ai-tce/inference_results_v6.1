"""End-to-end dry-run against real ``mlperf_loadgen``.

Skipped automatically (via the ``loadgen`` marker registered in
``conftest.py``) when the module isn't importable, so this test file is safe
to ship in CI runners that don't have LoadGen installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wan_harness.config import HarnessConfig

pytestmark = pytest.mark.loadgen


def _run(tmp_path: Path, *, scenario: str, mode: str) -> Path:
    from wan_harness.loadgen_runner import run

    out_dir = tmp_path / f"{scenario}-{mode}"
    cfg = HarnessConfig(
        backend="mock",
        scenario=scenario,
        mode=mode,
        # Tiny payload so the test runs in well under a second.
        height=8,
        width=16,
        num_frames=2,
        # Use a synthetic prompts set; we don't have vbench_prompts.txt in CI.
        prompts_path=tmp_path / "no-such-file.txt",
        output_dir=out_dir,
        min_query_count=4,
        min_duration_ms=10,
        performance_sample_count=8,
    )
    result = run(cfg, rank=0, world_size=1)
    assert result.completed >= 4
    return out_dir


def test_offline_performance(tmp_path: Path) -> None:
    out_dir = _run(tmp_path, scenario="Offline", mode="performance")
    assert (out_dir / "mlperf_log_summary.txt").exists()
    assert (out_dir / "mlperf_log_detail.txt").exists()
    summary = (out_dir / "mlperf_log_summary.txt").read_text()
    assert "Scenario : Offline" in summary
    assert "PerformanceOnly" in summary

    detail = (out_dir / "mlperf_log_detail.txt").read_text()
    assert "Multiple conf files are used" not in detail

    meta = json.loads((out_dir / "harness_metadata.json").read_text())
    assert meta["result"]["backend"] == "mock"
    assert meta["config"]["scenario"] == "Offline"


def test_singlestream_accuracy(tmp_path: Path) -> None:
    out_dir = _run(tmp_path, scenario="SingleStream", mode="accuracy")
    assert (out_dir / "mlperf_log_summary.txt").exists()
    artefacts = out_dir / "artefacts"
    assert artefacts.is_dir()
    bins = list(artefacts.glob("*.bin"))
    assert bins, "expected at least one artefact bin file"
    index_lines = (artefacts / "artefacts.jsonl").read_text().strip().splitlines()
    assert len(index_lines) == len(bins)
    record = json.loads(index_lines[0])
    assert record["kind"] == "bin"
    assert record["bytes"] == 2 * 8 * 16 * 3

    acc_log = out_dir / "mlperf_log_accuracy.json"
    assert acc_log.is_file()
    linked = artefacts / "mlperf_log_accuracy.json"
    assert linked.is_symlink()
    assert linked.resolve() == acc_log.resolve()


def test_audit_conf_wan_min_query_count(tmp_path: Path) -> None:
    """Wan-specific audit keys require FromConfig(audit, model_name, scenario)."""
    import re

    from wan_harness.loadgen_runner import run

    audit_conf = tmp_path / "audit.config"
    audit_conf.write_text(
        "*.SingleStream.mode = 2\n"
        "*.SingleStream.performance_issue_unique = 0\n"
        "*.SingleStream.performance_issue_same = 1\n"
        "*.SingleStream.performance_issue_same_index = 3\n"
        "wan-2.2-t2v-a14b.SingleStream.min_query_count = 20\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "SingleStream-audit"
    cfg = HarnessConfig(
        backend="mock",
        scenario="SingleStream",
        mode="performance",
        height=8,
        width=16,
        num_frames=2,
        prompts_path=tmp_path / "no-such-file.txt",
        output_dir=out_dir,
        min_duration_ms=10,
        performance_sample_count=8,
        audit_conf_path=audit_conf,
    )
    run(cfg, rank=0, world_size=1)

    detail = (out_dir / "mlperf_log_detail.txt").read_text()
    m = re.search(r'"key": "requested_min_query_count", "value": (\d+)', detail)
    assert m is not None, "requested_min_query_count missing from detail log"
    assert int(m.group(1)) == 20
    assert "Found Audit Config file (audit.config)" in detail
    # Audit FromConfig is a second conf_type=1 call; LoadGen logs
    # error_invalid_config (expected on compliance, harmless for TEST04).
    assert "Multiple conf files are used" in detail
