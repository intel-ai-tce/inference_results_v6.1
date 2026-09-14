"""Unit tests for :mod:`wan_harness.post_run_overhead`."""

from __future__ import annotations

from wan_harness.post_run_overhead import (
    PHASE_RESULT_PACK,
    PHASE_RUN_UNIT,
    PHASE_SUT_RESPONSE_COMPLETE,
    PHASE_WIRE_TRANSFER,
    get_collector,
)


def test_collector_disabled_is_noop() -> None:
    collector = get_collector()
    collector.configure(enabled=False, rank=0)
    collector.record(PHASE_RESULT_PACK, 1.0, sample_index=0, rank=0, nbytes=10)
    assert collector.summary_dict()["sample_count"] == 0


def test_collector_aggregates_phases() -> None:
    collector = get_collector()
    collector.configure(enabled=True, rank=2)
    collector.record(PHASE_RUN_UNIT, 0.100, sample_index=1, rank=2)
    collector.record(PHASE_RESULT_PACK, 0.010, sample_index=1, rank=2, nbytes=1024)
    collector.record(PHASE_WIRE_TRANSFER, 0.200, sample_index=1, rank=2, nbytes=1024)
    collector.record(
        PHASE_SUT_RESPONSE_COMPLETE, 0.050, sample_index=1, rank=0, nbytes=1024
    )

    summary = collector.summary_dict()
    assert summary["rank"] == 2
    assert summary["sample_count"] == 4
    assert summary["post_run_total_s"] > 0.0

    phases = {row["phase"]: row for row in summary["phases"]}
    assert phases[PHASE_WIRE_TRANSFER]["count"] == 1
    assert phases[PHASE_WIRE_TRANSFER]["mean_ms"] == 200.0
    assert phases[PHASE_WIRE_TRANSFER]["mib_per_s"] is not None

    collector.configure(enabled=False, rank=0)
