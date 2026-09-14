"""Tests for VBench visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.vbench_viz import (
    DimensionSeries,
    VideoScore,
    align_video_scores,
    dimension_order,
    load_dimension_series,
    resolve_vbench_run,
)


def _write_results(
    out_dir: Path,
    *,
    timestamp: str = "2026-06-05-08:30:00",
    payload: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if payload is None:
        payload = {
            "subject_consistency": [
                0.9,
                [
                    {"video_path": "/staged/prompt-a-0.mp4", "video_results": 0.8},
                    {"video_path": "/staged/prompt-b-0.mp4", "video_results": 1.0},
                ],
            ],
            "dynamic_degree": [
                0.75,
                [
                    {"video_path": "/staged/prompt-a-0.mp4", "video_results": True},
                    {"video_path": "/staged/prompt-b-0.mp4", "video_results": False},
                ],
            ],
        }
    path = out_dir / f"results_{timestamp}_eval_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_dimension_series_extracts_video_keys_and_bool_scores(tmp_path: Path) -> None:
    results = _write_results(tmp_path / "vbench")
    series = load_dimension_series(results)
    by_name = {s.name: s for s in series}
    assert by_name["subject_consistency"].mean == pytest.approx(0.9)
    assert [v.key for v in by_name["subject_consistency"].videos] == [
        "prompt-a-0.mp4",
        "prompt-b-0.mp4",
    ]
    assert [v.score for v in by_name["dynamic_degree"].videos] == [1.0, 0.0]


def test_align_video_scores_joins_on_key() -> None:
    a = (VideoScore("x-0.mp4", 0.2), VideoScore("y-0.mp4", 0.4))
    b = (VideoScore("y-0.mp4", 0.5), VideoScore("x-0.mp4", 0.3))
    keys, sa, sb = align_video_scores(a, b)
    assert keys == ["x-0.mp4", "y-0.mp4"]
    assert sa == [0.2, 0.4]
    assert sb == [0.3, 0.5]


def test_dimension_order_puts_defaults_first() -> None:
    series = (
        DimensionSeries("zebra", 0.1, ()),
        DimensionSeries("subject_consistency", 0.9, ()),
        DimensionSeries("scene", 0.5, ()),
    )
    ordered = dimension_order(series)
    assert [s.name for s in ordered] == ["subject_consistency", "scene", "zebra"]


def test_resolve_vbench_run_from_accuracy_dir(tmp_path: Path) -> None:
    accuracy = tmp_path / "runs" / "exp1" / "Offline" / "accuracy"
    _write_results(accuracy / "vbench")
    resolved = resolve_vbench_run(accuracy)
    assert resolved.scenario == "Offline"
    assert resolved.vbench_dir == accuracy / "vbench"
    assert resolved.label == "exp1"


def test_resolve_vbench_run_from_vbench_dir(tmp_path: Path) -> None:
    accuracy = tmp_path / "runs" / "exp2" / "SingleStream" / "accuracy"
    vbench = accuracy / "vbench"
    _write_results(vbench)
    resolved = resolve_vbench_run(vbench)
    assert resolved.scenario == "SingleStream"
    assert resolved.run_dir == accuracy
