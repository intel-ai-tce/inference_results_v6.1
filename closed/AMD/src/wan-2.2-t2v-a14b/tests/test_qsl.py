"""Tests for :mod:`wan_harness.qsl` and the prompt loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from wan_harness.config import REPO_ROOT
from wan_harness.data.prompts import load_prompts, synthetic_prompts
from wan_harness.qsl import WanQSL


def test_synthetic_prompts() -> None:
    ds = synthetic_prompts(8)
    assert len(ds) == 8
    assert ds[0].startswith("synthetic-prompt-")
    assert ds.get_many([0, 7]) == [ds[0], ds[7]]


def test_load_prompts(tmp_path: Path) -> None:
    p = tmp_path / "prompts.txt"
    p.write_text("first prompt\n\n   \nsecond prompt\nthird\n", encoding="utf-8")
    ds = load_prompts(p)
    assert ds.prompts == ("first prompt", "second prompt", "third")


def test_load_prompts_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path / "does-not-exist.txt")


def test_qsl_load_unload() -> None:
    ds = synthetic_prompts(10)
    qsl = WanQSL(ds, performance_sample_count=5)
    assert qsl.total_sample_count == 10
    assert qsl.performance_sample_count == 5
    assert qsl.loaded_indices == frozenset()

    qsl.load_query_samples([0, 1, 2])
    assert qsl.loaded_indices == frozenset({0, 1, 2})

    qsl.load_query_samples([2, 3])
    assert qsl.loaded_indices == frozenset({0, 1, 2, 3})

    qsl.unload_query_samples([1])
    assert qsl.loaded_indices == frozenset({0, 2, 3})


def test_qsl_get_prompts_order() -> None:
    ds = synthetic_prompts(5)
    qsl = WanQSL(ds)
    assert qsl.get_prompts([4, 0, 2]) == [ds[4], ds[0], ds[2]]


def test_qsl_out_of_range() -> None:
    ds = synthetic_prompts(4)
    qsl = WanQSL(ds)
    with pytest.raises(IndexError):
        qsl.load_query_samples([4])


def test_qsl_perf_count_exceeds_total() -> None:
    ds = synthetic_prompts(4)
    with pytest.raises(ValueError):
        WanQSL(ds, performance_sample_count=8)


def test_vendored_synthetic_prompts_file_loads() -> None:
    """The smoke test points --prompts at this file; keep it parseable."""
    path = REPO_ROOT / "data" / "synthetic_prompts.txt"
    ds = load_prompts(path)
    assert len(ds) >= 8, f"expected >= 8 prompts, got {len(ds)}"
    # Every entry is a non-trivial string.
    for i, prompt in enumerate(ds.prompts):
        assert isinstance(prompt, str) and len(prompt) >= 8, (
            f"prompt {i} is too short / not a string: {prompt!r}"
        )
    # No duplicates – avoid feeding the model the same prompt twice in a
    # smoke run.
    assert len(set(ds.prompts)) == len(ds.prompts), "duplicate prompts found"
