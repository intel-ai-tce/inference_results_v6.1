"""Tests for :mod:`wan_harness.config`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wan_harness.config import HarnessConfig, coerce_field_value, load_harness_config


def test_defaults_validate() -> None:
    cfg = HarnessConfig()
    assert cfg.scenario == "Offline"
    assert cfg.mode == "performance"
    assert cfg.backend == "mock"
    assert cfg.height % 8 == 0 and cfg.width % 8 == 0
    # as_dict turns Paths into strings so it's JSON-serialisable.
    d = cfg.as_dict()
    assert isinstance(d["prompts_path"], str)
    assert isinstance(d["output_dir"], str)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scenario", "Garbage"),
        ("mode", "weird"),
        ("backend", "tensorrt"),
        ("height", 0),
        ("width", 1281),  # not a multiple of 8
        ("num_frames", -1),
        ("sample_steps", 0),
        ("boundary_ratio", 0.0),
        ("boundary_ratio", 1.5),
        ("seed", -7),
        ("mock_delay_ms", -3),
        ("mock_payload", "weird"),
    ],
)
def test_invalid_values_raise(field: str, value) -> None:
    with pytest.raises(ValueError):
        HarnessConfig(**{field: value})


def test_yaml_overrides(tmp_path: Path) -> None:
    cfg_path = tmp_path / "inference_config.yaml"
    cfg_path.write_text(
        "height: 480\nwidth: 832\nnum_frames: 41\nseed: 7\n",
        encoding="utf-8",
    )
    cfg = load_harness_config(inference_config_path=cfg_path)
    assert cfg.height == 480
    assert cfg.width == 832
    assert cfg.num_frames == 41
    assert cfg.seed == 7
    # Untouched fields keep their defaults.
    assert cfg.scenario == "Offline"


def test_yaml_then_cli(tmp_path: Path) -> None:
    cfg_path = tmp_path / "inference_config.yaml"
    cfg_path.write_text("seed: 7\n", encoding="utf-8")
    cfg = load_harness_config(
        inference_config_path=cfg_path,
        cli_overrides={"seed": 99, "scenario": "SingleStream"},
    )
    assert cfg.seed == 99
    assert cfg.scenario == "SingleStream"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "inference_config.yaml"
    cfg_path.write_text("seed: 7\n", encoding="utf-8")
    monkeypatch.setenv("WAN_HARNESS_SEED", "123")
    monkeypatch.setenv("WAN_HARNESS_SCENARIO", "SingleStream")
    monkeypatch.setenv("WAN_HARNESS_TARGET_QPS", "0.04")
    monkeypatch.setenv("WAN_HARNESS_FOO_BAR", "ignored")  # unknown fields ignored

    cfg = load_harness_config(inference_config_path=cfg_path)
    assert cfg.seed == 123
    assert cfg.scenario == "SingleStream"
    assert cfg.target_qps == pytest.approx(0.04)


def test_cli_beats_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "inference_config.yaml"
    cfg_path.write_text("seed: 7\n", encoding="utf-8")
    monkeypatch.setenv("WAN_HARNESS_SEED", "123")
    cfg = load_harness_config(
        inference_config_path=cfg_path, cli_overrides={"seed": 999}
    )
    assert cfg.seed == 999


def test_merged_ignores_none() -> None:
    cfg = HarnessConfig()
    new = cfg.merged(seed=None, mock_delay_ms=10)
    assert new.seed == cfg.seed
    assert new.mock_delay_ms == 10


def test_coerce_field_value_typed() -> None:
    assert coerce_field_value("height", "720") == 720
    assert coerce_field_value("guidance_scale", "4.5") == pytest.approx(4.5)
    assert coerce_field_value("enable_loadgen_trace", "true") is True
    assert coerce_field_value("enable_loadgen_trace", "0") is False
    assert isinstance(coerce_field_value("output_dir", "/tmp/x"), Path)
    assert coerce_field_value("backend", "mock") == "mock"


def test_coerce_field_value_unknown() -> None:
    with pytest.raises(KeyError):
        coerce_field_value("not_a_field", "1")
