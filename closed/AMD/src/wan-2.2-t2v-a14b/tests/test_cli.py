"""Smoke tests for the CLI argument parser and ``--set key=value`` escape hatch."""

from __future__ import annotations

import json

import pytest

from wan_harness.cli import _build_parser, _resolve_config


def _parse(args: list[str]):
    return _build_parser().parse_args(args)


def test_set_overrides_int_and_float() -> None:
    args = _parse(
        [
            "run",
            "--set",
            "height=8",
            "--set",
            "width=16",
            "--set",
            "num_frames=2",
            "--set",
            "guidance_scale=3.5",
        ]
    )
    cfg = _resolve_config(args)
    assert cfg.height == 8
    assert cfg.width == 16
    assert cfg.num_frames == 2
    assert cfg.guidance_scale == pytest.approx(3.5)


def test_set_overrides_unknown_field_exits() -> None:
    args = _parse(["run", "--set", "not_a_field=1"])
    with pytest.raises(SystemExit):
        _resolve_config(args)


def test_set_overrides_missing_equals_exits() -> None:
    args = _parse(["run", "--set", "garbage"])
    with pytest.raises(SystemExit):
        _resolve_config(args)


def test_set_beats_named_flag() -> None:
    """--set wins over the curated --scenario flag (consistent precedence)."""
    args = _parse(
        ["run", "--scenario", "Offline", "--set", "scenario=SingleStream"]
    )
    cfg = _resolve_config(args)
    assert cfg.scenario == "SingleStream"


def test_dry_run_implies_mock() -> None:
    args = _parse(["run", "--dry-run"])
    cfg = _resolve_config(args)
    assert cfg.backend == "mock"


def test_dry_run_conflict_exits() -> None:
    args = _parse(["run", "--dry-run", "--backend", "wan22"])
    with pytest.raises(SystemExit):
        _resolve_config(args)


def test_print_config_json(capsys: pytest.CaptureFixture[str]) -> None:
    from wan_harness.cli import _cmd_print_config

    args = _parse(
        ["print-config", "--set", "height=8", "--set", "width=16", "--format", "json"]
    )
    rc = _cmd_print_config(args)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["height"] == 8
    assert payload["width"] == 16
