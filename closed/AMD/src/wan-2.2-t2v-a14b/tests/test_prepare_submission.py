"""Tests for :mod:`tools.prepare_submission`."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.package_code import validate_git_state
from tools.prepare_submission import (
    DEFAULT_SYSTEM,
    WAN_AUDIT_VIDEO_INDICES,
    PrepareConfig,
    install_system_desc,
    prepare_submission,
    truncate_accuracy_log,
)
from wan_harness.vbench import MAX_ACCURACY_LOG_SIZE, write_accuracy_txt

from tests.test_vbench import (
    SUBMISSION_HASH_RE,
    SUBMISSION_VBENCH_SCORE_RE,
    _make_accuracy_run,
    _make_result,
    _make_vbench_results,
)


_MLPERF_INFERENCE_VERSION = "v6.1"
_WAN_AUDIT_VIDEO_INDICES = WAN_AUDIT_VIDEO_INDICES[_MLPERF_INFERENCE_VERSION]

def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _write_loadgen_logs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mlperf_log_summary.txt").write_text(
        "RESULTS: mode=PerformanceOnly\n", encoding="utf-8"
    )
    (directory / "mlperf_log_detail.txt").write_text(
        ':::MLLOG {"key": "value"}\n', encoding="utf-8"
    )


def _write_experiment_manifest(
    experiment_root: Path,
    *,
    sha: str | None = None,
    dirty: bool = False,
) -> None:
    experiment_root.mkdir(parents=True, exist_ok=True)
    head = sha or _git_head()
    manifest = {
        "started_at_utc": "2026-06-08T13-53-29Z",
        "backend": "wan22",
        "root": str(experiment_root),
        "git": {
            "sha": head,
            "short_sha": head[:7],
            "branch": "main",
            "dirty": dirty,
            "commit_subject": "test experiment",
        },
    }
    (experiment_root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _user_conf(tmp_path: Path) -> Path:
    path = tmp_path / "user.conf"
    path.write_text("# loadgen overrides\n", encoding="utf-8")
    return path


def _measurements(tmp_path: Path) -> Path:
    path = tmp_path / "measurements.json"
    path.write_text(
        json.dumps(
            {
                "input_data_types": "bf16",
                "retraining": "No",
                "starting_weights_filename": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "weight_data_types": "bf16, fp8, fp4",
                "weight_transformations": "quantization",
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _readme_template(tmp_path: Path) -> Path:
    path = tmp_path / "scenario_README.md"
    path.write_text("# $scenario\n\nsha=$git_sha\n", encoding="utf-8")
    return path


def _system_desc(tmp_path: Path) -> Path:
    path = tmp_path / "systems" / f"{DEFAULT_SYSTEM}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"submitter":"TestOrg","division":"closed","accelerators_per_node":8}\n',
        encoding="utf-8",
    )
    return path


def _prepare_cfg(
    tmp_path: Path,
    experiment: Path,
    *,
    skip_compliance: bool = False,
    skip_vbench_refresh: bool = False,
    skip_code: bool = True,
    skip_measurements: bool = False,
    skip_readme: bool = False,
    dry_run: bool = False,
) -> PrepareConfig:
    repo_root = Path(__file__).resolve().parent.parent
    return PrepareConfig(
        experiment_root=experiment,
        version=_MLPERF_INFERENCE_VERSION,
        output_root=tmp_path / "submission",
        division="closed",
        submitter="TestOrg",
        system=DEFAULT_SYSTEM,
        benchmark="wan-2.2-t2v-a14b",
        user_conf=_user_conf(tmp_path),
        system_desc=_system_desc(tmp_path),
        measurements_template=_measurements(tmp_path),
        readme_template=_readme_template(tmp_path),
        repo_root=repo_root,
        skip_compliance=skip_compliance,
        skip_vbench_refresh=skip_vbench_refresh,
        skip_code=skip_code,
        skip_measurements=skip_measurements,
        skip_readme=skip_readme,
        dry_run=dry_run,
    )


def _ensure_audit_videos(artefacts: Path) -> None:
    prompts = json.loads((artefacts / "prompts.json").read_text(encoding="utf-8"))
    stub = b"\x00\x00\x00\x18ftypmp42"
    for idx in _WAN_AUDIT_VIDEO_INDICES:
        name = f"{idx}.mp4"
        prompts.setdefault(name, f"prompt {idx}")
        path = artefacts / name
        if not path.is_file():
            path.write_bytes(stub)
    (artefacts / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")


def _make_experiment(root: Path, *, with_compliance: bool = True) -> Path:
    _write_experiment_manifest(root)
    for scenario in ("Offline", "SingleStream"):
        perf = root / scenario / "performance" / "run_1"
        _write_loadgen_logs(perf)

        acc = _make_accuracy_run(root / scenario, mp4_count=2)
        _ensure_audit_videos(acc / "artefacts")
        _write_loadgen_logs(acc)
        acc_json = acc / "mlperf_log_accuracy.json"
        acc_json.write_bytes(b"x" * (MAX_ACCURACY_LOG_SIZE + 512))

        vbench_dir = acc / "vbench"
        _make_vbench_results(vbench_dir)
        result = _make_result(
            70.48,
            results_file=vbench_dir / "results_2026-06-05-08:30:00_eval_results.json",
        )
        write_accuracy_txt(result, acc / "accuracy.txt", acc_json_path=acc_json)

        if with_compliance:
            test04 = root / scenario / "TEST04"
            _write_loadgen_logs(test04)
            verify = root / scenario / "compliance" / "TEST04"
            verify.mkdir(parents=True, exist_ok=True)
            (verify / "verify_performance.txt").write_text("TEST PASS\n", encoding="utf-8")
    return root


def test_truncate_accuracy_log_shrinks_large_file(tmp_path: Path) -> None:
    path = tmp_path / "mlperf_log_accuracy.json"
    path.write_bytes(b"[" + b'{"q":1},' * 5000 + b"]")
    before = path.stat().st_size
    assert before > MAX_ACCURACY_LOG_SIZE

    size_before, size_after = truncate_accuracy_log(path)

    assert size_before == before
    assert size_after <= MAX_ACCURACY_LOG_SIZE
    assert "..." in path.read_text(encoding="utf-8")


def test_truncate_accuracy_log_idempotent_on_small_file(tmp_path: Path) -> None:
    path = tmp_path / "mlperf_log_accuracy.json"
    payload = b'[{"qsl_idx": 0}]'
    path.write_bytes(payload)

    size_before, size_after = truncate_accuracy_log(path)

    assert size_before == len(payload)
    assert size_after == len(payload)
    assert path.read_bytes() == payload


def test_install_system_desc_overwrites_submitter_and_division(tmp_path: Path) -> None:
    cfg = _prepare_cfg(tmp_path, tmp_path / "exp", dry_run=False)
    summary = install_system_desc(cfg)
    dest = Path(summary["destination"])
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["submitter"] == "TestOrg"
    assert payload["division"] == "closed"


def test_validate_git_state_rejects_dirty_manifest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    manifest = {
        "git": {
            "sha": _git_head(),
            "dirty": True,
        }
    }
    with pytest.raises(RuntimeError, match="dirty working tree"):
        validate_git_state(repo_root, manifest)


def test_prepare_submission_layout(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path / "exp")
    cfg = _prepare_cfg(tmp_path, experiment)

    manifest_path = prepare_submission(cfg)
    assert manifest_path.is_file()

    system_json = (
        cfg.output_root / "closed" / "TestOrg" / "systems" / f"{DEFAULT_SYSTEM}.json"
    )
    assert system_json.is_file()

    for scenario in ("Offline", "SingleStream"):
        base = (
            cfg.output_root
            / "closed"
            / "TestOrg"
            / "results"
            / DEFAULT_SYSTEM
            / "wan-2.2-t2v-a14b"
            / scenario
        )
        perf = base / "performance" / "run_1"
        acc = base / "accuracy"
        test04 = base / "TEST04" / "performance" / "run_1"

        assert (perf / "mlperf_log_summary.txt").is_file()
        assert (acc / "mlperf_log_accuracy.json").is_file()
        assert (acc / "accuracy.txt").is_file()
        assert (test04 / "mlperf_log_summary.txt").is_file()
        assert (base / "TEST04" / "verify_performance.txt").is_file()
        assert (base / "user.conf").is_file()
        assert (base / "measurements.json").is_file()
        assert (base / "README.md").is_file()

        measurements = json.loads((base / "measurements.json").read_text(encoding="utf-8"))
        assert "fp8" in measurements["weight_data_types"]

        acc_size = (acc / "mlperf_log_accuracy.json").stat().st_size
        assert acc_size <= MAX_ACCURACY_LOG_SIZE

        acc_txt = (acc / "accuracy.txt").read_text(encoding="utf-8")
        assert SUBMISSION_VBENCH_SCORE_RE.search(acc_txt)
        hash_match = None
        for line in acc_txt.splitlines():
            hash_match = SUBMISSION_HASH_RE.match(line)
            if hash_match:
                break
        assert hash_match is not None

        digest = hashlib.sha256((acc / "mlperf_log_accuracy.json").read_bytes()).hexdigest()
        assert hash_match.group(1) == digest

        videos = acc / "videos"
        assert videos.is_dir()
        for idx in _WAN_AUDIT_VIDEO_INDICES:
            assert (videos / f"{idx}.mp4").is_file()
        assert (videos / "captions.txt").is_file()
        assert not (acc / "artefacts").exists()


def test_prepare_submission_validates_git_before_scenario_work(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path / "exp")
    manifest = json.loads((experiment / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["git"]["sha"] = "0" * 40
    (experiment / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cfg = _prepare_cfg(tmp_path, experiment, skip_code=False)

    with pytest.raises(RuntimeError, match="does not match experiment git.sha"):
        prepare_submission(cfg)

    results_root = (
        cfg.output_root
        / "closed"
        / "TestOrg"
        / "results"
        / DEFAULT_SYSTEM
        / "wan-2.2-t2v-a14b"
    )
    assert not results_root.exists()


def test_prepare_submission_packages_code(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path / "exp")
    cfg = _prepare_cfg(tmp_path, experiment, skip_code=False)

    fake_summary = {
        "destination": str(
            cfg.output_root / "closed" / "TestOrg" / "src" / "wan-2.2-t2v-a14b"
        ),
        "git_sha": _git_head(),
    }
    with patch("tools.prepare_submission.package_code_snapshot", return_value=fake_summary):
        prepare_submission(cfg)

    prepare_manifest = json.loads(
        (cfg.output_root / "PREPARE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert prepare_manifest["code"]["git_sha"] == _git_head()


def test_prepare_submission_dry_run_writes_nothing(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path / "exp")
    cfg = _prepare_cfg(
        tmp_path,
        experiment,
        skip_compliance=True,
        skip_vbench_refresh=True,
        dry_run=True,
    )

    prepare_submission(cfg)
    assert not cfg.output_root.exists()


def test_prepare_submission_refuses_nonempty_destination(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path / "exp")
    cfg = _prepare_cfg(
        tmp_path,
        experiment,
        skip_compliance=True,
        skip_vbench_refresh=True,
        dry_run=False,
    )
    dest = (
        cfg.output_root
        / "closed"
        / "TestOrg"
        / "results"
        / DEFAULT_SYSTEM
        / "wan-2.2-t2v-a14b"
        / "Offline"
        / "performance"
        / "run_1"
    )
    dest.mkdir(parents=True)
    (dest / "mlperf_log_summary.txt").write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_submission(cfg)
