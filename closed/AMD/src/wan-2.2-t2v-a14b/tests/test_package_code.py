"""Tests for :mod:`tools.package_code`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.package_code import (
    build_reproducibility_json,
    load_experiment_manifest,
    package_code_snapshot,
    parse_dockerfile_args,
    prune_archived_source,
    run_git_archive,
    validate_git_state,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "safe.directory=*", *args],
        cwd=repo,
        text=True,
    ).strip()


def _init_repo(tmp_path: Path, *, with_github: bool = True) -> tuple[Path, str]:
    """Create a minimal git repo and return ``(repo_root, commit_sha)``."""
    repo = tmp_path / "harness"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "README.md").write_text("# harness\n", encoding="utf-8")
    (repo / "launch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / ".dockerignore").write_text(".git\n", encoding="utf-8")
    (repo / ".gitignore").write_text("/runs/\n", encoding="utf-8")
    (repo / "data").mkdir()
    (repo / "data" / ".gitkeep").write_text("", encoding="utf-8")
    if with_github:
        workflows = repo / ".github" / "workflows"
        workflows.mkdir(parents=True)
        (workflows / "ci.yml").write_text("on: push\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init harness"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo, _git(repo, "rev-parse", "HEAD")


def _write_manifest(experiment_root: Path, *, sha: str, dirty: bool = False) -> None:
    experiment_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at_utc": "2026-06-08T13-53-29Z",
        "git": {
            "sha": sha,
            "short_sha": sha[:7],
            "branch": "main",
            "dirty": dirty,
            "commit_subject": "test experiment",
        },
    }
    (experiment_root / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_parse_dockerfile_args_reads_pinned_commits() -> None:
    args = parse_dockerfile_args(_REPO_ROOT / "docker" / "Dockerfile")
    assert args["MLPERF_INFERENCE_COMMIT"] == "bdc4116022872258d7b70d6ddcc3fe6000ea16fe"
    assert args["VBENCH_COMMIT"]
    assert args["XDIT_COMMIT"]


def test_load_experiment_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MANIFEST.json"):
        load_experiment_manifest(tmp_path / "no-such-exp")


def test_validate_git_state_rejects_dirty_manifest(tmp_path: Path) -> None:
    repo, sha = _init_repo(tmp_path, with_github=False)
    manifest = {"git": {"sha": sha, "dirty": True}}
    with pytest.raises(RuntimeError, match="dirty working tree"):
        validate_git_state(repo, manifest)


def test_validate_git_state_rejects_head_mismatch(tmp_path: Path) -> None:
    repo, sha = _init_repo(tmp_path, with_github=False)
    manifest = {"git": {"sha": "0" * 40, "dirty": False}}
    with pytest.raises(RuntimeError, match="does not match experiment git.sha"):
        validate_git_state(repo, manifest)


def test_validate_git_state_accepts_matching_head(tmp_path: Path) -> None:
    repo, sha = _init_repo(tmp_path, with_github=False)
    manifest = {"git": {"sha": sha, "dirty": False, "branch": "main"}}
    git_info = validate_git_state(repo, manifest)
    assert git_info["sha"] == sha


def test_prune_archived_source_removes_dot_artifacts(tmp_path: Path) -> None:
    dest = tmp_path / "src"
    dest.mkdir()
    (dest / "launch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (dest / ".dockerignore").write_text(".git\n", encoding="utf-8")
    (dest / ".gitignore").write_text("/runs/\n", encoding="utf-8")
    (dest / "data").mkdir()
    (dest / "data" / ".gitkeep").write_text("", encoding="utf-8")
    github = dest / ".github" / "workflows"
    github.mkdir(parents=True)
    (github / "ci.yml").write_text("on: push\n", encoding="utf-8")

    removed = prune_archived_source(dest)

    assert set(removed) == {".github", ".dockerignore", ".gitignore", "data/.gitkeep"}
    assert not (dest / ".github").exists()
    assert not (dest / ".dockerignore").exists()
    assert not (dest / ".gitignore").exists()
    assert not (dest / "data" / ".gitkeep").exists()
    assert (dest / "launch.sh").is_file()


def test_run_git_archive_omits_dot_artifacts(tmp_path: Path) -> None:
    repo, sha = _init_repo(tmp_path, with_github=True)
    dest = tmp_path / "archive"

    run_git_archive(repo, sha, dest)

    assert (dest / "README.md").is_file()
    assert (dest / "launch.sh").is_file()
    assert (dest / "data").is_dir()
    assert not (dest / ".github").exists()
    assert not (dest / ".dockerignore").exists()
    assert not (dest / ".gitignore").exists()
    assert not (dest / "data" / ".gitkeep").exists()


def test_package_code_snapshot_writes_reproducibility_json(tmp_path: Path) -> None:
    repo, sha = _init_repo(tmp_path, with_github=True)
    experiment = tmp_path / "experiment"
    _write_manifest(experiment, sha=sha)
    output = tmp_path / "submission"

    summary = package_code_snapshot(
        repo_root=repo,
        experiment_root=experiment,
        output_root=output,
        division="closed",
        submitter="TestOrg",
        fetch_data_commit="deadbeef",
    )

    dest = Path(summary["destination"])
    assert summary["git_sha"] == sha
    assert (dest / "README.md").is_file()
    assert not (dest / ".github").exists()
    assert not (dest / ".gitignore").exists()
    repro = json.loads((dest / "REPRODUCIBILITY.json").read_text(encoding="utf-8"))
    assert repro["harness_git_sha"] == sha
    assert repro["data_fetch"]["pinned_commit"] == "deadbeef"


def test_build_reproducibility_json_includes_model_and_docker_pins() -> None:
    manifest = {"started_at_utc": "2026-06-08T13-53-29Z"}
    git_info = {
        "sha": "abc",
        "branch": "main",
        "commit_subject": "test",
    }
    repro = build_reproducibility_json(
        manifest=manifest,
        git_info=git_info,
        repo_root=_REPO_ROOT,
        experiment_root=_REPO_ROOT / "runs" / "wan22" / "exp",
        fetch_data_commit="fetch-sha",
    )
    assert repro["harness_git_sha"] == "abc"
    assert repro["docker_build_args"]["MLPERF_INFERENCE_COMMIT"]
    assert repro["model"]["repo_id"].startswith("Wan-AI/")
