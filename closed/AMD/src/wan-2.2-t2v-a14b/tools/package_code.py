"""Package a reproducible harness source snapshot for MLPerf submission.

Creates ``closed/<submitter>/src/<benchmark>/`` via ``git archive`` at the
commit recorded in the experiment ``MANIFEST.json``. Writes
``REPRODUCIBILITY.json`` alongside the archived tree with Docker build pins
and data-fetch metadata.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_log = logging.getLogger("package_code")

DEFAULT_BENCHMARK_DIR = "wan-2.2-t2v-a14b"
DOCKERFILE = Path("docker/Dockerfile")

# ARG names whose defaults we record in REPRODUCIBILITY.json (build-time pins).
_PINNED_DOCKER_ARGS = (
    "AITER_COMMIT",
    "XDIT_COMMIT",
    "MLPERF_INFERENCE_COMMIT",
    "VBENCH_COMMIT",
    "ROCM_VERSION",
)


def load_experiment_manifest(experiment_root: Path) -> dict[str, Any]:
    path = experiment_root.resolve() / "MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"experiment MANIFEST.json not found: {path} "
            f"(was this directory produced by run_all.sh?)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "safe.directory=*", *args],
        cwd=repo_root,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def parse_dockerfile_args(dockerfile: Path) -> dict[str, str]:
    """Return default values for pinned ``ARG`` lines in the Dockerfile."""
    if not dockerfile.is_file():
        return {}
    text = dockerfile.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for name in _PINNED_DOCKER_ARGS:
        match = re.search(rf"^ARG\s+{re.escape(name)}=(.+)$", text, re.MULTILINE)
        if match:
            out[name] = match.group(1).strip()
    return out


def _read_model_revision(repo_root: Path) -> dict[str, str]:
    """Best-effort read of model path/revision from Offline backend config."""
    cfg_path = repo_root / "configs" / "wan22" / "Offline.yaml"
    if not cfg_path.is_file():
        return {}
    text = cfg_path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for key in ("path", "revision"):
        match = re.search(rf"^\s+{key}:\s+(\S+)\s*$", text, re.MULTILINE)
        if match:
            out[key] = match.group(1)
    return out


def validate_git_state(
    repo_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Ensure the experiment git metadata is safe to archive. Returns git block."""
    git_info = manifest.get("git")
    if not isinstance(git_info, dict):
        raise RuntimeError("experiment MANIFEST.json missing 'git' block")

    sha = git_info.get("sha")
    if not sha:
        raise RuntimeError("experiment MANIFEST.json missing git.sha")

    if git_info.get("dirty"):
        raise RuntimeError(
            "experiment was run from a dirty working tree (MANIFEST.json git.dirty=true); "
            "commit or stash changes and re-run stage 1 before packaging"
        )

    try:
        head = _git(repo_root, "rev-parse", "HEAD")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"cannot read git HEAD under {repo_root}: {exc.stderr}") from exc

    if head != sha:
        raise RuntimeError(
            f"current HEAD ({head[:12]}) does not match experiment git.sha ({sha[:12]}); "
            f"checkout {sha} before packaging so the archived source matches the run"
        )

    return git_info


def run_git_archive(repo_root: Path, sha: str, dest: Path) -> None:
    """Extract ``git archive`` for *sha* into *dest* (created if needed)."""
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "-c", "safe.directory=*", "archive", sha],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git archive {sha} failed: {proc.stderr.decode('utf-8', errors='replace').strip()}"
        )
    extract = subprocess.run(
        ["tar", "-x", "-C", str(dest)],
        input=proc.stdout,
        stderr=subprocess.PIPE,
        check=False,
    )
    if extract.returncode != 0:
        raise RuntimeError(
            f"tar extract failed: {extract.stderr.decode('utf-8', errors='replace').strip()}"
        )
    prune_archived_source(dest)


# Root dotfiles and placeholder paths dropped after ``git archive`` — not needed
# to rebuild/replay and some submission checkers reject "git unfriendly" names.
_PRUNE_ROOT_DOTFILES = (".dockerignore", ".gitignore")


def prune_archived_source(dest: Path) -> list[str]:
    """Drop paths from a ``git archive`` tree that are not needed for reproduction.

    ``git archive`` includes everything tracked at the experiment SHA (e.g.
    ``.github/`` for CI, ``.gitignore``, ``data/.gitkeep``). Submission
    packaging only needs the harness source files a submitter would use to
    rebuild and replay runs.
    """
    removed: list[str] = []
    github = dest / ".github"
    if github.is_dir():
        shutil.rmtree(github)
        removed.append(".github")
    for name in _PRUNE_ROOT_DOTFILES:
        path = dest / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    for gitkeep in sorted(dest.rglob(".gitkeep")):
        rel = gitkeep.relative_to(dest).as_posix()
        gitkeep.unlink()
        removed.append(rel)
    return removed


def build_reproducibility_json(
    *,
    manifest: dict[str, Any],
    git_info: dict[str, Any],
    repo_root: Path,
    experiment_root: Path,
    fetch_data_commit: str,
) -> dict[str, Any]:
    docker_args = parse_dockerfile_args(repo_root / DOCKERFILE)
    model = _read_model_revision(repo_root)
    return {
        "harness_git_sha": git_info.get("sha"),
        "harness_git_branch": git_info.get("branch"),
        "harness_git_commit_subject": git_info.get("commit_subject"),
        "experiment_root": str(experiment_root.resolve()),
        "experiment_started_at_utc": manifest.get("started_at_utc"),
        "docker_base_image": "amdsiloai/pytorch-xdit:v26.6",
        "docker_build_args": docker_args,
        "data_fetch": {
            "tool": "python3 -m tools.fetch_data",
            "pinned_commit": fetch_data_commit,
        },
        "model": {
            "repo_id": model.get("path", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
            "revision": model.get("revision"),
        },
    }


def package_code_snapshot(
    *,
    repo_root: Path,
    experiment_root: Path,
    output_root: Path,
    division: str,
    submitter: str,
    benchmark: str = DEFAULT_BENCHMARK_DIR,
    fetch_data_commit: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Archive harness source into ``<output>/closed/<submitter>/src/<benchmark>/``."""
    manifest = load_experiment_manifest(experiment_root)
    git_info = validate_git_state(repo_root, manifest)
    sha = git_info["sha"]

    dest = output_root / division / submitter / "src" / benchmark
    if dest.exists() and any(dest.iterdir()) and not dry_run:
        raise FileExistsError(f"refusing to overwrite non-empty code tree: {dest}")

    if dry_run:
        _log.info("would git archive %s -> %s", sha[:12], dest)
        _log.info("would write %s/REPRODUCIBILITY.json", dest)
        return {
            "destination": str(dest),
            "git_sha": sha,
            "dry_run": True,
        }

    run_git_archive(repo_root, sha, dest)
    repro = build_reproducibility_json(
        manifest=manifest,
        git_info=git_info,
        repo_root=repo_root,
        experiment_root=experiment_root,
        fetch_data_commit=fetch_data_commit,
    )
    repro_path = dest / "REPRODUCIBILITY.json"
    repro_path.write_text(json.dumps(repro, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log.info("wrote harness source snapshot to %s (git %s)", dest, sha[:12])
    _log.info("wrote %s", repro_path)
    return {
        "destination": str(dest),
        "git_sha": sha,
        "reproducibility_json": str(repro_path),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from tools.fetch_data import DEFAULT_COMMIT

    parser = argparse.ArgumentParser(description="Package harness source for MLPerf submission.")
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submitter", required=True)
    parser.add_argument("--division", default="closed")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        package_code_snapshot(
            repo_root=args.repo_root,
            experiment_root=args.experiment_root,
            output_root=args.output,
            division=args.division,
            submitter=args.submitter,
            benchmark=args.benchmark,
            fetch_data_commit=DEFAULT_COMMIT,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError) as exc:
        _log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
