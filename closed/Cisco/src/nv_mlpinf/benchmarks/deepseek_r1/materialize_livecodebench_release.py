# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Materialize the pinned LiveCodeBench release_v1 snapshot without network I/O."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable, ContextManager, Iterable, Mapping


try:
    from .eval_accuracy_hardened import (
        CANARY_QUESTION_ID,
        EXPECTED_PROBLEM_COUNT,
        RELEASE_VERSION,
        _preflight_livecodebench,
        _problem_id,
        _sha256_file,
        _sha256_tree,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_accuracy_hardened import (  # type: ignore[no-redef]
        CANARY_QUESTION_ID,
        EXPECTED_PROBLEM_COUNT,
        RELEASE_VERSION,
        _preflight_livecodebench,
        _problem_id,
        _sha256_file,
        _sha256_tree,
    )


DATASET_ID = "livecodebench/code_generation_lite"
LIVE_CODE_BENCH_REVISION = "b1e7cab44d610bbc2e10d36d270cd0c89c600492"
CACHE_FINGERPRINT = "4c038560f391c4c05fdf7fd7ae61ae0e6dbd8672f8fe5b95597b78a8dc40a417"
DEFAULT_ARROW_FILE = Path(
    os.environ.get(
        "MLPERF_LCB_ARROW_FILE",
        "/home/mlperf_inference_storage/data/livecodebench/"
        "livecodebench___code_generation_lite/"
        "release_latest-version_tag=release_v1/0.0.0/"
        f"{CACHE_FINGERPRINT}/code_generation_lite-test.arrow",
    )
)
DEFAULT_LCB_DIR = Path(
    "/work/3rdparty/mlc-inference/language/deepseek-r1/submodules/LiveCodeBench"
)
DEFAULT_OUTPUT_FILE = Path(
    "/work/build/accuracy_assets/deepseek-r1/livecodebench/release_v1.pkl"
)

OFFLINE_FLAGS = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
ISOLATED_PATHS = {
    "HOME": "home",
    "XDG_CACHE_HOME": "xdg-cache",
    "XDG_CONFIG_HOME": "xdg-config",
    "XDG_DATA_HOME": "xdg-data",
    "HF_HOME": "hf-home",
    "HF_HUB_CACHE": "hf-hub",
    "TRANSFORMERS_CACHE": "transformers",
    "TMPDIR": "tmp",
}


def _validate_arrow_cache_path(arrow_file: Path) -> Path:
    expected_tail = (
        "livecodebench___code_generation_lite",
        "release_latest-version_tag=release_v1",
        "0.0.0",
        CACHE_FINGERPRINT,
        "code_generation_lite-test.arrow",
    )
    if tuple(arrow_file.parts[-5:]) != expected_tail:
        raise ValueError(
            "LiveCodeBench Arrow cache path does not match the pinned release_v1 "
            f"fingerprint {CACHE_FINGERPRINT}: {arrow_file}"
        )
    if not arrow_file.is_file():
        raise FileNotFoundError(f"Offline LiveCodeBench Arrow cache not found: {arrow_file}")
    return arrow_file.parents[4]


def _validate_environment(
    arrow_file: Path,
    state_root: Path,
    *,
    environ: Mapping[str, str] = os.environ,
) -> None:
    if not state_root.is_absolute():
        raise ValueError(f"Materializer state root must be absolute: {state_root}")
    for name, expected in OFFLINE_FLAGS.items():
        if environ.get(name) != expected:
            raise RuntimeError(f"{name}={expected} is required for offline materialization")

    datasets_cache = _validate_arrow_cache_path(arrow_file)
    configured_cache = environ.get("HF_DATASETS_CACHE")
    if not configured_cache or Path(configured_cache).resolve() != datasets_cache.resolve():
        raise RuntimeError(f"HF_DATASETS_CACHE must be {datasets_cache}")

    for name, suffix in ISOLATED_PATHS.items():
        expected_path = (state_root / suffix).resolve()
        configured = environ.get(name)
        if not configured or Path(configured).resolve() != expected_path:
            raise RuntimeError(f"{name} must be isolated at {expected_path}")


def _prepare_state_root(state_root: Path) -> None:
    for suffix in ISOLATED_PATHS.values():
        (state_root / suffix).mkdir(parents=True, exist_ok=True)


def _validate_source_checkout(
    lcb_dir: Path,
    *,
    execute: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    source_root = lcb_dir / "lcb_runner"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Vendored LiveCodeBench source not found: {source_root}")

    revision_result = execute(
        ["git", "-C", str(lcb_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if revision_result.returncode != 0:
        raise RuntimeError(
            f"Cannot read vendored LiveCodeBench revision: {revision_result.stderr.strip()}"
        )
    revision = revision_result.stdout.strip()
    if revision != LIVE_CODE_BENCH_REVISION:
        raise RuntimeError(
            "Vendored LiveCodeBench revision mismatch: "
            f"expected {LIVE_CODE_BENCH_REVISION}, found {revision}"
        )

    status_result = execute(
        ["git", "-C", str(lcb_dir), "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status_result.returncode != 0:
        raise RuntimeError(
            f"Cannot validate vendored LiveCodeBench source: {status_result.stderr.strip()}"
        )
    dirty = []
    for line in status_result.stdout.splitlines():
        path = line[3:].strip()
        if "__pycache__/" in path or path.endswith(".pyc"):
            continue
        dirty.append(line)
    if dirty:
        raise RuntimeError(
            "Vendored LiveCodeBench checkout is not clean: " + "; ".join(dirty[:5])
        )
    return revision


@contextmanager
def _network_blocked() -> Iterable[None]:
    """Reject socket creation paths even if an offline library regresses."""

    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

    def deny_network(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Network access is disabled for LiveCodeBench materialization")

    class OfflineSocket(original_socket):
        def connect(self, *args: Any, **kwargs: Any) -> Any:
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                return deny_network()
            return super().connect(*args, **kwargs)

        def connect_ex(self, *args: Any, **kwargs: Any) -> Any:
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                return deny_network()
            return super().connect_ex(*args, **kwargs)

    socket.socket = OfflineSocket
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network
    try:
        yield
    finally:
        socket.socket = original_socket
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo


def _build_prompt_problems(
    arrow_file: Path,
    lcb_dir: Path,
    state_root: Path,
    *,
    datasets_module: ModuleType | Any | None = None,
    build_api: Callable[[argparse.Namespace], tuple[Iterable[Any], Any]] | None = None,
    scenario: Any | None = None,
) -> list[Any]:
    if datasets_module is None:
        datasets_module = importlib.import_module("datasets")

    load_calls = []

    def offline_load_dataset(
        dataset_name: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if dataset_name is None:
            dataset_name = kwargs.pop("path", None)
        if dataset_name != DATASET_ID or args:
            raise RuntimeError(
                f"Unexpected LiveCodeBench dataset request: {dataset_name!r}, {args!r}"
            )
        split = kwargs.pop("split", None)
        version_tag = kwargs.pop("version_tag", None)
        kwargs.pop("trust_remote_code", None)
        if kwargs:
            raise RuntimeError(
                f"Unsupported LiveCodeBench dataset options in offline mode: {sorted(kwargs)}"
            )
        if version_tag != RELEASE_VERSION:
            raise RuntimeError(
                f"LiveCodeBench requested version_tag={version_tag!r}, expected {RELEASE_VERSION}"
            )
        if split is not None and str(split) != "test":
            raise RuntimeError(f"LiveCodeBench requested unsupported split: {split}")

        load_calls.append((dataset_name, str(split) if split is not None else None))
        dataset = datasets_module.Dataset.from_file(str(arrow_file))
        if split is None:
            return datasets_module.DatasetDict({"test": dataset})
        return dataset

    if build_api is None:
        if any(name == "lcb_runner" or name.startswith("lcb_runner.") for name in sys.modules):
            raise RuntimeError(
                "lcb_runner was imported before the offline dataset overlay was installed"
            )
        if str(lcb_dir) not in sys.path:
            sys.path.insert(0, str(lcb_dir))

    original_load_dataset = datasets_module.load_dataset
    original_cwd = Path.cwd()
    original_tqdm_disable = os.environ.get("TQDM_DISABLE")
    datasets_module.load_dataset = offline_load_dataset
    try:
        os.chdir(lcb_dir)
        os.environ["TQDM_DISABLE"] = "1"
        if build_api is None:
            from lcb_runner.runner.scenario_router import build_prompt_benchmark
            from lcb_runner.utils.scenarios import Scenario

            build_api = build_prompt_benchmark
            scenario = Scenario.codegeneration

        build_args = argparse.Namespace(
            scenario=scenario,
            release_version=RELEASE_VERSION,
            subset="code_generation",
            language="python",
            not_fast=False,
            start_date=None,
            end_date=None,
            k=[1],
            num_samples=1,
            timeout=60,
            num_workers=1,
            num_process_evaluate=1,
            model_name="mlperf_deepseek_snapshot",
            output_dir=str(state_root / "build-output"),
            prompt_type="custom",
            continue_existing=False,
            evaluate=True,
        )
        full_benchmark, _ = build_api(build_args)
    finally:
        os.chdir(original_cwd)
        if original_tqdm_disable is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = original_tqdm_disable
        datasets_module.load_dataset = original_load_dataset

    if len(load_calls) != 1:
        raise RuntimeError(
            "Vendored build_prompt_benchmark must make exactly one pinned Arrow load; "
            f"observed {len(load_calls)}"
        )

    problems_by_id: dict[str, Any] = {}
    for problem in full_benchmark:
        question_id = _problem_id(problem)
        if question_id in problems_by_id:
            raise ValueError(f"Duplicate LiveCodeBench question_id: {question_id}")
        problems_by_id[question_id] = problem
    if len(problems_by_id) != EXPECTED_PROBLEM_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_PROBLEM_COUNT} unique release_v1 IDs, "
            f"found {len(problems_by_id)}"
        )
    if CANARY_QUESTION_ID not in problems_by_id:
        raise ValueError(f"release_v1 is missing canary {CANARY_QUESTION_ID}")
    return [problems_by_id[question_id] for question_id in sorted(problems_by_id)]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temp(path: Path, content: bytes) -> Path:
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _publish_snapshot(payload: dict[str, Any], output_file: Path) -> tuple[Path, str]:
    if output_file.name != "release_v1.pkl":
        raise ValueError("Materializer output filename must be release_v1.pkl")
    checksum_file = Path(f"{output_file}.sha256")
    if output_file.exists() or checksum_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite LiveCodeBench snapshot or checksum: {output_file}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    snapshot_bytes = pickle.dumps(payload, protocol=4)
    snapshot_temp = _write_temp(output_file, snapshot_bytes)
    snapshot_sha256 = _sha256_file(snapshot_temp)
    checksum_bytes = f"{snapshot_sha256}  {output_file.name}\n".encode("ascii")
    checksum_temp = _write_temp(checksum_file, checksum_bytes)

    try:
        os.link(snapshot_temp, output_file)
        try:
            os.link(checksum_temp, checksum_file)
        except Exception:
            output_file.unlink(missing_ok=True)
            raise
        _fsync_directory(output_file.parent)
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite LiveCodeBench snapshot or checksum: {output_file}"
        ) from error
    finally:
        snapshot_temp.unlink(missing_ok=True)
        checksum_temp.unlink(missing_ok=True)
    return checksum_file, snapshot_sha256


def materialize_snapshot(
    *,
    arrow_file: Path,
    lcb_dir: Path,
    output_file: Path,
    state_root: Path,
    datasets_module: ModuleType | Any | None = None,
    build_api: Callable[[argparse.Namespace], tuple[Iterable[Any], Any]] | None = None,
    scenario: Any | None = None,
    source_validator: Callable[[Path], str] = _validate_source_checkout,
    network_guard: Callable[[], ContextManager[Any]] = _network_blocked,
    preflight: Callable[..., tuple[str, int]] = _preflight_livecodebench,
) -> dict[str, Any]:
    """Build, atomically publish, and execute-canary-check a release_v1 snapshot."""

    if output_file.exists() or Path(f"{output_file}.sha256").exists():
        raise FileExistsError(f"Refusing to overwrite LiveCodeBench snapshot: {output_file}")
    _validate_environment(arrow_file, state_root)
    _prepare_state_root(state_root)
    revision_before = source_validator(lcb_dir)
    arrow_sha256_before = _sha256_file(arrow_file)
    source_sha256_before = _sha256_tree(lcb_dir / "lcb_runner")

    with network_guard():
        problems = _build_prompt_problems(
            arrow_file,
            lcb_dir,
            state_root,
            datasets_module=datasets_module,
            build_api=build_api,
            scenario=scenario,
        )
        revision_after = source_validator(lcb_dir)
        arrow_sha256_after = _sha256_file(arrow_file)
        source_sha256_after = _sha256_tree(lcb_dir / "lcb_runner")
        if revision_after != revision_before or revision_after != LIVE_CODE_BENCH_REVISION:
            raise RuntimeError("Vendored LiveCodeBench revision changed during materialization")
        if arrow_sha256_after != arrow_sha256_before:
            raise RuntimeError("Offline LiveCodeBench Arrow cache changed during materialization")
        if source_sha256_after != source_sha256_before:
            raise RuntimeError("Vendored LiveCodeBench source changed during materialization")

        payload = {
            "schema_version": 1,
            "release_version": RELEASE_VERSION,
            "arrow_sha256": arrow_sha256_before,
            "cache_fingerprint": CACHE_FINGERPRINT,
            "livecodebench_revision": revision_before,
            "livecodebench_source_sha256": source_sha256_before,
            "problems": problems,
        }
        checksum_file, snapshot_sha256 = _publish_snapshot(payload, output_file)
        try:
            preflight(
                output_file,
                checksum_file,
                lcb_dir,
                allowed_root=output_file.parent,
            )
            if source_validator(lcb_dir) != revision_before:
                raise RuntimeError("Vendored LiveCodeBench revision changed during canary")
            if _sha256_tree(lcb_dir / "lcb_runner") != source_sha256_before:
                raise RuntimeError("Vendored LiveCodeBench source changed during canary")
        except Exception:
            checksum_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)
            _fsync_directory(output_file.parent)
            raise

    return {
        "status": "SUCCESS",
        "release_version": RELEASE_VERSION,
        "problem_count": len(problems),
        "canary_question_id": CANARY_QUESTION_ID,
        "snapshot_path": str(output_file.resolve()),
        "snapshot_sha256": snapshot_sha256,
        "checksum_path": str(checksum_file.resolve()),
        "arrow_path": str(arrow_file.resolve()),
        "arrow_sha256": arrow_sha256_before,
        "cache_fingerprint": CACHE_FINGERPRINT,
        "livecodebench_path": str(lcb_dir.resolve()),
        "livecodebench_revision": revision_before,
        "livecodebench_source_sha256": source_sha256_before,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the offline LiveCodeBench release_v1 snapshot"
    )
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--arrow-file", type=Path, default=DEFAULT_ARROW_FILE)
    parser.add_argument("--livecodebench-dir", type=Path, default=DEFAULT_LCB_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = materialize_snapshot(
            arrow_file=args.arrow_file,
            lcb_dir=args.livecodebench_dir,
            output_file=args.output_file,
            state_root=args.state_root,
        )
    except Exception as error:
        print(
            f"ERROR: LiveCodeBench release_v1 materialization failed closed: {error}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
