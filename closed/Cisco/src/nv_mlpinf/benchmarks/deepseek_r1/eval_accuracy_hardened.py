# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Fail-closed overlay for the MLCommons DeepSeek-R1 accuracy evaluator.

The release_v1 snapshot is a pickle envelope containing ``release_version`` and
``problems``. The problems are the 400 instances returned by LiveCodeBench's
``build_prompt_benchmark``. A sibling ``.sha256`` file pins the snapshot bytes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
import traceback
from types import ModuleType
from typing import Any, Callable, Iterable


RELEASE_VERSION = "release_v1"
EXPECTED_PROBLEM_COUNT = 400
CANARY_QUESTION_ID = "abc306_a"
CANARY_SOLUTION = """\
n = int(input())
s = input().strip()
print(''.join(character * 2 for character in s))
"""
DEFAULT_ASSET_ROOT = Path("/work/build/accuracy_assets")
DEFAULT_LCB_ARTIFACT = (
    DEFAULT_ASSET_ROOT / "deepseek-r1/livecodebench/release_v1.pkl"
)
DEFAULT_LCB_CHECKSUM = Path(f"{DEFAULT_LCB_ARTIFACT}.sha256")

WORKER_CORRECT = "CORRECT"
WORKER_WRONG = "WRONG"
WORKER_ERROR = "ERROR"


class LiveCodeBenchEvaluationError(RuntimeError):
    """Raised when LiveCodeBench cannot produce a trustworthy score."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluator source tree not found: {root}")

    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    if not files:
        raise RuntimeError(f"Evaluator source tree is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must be under {root}: {path}") from error


def _read_expected_sha256(artifact_path: Path, checksum_path: Path) -> str:
    _require_file(checksum_path, "LiveCodeBench checksum")
    lines = [line.strip() for line in checksum_path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Checksum file must contain exactly one entry: {checksum_path}")

    fields = lines[0].split()
    expected = fields[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid SHA256 in {checksum_path}")
    if len(fields) > 1 and fields[1].lstrip("*") != artifact_path.name:
        raise ValueError(
            f"Checksum entry names {fields[1]!r}, expected {artifact_path.name!r}"
        )
    return expected


def _problem_id(problem: Any) -> str:
    if isinstance(problem, dict):
        question_id = problem.get("question_id")
    else:
        question_id = getattr(problem, "question_id", None)
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("Every LiveCodeBench problem must have a non-empty question_id")
    return question_id


def _load_release_snapshot(
    artifact_path: str,
    expected_sha256: str,
    lcb_dir: str,
) -> dict[str, Any]:
    """Load and validate the pinned release without calling datasets.load_dataset."""

    artifact = Path(artifact_path)
    source_dir = Path(lcb_dir)
    _require_file(artifact, "LiveCodeBench release_v1 snapshot")
    if _sha256_file(artifact) != expected_sha256:
        raise ValueError(f"LiveCodeBench snapshot checksum mismatch: {artifact}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"LiveCodeBench source not found: {source_dir}")
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    with artifact.open("rb") as stream:
        payload = pickle.load(stream)

    if not isinstance(payload, dict) or payload.get("release_version") != RELEASE_VERSION:
        raise ValueError(
            f"LiveCodeBench snapshot must declare release_version={RELEASE_VERSION!r}"
        )
    problems = payload.get("problems")
    if isinstance(problems, dict):
        items: Iterable[Any] = problems.values()
    elif isinstance(problems, (list, tuple)):
        items = problems
    else:
        raise ValueError("LiveCodeBench snapshot must contain a problems list or mapping")

    benchmark: dict[str, Any] = {}
    for problem in items:
        question_id = _problem_id(problem)
        if question_id in benchmark:
            raise ValueError(f"Duplicate LiveCodeBench question_id: {question_id}")
        benchmark[question_id] = problem

    if len(benchmark) != EXPECTED_PROBLEM_COUNT:
        raise ValueError(
            "LiveCodeBench release_v1 snapshot must contain exactly "
            f"{EXPECTED_PROBLEM_COUNT} unique IDs; found {len(benchmark)}"
        )
    if CANARY_QUESTION_ID not in benchmark:
        raise ValueError(
            f"LiveCodeBench release_v1 snapshot is missing canary {CANARY_QUESTION_ID}"
        )
    return benchmark


def _grade_livecodebench(
    code: str,
    question_id: str,
    artifact_path: str,
    expected_sha256: str,
    lcb_dir: str,
) -> bool:
    benchmark = _load_release_snapshot(artifact_path, expected_sha256, lcb_dir)
    instance = benchmark.get(question_id)
    if instance is None:
        raise KeyError(f"LiveCodeBench question_id not in release_v1: {question_id}")

    source_dir = Path(lcb_dir)
    original_cwd = Path.cwd()
    original_tqdm_disable = os.environ.get("TQDM_DISABLE")
    with tempfile.TemporaryDirectory(prefix=f"lcb-{question_id}-", dir="/tmp") as temp_dir:
        try:
            os.chdir(source_dir)
            os.environ["TQDM_DISABLE"] = "1"

            from lcb_runner.evaluation import extract_instance_results
            from lcb_runner.runner.scenario_router import (
                get_metrics,
                sort_and_extract_save_results,
            )
            from lcb_runner.utils.scenarios import Scenario

            evaluator_args = argparse.Namespace(
                scenario=Scenario.codegeneration,
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
                model_name="mlperf_deepseek_eval",
                output_dir=temp_dir,
                prompt_type="custom",
                continue_existing=False,
                evaluate=True,
            )
            outputs = [[code]]
            save_results = [instance.insert_output(outputs[0], outputs[0])]
            _, combined_results = sort_and_extract_save_results(
                evaluator_args.scenario, save_results
            )
            _, instance_results, _ = get_metrics(
                evaluator_args.scenario,
                evaluator_args,
                [instance],
                combined_results,
            )
            graded = extract_instance_results(instance_results)
            if not graded or not graded[0]:
                raise RuntimeError(
                    f"LiveCodeBench returned no grade for question_id {question_id}"
                )
            return bool(graded[0][0])
        finally:
            os.chdir(original_cwd)
            if original_tqdm_disable is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = original_tqdm_disable


def _evaluate_livecodebench_worker(
    args: tuple[str, str, str, str, str],
) -> tuple[str, str, str | None]:
    code, question_id, artifact_path, expected_sha256, lcb_dir = args
    try:
        correct = _grade_livecodebench(
            code, question_id, artifact_path, expected_sha256, lcb_dir
        )
    except Exception as error:
        detail = "".join(traceback.format_exception(error, limit=8)).strip()
        return question_id, WORKER_ERROR, detail
    return question_id, WORKER_CORRECT if correct else WORKER_WRONG, None


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(value != value)
    except (TypeError, ValueError):
        return True


def _shutdown_executor(executor: Any, *, wait: bool) -> None:
    try:
        executor.shutdown(wait=wait, cancel_futures=not wait)
    except TypeError:
        executor.shutdown(wait=wait)


def _process_livecodebench_parallel(
    df: Any,
    group_indices: Iterable[Any],
    *,
    artifact_path: Path,
    expected_sha256: str,
    lcb_dir: Path,
    executor_factory: Callable[..., Any] = ProcessPoolExecutor,
) -> tuple[int, int]:
    """Grade LiveCodeBench rows, aborting the whole evaluator on any ERROR."""

    benchmark = _load_release_snapshot(
        str(artifact_path), expected_sha256, str(lcb_dir)
    )
    work_items = []
    for index in group_indices:
        row = df.loc[index]
        code = row.get("extracted_answer")
        question_id = row.get("ground_truth")
        if _is_present(code) and _is_present(question_id):
            work_items.append((index, str(code), str(question_id)))

    missing_ids = sorted(
        {question_id for _, _, question_id in work_items} - benchmark.keys()
    )
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise LiveCodeBenchEvaluationError(
            f"ERROR: {len(missing_ids)} requested LiveCodeBench IDs are absent from "
            f"the pinned {RELEASE_VERSION} snapshot: {preview}"
        )
    if not work_items:
        return 0, 0

    max_workers = min(multiprocessing.cpu_count(), len(work_items))
    executor = executor_factory(max_workers=max_workers)
    future_to_item = {
        executor.submit(
            _evaluate_livecodebench_worker,
            (code, question_id, str(artifact_path), expected_sha256, str(lcb_dir)),
        ): (index, question_id)
        for index, code, question_id in work_items
    }

    correct_count = 0
    total_evaluated = 0
    try:
        for future in as_completed(future_to_item, timeout=1200):
            index, expected_question_id = future_to_item[future]
            try:
                question_id, status, detail = future.result()
            except Exception as error:
                raise LiveCodeBenchEvaluationError(
                    f"ERROR: LiveCodeBench worker crashed for row {index}: {error}"
                ) from error
            if question_id != expected_question_id:
                raise LiveCodeBenchEvaluationError(
                    "ERROR: LiveCodeBench worker returned a mismatched question_id: "
                    f"expected {expected_question_id}, got {question_id}"
                )
            if status == WORKER_ERROR:
                raise LiveCodeBenchEvaluationError(
                    f"ERROR: LiveCodeBench worker failed for row {index} "
                    f"({question_id}): {detail}"
                )
            if status not in {WORKER_CORRECT, WORKER_WRONG}:
                raise LiveCodeBenchEvaluationError(
                    f"ERROR: LiveCodeBench worker returned invalid status {status!r}"
                )

            is_correct = status == WORKER_CORRECT
            df.at[index, "prompt_accuracy"] = 100.0 if is_correct else 0.0
            total_evaluated += 1
            correct_count += int(is_correct)
    except LiveCodeBenchEvaluationError:
        for future in future_to_item:
            future.cancel()
        _shutdown_executor(executor, wait=False)
        raise
    except Exception as error:
        for future in future_to_item:
            future.cancel()
        _shutdown_executor(executor, wait=False)
        raise LiveCodeBenchEvaluationError(
            f"ERROR: LiveCodeBench worker infrastructure failed: {error}"
        ) from error
    else:
        _shutdown_executor(executor, wait=True)

    return correct_count, total_evaluated


def _preflight_livecodebench(
    artifact_path: Path,
    checksum_path: Path,
    lcb_dir: Path,
    *,
    allowed_root: Path,
    executor_factory: Callable[..., Any] = ProcessPoolExecutor,
) -> tuple[str, int]:
    _require_under(artifact_path, allowed_root, "LiveCodeBench snapshot")
    _require_under(checksum_path, allowed_root, "LiveCodeBench checksum")
    if checksum_path != Path(f"{artifact_path}.sha256"):
        raise ValueError(
            f"LiveCodeBench checksum must be the snapshot sidecar {artifact_path}.sha256"
        )

    expected_sha256 = _read_expected_sha256(artifact_path, checksum_path)
    benchmark = _load_release_snapshot(
        str(artifact_path), expected_sha256, str(lcb_dir)
    )

    executor = executor_factory(max_workers=1)
    future = executor.submit(
        _evaluate_livecodebench_worker,
        (
            CANARY_SOLUTION,
            CANARY_QUESTION_ID,
            str(artifact_path),
            expected_sha256,
            str(lcb_dir),
        ),
    )
    try:
        question_id, status, detail = future.result(timeout=180)
    except Exception as error:
        _shutdown_executor(executor, wait=False)
        raise LiveCodeBenchEvaluationError(
            f"ERROR: LiveCodeBench canary worker crashed: {error}"
        ) from error
    _shutdown_executor(executor, wait=True)

    if question_id != CANARY_QUESTION_ID or status != WORKER_CORRECT:
        raise LiveCodeBenchEvaluationError(
            f"ERROR: known-correct {CANARY_QUESTION_ID} canary failed with "
            f"status={status}: {detail}"
        )
    return expected_sha256, len(benchmark)


def _load_upstream_evaluator(path: Path) -> ModuleType:
    _require_file(path, "Upstream DeepSeek-R1 evaluator")
    module_name = "_mlcommons_deepseek_r1_eval_accuracy"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import upstream evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _publish_no_overwrite(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite evaluator output: {destination}") from error


def run_evaluation(
    *,
    input_file: Path,
    dataset_file: Path,
    output_file: Path,
    checkpoint_path: Path,
    upstream_evaluator: Path,
    livecodebench_artifact: Path,
    livecodebench_checksum: Path,
    allowed_data_root: Path = DEFAULT_ASSET_ROOT,
) -> Path:
    """Evaluate an MLPerf accuracy log and publish the scored output."""

    for path, label in (
        (input_file, "MLPerf accuracy log"),
        (dataset_file, "DeepSeek-R1 evaluation dataset"),
        (upstream_evaluator, "Upstream DeepSeek-R1 evaluator"),
    ):
        _require_file(path, label)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Pinned DeepSeek-R1 tokenizer not found: {checkpoint_path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite re-evaluation output: {output_file}"
        )
    input_resolved = input_file.resolve()
    if input_resolved == output_file.resolve():
        raise ValueError("Evaluator output must not overwrite the MLPerf accuracy log")

    input_sha256 = _sha256_file(input_file)
    dataset_sha256 = _sha256_file(dataset_file)
    lcb_dir = upstream_evaluator.parent / "submodules" / "LiveCodeBench"
    lcb_sha256, _ = _preflight_livecodebench(
        livecodebench_artifact,
        livecodebench_checksum,
        lcb_dir,
        allowed_root=allowed_data_root,
    )

    upstream = _load_upstream_evaluator(upstream_evaluator)
    upstream.process_livecodebench_parallel = lambda df, group_indices: (
        _process_livecodebench_parallel(
            df,
            group_indices,
            artifact_path=livecodebench_artifact,
            expected_sha256=lcb_sha256,
            lcb_dir=lcb_dir,
        )
    )

    with tempfile.TemporaryDirectory(
        prefix=".deepseek-r1-reeval-", dir=output_file.parent
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        evaluated_df, saved_file_path = upstream.process_mlperf_log_accuracy(
            mlperf_log_file=input_file,
            dataset_file=dataset_file,
            checkpoint_path=str(checkpoint_path),
            output_dir=temp_dir,
            base_filename="deepseek-r1_evaluated.pkl",
        )
        saved_file = Path(saved_file_path)
        _require_file(saved_file, "Evaluated DeepSeek-R1 output")

        if _sha256_file(input_file) != input_sha256:
            raise RuntimeError("MLPerf accuracy log changed during evaluation")
        if _sha256_file(dataset_file) != dataset_sha256:
            raise RuntimeError("DeepSeek-R1 evaluation dataset changed during re-evaluation")

        _publish_no_overwrite(saved_file, output_file)

    upstream.print_evaluation_results(evaluated_df, upstream.logger)
    return output_file


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed DeepSeek-R1 MLPerf accuracy evaluator"
    )
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--dataset-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--upstream-evaluator", required=True, type=Path)
    parser.add_argument(
        "--livecodebench-artifact", type=Path, default=DEFAULT_LCB_ARTIFACT
    )
    parser.add_argument(
        "--livecodebench-checksum", type=Path, default=DEFAULT_LCB_CHECKSUM
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_evaluation(
            input_file=args.input_file,
            dataset_file=args.dataset_file,
            output_file=args.output_file,
            checkpoint_path=args.checkpoint_path,
            upstream_evaluator=args.upstream_evaluator,
            livecodebench_artifact=args.livecodebench_artifact,
            livecodebench_checksum=args.livecodebench_checksum,
        )
    except Exception as error:
        print(
            f"ERROR: DeepSeek-R1 accuracy evaluation failed closed: {error}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
