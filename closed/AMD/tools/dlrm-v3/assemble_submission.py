#!/usr/bin/env python3
"""Assemble a runner-staged q12,200 DLRM-v3 payload into a final MLPerf tree.

This script does not modify the upstream mlcommons/inference repository. It
creates a separate output directory with the AMD-style layout:

    <output>/closed/AMD/{documentation,results,setup,src,systems,tools}

If an mlcommons/inference checkout is supplied, it can also run the official
accuracy-log truncator and submission checker against the generated tree.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SYSTEM_NAME = "8xMI355X_2xEPYC_9575F"
BENCHMARK = "dlrm-v3"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing source tree: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        ),
    )


def copy_file_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_log_set(src_dir: Path, dst_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in (
        "mlperf_log_summary.txt",
        "mlperf_log_detail.txt",
        "mlperf_log_accuracy.json",
        "mlperf_log_trace.json",
    ):
        if copy_file_if_exists(src_dir / name, dst_dir / name):
            copied.append(name)
    return copied


def write_measurements(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_data_types": "int64,fp16,bf16,fp8",
        "retraining": "No",
        "starting_weights_filename": "MLCommons DLRM-v3 checkpoint",
        "weight_data_types": "fp16,bf16,fp8",
        "weight_transformations": "quantization, fusion",
    }
    path.write_text(json.dumps(payload, indent=4, sort_keys=False) + "\n")


def write_scenario_readme(path: Path, scenario: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Measurement - DLRM-v3 {scenario}\n\n"
        "Please proceed to the setup section (`AMD/setup/dlrm-v3`) for q12,200 "
        "DLRM-v3 ROCm build/run instructions and to the source section "
        "(`AMD/src/dlrm-v3`) for benchmark implementation details.\n"
    )


def latest_artifact(root: Path, prefix: str) -> Path:
    matches = sorted(root.glob(f"{prefix}*"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return matches[-1] if matches else root / prefix


def copy_user_conf(staging: Path, dst_dir: Path) -> None:
    cfg = staging / "src/dlrm-v3/harness/benchmarks/user_mi355x8_nve_b64_qps12200_PROD10min.conf"
    copy_file_if_exists(cfg, dst_dir / "user.conf")


def assemble_results(args: argparse.Namespace, amd: Path) -> None:
    results_root = amd / "results" / SYSTEM_NAME / BENCHMARK
    server = results_root / "Server"
    offline = results_root / "Offline"

    # Server PerformanceOnly
    perf_run = server / "performance" / "run_1"
    copied = copy_log_set(args.server_perf_artifact, perf_run)
    if not copied:
        raise FileNotFoundError(f"no server performance logs copied from {args.server_perf_artifact}")
    write_measurements(server / "measurements.json")
    write_scenario_readme(server / "README.md", "Server")
    copy_user_conf(args.staging, server)

    # Server AccuracyOnly. v6.1 checker expects accuracy logs for each submitted
    # scenario, not just Offline GAUC.
    server_acc = server / "accuracy"
    copied = copy_log_set(args.server_accuracy_artifact, server_acc)
    if not copied:
        raise FileNotFoundError(f"no server accuracy logs copied from {args.server_accuracy_artifact}")
    copy_file_if_exists(args.server_accuracy_artifact / "accuracy_metrics.txt", server_acc / "accuracy.txt")

    # Offline PerformanceOnly, when available.
    if args.offline_perf_artifact.exists():
        offline_perf = offline / "performance" / "run_1"
        copied = copy_log_set(args.offline_perf_artifact, offline_perf)
        if copied:
            print(f"[assemble] copied Offline performance logs from {args.offline_perf_artifact}")
    else:
        print(f"[assemble] Offline performance artifact not found: {args.offline_perf_artifact}")

    # Offline AccuracyOnly / GAUC
    acc_dir = offline / "accuracy"
    copied = copy_log_set(args.offline_accuracy_artifact, acc_dir)
    if not copied:
        raise FileNotFoundError(f"no offline accuracy logs copied from {args.offline_accuracy_artifact}")
    if copy_file_if_exists(args.offline_accuracy_artifact / "accuracy_metrics.txt", acc_dir / "accuracy.txt"):
        pass
    write_measurements(offline / "measurements.json")
    write_scenario_readme(offline / "README.md", "Offline")
    copy_user_conf(args.staging, offline)

    # TEST08 compliance. Public v6.0/v6.1 submissions store only the verifier
    # result under each scenario's TEST08 directory. Keep the large reference
    # and sampled audit accuracy logs out of the final tree.
    server_test = server / "TEST08"
    copy_file_if_exists(args.test08_audit_artifact / "verify_accuracy.txt", server_test / "verify_accuracy.txt")

    offline_test = offline / "TEST08"
    copy_file_if_exists(args.test08_audit_artifact / "verify_accuracy.txt", offline_test / "verify_accuracy.txt")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("[assemble]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def maybe_run_mlcommons_tools(args: argparse.Namespace, output: Path) -> None:
    if not args.mlcommons_inference_root:
        return
    mlc = args.mlcommons_inference_root
    if not mlc.exists():
        raise FileNotFoundError(f"MLCommons inference checkout not found: {mlc}")
    if args.run_truncate:
        if args.truncated_output.exists():
            shutil.rmtree(args.truncated_output)
        run_cmd(
            [
                sys.executable,
                str(mlc / "tools/submission/truncate_accuracy_log.py"),
                "--input",
                str(output),
                "--output",
                str(args.truncated_output),
                "--submitter",
                args.submitter,
            ]
        )
    checker_input = args.truncated_output if args.run_truncate else output
    if args.run_checker:
        run_cmd(
            [
                sys.executable,
                str(mlc / "tools/submission/submission_checker/main.py"),
                "--input",
                str(checker_input),
                "--submitter",
                args.submitter,
                "--version",
                args.checker_version,
                "--csv",
                str(args.checker_csv),
            ],
            cwd=mlc.parent,
        )


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_artifacts = root / "artifacts"
    if not default_artifacts.exists():
        sibling_runner = root.parent / "dlrm-v3-rocm-runner" / "artifacts"
        if sibling_runner.exists():
            default_artifacts = sibling_runner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output submission root directory")
    parser.add_argument("--staging", type=Path, default=root / "submission", help="Runner-owned staging directory")
    parser.add_argument("--artifacts-root", type=Path, default=default_artifacts)
    parser.add_argument("--submitter", default="AMD")
    parser.add_argument("--checker-version", default="v6.1")
    parser.add_argument("--server-perf-artifact", type=Path, default=None)
    parser.add_argument("--server-accuracy-artifact", type=Path, default=None)
    parser.add_argument("--offline-perf-artifact", type=Path, default=None)
    parser.add_argument("--offline-accuracy-artifact", type=Path, default=None)
    parser.add_argument("--test08-ref-artifact", type=Path, default=None)
    parser.add_argument("--test08-audit-artifact", type=Path, default=None)
    parser.add_argument("--mlcommons-inference-root", type=Path, default=None)
    parser.add_argument("--run-truncate", action="store_true")
    parser.add_argument("--run-checker", action="store_true")
    parser.add_argument("--truncated-output", type=Path, default=Path("submission_truncated"))
    parser.add_argument("--checker-csv", type=Path, default=Path("submission_checker.csv"))
    parser.add_argument("--clean", action="store_true", help="Remove output directory before assembling")
    parser.add_argument("--skip-results", action="store_true", help="Only assemble code/setup/src/systems/tools, not result logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.staging = args.staging.resolve()
    args.artifacts_root = args.artifacts_root.resolve()
    args.server_perf_artifact = (
        args.server_perf_artifact
        or args.artifacts_root / "gold_prod_deg5_q12200_PROD10min_20260716T093104"
    ).resolve()
    args.server_accuracy_artifact = (
        args.server_accuracy_artifact
        or latest_artifact(args.artifacts_root, "gold_formal_q12200_server_accuracy")
    ).resolve()
    args.offline_accuracy_artifact = (
        args.offline_accuracy_artifact
        or args.artifacts_root / "gold_acc_prod_deg5_20260716T054951"
    ).resolve()
    args.offline_perf_artifact = (
        args.offline_perf_artifact
        or latest_artifact(args.artifacts_root, "gold_offline_q12200_PROD10min")
    ).resolve()
    args.test08_ref_artifact = (
        args.test08_ref_artifact
        or args.artifacts_root / "gold_test08_q12200_deg5_ref_offline_acc_20260716T095752"
    ).resolve()
    args.test08_audit_artifact = (
        args.test08_audit_artifact
        or args.artifacts_root / "gold_test08_q12200_deg5_srv_perf_audit_20260716T100434"
    ).resolve()
    output = args.output.resolve()
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    amd = output / "closed" / "AMD"
    for subdir in ("documentation", "setup", "src", "systems", "tools"):
        copy_tree(args.staging / subdir, amd / subdir)
    if args.skip_results:
        print("[assemble] skipped result log copy")
    else:
        assemble_results(args, amd)
    maybe_run_mlcommons_tools(args, output)

    print(f"[assemble] wrote {output}")
    print(f"[assemble] results: {amd / 'results' / SYSTEM_NAME / BENCHMARK}")


if __name__ == "__main__":
    main()
