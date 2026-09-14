#!/usr/bin/env python3
"""Create a clean DeepSeek-R1 Offline+Server MLPerf submission staging tree.

This deliberately stages only the artifacts selected on the command line.  It
does not choose a "best" result automatically, and it never modifies the raw
SFlow outputs or the existing build/artifacts tree.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


SYSTEM = "GB200-NVL72_GB200-186GB_aarch64x72_TRT"
BENCHMARK = "deepseek-r1"
PERFORMANCE_FILES = (
    "mlperf_log_detail.txt",
    "mlperf_log_summary.txt",
    "mlperf_log_accuracy.json",
)
ACCURACY_FILES = (
    "accuracy.txt",
    "mlperf_log_accuracy.json",
    "mlperf_log_detail.txt",
    "mlperf_log_summary.txt",
)
MEASUREMENT_FILES = (
    "README.md",
    "calibration_process.adoc",
    "measurements.json",
    "mlperf.conf",
    "user.conf",
)


def existing_directory(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return path


def copy_required_files(source: Path, destination: Path, filenames: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    missing = [name for name in filenames if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source} is missing required files: {', '.join(missing)}")
    for name in filenames:
        shutil.copy2(source / name, destination / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-root",
        required=True,
        type=Path,
        help="New, empty staging root. The script refuses to overwrite it.",
    )
    parser.add_argument(
        "--offline-performance",
        required=True,
        type=existing_directory,
        help="Offline PerformanceOnly result directory for the selected score.",
    )
    parser.add_argument(
        "--offline-accuracy",
        required=True,
        type=existing_directory,
        help="Matching Offline AccuracyOnly result directory.",
    )
    parser.add_argument(
        "--offline-test06",
        required=True,
        type=existing_directory,
        help="Matching Offline PerformanceOnly result directory containing TEST06/.",
    )
    parser.add_argument(
        "--offline-config",
        required=True,
        type=existing_directory,
        help="Generated Offline loadgen-config directory for the selected score.",
    )
    parser.add_argument(
        "--server-source",
        required=True,
        type=existing_directory,
        help="Already curated, internally aligned Server scenario directory.",
    )
    parser.add_argument("--submitter", default="NVIDIA")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_root = args.stage_root.resolve()
    if stage_root.exists():
        print(f"Refusing to overwrite existing stage root: {stage_root}", file=sys.stderr)
        return 2

    scenario_root = (
        stage_root
        / "closed"
        / args.submitter
        / "results"
        / SYSTEM
        / BENCHMARK
    )
    offline_destination = scenario_root / "Offline"
    server_destination = scenario_root / "Server"

    try:
        copy_required_files(
            args.offline_performance,
            offline_destination / "performance" / "run_1",
            PERFORMANCE_FILES,
        )
        copy_required_files(args.offline_accuracy, offline_destination / "accuracy", ACCURACY_FILES)
        copy_required_files(args.offline_config, offline_destination, MEASUREMENT_FILES)

        test06_source = args.offline_test06 / "TEST06"
        copy_required_files(
            test06_source,
            offline_destination / "TEST06",
            ("verify_accuracy.txt",),
        )
        copy_required_files(
            test06_source / "accuracy",
            offline_destination / "TEST06" / "accuracy",
            ("mlperf_log_accuracy.json",),
        )

        shutil.copytree(args.server_source, server_destination)
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise

    manifest = {
        "submitter": args.submitter,
        "system": SYSTEM,
        "benchmark": BENCHMARK,
        "scenarios": ["Offline", "Server"],
        "offline_performance_source": str(args.offline_performance),
        "offline_accuracy_source": str(args.offline_accuracy),
        "offline_test06_source": str(args.offline_test06),
        "offline_config_source": str(args.offline_config),
        "server_source": str(args.server_source),
    }
    (stage_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created clean submission staging tree: {stage_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
