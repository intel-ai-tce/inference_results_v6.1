#!/usr/bin/env python3
"""Check DeepSeek-R1 AccuracyOnly artifacts and accuracy thresholds."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


MIN_EXACT_MATCH = 80.544618
MIN_TOKENS_PER_SAMPLE = 3497.60466
MAX_TOKENS_PER_SAMPLE = 4274.85014
MIN_SAMPLES = 4_388
REQUIRED_FILES = (
    "accuracy.txt",
    "metadata.json",
    "mlperf_log_accuracy.json",
    "mlperf_log_detail.txt",
    "mlperf_log_summary.txt",
)
MLLOG_PREFIX = ":::MLLOG "
SCENARIOS = ("Offline", "Server", "Interactive")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        type=Path,
        help="AccuracyOnly result directory or its accuracy.txt",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        help="Expected LoadGen scenario (defaults to the result directory name)",
    )
    return parser.parse_args()


def _load_metrics(accuracy_log: Path) -> dict[str, Any] | None:
    required_metrics = {"exact_match", "tokens_per_sample", "num-samples"}
    for line in reversed(accuracy_log.read_text(encoding="utf-8").splitlines()):
        try:
            record = ast.literal_eval(line.strip())
        except (SyntaxError, ValueError):
            continue
        if isinstance(record, dict) and required_metrics <= record.keys():
            return record
    return None


def _load_mllog_events(detail_log: Path) -> dict[str, Any]:
    events: dict[str, Any] = {}
    for line in detail_log.read_text(encoding="utf-8").splitlines():
        if not line.startswith(MLLOG_PREFIX):
            continue
        record = json.loads(line[len(MLLOG_PREFIX):])
        events[record["key"]] = record.get("value")
    return events


def main() -> int:
    args = _parse_args()
    result_dir = args.result if args.result.is_dir() else args.result.parent
    accuracy_log = result_dir / "accuracy.txt"
    raw_accuracy_log = result_dir / "mlperf_log_accuracy.json"
    detail_log = result_dir / "mlperf_log_detail.txt"
    metadata_file = result_dir / "metadata.json"

    missing_files = [name for name in REQUIRED_FILES if not (result_dir / name).is_file()]
    if missing_files:
        print("INCOMPLETE: required result files are missing: " + ", ".join(missing_files))
        return 2
    if raw_accuracy_log.stat().st_size == 0:
        print(f"INCOMPLETE: raw accuracy JSON is empty: {raw_accuracy_log}")
        return 2

    expected_scenario = args.scenario or result_dir.name
    if expected_scenario not in SCENARIOS:
        print(
            "INCOMPLETE: cannot infer the expected scenario from result directory "
            f"{result_dir}; pass --scenario"
        )
        return 2

    try:
        events = _load_mllog_events(detail_log)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError) as error:
        print(f"INCOMPLETE: malformed MLLOG event in {detail_log}: {error}")
        return 2

    required_event_keys = {"effective_scenario", "effective_test_mode"}
    missing_event_keys = sorted(required_event_keys - events.keys())
    if missing_event_keys:
        print(
            "INCOMPLETE: LoadGen has not written required result keys: "
            + ", ".join(missing_event_keys)
        )
        return 2

    metrics = _load_metrics(accuracy_log)
    if metrics is None:
        print(f"INCOMPLETE: DeepSeek accuracy metrics were not found in {accuracy_log}")
        return 2

    try:
        exact_match = float(metrics["exact_match"])
        tokens_per_sample = float(metrics["tokens_per_sample"])
        num_samples = int(metrics["num-samples"])
    except (TypeError, ValueError):
        print(f"INCOMPLETE: DeepSeek accuracy metrics are malformed in {accuracy_log}")
        return 2

    expected_effective_scenario = "Server" if expected_scenario == "Interactive" else expected_scenario
    checks = {
        "metadata_scenario": metadata.get("scenario") == expected_scenario,
        "metadata_accuracy_only_mode": metadata.get("test_mode") == "AccuracyOnly",
        "expected_effective_scenario": (
            events["effective_scenario"] == expected_effective_scenario
        ),
        "effective_accuracy_only_mode": events["effective_test_mode"] == "AccuracyOnly",
        "exact_match": exact_match >= MIN_EXACT_MATCH,
        "tokens_per_sample_lower_bound": tokens_per_sample >= MIN_TOKENS_PER_SAMPLE,
        "tokens_per_sample_upper_bound": tokens_per_sample <= MAX_TOKENS_PER_SAMPLE,
        "minimum_samples": num_samples >= MIN_SAMPLES,
    }

    print(f"accuracy_log={accuracy_log}")
    print(f"expected_scenario={expected_scenario}")
    print(f"expected_effective_scenario={expected_effective_scenario}")
    print(f"effective_scenario={events['effective_scenario']}")
    print(f"effective_test_mode={events['effective_test_mode']}")
    print(f"exact_match={exact_match:.6f}")
    print(f"tokens_per_sample={tokens_per_sample:.6f}")
    print(f"num_samples={num_samples}")
    for name, passed in checks.items():
        print(f"check.{name}={'PASS' if passed else 'FAIL'}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("FAIL: " + ", ".join(failed))
        return 1

    print("PASS: DeepSeek-R1 AccuracyOnly result satisfies the submission accuracy gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
