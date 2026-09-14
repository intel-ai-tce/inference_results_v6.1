#!/usr/bin/env python3
"""Check DeepSeek-R1 TEST06 artifacts, metadata, and scenario-specific gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MLLOG_PREFIX = ":::MLLOG "
SCENARIOS = ("Offline", "Server", "Interactive")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        type=Path,
        help="TEST06 PerformanceOnly scenario result directory",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        help="Expected scenario (defaults to the result directory name)",
    )
    return parser.parse_args()


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
    result_dir = args.result
    expected_scenario = args.scenario or result_dir.name
    if expected_scenario not in SCENARIOS:
        print(
            "INCOMPLETE: cannot infer the expected scenario from result directory "
            f"{result_dir}; pass --scenario"
        )
        return 2

    paths = {
        "metadata": result_dir / "metadata.json",
        "detail": result_dir / "mlperf_log_detail.txt",
        "verification": result_dir / "TEST06" / "verify_accuracy.txt",
        "accuracy_json": result_dir / "TEST06" / "accuracy" / "mlperf_log_accuracy.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        print("INCOMPLETE: required TEST06 files are missing: " + ", ".join(missing))
        return 2
    if paths["accuracy_json"].stat().st_size == 0:
        print(f"INCOMPLETE: TEST06 accuracy JSON is empty: {paths['accuracy_json']}")
        return 2

    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        events = _load_mllog_events(paths["detail"])
    except (json.JSONDecodeError, KeyError) as error:
        print(f"INCOMPLETE: malformed JSON or MLLOG event: {error}")
        return 2

    required_events = {"effective_scenario", "effective_test_mode"}
    missing_events = sorted(required_events - events.keys())
    if missing_events:
        print(
            "INCOMPLETE: LoadGen has not written required result keys: "
            + ", ".join(missing_events)
        )
        return 2

    verification = paths["verification"].read_text(encoding="utf-8")
    expected_effective_scenario = "Server" if expected_scenario == "Interactive" else expected_scenario
    first_token_true = "First token check pass: True" in verification
    first_token_skipped = "First token check pass: Skipped" in verification
    first_token_pass = first_token_true
    if expected_scenario == "Offline":
        first_token_pass = first_token_pass or first_token_skipped

    checks = {
        "metadata_scenario": metadata.get("scenario") == expected_scenario,
        "metadata_performance_only_mode": metadata.get("test_mode") == "PerformanceOnly",
        "metadata_audit_test_TEST06": metadata.get("audit_test") == "TEST06",
        "metadata_audit_result_PASS": metadata.get("audit_result") == "TEST06_PASS",
        "metadata_audit_success": metadata.get("audit_success") is True,
        "effective_scenario": events["effective_scenario"] == expected_effective_scenario,
        "effective_performance_only_mode": events["effective_test_mode"] == "PerformanceOnly",
        "scenario_appropriate_first_token": first_token_pass,
        "eos": "EOS check pass: True" in verification,
        "sample_length": "Sample length check pass: True" in verification,
        "verification_complete": "TEST06 verification complete" in verification,
    }

    print(f"result_dir={result_dir}")
    print(f"expected_scenario={expected_scenario}")
    print(f"expected_effective_scenario={expected_effective_scenario}")
    print(f"first_token_true={first_token_true}")
    print(f"first_token_skipped={first_token_skipped}")
    for name, passed in checks.items():
        print(f"check.{name}={'PASS' if passed else 'FAIL'}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("FAIL: " + ", ".join(failed))
        return 1

    print("PASS: DeepSeek-R1 TEST06 result satisfies the compliance gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
