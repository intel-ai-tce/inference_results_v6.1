#!/usr/bin/env python3
"""Check DeepSeek-R1 Interactive PerformanceOnly validity and latency gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MLLOG_PREFIX = ":::MLLOG "
TTFT_99_KEY = "result_first_token_99.00_percentile_latency_ns"
TPOT_99_KEY = "result_time_per_output_token_99.00_percentile_ns"
SCORE_KEY = "result_completed_tokens_per_second"
MAX_TTFT_NS = 1_500_000_000
MAX_TPOT_NS = 15_000_000
MIN_DURATION_MS = 600_000
MIN_QUERY_COUNT = 144_000
MIN_PERFORMANCE_SAMPLES = 4_388
EXPECTED_SEEDS = {
    "effective_qsl_rng_seed": 2085463073848966840,
    "effective_sample_index_rng_seed": 2747215439041700203,
    "effective_schedule_rng_seed": 16159082839903944936,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        type=Path,
        help="Interactive PerformanceOnly result directory or its detail log",
    )
    return parser.parse_args()


def _load_events(detail_log: Path) -> dict[str, Any]:
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
    detail_log = result_dir / "mlperf_log_detail.txt"
    metadata_file = result_dir / "metadata.json"

    missing_files = [
        str(path)
        for path in (detail_log, metadata_file)
        if not path.is_file()
    ]
    if missing_files:
        print("INCOMPLETE: required result files are missing: " + ", ".join(missing_files))
        return 2

    try:
        events = _load_events(detail_log)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError) as error:
        print(f"INCOMPLETE: malformed result data: {error}")
        return 2

    required_keys = {
        "effective_scenario",
        "effective_test_mode",
        "result_validity",
        "result_min_duration_met",
        "result_min_queries_met",
        "early_stopping_met",
        "requested_use_token_latencies",
        "effective_min_duration_ms",
        "effective_min_query_count",
        "effective_performance_sample_count",
        SCORE_KEY,
        TTFT_99_KEY,
        TPOT_99_KEY,
        *EXPECTED_SEEDS,
    }
    missing_keys = sorted(required_keys - events.keys())
    if missing_keys:
        print(
            "INCOMPLETE: LoadGen has not written required result keys: "
            + ", ".join(missing_keys)
        )
        return 2

    try:
        completed_tokens_per_second = float(events[SCORE_KEY])
        ttft_99_ns = float(events[TTFT_99_KEY])
        tpot_99_ns = float(events[TPOT_99_KEY])
    except (TypeError, ValueError):
        print("INCOMPLETE: LoadGen score or latency values are malformed")
        return 2

    checks = {
        "interactive_result_directory": result_dir.name == "Interactive",
        "metadata_interactive_scenario": metadata.get("scenario") == "Interactive",
        "metadata_performance_only_mode": metadata.get("test_mode") == "PerformanceOnly",
        # Interactive is temporarily represented as Server inside LoadGen.
        "effective_loadgen_server_scenario": events["effective_scenario"] == "Server",
        "effective_performance_only_mode": events["effective_test_mode"] == "PerformanceOnly",
        "result_validity_is_VALID": events["result_validity"] == "VALID",
        "minimum_duration_met": events["result_min_duration_met"] is True,
        "minimum_queries_met": events["result_min_queries_met"] is True,
        "early_stopping_met": events["early_stopping_met"] is True,
        "positive_finite_score": (
            math.isfinite(completed_tokens_per_second) and completed_tokens_per_second > 0
        ),
        "token_latencies_enabled": events["requested_use_token_latencies"] is True,
        "configured_min_duration": events["effective_min_duration_ms"] >= MIN_DURATION_MS,
        "configured_min_queries": events["effective_min_query_count"] >= MIN_QUERY_COUNT,
        "performance_sample_set": (
            events["effective_performance_sample_count"] >= MIN_PERFORMANCE_SAMPLES
        ),
        "fixed_mlperf_seeds": all(events[key] == value for key, value in EXPECTED_SEEDS.items()),
        "ttft_99_below_1500_ms": (
            math.isfinite(ttft_99_ns) and 0 <= ttft_99_ns < MAX_TTFT_NS
        ),
        "tpot_99_below_15_ms": (
            math.isfinite(tpot_99_ns) and 0 <= tpot_99_ns < MAX_TPOT_NS
        ),
    }

    print(f"detail_log={detail_log}")
    print(f"official_result_completed_tokens_per_second={completed_tokens_per_second:.6f}")
    print(f"result_first_token_99_percentile_latency_ms={ttft_99_ns / 1_000_000:.6f}")
    print(f"result_time_per_output_token_99_percentile_ms={tpot_99_ns / 1_000_000:.6f}")
    for name, passed in checks.items():
        print(f"check.{name}={'PASS' if passed else 'FAIL'}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("FAIL: " + ", ".join(failed))
        return 1

    print("PASS: Interactive PerformanceOnly result satisfies all submission gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
