#!/usr/bin/env python3
"""Check a DeepSeek-R1 Offline LoadGen result and compare its sizing target."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MLLOG_PREFIX = ":::MLLOG "
EXPECTED_SEEDS = {
    "effective_qsl_rng_seed": 2085463073848966840,
    "effective_sample_index_rng_seed": 2747215439041700203,
    "effective_schedule_rng_seed": 16159082839903944936,
}


def _load_events(detail_log: Path) -> dict[str, Any]:
    events: dict[str, Any] = {}
    for line in detail_log.read_text(encoding="utf-8").splitlines():
        if not line.startswith(MLLOG_PREFIX):
            continue
        record = json.loads(line[len(MLLOG_PREFIX):])
        events[record["key"]] = record.get("value")
    return events


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        type=Path,
        help="Offline result directory or its mlperf_log_detail.txt",
    )
    parser.add_argument("--expected-samples-per-second", type=float, default=135.0)
    parser.add_argument("--min-duration-ms", type=int, default=600_000)
    parser.add_argument("--min-samples", type=int, default=631_872)
    parser.add_argument("--min-performance-samples", type=int, default=4_388)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    detail_log = args.result
    if detail_log.is_dir():
        detail_log = detail_log / "mlperf_log_detail.txt"
    if not detail_log.is_file():
        raise FileNotFoundError(detail_log)

    events = _load_events(detail_log)
    final_keys = {
        "result_validity",
        "result_min_duration_met",
        "result_min_queries_met",
        "result_samples_per_second",
        "result_tokens_per_second",
        "result_query_count",
        "generated_query_count",
        "num_errors",
    }
    missing_final = sorted(final_keys - events.keys())
    if missing_final:
        print("INCOMPLETE: LoadGen has not written final result keys: " + ", ".join(missing_final))
        return 2

    try:
        samples_per_second = float(events["result_samples_per_second"])
        tokens_per_second = float(events["result_tokens_per_second"])
    except (TypeError, ValueError):
        print("INCOMPLETE: LoadGen score values are malformed")
        return 2

    checks = {
        "result_validity_is_VALID": events["result_validity"] == "VALID",
        "minimum_duration_met": events["result_min_duration_met"] is True,
        "minimum_queries_met": events["result_min_queries_met"] is True,
        "single_offline_query": (
            events["result_query_count"] == 1 and events["generated_query_count"] == 1
        ),
        "positive_finite_scores": (
            math.isfinite(samples_per_second)
            and samples_per_second > 0
            and math.isfinite(tokens_per_second)
            and tokens_per_second > 0
        ),
        "no_loadgen_errors": events["num_errors"] == 0,
        "offline_scenario": events.get("effective_scenario") == "Offline",
        "performance_only_mode": events.get("effective_test_mode") == "PerformanceOnly",
        "token_latencies_enabled": events.get("requested_use_token_latencies") is True,
        "configured_min_duration": events.get("effective_min_duration_ms", 0) >= args.min_duration_ms,
        "configured_expected_qps": (
            events.get("requested_offline_expected_qps") == args.expected_samples_per_second
            and events.get("effective_target_qps") == args.expected_samples_per_second
        ),
        "configured_min_samples": (
            events.get("requested_min_query_count") == args.min_samples
            and events.get("effective_min_sample_count") == args.min_samples
        ),
        "offline_samples_per_query": events.get("effective_samples_per_query") == args.min_samples,
        "generated_samples_per_query": events.get("generated_samples_per_query") == args.min_samples,
        "performance_sample_set": (
            events.get("effective_performance_sample_count", 0) >= args.min_performance_samples
        ),
        "fixed_mlperf_seeds": all(events.get(key) == value for key, value in EXPECTED_SEEDS.items()),
    }

    target_ratio = samples_per_second / args.expected_samples_per_second

    print(f"detail_log={detail_log}")
    print(f"official_result_tokens_per_second={tokens_per_second:.6f}")
    print(f"secondary_result_samples_per_second={samples_per_second:.6f}")
    print(f"configured_expected_samples_per_second={args.expected_samples_per_second:.6f}")
    print(f"samples_per_second_vs_configured_target={target_ratio:.6f}x")
    for name, passed in checks.items():
        print(f"check.{name}={'PASS' if passed else 'FAIL'}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("FAIL: " + ", ".join(failed))
        return 1

    print("PASS: LoadGen Offline PerformanceOnly result satisfies the configured validity gates")
    print(
        "NOTE: This validates only Offline PerformanceOnly; Offline AccuracyOnly/TEST06 "
        "and the required Server-or-Interactive result set are not checked here"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
