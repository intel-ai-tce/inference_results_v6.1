#!/usr/bin/env python3
"""Append one CSV row of ``[parameters | metrics]`` per benchmark run, for experiment tracking.

Called automatically at the end of ``benchmark_mlperf6pt1.py``, and runnable standalone on any run
folder under ``outputs/<scenario_type>/single_run/<ts>/``::

    python parse_results.py --run outputs/offline/single_run/20260815-101500
    python parse_results.py --run outputs/server/single_run/*

Everything is read from artifacts each run already persisted: ``.hydra/config.yaml`` (the
resolved benchmark.yaml: scenario + server + quantization + env), ``status.txt`` (run
outcome), and the endpoints ``report_dir`` (``config.yaml``, ``result_summary.json``,
``results.json``). No benchmark is re-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CSV = "results.csv"

# Category Hierarchical F1 thresholds (v6.1 reference spec); used for the accuracy_pass flag.
ACCURACY_TARGET = {"offline": 0.7824, "server": 0.7824, "interactive": 0.7799}

# Normalize the harness/scenario "type" to one of {offline, server, interactive}.
SCENARIO_ALIAS = {
    "offline": "offline",
    "server": "server",
    "interactive": "interactive",
    "online": "server",
}

COLUMNS = [
    # --- parameters ---
    "timestamp",
    "mode",
    "scenario",
    "status",
    "gpu",
    "model",
    "model_revision",
    "quant_source",
    "quant_method",
    "weight_element",
    "weight_scale",
    "weight_granularity",
    "activation_element",
    "activation_scale",
    "activation_granularity",
    "kv_cache_element",
    "kv_cache_scale",
    "kv_cache_granularity",
    "gemm_element",
    "gemm_accumulation",
    "tensor_parallel_size",
    "enable_expert_parallel",
    "gpu_memory_utilization",
    "max_model_len",
    "max_number_of_batched_tokens",
    "max_num_seqs",
    "target_qps",
    "n_samples_to_issue",
    "env",
    "run",
    # --- metrics ---
    "accuracy_f1",
    "accuracy_target",
    "accuracy_pass",
    "qps",
    "tokens_per_s",
    "osl_mean",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p99_ms",
    "e2e_p50_ms",
    "e2e_p99_ms",
    "e2e_mean_ms",
    "n_completed",
    "n_failed",
    "duration_s",
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _find_report_dir(run_dir: Path) -> Path | None:
    hits = sorted(run_dir.glob("**/result_summary.json")) or sorted(
        run_dir.glob("**/results.json")
    )
    return hits[0].parent if hits else None


def _pct(metric: dict[str, Any], p: float) -> float | None:
    """Percentile ``p`` from a harness series dict, tolerant of "99"/"99.0" keys."""
    perc = (metric or {}).get("percentiles", {}) or {}
    # The harness writes one-decimal keys ("99.0", "50.0"); also tolerate "99"/"99.00".
    for key in (str(int(p)), str(p), f"{p:.1f}", f"{p:.2f}"):
        if key in perc and perc[key] is not None:
            return float(perc[key])
    return None


def _ms(ns: float | None) -> float | None:
    return None if ns is None else ns / 1e6


def _round(v: Any, n: int = 4) -> Any:
    return round(v, n) if isinstance(v, float) else v


def _timestamp(run_dir: Path) -> str:
    stamp = r"\d{8}-\d{6}"
    if re.fullmatch(stamp, run_dir.name):  # single run
        return run_dir.name
    if re.fullmatch(stamp, run_dir.parent.name):  # sweep combo: parent holds it
        return run_dir.parent.name
    return datetime.fromtimestamp(run_dir.stat().st_mtime).strftime(
        "%Y%m%d-%H%M%S"
    )  # fallback: when parser_results.py is run standalone


# A run whose served samples mostly failed (e.g. the engine died mid-run -> mass 500s) is NOT ok,
# even if the client process exited 0 and the server was healthy at start. Fraction of returned
# samples that may error before the run is downgraded to "failed".
FAILED_SAMPLE_FRACTION = 0.5


def status_with_failures(base: str, n_completed: int, n_failed: int) -> str:
    """Downgrade an 'ok' status to 'failed' when most returned samples errored (mid-run crash)."""
    if base == "ok" and n_completed and n_failed / n_completed > FAILED_SAMPLE_FRACTION:
        return "failed"
    return base


def report_sample_counts(run_dir: Path) -> tuple[int, int]:
    """(n_completed, n_failed) from the run's result_summary.json; (0, 0) if absent."""
    report_dir = _find_report_dir(run_dir)
    summary = _load_json(report_dir / "result_summary.json") if report_dir else {}
    return int(summary.get("n_samples_completed", 0)), int(
        summary.get("n_samples_failed", 0)
    )


def collect_row(run_dir: Path) -> dict[str, Any]:
    """Build one ``{column: value}`` row from a single run folder."""
    run_dir = run_dir.resolve()
    hydra_cfg = _load_yaml(
        run_dir / ".hydra" / "config.yaml"
    )  # resolved benchmark.yaml
    server = hydra_cfg.get("server", {}) or {}
    # env vars live under server.env (consolidated); still merge any top-level env if present
    env = {**(server.get("env") or {}), **(hydra_cfg.get("env") or {})}

    report_dir = _find_report_dir(run_dir)
    bench_cfg = _load_yaml(report_dir / "config.yaml") if report_dir else {}
    summary = _load_json(report_dir / "result_summary.json") if report_dir else {}
    results = _load_json(report_dir / "results.json") if report_dir else {}

    # Scenario type (offline/server/interactive). The endpoints "type" field is "online" for
    # BOTH server and interactive, so it cannot tell them apart. Prefer the unambiguous hydra
    # scenario name (offline_*/server_*/interactive_*); fall back to the type field only if the
    # name is unavailable.
    name = str(hydra_cfg.get("scenario", run_dir.name)).lower()
    scenario = next((v for k, v in SCENARIO_ALIAS.items() if name.startswith(k)), "")
    if not scenario:
        raw_type = str(bench_cfg.get("type", "")).strip().lower()
        scenario = SCENARIO_ALIAS.get(raw_type, raw_type)

    model = (bench_cfg.get("model_params", {}) or {}).get("name", "")
    settings = bench_cfg.get("settings", {}) or {}
    target_qps = (settings.get("load_pattern", {}) or {}).get("target_qps")
    n_samples_to_issue = (settings.get("runtime", {}) or {}).get("n_samples_to_issue")
    mode = hydra_cfg.get("mode", "reference")  # this submission copy is reference-only; `mode` removed from config

    status_file = run_dir / "status.txt"
    status = (
        status_file.read_text().strip()
        if status_file.is_file()
        else ("ok" if report_dir else "")
    )

    run_meta = _load_json(run_dir / "run_meta.json")
    quant = run_meta.get("quantization", {}) or {}

    def _q(target: str, field: str) -> Any:
        return (quant.get(target) or {}).get(field)

    # Metrics from result_summary.json (+ accuracy from results.json).
    dur_ns = summary.get("duration_ns") or 0
    dur_s = dur_ns / 1e9 if dur_ns else None
    completed = int(summary.get("n_samples_completed", 0))
    osl = summary.get("output_sequence_lengths") or {}
    osl_total = osl.get("total") or 0
    latency, ttft, tpot = (summary.get(k) or {} for k in ("latency", "ttft", "tpot"))

    # qps: prefer the harness-reported value; if absent, derive from completed/duration. This is
    # a derivation from already-persisted artifacts;
    # tokens_per_s below is likewise always derived (no primary field is recorded).
    qps = (results.get("results", {}) or {}).get("qps")
    if qps is None and dur_s:
        qps = completed / dur_s

    f1 = None
    for entry in (results.get("accuracy_scores") or {}).values():
        if isinstance(entry, dict) and entry.get("score") is not None:
            f1 = float(entry["score"])
            break
    target = ACCURACY_TARGET.get(scenario)

    # A run the harness called "ok" but whose samples mostly failed (e.g. server died mid-run ->
    # 500s) is not valid; record it as failed so it isn't mistaken for a good run.
    status = status_with_failures(
        status, completed, int(summary.get("n_samples_failed", 0))
    )

    row = {
        "timestamp": _timestamp(run_dir),
        "mode": mode,
        "scenario": scenario,
        "status": status,
        "gpu": run_meta.get("gpu", ""),
        "model": model,
        "model_revision": run_meta.get("model_revision", ""),
        "quant_source": run_meta.get("quant_source", ""),
        "quant_method": quant.get("method"),
        "weight_element": _q("weight", "element"),
        "weight_scale": _q("weight", "scale"),
        "weight_granularity": _q("weight", "granularity"),
        "activation_element": _q("activation", "element"),
        "activation_scale": _q("activation", "scale"),
        "activation_granularity": _q("activation", "granularity"),
        "kv_cache_element": _q("kv_cache", "element"),
        "kv_cache_scale": _q("kv_cache", "scale"),
        "kv_cache_granularity": _q("kv_cache", "granularity"),
        "gemm_element": _q("gemm", "element"),
        "gemm_accumulation": _q("gemm", "accumulation"),
        "tensor_parallel_size": server.get("tensor_parallel_size"),
        "enable_expert_parallel": server.get("enable_expert_parallel"),
        "gpu_memory_utilization": server.get("gpu_memory_utilization"),
        "max_model_len": server.get("max_model_len"),
        "max_number_of_batched_tokens": server.get("max_number_of_batched_tokens"),
        "max_num_seqs": server.get("max_num_seqs"),
        "target_qps": target_qps,
        "n_samples_to_issue": n_samples_to_issue,
        "env": ";".join(f"{k}={v}" for k, v in env.items()),
        "run": run_dir.name,
        "accuracy_f1": f1,
        "accuracy_target": target,
        "accuracy_pass": (
            (f1 >= target) if (f1 is not None and target is not None) else None
        ),
        "qps": qps,
        "tokens_per_s": (osl_total / dur_s) if dur_s else None,
        "osl_mean": osl.get("avg"),
        "ttft_p50_ms": _ms(_pct(ttft, 50)),
        "ttft_p99_ms": _ms(_pct(ttft, 99)),
        "tpot_p50_ms": _ms(_pct(tpot, 50)),
        "tpot_p99_ms": _ms(_pct(tpot, 99)),
        "e2e_p50_ms": _ms(_pct(latency, 50)),
        "e2e_p99_ms": _ms(_pct(latency, 99)),
        "e2e_mean_ms": _ms(latency.get("avg")),
        "n_completed": completed,
        "n_failed": int(summary.get("n_samples_failed", 0)),
        "duration_s": dur_s,
    }
    return {k: _round(v) for k, v in row.items()}


def append_row(run_dir: Path, csv_path: Path | None = None) -> Path:
    """Append one row for ``run_dir`` to the shared CSV (creating it + header if needed)."""
    run_dir = Path(run_dir).resolve()
    csv_path = Path(csv_path) if csv_path else (run_dir.parent / DEFAULT_CSV)
    row = collect_row(run_dir)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(
            {k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS}
        )
    return csv_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run",
        nargs="+",
        required=True,
        type=Path,
        metavar="DIR",
        help="one or more run folders, e.g. outputs/<mode>/<scenario_type>/single_run/<ts>",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV to append to (default: results.csv next to the run folders)",
    )
    ap.add_argument(
        "--print", action="store_true", help="also print each row to stdout"
    )
    args = ap.parse_args(argv)

    csv_path = None
    for run_dir in args.run:
        if not run_dir.is_dir():
            print(f"skip (not a dir): {run_dir}", file=sys.stderr)
            continue
        csv_path = append_row(run_dir, args.csv)
        if args.print:
            print(json.dumps(collect_row(run_dir), indent=2))
    if csv_path:
        print(f"appended to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
