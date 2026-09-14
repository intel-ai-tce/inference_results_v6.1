# Lightweight rolling timing stats for DLRMv3 bottleneck analysis.
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

_lock = threading.Lock()
_counts: Dict[str, int] = {}
_sums: Dict[str, float] = {}
_samples: Dict[str, List[float]] = {}
_log_path: Optional[str] = os.environ.get("DLRM_TIMING_LOG")
_report_every: int = int(os.environ.get("DLRM_TIMING_REPORT_EVERY", "25"))


def enabled() -> bool:
    return os.environ.get("DLRM_TIMING", "0") == "1"


def reset() -> None:
    """Clear accumulated samples (call after warmup, before timed section)."""
    with _lock:
        _counts.clear()
        _sums.clear()
        _samples.clear()


def record(name: str, seconds: float) -> None:
    if not enabled():
        return
    to_log: Optional[Dict[str, Any]] = None
    with _lock:
        _counts[name] = _counts.get(name, 0) + 1
        _sums[name] = _sums.get(name, 0.0) + seconds
        if name not in _samples:
            _samples[name] = []
        xs = _samples[name]
        xs.append(seconds)
        if len(xs) > 2000:
            del xs[: len(xs) - 2000]
        n = _counts[name]
        if _log_path and n % max(_report_every, 1) == 0:
            to_log = {"event": "record", "name": name, "n": n, "last_s": seconds}
    if to_log is not None:
        _append_jsonl(to_log)


def _append_jsonl(obj: Dict[str, Any]) -> None:
    assert _log_path
    obj["ts"] = time.time()
    with open(_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _snapshot_samples() -> Dict[str, List[float]]:
    with _lock:
        return {k: list(v) for k, v in _samples.items()}


def _compute_summary(snap: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name in sorted(snap.keys()):
        xs = snap[name]
        if not xs:
            continue
        arr = np.array(xs, dtype=np.float64)
        out[name] = {
            "count": float(len(xs)),
            "mean_ms": float(arr.mean() * 1000),
            "p50_ms": float(np.percentile(arr, 50) * 1000),
            "p90_ms": float(np.percentile(arr, 90) * 1000),
            "p99_ms": float(np.percentile(arr, 99) * 1000),
            "max_ms": float(arr.max() * 1000),
        }
    return out


def maybe_report(force: bool = False) -> None:
    if not enabled():
        return
    with _lock:
        total = _counts.get("batch.total", 0)
        if not force and total % max(_report_every, 1) != 0:
            return
    summary = _compute_summary(_snapshot_samples())
    line = format_summary(summary)
    print(line, flush=True)
    if _log_path:
        _append_jsonl({"event": "summary", "summary": summary})


def summarize() -> Dict[str, Dict[str, float]]:
    return _compute_summary(_snapshot_samples())


def format_summary(summary: Dict[str, Dict[str, float]]) -> str:
    order = [
        "batch.total",
        "batch.collate",
        "batch.predict",
        "batch.send_zmq",
        "batch.batching",
        "batch.queue",
        "predict.sparse",
        "predict.h2d",
        "predict.dense_e2e",
        "predict.dense_queue_wait",
        "predict.prediction",
        "loadgen.qsc_enqueue",
        "loadgen.response_pack",
    ]
    parts = ["DLRM_TIMING"]
    seen = set()
    for key in order + sorted(summary.keys()):
        if key in seen or key not in summary:
            continue
        seen.add(key)
        s = summary[key]
        parts.append(
            f"{key}: mean={s['mean_ms']:.1f}ms p99={s['p99_ms']:.1f}ms n={int(s['count'])}"
        )
    return " | ".join(parts)
