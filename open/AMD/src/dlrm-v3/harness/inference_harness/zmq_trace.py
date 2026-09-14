"""
Structured ZMQ pipeline tracing for ROCm harness debugging.

Enable with DLRM_ZMQ_TRACE=1. Logs go to stderr and optionally
DLRM_ZMQ_TRACE_LOG (default /tmp/dlrm_zmq_trace.jsonl in container).

Each production batch gets a monotonic seq (matches LoadGen out_batch_counter).
Correlate LoadGen and worker lines by seq=NNN.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_seq_assign = 0


def zmq_trace_enabled() -> bool:
    """On by default for ROCm harness; set DLRM_ZMQ_TRACE=0 to disable."""
    v = os.environ.get("DLRM_ZMQ_TRACE")
    if v == "1":
        return True
    if v == "0":
        return False
    return os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1"


def _log_path() -> Optional[str]:
    return os.environ.get("DLRM_ZMQ_TRACE_LOG", "").strip() or None


def _emit(
    side: str,
    rank: int,
    seq: int,
    stage: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not zmq_trace_enabled():
        return
    row = {
        "t": round(time.time(), 6),
        "side": side,
        "rank": rank,
        "seq": seq,
        "stage": stage,
    }
    if extra:
        row.update(extra)
    line = (
        f"[ZMQ_TRACE] {side} rank={rank} seq={seq} stage={stage}"
        + (f" {extra}" if extra else "")
    )
    with _lock:
        logger.info(line)
        path = _log_path()
        if path:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
            except OSError as e:
                logger.warning(f"[ZMQ_TRACE] failed to write {path}: {e}")


def next_seq() -> int:
    """Assign a new trace sequence id (warmup paths may use seq=0)."""
    global _seq_assign
    with _lock:
        _seq_assign += 1
        return _seq_assign


def trace_lg(rank: int, seq: int, stage: str, **extra: Any) -> None:
    _emit("LG", rank, seq, stage, extra if extra else None)


def trace_wk(rank: int, seq: int, stage: str, **extra: Any) -> None:
    _emit("WK", rank, seq, stage, extra if extra else None)


def packet_summary(
    query_ids: Optional[List[int]],
    ts_pairs: Optional[List[Tuple[int, int]]],
    real_n: Optional[int] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if query_ids:
        out["q0"] = int(query_ids[0])
        out["qN"] = len(query_ids)
        if len(query_ids) > 1:
            out["q_last"] = int(query_ids[-1])
    if ts_pairs:
        out["ts0"] = list(ts_pairs[0])
        if len(ts_pairs) > 1:
            out["ts_last"] = list(ts_pairs[-1])
    if real_n is not None:
        out["real_n"] = real_n
    return out
