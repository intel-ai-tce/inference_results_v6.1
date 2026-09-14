import json
import logging
import os
import time
from typing import Any, Dict, Optional

import torch


_ENABLED = os.environ.get("DLRM_MEMORY_TRACE", "0") == "1"
_SNAPSHOT = os.environ.get("DLRM_MEMORY_TRACE_SNAPSHOT", "0") == "1"


def memory_trace_enabled() -> bool:
    return _ENABLED


def _snapshot_summary(device: int) -> Optional[Dict[str, int]]:
    if not _SNAPSHOT:
        return None
    summary = {
        "snapshot_segments": 0,
        "snapshot_blocks": 0,
        "snapshot_active_blocks": 0,
        "snapshot_inactive_blocks": 0,
        "snapshot_active_B": 0,
        "snapshot_inactive_B": 0,
        "largest_torch_inactive_block_B": 0,
    }
    largest = 0
    for segment in torch.cuda.memory_snapshot():
        if segment.get("device", device) != device:
            continue
        summary["snapshot_segments"] += 1
        for block in segment.get("blocks", ()):
            size = int(block.get("size", 0))
            state = str(block.get("state", ""))
            summary["snapshot_blocks"] += 1
            if state == "inactive":
                summary["snapshot_inactive_blocks"] += 1
                summary["snapshot_inactive_B"] += size
                largest = max(largest, size)
            elif state.startswith("active"):
                summary["snapshot_active_blocks"] += 1
                summary["snapshot_active_B"] += size
    summary["largest_torch_inactive_block_B"] = largest
    return summary


def log_memory_phase(
    logger: logging.Logger,
    phase: str,
    *,
    rank: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not _ENABLED:
        return
    try:
        if not torch.cuda.is_available():
            logger.warning(
                "[memtrace] %s",
                json.dumps(
                    {
                        "ts": time.time(),
                        "phase": phase,
                        "rank": rank,
                        "cuda_available": False,
                        "extra": extra or {},
                    },
                    sort_keys=True,
                ),
            )
            return

        device = torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(device)
        allocated_b = torch.cuda.memory_allocated(device)
        reserved_b = torch.cuda.memory_reserved(device)
        record: Dict[str, Any] = {
            "ts": time.time(),
            "phase": phase,
            "rank": rank,
            "device": int(device),
            "driver_free_B": int(free_b),
            "driver_total_B": int(total_b),
            "driver_used_B": int(total_b - free_b),
            "torch_allocated_B": int(allocated_b),
            "torch_reserved_B": int(reserved_b),
            "torch_reserved_minus_allocated_B": int(reserved_b - allocated_b),
            "torch_max_allocated_B": int(torch.cuda.max_memory_allocated(device)),
            "torch_max_reserved_B": int(torch.cuda.max_memory_reserved(device)),
            "extra": extra or {},
        }
        snapshot_summary = _snapshot_summary(device)
        if snapshot_summary is not None:
            record.update(snapshot_summary)
        logger.warning("[memtrace] %s", json.dumps(record, sort_keys=True))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[memtrace] phase=%s failed: %r", phase, exc)
