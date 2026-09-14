"""Lightweight latency instrumentation (reuses open-path timing_stats when available)."""
from __future__ import annotations

try:
    from timing_stats import (
        enabled,
        format_summary,
        maybe_report,
        record,
        reset,
        summarize,
    )
except ImportError:

    def enabled() -> bool:
        return False

    def record(_name: str, _seconds: float) -> None:
        pass

    def maybe_report(force: bool = False) -> None:
        pass

    def reset() -> None:
        pass

    def summarize():
        return {}

    def format_summary(_summary) -> str:
        return ""
