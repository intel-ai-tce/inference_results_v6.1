"""Instrumentation for work that happens after ``backend.run_unit`` returns.

Offline data-parallel dispatchers spend non-trivial time packaging the
generated frame buffer into a wire :class:`~wan_harness.wire.Result`,
moving it across ranks (``gather_object`` / Gloo pt2pt), and handing it
to LoadGen via :class:`~wan_harness.sut.WanSUT`. This module collects
per-phase timings so those costs can be quantified independently of the
model call itself.

Enable via ``HarnessConfig.measure_post_run_overhead`` (CLI:
``--measure-post-run-overhead``). Each rank writes its own summary JSON;
rank 0 also logs an aggregate table at INFO level.

Quick probe (mock backend, real-sized frames, no GPU)::

    ./scripts/measure_post_run_overhead.sh
    ./scripts/measure_post_run_overhead.sh --dispatch wave

Stdout/stderr from all ranks are tee'd to ``${output_dir}/harness.log``.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

_log = logging.getLogger(__name__)

__all__ = [
    "PHASE_RESULT_PACK",
    "PHASE_RUN_UNIT",
    "PHASE_SUT_ARTEFACT_WRITE",
    "PHASE_SUT_RESPONSE_COMPLETE",
    "PHASE_WIRE_TRANSFER",
    "PostRunOverheadCollector",
    "get_collector",
]

# Phases recorded on worker ranks and rank 0 inside the dispatcher.
PHASE_RUN_UNIT = "run_unit"
PHASE_RESULT_PACK = "result_pack"
PHASE_WIRE_TRANSFER = "wire_transfer"

# Phases recorded on rank 0 inside the SUT after the dispatcher yields.
PHASE_SUT_RESPONSE_COMPLETE = "sut_response_complete"
PHASE_SUT_ARTEFACT_WRITE = "sut_artefact_write"


@dataclass(frozen=True)
class _Sample:
    phase: str
    seconds: float
    sample_index: int
    rank: int
    nbytes: int = 0


@dataclass
class PhaseStats:
    phase: str
    count: int
    total_s: float
    mean_ms: float
    p50_ms: float
    p99_ms: float
    max_ms: float
    total_mib: float
    mib_per_s: float | None


@dataclass
class PostRunOverheadCollector:
    """Process-local accumulator for post-``run_unit`` phase timings."""

    enabled: bool = False
    rank: int = 0
    _samples: list[_Sample] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def configure(self, *, enabled: bool, rank: int = 0) -> None:
        with self._lock:
            self.enabled = bool(enabled)
            self.rank = int(rank)
            if not self.enabled:
                self._samples.clear()

    def record(
        self,
        phase: str,
        seconds: float,
        *,
        sample_index: int,
        rank: int | None = None,
        nbytes: int = 0,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._samples.append(
                _Sample(
                    phase=str(phase),
                    seconds=float(seconds),
                    sample_index=int(sample_index),
                    rank=int(self.rank if rank is None else rank),
                    nbytes=max(int(nbytes), 0),
                )
            )

    @contextmanager
    def measure(
        self,
        phase: str,
        *,
        sample_index: int,
        rank: int | None = None,
        nbytes: int = 0,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                phase,
                time.perf_counter() - t0,
                sample_index=sample_index,
                rank=rank,
                nbytes=nbytes,
            )

    def _phase_stats(self) -> list[PhaseStats]:
        by_phase: dict[str, list[_Sample]] = {}
        with self._lock:
            for sample in self._samples:
                by_phase.setdefault(sample.phase, []).append(sample)

        stats: list[PhaseStats] = []
        for phase in sorted(by_phase):
            rows = by_phase[phase]
            ms = [s.seconds * 1000.0 for s in rows]
            total_s = sum(s.seconds for s in rows)
            total_bytes = sum(s.nbytes for s in rows)
            total_mib = total_bytes / (1024.0 * 1024.0)
            stats.append(
                PhaseStats(
                    phase=phase,
                    count=len(rows),
                    total_s=total_s,
                    mean_ms=statistics.mean(ms),
                    p50_ms=_percentile(ms, 50.0),
                    p99_ms=_percentile(ms, 99.0),
                    max_ms=max(ms),
                    total_mib=total_mib,
                    mib_per_s=(total_mib / total_s) if total_s > 0 else None,
                )
            )
        return stats

    def summary_dict(self) -> dict[str, object]:
        phases = self._phase_stats()
        post_run_phases = [
            p for p in phases if p.phase != PHASE_RUN_UNIT
        ]
        post_run_total_s = sum(p.total_s for p in post_run_phases)
        return {
            "rank": self.rank,
            "enabled": self.enabled,
            "sample_count": len(self._samples),
            "post_run_total_s": post_run_total_s,
            "phases": [asdict(p) for p in phases],
        }

    def log_summary(self) -> None:
        if not self.enabled:
            return
        phases = self._phase_stats()
        if not phases:
            _log.info(
                "post_run_overhead rank=%d: no samples recorded", self.rank
            )
            return

        post_run = [p for p in phases if p.phase != PHASE_RUN_UNIT]
        post_run_total = sum(p.total_s for p in post_run)
        _log.info(
            "post_run_overhead rank=%d: %d phase records, post-run total=%.3fs",
            self.rank,
            sum(p.count for p in post_run),
            post_run_total,
        )
        header = (
            f"{'phase':<24} {'count':>6} {'total_s':>9} {'mean_ms':>9} "
            f"{'p50_ms':>9} {'p99_ms':>9} {'max_ms':>9} {'mib/s':>9}"
        )
        _log.info("post_run_overhead rank=%d summary:\n%s", self.rank, header)
        for p in phases:
            rate = f"{p.mib_per_s:9.1f}" if p.mib_per_s is not None else f"{'—':>9}"
            _log.info(
                "post_run_overhead rank=%d  %-24s %6d %9.3f %9.1f %9.1f "
                "%9.1f %9.1f %s",
                self.rank,
                p.phase,
                p.count,
                p.total_s,
                p.mean_ms,
                p.p50_ms,
                p.p99_ms,
                p.max_ms,
                rate.strip(),
            )

    def write_summary(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.summary_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path


_collector = PostRunOverheadCollector()


def get_collector() -> PostRunOverheadCollector:
    return _collector


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)
