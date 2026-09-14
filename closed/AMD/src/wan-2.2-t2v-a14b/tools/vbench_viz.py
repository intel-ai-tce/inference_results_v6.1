"""Shared helpers for VBench score visualization tools.

Loads per-video scores from VBench ``results_*_eval_results.json`` files
produced under an accuracy run's ``vbench/`` directory.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wan_harness.vbench import (  # noqa: E402
    DEFAULT_DIMENSIONS,
    REFERENCE_ACCURACY,
    VBenchResult,
    _find_latest_results_file,
    parse_results,
)

_log = logging.getLogger("vbench_viz")

SCENARIOS = ("Offline", "SingleStream")


@dataclass(frozen=True)
class VideoScore:
    """One evaluated video within a dimension."""

    key: str
    score: float


@dataclass(frozen=True)
class DimensionSeries:
    """Per-video scores for one VBench dimension."""

    name: str
    mean: float
    videos: tuple[VideoScore, ...]


@dataclass(frozen=True)
class ResolvedVBenchRun:
    """Located accuracy-mode run with VBench artefacts."""

    scenario: str | None
    run_dir: Path
    vbench_dir: Path
    results_file: Path
    label: str


def _score_from_record(record: object, *, dimension: str) -> float | None:
    """Extract a numeric score from one VBench per-video record."""
    if isinstance(record, (int, float)):
        return float(record)
    if not isinstance(record, dict):
        return None
    for key in ("video_results", "score", dimension):
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _video_key_from_record(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    raw = record.get("video_path")
    if not isinstance(raw, str) or not raw:
        return None
    return Path(raw).name


def load_dimension_series(results_file: Path) -> tuple[DimensionSeries, ...]:
    """Parse per-video scores from one VBench results JSON file."""
    results_file = Path(results_file)
    with results_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{results_file} is empty or not a JSON object.")

    series: list[DimensionSeries] = []
    for name in sorted(payload):
        entry = payload[name]
        if not isinstance(entry, list) or not entry:
            _log.warning("%s: skipping dimension %s (unexpected shape)", results_file, name)
            continue
        mean = entry[0]
        if not isinstance(mean, (int, float)):
            _log.warning("%s: skipping dimension %s (non-numeric mean)", results_file, name)
            continue
        videos: list[VideoScore] = []
        if len(entry) > 1 and isinstance(entry[1], list):
            for idx, record in enumerate(entry[1]):
                score = _score_from_record(record, dimension=name)
                if score is None:
                    _log.debug("%s: %s record %d has no numeric score", results_file, name, idx)
                    continue
                key = _video_key_from_record(record) or f"video_{idx}"
                videos.append(VideoScore(key=key, score=score))
        series.append(DimensionSeries(name=name, mean=float(mean), videos=tuple(videos)))
    if not series:
        raise ValueError(f"{results_file} produced no parseable dimensions.")
    return tuple(series)


def _scenario_from_path(path: Path) -> str | None:
    for name in SCENARIOS:
        if name in path.resolve().parts:
            return name
    return None


def _default_label(run_dir: Path) -> str:
    parts = run_dir.resolve().parts
    if "accuracy" in parts:
        idx = parts.index("accuracy")
        if idx >= 2:
            return parts[idx - 2]
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def resolve_vbench_run(
    run_dir: Path,
    *,
    label: str | None = None,
) -> ResolvedVBenchRun:
    """Resolve an accuracy directory or ``vbench/`` output directory."""
    run_dir = Path(run_dir).resolve()
    if run_dir.name == "vbench" and run_dir.is_dir():
        vbench_dir = run_dir
        run_dir = run_dir.parent
    elif (run_dir / "vbench").is_dir():
        vbench_dir = run_dir / "vbench"
    else:
        raise FileNotFoundError(
            f"No vbench/ directory under {run_dir}. Run VBench first "
            f"(python -m tools.run_vbench {run_dir})."
        )

    results_file = _find_latest_results_file(vbench_dir)
    return ResolvedVBenchRun(
        scenario=_scenario_from_path(run_dir),
        run_dir=run_dir,
        vbench_dir=vbench_dir,
        results_file=results_file,
        label=label or _default_label(run_dir),
    )


def load_vbench_result(resolved: ResolvedVBenchRun) -> VBenchResult:
    """Load aggregate :class:`VBenchResult` via the harness parser."""
    return parse_results(
        resolved.vbench_dir,
        videos_path=resolved.vbench_dir / "videos_staged",
        prompts_path=resolved.run_dir / "artefacts" / "prompts.json",
        nproc_per_node=1,
    )


def dimension_order(series: tuple[DimensionSeries, ...]) -> tuple[DimensionSeries, ...]:
    """Order dimensions: MLPerf defaults first, then any extras alphabetically."""
    by_name = {s.name: s for s in series}
    ordered: list[DimensionSeries] = []
    for name in DEFAULT_DIMENSIONS:
        if name in by_name:
            ordered.append(by_name.pop(name))
    ordered.extend(by_name[name] for name in sorted(by_name))
    return tuple(ordered)


def align_video_scores(
    a: tuple[VideoScore, ...],
    b: tuple[VideoScore, ...],
) -> tuple[list[str], list[float], list[float]]:
    """Join two per-video series on ``VideoScore.key``."""
    map_a = {v.key: v.score for v in a}
    map_b = {v.key: v.score for v in b}
    common = sorted(set(map_a) & set(map_b))
    if not common:
        return [], [], []
    return common, [map_a[k] for k in common], [map_b[k] for k in common]


def print_run_summary(
    resolved: ResolvedVBenchRun,
    series: tuple[DimensionSeries, ...],
    *,
    result: VBenchResult | None = None,
) -> None:
    """Emit a short text summary to stdout."""
    lines = [
        "",
        f"=== VBench scores: {resolved.label} ===",
        f"run_dir: {resolved.run_dir}",
        f"results: {resolved.results_file.name}",
    ]
    if result is not None:
        lines.append(
            f"vbench_score={result.vbench_score:.4f}  "
            f"reference={REFERENCE_ACCURACY:.2f}  "
            f"pass_99={'Yes' if result.pass_99 else 'No'}"
        )
    lines.append("")
    for dim in dimension_order(series):
        scores = [v.score for v in dim.videos]
        if scores:
            lo, hi = min(scores), max(scores)
            lines.append(
                f"  {dim.name:30s}  mean={dim.mean:.4f}  n={len(scores):3d}  "
                f"min={lo:.4f}  max={hi:.4f}"
            )
        else:
            lines.append(f"  {dim.name:30s}  mean={dim.mean:.4f}  n=0")
    print("\n".join(lines), flush=True)
