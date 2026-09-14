"""Plot per-sample inference times from a finished performance run.

Supports **Offline** and **SingleStream**. Reads dispatcher ``complete`` lines
from the scenario's ``run.log`` (or ``run_unit`` totals for single-process
mock runs), omits harness warmup (negative ``sample_index``), and renders a
histogram with percentile markers and/or a **wall-clock time series** (log
line timestamps at completion, not prompt index).

This reports **backend inference wall time per sample** (``run_unit`` /
dispatcher ``latency=``), not LoadGen schedule latency from
``mlperf_log_trace.json``.

Requires ``matplotlib`` (not shipped in the base wan-harness image)::

    pip install matplotlib
    python -m tools.plot_sample_runtimes --scenario SingleStream
    python -m tools.plot_sample_runtimes --scenario Offline
    python -m tools.plot_sample_runtimes --run-dir runs/wan22/latest/Offline/performance/run_1
    python -m tools.plot_sample_runtimes --run-dir …/run_1 --plot timeseries
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_log = logging.getLogger("plot_sample_runtimes")

SCENARIOS = ("Offline", "SingleStream")

# Worker ranks prefix log records with [rank=N]; rank 0 does not.
_RANK_PREFIX_RE = re.compile(r"^\s*\[rank=\d+\]")

# Standard logging timestamp at the start of ``run.log`` lines (wall clock).
_LOG_TS_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b")

# Dispatcher "complete" lines that include per-sample inference time.
_COMPLETE_LATENCY_RE = re.compile(
    r"(?:UlyssesDispatcher|AsyncDPDispatcher)\.complete\s+"
    r"sample=(-?\d+)\b.*?latency=([0-9.]+)s"
)
_WAVE_LATENCY_RE = re.compile(
    r"WaveDispatcher\.complete\s+sample=(-?\d+)\b.*?wave_latency=([0-9.]+)s"
)

# Fallback for single-process runs: backend run_unit total on rank 0.
_RUN_UNIT_TOTAL_RE = re.compile(
    r"run_unit\s+rank=0\s+sample=(-?\d+)\b.*?total=([0-9.]+)s"
)

DEFAULT_PERCENTILES = (50.0, 90.0, 95.0, 97.0, 99.0)


@dataclass(frozen=True)
class SampleRuntime:
    """One measured sample after warmup is stripped."""

    sample_index: int
    seconds: float
    #: Parsed from the log line prefix when present (``YYYY-MM-DD HH:MM:SS``).
    wall_time: datetime | None = None


@dataclass(frozen=True)
class ResolvedRun:
    scenario: str
    run_dir: Path
    run_log: Path | None
    prompts_path: Path | None


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of an empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def _percentiles(values: list[float], levels: tuple[float, ...]) -> dict[float, float]:
    ordered = sorted(values)
    return {pct: _percentile(ordered, pct) for pct in levels}


def _scenario_from_path(path: Path) -> str | None:
    parts = path.resolve().parts
    for name in SCENARIOS:
        if name in parts:
            return name
    return None


def _scenario_from_metadata(run_dir: Path) -> str | None:
    meta_path = run_dir / "harness_metadata.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    scenario = meta.get("config", {}).get("scenario")
    return scenario if scenario in SCENARIOS else None


def _scenario_from_detail(run_dir: Path) -> str | None:
    detail_path = run_dir / "mlperf_log_detail.txt"
    if not detail_path.is_file():
        return None
    for line in detail_path.read_text(encoding="utf-8").splitlines():
        if "requested_scenario" not in line:
            continue
        if not line.startswith(":::MLLOG "):
            continue
        try:
            payload = json.loads(line[len(":::MLLOG ") :])
        except json.JSONDecodeError:
            continue
        scenario = payload.get("value")
        return scenario if scenario in SCENARIOS else None
    return None


def _prompts_path_from_run(run_dir: Path, repo_root: Path) -> Path | None:
    meta_path = run_dir / "harness_metadata.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    raw = meta.get("config", {}).get("prompts_path")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path if path.is_file() else None


def _load_prompts(path: Path) -> tuple[str, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = tuple(line.strip() for line in lines if line.strip())
    if not prompts:
        raise ValueError(f"Prompts file is empty: {path}")
    return prompts


def _format_prompt(prompt: str, *, max_width: int) -> str:
    if max_width <= 0 or len(prompt) <= max_width:
        return prompt
    return prompt[: max_width - 1] + "…"


def detect_scenario(run_dir: Path) -> str:
    for detector in (_scenario_from_metadata, _scenario_from_detail, _scenario_from_path):
        scenario = detector(run_dir)
        if scenario is not None:
            return scenario
    raise ValueError(
        f"Could not determine scenario for {run_dir}; pass --scenario explicitly."
    )


def _parse_log_line_wall_time(line: str) -> datetime | None:
    match = _LOG_TS_PREFIX_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def resolve_run(
    *,
    repo_root: Path,
    scenario: str | None,
    run_dir: Path | None,
    experiment_root: Path | None,
) -> ResolvedRun:
    """Locate a scenario performance ``run_1`` directory and its ``run.log``."""
    if run_dir is not None:
        resolved = run_dir.resolve()
        resolved_scenario = scenario or detect_scenario(resolved)
        perf_parent = resolved.parent
        run_log = (
            perf_parent / "run.log"
            if perf_parent.name == "performance"
            else None
        )
        if run_log is not None and not run_log.is_file():
            run_log = None
        return ResolvedRun(
            scenario=resolved_scenario,
            run_dir=resolved,
            run_log=run_log,
            prompts_path=_prompts_path_from_run(resolved, repo_root),
        )

    if scenario is None:
        scenario = "SingleStream"
        _log.info("no --scenario given; defaulting to %s", scenario)

    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported scenario {scenario!r}; pick one of {SCENARIOS!r}")

    root = (experiment_root or (repo_root / "runs/wan22/latest")).resolve()
    perf = root / scenario / "performance"
    if not perf.is_dir():
        raise FileNotFoundError(
            f"{scenario} performance directory not found at {perf}. "
            "Pass --run-dir or --experiment-root explicitly."
        )
    run_1 = perf / "run_1"
    if not run_1.is_dir():
        raise FileNotFoundError(f"Expected run output at {run_1}")
    run_log = perf / "run.log"
    return ResolvedRun(
        scenario=scenario,
        run_dir=run_1,
        run_log=run_log if run_log.is_file() else None,
        prompts_path=_prompts_path_from_run(run_1, repo_root),
    )


def _parse_run_log(run_log: Path) -> list[SampleRuntime]:
    if not run_log.is_file():
        return []

    text = run_log.read_text(encoding="utf-8")
    samples: list[SampleRuntime] = []
    for raw_line in text.splitlines():
        if _RANK_PREFIX_RE.match(raw_line):
            continue
        wall_time = _parse_log_line_wall_time(raw_line)
        for pattern in (_COMPLETE_LATENCY_RE, _WAVE_LATENCY_RE):
            match = pattern.search(raw_line)
            if match:
                idx = int(match.group(1))
                if idx < 0:
                    continue
                samples.append(
                    SampleRuntime(
                        sample_index=idx,
                        seconds=float(match.group(2)),
                        wall_time=wall_time,
                    )
                )
                break

    if samples:
        return samples

    for raw_line in text.splitlines():
        if _RANK_PREFIX_RE.match(raw_line):
            continue
        match = _RUN_UNIT_TOTAL_RE.search(raw_line)
        if not match:
            continue
        idx = int(match.group(1))
        if idx < 0:
            continue
        samples.append(
            SampleRuntime(
                sample_index=idx,
                seconds=float(match.group(2)),
                wall_time=_parse_log_line_wall_time(raw_line),
            )
        )
    return samples


def load_inference_times(resolved: ResolvedRun) -> list[SampleRuntime]:
    if resolved.run_log is None:
        raise FileNotFoundError(
            f"No run.log beside {resolved.run_dir}; cannot extract inference times."
        )
    samples = _parse_run_log(resolved.run_log)
    if not samples:
        raise FileNotFoundError(
            f"No per-sample inference times found in {resolved.run_log}. "
            "Expected dispatcher complete lines with latency= / wave_latency=, "
            "or run_unit totals for single-process runs."
        )
    return samples


def plot_histogram(
    runtimes_s: list[float],
    *,
    scenario: str,
    percentiles: tuple[float, ...],
    title: str,
    output: Path,
) -> dict[float, float]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    pct_values = _percentiles(runtimes_s, percentiles)
    mean_s = statistics.mean(runtimes_s)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        runtimes_s,
        bins="auto",
        orientation="horizontal",
        color="#4C78A8",
        edgecolor="white",
        alpha=0.85,
        label=f"n={len(runtimes_s)}",
    )

    colors = {
        50.0: "#F58518",
        90.0: "#E45756",
        95.0: "#72B7B2",
        97.0: "#54A24B",
        99.0: "#B279A2",
    }
    xmax = ax.get_xlim()[1]
    for pct in percentiles:
        value = pct_values[pct]
        color = colors.get(pct, "#333333")
        ax.axhline(value, color=color, linewidth=1.5, linestyle="--", alpha=0.9)
        ax.annotate(
            f"p{pct:g} = {value:.3f}s",
            xy=(xmax * 0.98, value),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color=color,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.8},
        )

    ax.set_ylabel("Per-sample inference time (seconds)")
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    subtitle = (
        f"metric=inference_time  n={len(runtimes_s)}  mean={mean_s:.3f}s  "
        f"min={min(runtimes_s):.3f}s  max={max(runtimes_s):.3f}s"
    )
    ax.text(0.01, 0.98, subtitle, transform=ax.transAxes, va="top", fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    _log.info("wrote %s", output)
    return pct_values


def plot_wall_clock_timeseries(
    samples: list[SampleRuntime],
    *,
    title: str,
    output: Path,
) -> None:
    """Scatter/lines of inference time vs log wall-clock completion time."""
    timed = [s for s in samples if s.wall_time is not None]
    if len(timed) != len(samples):
        raise ValueError(
            "Wall-clock time series needs a timestamp on every sample line "
            f"({len(timed)}/{len(samples)} parsed). Ensure run.log lines start "
            "with 'YYYY-MM-DD HH:MM:SS'."
        )

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    timed_sorted = sorted(timed, key=lambda s: s.wall_time)
    times = [s.wall_time for s in timed_sorted]
    runtimes_s = [s.seconds for s in timed_sorted]
    mean_s = statistics.mean(runtimes_s)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, runtimes_s, marker="o", linestyle="-", markersize=3, color="#4C78A8")
    ax.set_ylabel("Per-sample inference time (seconds)")
    ax.set_xlabel("Wall clock (log line timestamp at completion)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    ax.grid(axis="both", alpha=0.25)
    subtitle = (
        f"metric=inference_time  n={len(runtimes_s)}  mean={mean_s:.3f}s  "
        f"min={min(runtimes_s):.3f}s  max={max(runtimes_s):.3f}s  "
        "(ordered by completion time, not prompt index)"
    )
    ax.text(0.01, 0.98, subtitle, transform=ax.transAxes, va="top", fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    _log.info("wrote %s", output)


def _prompt_lookup(
    prompts: tuple[str, ...] | None,
    sample_index: int,
) -> str | None:
    if prompts is None:
        return None
    if 0 <= sample_index < len(prompts):
        return prompts[sample_index]
    return None


def _extreme_samples(
    samples: list[SampleRuntime],
    *,
    pick: str,
) -> list[SampleRuntime]:
    if not samples:
        return []
    if pick == "min":
        target = min(s.seconds for s in samples)
    else:
        target = max(s.seconds for s in samples)
    return [s for s in samples if s.seconds == target]


def print_inference_statistics(
    samples: list[SampleRuntime],
    *,
    scenario: str,
    percentiles: dict[float, float],
    prompts: tuple[str, ...] | None,
    prompts_path: Path | None,
    prompt_width: int,
) -> None:
    """Emit a human-readable summary to stdout (independent of log level)."""
    runtimes = [s.seconds for s in samples]
    stdev = statistics.pstdev(runtimes) if len(runtimes) > 1 else 0.0
    lines = [
        "",
        f"=== {scenario} per-sample inference statistics (n={len(samples)}) ===",
        (
            f"mean={statistics.mean(runtimes):.3f}s  "
            f"min={min(runtimes):.3f}s  max={max(runtimes):.3f}s  "
            f"stdev={stdev:.3f}s"
        ),
    ]
    pct_parts = [
        f"p{int(p) if p == int(p) else p:g}={percentiles[p]:.3f}s"
        for p in sorted(percentiles)
    ]
    lines.append("  ".join(pct_parts))

    if prompts_path is not None:
        lines.append(f"prompts: {prompts_path}")
    elif prompts is None:
        lines.append("prompts: unavailable (no prompts_path in harness_metadata.json)")

    for label, pick in (("Fastest", "min"), ("Slowest", "max")):
        rows = _extreme_samples(samples, pick=pick)
        lines.append("")
        for row in rows:
            prompt = _prompt_lookup(prompts, row.sample_index)
            lines.append(
                f"{label} ({row.seconds:.3f}s) — sample_index={row.sample_index}:"
            )
            if prompt is None:
                lines.append("  <prompt unavailable>")
            else:
                lines.append(f"  {_format_prompt(prompt, max_width=prompt_width)}")

    print("\n".join(lines), flush=True)


def _default_output_path(resolved: ResolvedRun) -> Path:
    slug = resolved.scenario.lower()
    return resolved.run_dir / f"{slug}_sample_runtimes.png"


def _default_timeseries_output_path(resolved: ResolvedRun) -> Path:
    slug = resolved.scenario.lower()
    return resolved.run_dir / f"{slug}_sample_runtimes_wallclock.png"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.plot_sample_runtimes",
        description=(
            "Plot per-sample inference times (warmup omitted) from an "
            "Offline or SingleStream performance run: histogram and/or "
            "wall-clock time series."
        ),
    )
    p.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default=None,
        help="LoadGen scenario when resolving from an experiment root. "
             "Auto-detected when --run-dir is set. Default: SingleStream.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Performance run directory (…/<Scenario>/performance/run_1).",
    )
    p.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help="Experiment root (contains Offline/ and SingleStream/). "
             "Default: runs/wan22/latest",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "PNG output path. For --plot histogram (default): "
            "<run_dir>/<scenario>_sample_runtimes.png. For --plot timeseries: "
            "<run_dir>/<scenario>_sample_runtimes_wallclock.png. For --plot both, "
            "this path is the histogram; the timeseries defaults to "
            "<stem>_wallclock<suffix> beside it unless --timeseries-output is set."
        ),
    )
    p.add_argument(
        "--plot",
        choices=("histogram", "timeseries", "both"),
        default="histogram",
        help=(
            "histogram: horizontal histogram with percentile markers (default). "
            "timeseries: inference time vs log wall-clock completion timestamp. "
            "both: write two PNGs."
        ),
    )
    p.add_argument(
        "--timeseries-output",
        type=Path,
        default=None,
        help="PNG path for the wall-clock plot when --plot both (optional).",
    )
    p.add_argument(
        "--percentile",
        type=float,
        action="append",
        dest="percentiles",
        default=None,
        help="Percentile marker to draw (repeatable). Default: 50, 90, 95, 97, 99.",
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Override prompts file (default: prompts_path from harness_metadata.json).",
    )
    p.add_argument(
        "--prompt-width",
        type=int,
        default=120,
        help="Truncate printed prompts to this many characters (0 = no limit).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive plot window (requires a display).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    percentiles = (
        tuple(args.percentiles) if args.percentiles else DEFAULT_PERCENTILES
    )

    try:
        resolved = resolve_run(
            repo_root=_REPO_ROOT,
            scenario=args.scenario,
            run_dir=args.run_dir,
            experiment_root=args.experiment_root,
        )
        samples = load_inference_times(resolved)
        runtimes_s = [s.seconds for s in samples]
        _log.info(
            "loaded %d inference times for %s from %s",
            len(runtimes_s),
            resolved.scenario,
            resolved.run_log,
        )

        prompts_path = args.prompts or resolved.prompts_path
        prompts: tuple[str, ...] | None = None
        if prompts_path is not None:
            try:
                prompts = _load_prompts(prompts_path.resolve())
            except (OSError, ValueError) as exc:
                _log.warning("could not load prompts from %s: %s", prompts_path, exc)
                prompts_path = None

        default_hist = _default_output_path(resolved)
        default_ts = _default_timeseries_output_path(resolved)

        if args.plot == "histogram":
            hist_path = args.output or default_hist
            ts_path: Path | None = None
        elif args.plot == "timeseries":
            hist_path = None
            ts_path = args.output or default_ts
        else:
            hist_path = args.output or default_hist
            ts_path = args.timeseries_output or (
                hist_path.parent / f"{hist_path.stem}_wallclock{hist_path.suffix}"
            )

        title = (
            f"{resolved.scenario} per-sample inference times\n"
            f"{resolved.run_dir}"
        )
        ts_title = (
            f"{resolved.scenario} inference time vs wall clock\n"
            f"{resolved.run_dir}"
        )

        pct_values: dict[float, float] = _percentiles(runtimes_s, percentiles)

        if hist_path is not None:
            pct_values = plot_histogram(
                runtimes_s,
                scenario=resolved.scenario,
                percentiles=percentiles,
                title=title,
                output=hist_path,
            )
        if ts_path is not None:
            plot_wall_clock_timeseries(samples, title=ts_title, output=ts_path)

        print_inference_statistics(
            samples,
            scenario=resolved.scenario,
            percentiles=pct_values,
            prompts=prompts,
            prompts_path=prompts_path,
            prompt_width=args.prompt_width,
        )

        if args.show:
            try:
                import matplotlib.pyplot as plt

                for path in (hist_path, ts_path):
                    if path is None:
                        continue
                    img = plt.imread(path)
                    plt.figure(figsize=(12, 6))
                    plt.imshow(img)
                    plt.axis("off")
                    plt.show()
            except Exception as exc:  # noqa: BLE001
                _log.warning("--show failed (%s); PNG(s) saved", exc)
    except SystemExit:
        raise
    except (FileNotFoundError, ValueError) as exc:
        _log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
