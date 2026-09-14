"""Compare per-video VBench scores between two accuracy runs.

Joins on staged video filename (``{prompt}-{iteration}.mp4``) and plots two
lines per dimension subplot. Requires ``matplotlib``::

    pip install matplotlib
    python -m tools.compare_vbench \\
        --run-a runs/wan22/expA/Offline/accuracy \\
        --run-b runs/wan22/expB/Offline/accuracy
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path

from tools.vbench_viz import (
    align_video_scores,
    dimension_order,
    load_dimension_series,
    load_vbench_result,
    print_run_summary,
    resolve_vbench_run,
)

_log = logging.getLogger("compare_vbench")


def _default_output_path(run_a: Path) -> Path:
    resolved = resolve_vbench_run(run_a)
    slug = (resolved.scenario or "vbench").lower()
    return resolved.vbench_dir / f"{slug}_vbench_compare.png"


def plot_comparison(
    *,
    series_a: tuple,
    series_b: tuple,
    label_a: str,
    label_b: str,
    title: str,
    output: Path,
    show_mean: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from exc

    by_a = {s.name: s for s in series_a}
    by_b = {s.name: s for s in series_b}
    names = [s.name for s in dimension_order(series_a)]
    for name in sorted(set(by_b) - set(by_a)):
        names.append(name)

    n_dims = len(names)
    ncols = 2 if n_dims > 1 else 1
    nrows = (n_dims + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 3.4 * nrows),
        squeeze=False,
        layout="constrained",
    )
    color_a = "#4C78A8"
    color_b = "#F58518"

    for idx, name in enumerate(names):
        ax = axes[idx // ncols][idx % ncols]
        dim_a = by_a.get(name)
        dim_b = by_b.get(name)
        if dim_a is None or dim_b is None:
            missing = label_a if dim_a is None else label_b
            ax.text(
                0.5,
                0.5,
                f"dimension missing in {missing}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(name)
            continue

        keys, scores_a, scores_b = align_video_scores(dim_a.videos, dim_b.videos)
        if not keys:
            ax.text(
                0.5,
                0.5,
                "no overlapping videos",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{name}\nmatched n=0")
            continue

        xs = list(range(len(keys)))
        ax.plot(xs, scores_a, marker="o", markersize=2.5, linewidth=1.0, color=color_a, label=label_a)
        ax.plot(xs, scores_b, marker="o", markersize=2.5, linewidth=1.0, color=color_b, label=label_b)
        if show_mean:
            ax.axhline(dim_a.mean, color=color_a, linestyle="--", linewidth=0.9, alpha=0.6)
            ax.axhline(dim_b.mean, color=color_b, linestyle="--", linewidth=0.9, alpha=0.6)

        deltas = [b - a for a, b in zip(scores_a, scores_b, strict=True)]
        mean_delta = statistics.mean(deltas)
        ax.set_xlabel("Matched video index")
        ax.set_ylabel("Score")
        ax.set_title(
            f"{name}\n"
            f"matched n={len(keys)}  "
            f"mean Δ({label_b}−{label_a})={mean_delta:+.4f}"
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xlim(-0.5, max(xs) + 0.5)

    for idx in range(n_dims, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log.info("wrote %s", output)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.compare_vbench",
        description="Compare per-video VBench scores between two accuracy runs.",
    )
    p.add_argument(
        "--run-a",
        type=Path,
        required=True,
        help="First accuracy run directory (…/<Scenario>/accuracy).",
    )
    p.add_argument(
        "--run-b",
        type=Path,
        required=True,
        help="Second accuracy run directory.",
    )
    p.add_argument(
        "--label-a",
        default=None,
        help="Short name for run A (default: experiment dir name).",
    )
    p.add_argument(
        "--label-b",
        default=None,
        help="Short name for run B (default: experiment dir name).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="PNG path (default: <run-a>/vbench/<scenario>_vbench_compare.png).",
    )
    p.add_argument(
        "--no-mean-line",
        action="store_true",
        help="Omit horizontal mean lines in each subplot.",
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

    try:
        resolved_a = resolve_vbench_run(args.run_a, label=args.label_a)
        resolved_b = resolve_vbench_run(args.run_b, label=args.label_b)
        label_a = resolved_a.label
        label_b = resolved_b.label
        output = args.output or _default_output_path(args.run_a)

        series_a = load_dimension_series(resolved_a.results_file)
        series_b = load_dimension_series(resolved_b.results_file)
        result_a = load_vbench_result(resolved_a)
        result_b = load_vbench_result(resolved_b)

        if resolved_a.scenario and resolved_b.scenario and resolved_a.scenario != resolved_b.scenario:
            _log.warning(
                "scenarios differ: %s vs %s (continuing anyway)",
                resolved_a.scenario,
                resolved_b.scenario,
            )

        scenario_title = resolved_a.scenario or resolved_b.scenario or "VBench"
        title = (
            f"{scenario_title}: {label_a} vs {label_b}\n"
            f"A={resolved_a.run_dir}  B={resolved_b.run_dir}"
        )
        plot_comparison(
            series_a=series_a,
            series_b=series_b,
            label_a=label_a,
            label_b=label_b,
            title=title,
            output=output,
            show_mean=not args.no_mean_line,
        )

        print(
            f"\n=== VBench comparison: {label_a} vs {label_b} ===\n"
            f"vbench_score  {label_a}={result_a.vbench_score:.4f}  "
            f"{label_b}={result_b.vbench_score:.4f}  "
            f"Δ={result_b.vbench_score - result_a.vbench_score:+.4f}\n",
            flush=True,
        )
        print_run_summary(resolved_a, series_a, result=result_a)
        print_run_summary(resolved_b, series_b, result=result_b)
    except SystemExit:
        raise
    except (FileNotFoundError, ValueError) as exc:
        _log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
