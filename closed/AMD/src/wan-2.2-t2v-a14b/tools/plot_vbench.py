"""Plot per-video VBench scores for one accuracy run.

Reads ``vbench/results_*_eval_results.json`` and renders one line subplot per
dimension (separate y-scales). Requires ``matplotlib``::

    pip install matplotlib
    python -m tools.plot_vbench runs/wan22/latest/Offline/accuracy
    python -m tools.plot_vbench --scenario SingleStream
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tools.vbench_viz import (
    SCENARIOS,
    ResolvedVBenchRun,
    dimension_order,
    load_dimension_series,
    load_vbench_result,
    print_run_summary,
    resolve_vbench_run,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

_log = logging.getLogger("plot_vbench")


def _default_output_path(resolved: ResolvedVBenchRun) -> Path:
    slug = (resolved.scenario or "vbench").lower()
    return resolved.vbench_dir / f"{slug}_vbench_per_video.png"


def plot_single_run(
    series: tuple,
    *,
    title: str,
    output: Path,
    show_mean: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    ordered = dimension_order(series)
    n_dims = len(ordered)
    ncols = 2 if n_dims > 1 else 1
    nrows = (n_dims + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 3.2 * nrows),
        squeeze=False,
        layout="constrained",
    )
    color = "#4C78A8"

    for idx, dim in enumerate(ordered):
        ax = axes[idx // ncols][idx % ncols]
        xs = list(range(len(dim.videos)))
        ys = [v.score for v in dim.videos]
        if not xs:
            ax.text(0.5, 0.5, "no per-video scores", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{dim.name}\nmean={dim.mean:.4f}  n=0")
            continue

        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.0, color=color)
        if show_mean:
            ax.axhline(dim.mean, color="#E45756", linestyle="--", linewidth=1.0, alpha=0.85, label=f"mean={dim.mean:.4f}")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlabel("Video index")
        ax.set_ylabel("Score")
        ax.set_title(f"{dim.name}\nmean={dim.mean:.4f}  n={len(ys)}")
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.5, max(xs) + 0.5)

    for idx in range(n_dims, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log.info("wrote %s", output)


def _resolve_from_args(
    *,
    run_dir: Path | None,
    scenario: str | None,
    experiment_root: Path | None,
    label: str | None,
) -> ResolvedVBenchRun:
    if run_dir is not None:
        return resolve_vbench_run(run_dir, label=label)

    if scenario is None:
        scenario = "Offline"
        _log.info("no --scenario given; defaulting to %s", scenario)
    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported scenario {scenario!r}; pick one of {SCENARIOS!r}")

    root = (experiment_root or (_REPO_ROOT / "runs/wan22/latest")).resolve()
    accuracy = root / scenario / "accuracy"
    if not accuracy.is_dir():
        raise FileNotFoundError(
            f"{scenario} accuracy directory not found at {accuracy}. "
            "Pass --run-dir or --experiment-root explicitly."
        )
    return resolve_vbench_run(accuracy, label=label)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.plot_vbench",
        description="Plot per-video VBench scores (one subplot per dimension).",
    )
    p.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Accuracy run directory (…/<Scenario>/accuracy) or vbench/ output dir.",
    )
    p.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default=None,
        help="Scenario when resolving from an experiment root (default: Offline).",
    )
    p.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help="Experiment root containing Offline/ and SingleStream/ "
             "(default: runs/wan22/latest).",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Short name for plot title (default: experiment directory name).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="PNG output path (default: <vbench_dir>/<scenario>_vbench_per_video.png).",
    )
    p.add_argument(
        "--no-mean-line",
        action="store_true",
        help="Omit the horizontal mean line in each subplot.",
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

    try:
        resolved = _resolve_from_args(
            run_dir=args.run_dir,
            scenario=args.scenario,
            experiment_root=args.experiment_root,
            label=args.label,
        )
        series = load_dimension_series(resolved.results_file)
        result = load_vbench_result(resolved)
        output = args.output or _default_output_path(resolved)

        scenario_part = resolved.scenario or "run"
        title = f"{scenario_part} VBench per-video scores — {resolved.label}\n{resolved.run_dir}"
        plot_single_run(
            series,
            title=title,
            output=output,
            show_mean=not args.no_mean_line,
        )
        print_run_summary(resolved, series, result=result)

        if args.show:
            try:
                import matplotlib.pyplot as plt

                img = plt.imread(output)
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.axis("off")
                plt.show()
            except Exception as exc:  # noqa: BLE001
                _log.warning("--show failed (%s); PNG saved", exc)
    except SystemExit:
        raise
    except (FileNotFoundError, ValueError) as exc:
        _log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
