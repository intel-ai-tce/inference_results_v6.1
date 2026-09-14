"""Standalone VBench evaluator entry point.

Score the videos written by an accuracy-mode run and emit the MLPerf
submission artefacts (``accuracy.txt``) plus a structured sidecar
(``vbench/vbench_summary.json``). See :mod:`wan_harness.vbench` for the
substance; this module is a thin CLI wrapper so users can run

    python -m tools.run_vbench runs/wan22/latest/Offline/accuracy

without installing the harness package, matching the style of
``tools.fetch_data``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable when invoked before `pip install -e .`. Matches the
# pattern tests/conftest.py uses; keeps `python -m tools.run_vbench` usable
# straight after a fresh clone.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wan_harness.vbench import (  # noqa: E402  -- sys.path bootstrap above
    DEFAULT_DIMENSIONS,
    run_evaluation,
)

_log = logging.getLogger("run_vbench")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.run_vbench",
        description=(
            "Score an accuracy-mode wan-2.2-t2v-a14b run with VBench and "
            "emit accuracy.txt + vbench_summary.json."
        ),
    )
    p.add_argument(
        "run_dir",
        type=Path,
        help="Accuracy-mode run directory (contains artefacts/ and "
             "mlperf_log_accuracy.json), e.g. runs/wan22/<exp>/Offline/accuracy.",
    )
    p.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Directory of generated .mp4 files (default: <run_dir>/artefacts).",
    )
    p.add_argument(
        "--prompts-json",
        type=Path,
        default=None,
        help="VBench custom_input prompt map (default: <run_dir>/artefacts/prompts.json).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where VBench writes its results_*_eval_results.json plus our "
             "vbench_summary.json (default: <run_dir>/vbench).",
    )
    p.add_argument(
        "--accuracy-txt",
        type=Path,
        default=None,
        help="Path for the submission accuracy.txt (default: <run_dir>/accuracy.txt).",
    )
    p.add_argument(
        "--accuracy-json",
        type=Path,
        default=None,
        help="Path to mlperf_log_accuracy.json; its sha256 goes into the "
             "accuracy.txt hash= line (default: <run_dir>/mlperf_log_accuracy.json).",
    )
    p.add_argument(
        "--dimension",
        action="append",
        default=None,
        dest="dimensions",
        help="VBench dimension to score (repeatable). Default: the 6 reference "
             f"dimensions {list(DEFAULT_DIMENSIONS)}.",
    )
    p.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="torch.distributed.run --nproc_per_node (default: 1). "
             "Single-rank by default to sidestep VBench's dynamic_degree "
             "multi-process bug and the per-dimension checkpoint-download "
             "race; bump if you have a warm cache and want the speedup.",
    )
    p.add_argument(
        "--vbench-dir",
        type=Path,
        default=None,
        help="Override the VBench checkout (default: $WAN_VBENCH_DIR or "
             "submodules/VBench).",
    )
    p.add_argument(
        "--no-with-vbench",
        action="store_true",
        help="Do not prepend the `with-vbench` wrapper. Use this when running "
             "outside the wan-harness Docker image (e.g. in a host venv).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved VBench command and exit without running it.",
    )
    p.add_argument(
        "--parse-only",
        type=Path,
        default=None,
        help="Skip the VBench subprocess and parse an existing output directory. "
             "Useful for re-emitting accuracy.txt without re-scoring.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    dimensions = tuple(args.dimensions) if args.dimensions else DEFAULT_DIMENSIONS

    try:
        run_evaluation(
            args.run_dir,
            videos_dir=args.videos_dir,
            prompts_json=args.prompts_json,
            output_dir=args.output_dir,
            accuracy_txt=args.accuracy_txt,
            accuracy_json=args.accuracy_json,
            dimensions=dimensions,
            nproc_per_node=args.nproc_per_node,
            vbench_dir=args.vbench_dir,
            use_with_vbench=not args.no_with_vbench,
            parse_only=args.parse_only,
            dry_run=args.dry_run,
        )
    except SystemExit:
        raise
    except (FileNotFoundError, ValueError) as exc:
        _log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
