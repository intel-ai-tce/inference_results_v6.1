"""Command-line entry point for the harness.

Exposes the ``wan-harness`` console script. Subcommands:

  - ``run``          – execute a single (scenario, mode) LoadGen test.
  - ``print-config`` – print the fully resolved :class:`HarnessConfig` and exit.

A ``--dry-run`` switch on ``run`` is a thin alias for ``--backend mock``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from .backends import registered_backends
from .config import HarnessConfig, coerce_field_value, load_harness_config
from .logging_utils import configure_logging

_log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Parser construction.
# ----------------------------------------------------------------------


def _add_run_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--inference-config",
        type=Path,
        default=None,
        help="Path to inference_config.yaml. Defaults to configs/inference_config.yaml.",
    )
    p.add_argument(
        "--backend",
        choices=registered_backends(),
        default=None,
        help="Backend to use. Default: 'mock'.",
    )
    p.add_argument(
        "--scenario",
        choices=("Offline", "SingleStream"),
        default=None,
        help="MLPerf scenario.",
    )
    p.add_argument(
        "--mode",
        choices=("performance", "accuracy"),
        default=None,
        help="MLPerf test mode.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Shorthand for --backend mock.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for LoadGen logs and artefacts.",
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=None,
        dest="prompts_path",
        help="Path to the prompts text file (one prompt per line).",
    )
    p.add_argument(
        "--user-conf",
        type=Path,
        default=None,
        dest="user_conf_path",
        help="Path to user.conf with LoadGen overrides.",
    )
    p.add_argument(
        "--mlperf-conf",
        type=Path,
        default=None,
        dest="mlperf_conf_path",
        help="Ignored: LoadGen loads its built-in mlperf.conf via user.conf.",
    )
    p.add_argument(
        "--audit-conf",
        type=Path,
        default=None,
        dest="audit_conf_path",
        help="Path to audit.config for compliance tests.",
    )
    p.add_argument(
        "--backend-config",
        type=Path,
        default=None,
        dest="backend_config_path",
        help=(
            "Per-backend YAML config. For --backend wan22 this defaults to "
            "configs/wan22/<scenario>.yaml."
        ),
    )

    # Optional LoadGen tunables.
    p.add_argument("--performance-sample-count", type=int, default=None)
    p.add_argument("--min-query-count", type=int, default=None)
    p.add_argument("--min-duration-ms", type=int, default=None)
    p.add_argument("--max-duration-ms", type=int, default=None)
    p.add_argument("--target-qps", type=float, default=None)
    p.add_argument("--target-latency-ns", type=int, default=None)

    # Mock backend knobs.
    p.add_argument(
        "--mock-delay-ms",
        type=int,
        default=None,
        help="Artificial per-sample latency for the Mock backend.",
    )
    p.add_argument(
        "--mock-payload",
        choices=("zeros", "noise"),
        default=None,
        help="Mock backend payload kind.",
    )
    p.add_argument(
        "--mock-dispatch",
        choices=("wave", "async"),
        default=None,
        help=(
            "When using the mock backend under torchrun (world_size>1), pick "
            "the Offline data-parallel dispatcher for post-run_unit profiling."
        ),
    )
    p.add_argument(
        "--result-transport",
        choices=("shm", "gloo"),
        default=None,
        help=(
            "Bulk Result data plane for DP dispatchers: POSIX shared memory "
            "(default) or Gloo tensor transfer."
        ),
    )

    # Observability.
    p.add_argument("--enable-loadgen-trace", action="store_true", default=None)
    p.add_argument(
        "--measure-post-run-overhead",
        action="store_true",
        default=None,
        help=(
            "Record per-phase timings for work after backend.run_unit returns "
            "(result packaging, cross-rank transfer, LoadGen completion)."
        ),
    )
    p.add_argument(
        "--log-level",
        default=None,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )

    # Generic escape hatch for any HarnessConfig field that does not have
    # its own dedicated flag. Useful for ablations and CI smoke tests.
    # Example: --set height=8 --set width=16 --set num_frames=2
    p.add_argument(
        "--set",
        action="append",
        default=[],
        dest="set_overrides",
        metavar="KEY=VALUE",
        help=(
            "Override any HarnessConfig field. Repeatable. "
            "Reserved for advanced use; closed-division submissions must "
            "not change model-side fields (height, width, num_frames, ...)."
        ),
    )


def _add_vbench_arguments(p: argparse.ArgumentParser) -> None:
    """Flags for the ``wan-harness vbench`` subcommand.

    Mirrors ``tools/run_vbench.py``; keep them in sync so users can
    reach the standalone evaluator via either entry point.
    """
    p.add_argument(
        "run_dir",
        type=Path,
        help="Accuracy-mode run directory (contains artefacts/ and "
             "mlperf_log_accuracy.json), e.g. runs/wan22/<exp>/Offline/accuracy.",
    )
    p.add_argument("--videos-dir", type=Path, default=None)
    p.add_argument("--prompts-json", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None, dest="vbench_output_dir")
    p.add_argument("--accuracy-txt", type=Path, default=None)
    p.add_argument("--accuracy-json", type=Path, default=None)
    p.add_argument(
        "--dimension",
        action="append",
        default=None,
        dest="dimensions",
        help="VBench dimension to score (repeatable). Default: the 6 "
             "reference dimensions.",
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
    p.add_argument("--vbench-dir", type=Path, default=None)
    p.add_argument(
        "--no-with-vbench",
        action="store_true",
        help="Do not prepend `with-vbench`; use the current Python directly.",
    )
    p.add_argument("--dry-run", action="store_true", dest="vbench_dry_run")
    p.add_argument("--parse-only", type=Path, default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wan-harness",
        description="MLPerf inference harness for wan-2.2-t2v-a14b.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run one LoadGen test.")
    _add_run_arguments(run_p)

    cfg_p = sub.add_parser("print-config", help="Resolve and print the HarnessConfig.")
    _add_run_arguments(cfg_p)
    cfg_p.add_argument(
        "--format",
        choices=("json", "yaml"),
        default="json",
        help="Output format. Default: json.",
    )

    vbench_p = sub.add_parser(
        "vbench",
        help="Score an accuracy-mode run with VBench and emit accuracy.txt.",
    )
    _add_vbench_arguments(vbench_p)

    return parser


# ----------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------


_CLI_FIELDS = (
    "backend",
    "scenario",
    "mode",
    "output_dir",
    "prompts_path",
    "user_conf_path",
    "mlperf_conf_path",
    "audit_conf_path",
    "backend_config_path",
    "performance_sample_count",
    "min_query_count",
    "min_duration_ms",
    "max_duration_ms",
    "target_qps",
    "target_latency_ns",
    "mock_delay_ms",
    "mock_payload",
    "mock_dispatch",
    "result_transport",
    "enable_loadgen_trace",
    "measure_post_run_overhead",
    "log_level",
)


def _parse_set_overrides(set_args: list[str]) -> dict[str, object]:
    """Parse the repeatable ``--set key=value`` flag into a kwargs dict."""
    out: dict[str, object] = {}
    for entry in set_args:
        if "=" not in entry:
            raise SystemExit(
                f"--set expects KEY=VALUE, got {entry!r} (no '=' found)"
            )
        key, _, value = entry.partition("=")
        key = key.strip()
        try:
            out[key] = coerce_field_value(key, value)
        except KeyError:
            raise SystemExit(
                f"--set {entry!r}: {key!r} is not a HarnessConfig field"
            ) from None
        except (ValueError, TypeError) as exc:
            raise SystemExit(
                f"--set {entry!r}: cannot coerce {value!r} to the type of {key!r}: {exc}"
            ) from None
    return out


def _resolve_config(args: argparse.Namespace) -> HarnessConfig:
    cli_overrides = {k: getattr(args, k, None) for k in _CLI_FIELDS}
    if getattr(args, "dry_run", False):
        if cli_overrides.get("backend") not in (None, "mock"):
            raise SystemExit(
                f"--dry-run is incompatible with --backend {cli_overrides['backend']!r}"
            )
        cli_overrides["backend"] = "mock"

    # Apply generic --set overrides last so they win over the named flags
    # (consistent with the CLI > env > YAML > defaults precedence).
    set_overrides = _parse_set_overrides(getattr(args, "set_overrides", []) or [])
    cli_overrides.update(set_overrides)

    return load_harness_config(
        inference_config_path=args.inference_config,
        cli_overrides=cli_overrides,
    )


def _print_config(config: HarnessConfig, fmt: str) -> None:
    data = config.as_dict()
    if fmt == "json":
        json.dump(data, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        import yaml

        yaml.safe_dump(data, sys.stdout, sort_keys=True)


# ----------------------------------------------------------------------
# Subcommand handlers.
# ----------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    configure_logging(config.log_level)
    _log.info("resolved config: backend=%s scenario=%s mode=%s output_dir=%s",
              config.backend, config.scenario, config.mode, config.output_dir)

    # Imported here so `wan-harness print-config` works without mlperf_loadgen.
    from .loadgen_runner import run

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    result = run(config, rank=rank, world_size=world_size)
    _log.info(
        "done: issued=%d completed=%d artefacts=%s",
        result.issued,
        result.completed,
        result.artefacts_dir,
    )
    return 0


def _cmd_print_config(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    _print_config(config, args.format)
    return 0


def _cmd_vbench(args: argparse.Namespace) -> int:
    # Imported lazily so `wan-harness print-config` and `wan-harness run`
    # keep working in environments that don't have the VBench venv on
    # PYTHONPATH (the inference and VBench environments are deliberately
    # separated; see docker/Dockerfile).
    from .vbench import DEFAULT_DIMENSIONS, run_evaluation

    configure_logging("INFO")

    dimensions = tuple(args.dimensions) if args.dimensions else DEFAULT_DIMENSIONS
    try:
        run_evaluation(
            args.run_dir,
            videos_dir=args.videos_dir,
            prompts_json=args.prompts_json,
            output_dir=args.vbench_output_dir,
            accuracy_txt=args.accuracy_txt,
            accuracy_json=args.accuracy_json,
            dimensions=dimensions,
            nproc_per_node=args.nproc_per_node,
            vbench_dir=args.vbench_dir,
            use_with_vbench=not args.no_with_vbench,
            parse_only=args.parse_only,
            dry_run=args.vbench_dry_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        _log.error("%s", exc)
        return 2
    return 0


# ----------------------------------------------------------------------
# Entry point.
# ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "print-config":
        return _cmd_print_config(args)
    if args.command == "vbench":
        return _cmd_vbench(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
