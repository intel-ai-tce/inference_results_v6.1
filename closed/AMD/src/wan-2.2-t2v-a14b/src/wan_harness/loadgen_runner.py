"""LoadGen runner: builds TestSettings, owns the SUT/QSL lifecycle, runs the test.

This is the only module that depends on ``mlperf_loadgen``. Everything else in
the harness is testable without it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__ as HARNESS_VERSION
from .artefacts import ArtefactWriter
from .backends import build_backend
from .dispatcher import build_dispatcher
from .post_run_overhead import get_collector
from .qsl import WanQSL
from .response import LoadgenResponseWriter
from .sut import WanSUT

if TYPE_CHECKING:
    from .backends.base import Backend
    from .config import HarnessConfig
    from .data.prompts import PromptDataset

_log = logging.getLogger(__name__)

__all__ = ["RunResult", "run"]


# Map our string scenario names onto the loadgen enum at call time so this
# module is importable without loadgen.
_SCENARIO_NAMES = ("Offline", "SingleStream")
_MODE_NAMES = ("performance", "accuracy")


@dataclass(frozen=True)
class RunResult:
    """Summary of a single ``run()`` invocation."""

    scenario: str
    mode: str
    backend: str
    output_dir: Path
    issued: int
    completed: int
    artefacts_dir: Path | None


def _scenario_enum(name: str):
    import mlperf_loadgen as lg  # type: ignore[import-not-found]

    if name == "Offline":
        return lg.TestScenario.Offline
    if name == "SingleStream":
        return lg.TestScenario.SingleStream
    raise ValueError(f"Unsupported scenario {name!r}; pick one of {_SCENARIO_NAMES!r}")


def _mode_enum(name: str):
    import mlperf_loadgen as lg  # type: ignore[import-not-found]

    if name == "performance":
        return lg.TestMode.PerformanceOnly
    if name == "accuracy":
        return lg.TestMode.AccuracyOnly
    raise ValueError(f"Unsupported mode {name!r}; pick one of {_MODE_NAMES!r}")


def _build_test_settings(config: "HarnessConfig"):
    import mlperf_loadgen as lg  # type: ignore[import-not-found]

    settings = lg.TestSettings()

    # LoadGen allows exactly one user conf (conf_type=1). The first FromConfig
    # call auto-loads the built-in mlperf.conf; user.conf supplies overrides.
    if not config.user_conf_path.exists():
        raise FileNotFoundError(
            f"user.conf not found at {config.user_conf_path}; "
            "LoadGen requires a user.conf for benchmark runs."
        )
    if config.mlperf_conf_path is not None:
        _log.warning(
            "--mlperf-conf is ignored: LoadGen loads its built-in mlperf.conf "
            "automatically on the first FromConfig(user.conf) call"
        )
    _log.info(
        "applying user.conf from %s (built-in mlperf.conf loaded by LoadGen)",
        config.user_conf_path,
    )
    settings.FromConfig(
        str(config.user_conf_path), config.model_name, config.scenario
    )
    # Compliance: apply audit.config with the model name so Wan-specific keys
    # (e.g. wan-2.2-t2v-a14b.SingleStream.min_query_count) are picked up.
    # StartTestWithLogSettings(audit_path) below applies wildcard TEST04 flags
    # with model="*". LoadGen logs one error_invalid_config for the second
    # conf_type=1 FromConfig; that is expected on compliance runs (see reference).
    if config.audit_conf_path is not None and config.audit_conf_path.exists():
        _log.info("applying audit.config from %s", config.audit_conf_path)
        settings.FromConfig(
            str(config.audit_conf_path), config.model_name, config.scenario
        )

    settings.scenario = _scenario_enum(config.scenario)
    settings.mode = _mode_enum(config.mode)

    # CLI overrides (each is None unless the user passed it).
    if config.min_query_count is not None:
        settings.min_query_count = int(config.min_query_count)
    if config.min_duration_ms is not None:
        settings.min_duration_ms = int(config.min_duration_ms)
    if config.max_duration_ms is not None:
        settings.max_duration_ms = int(config.max_duration_ms)
    if config.target_qps is not None:
        settings.offline_expected_qps = float(config.target_qps)
        settings.server_target_qps = float(config.target_qps)
    if config.target_latency_ns is not None:
        settings.single_stream_expected_latency_ns = int(config.target_latency_ns)
        settings.server_target_latency_ns = int(config.target_latency_ns)

    return settings


def _build_log_settings(output_dir: Path, enable_trace: bool):
    import mlperf_loadgen as lg  # type: ignore[import-not-found]

    output_dir.mkdir(parents=True, exist_ok=True)
    out = lg.LogOutputSettings()
    out.outdir = str(output_dir)
    out.prefix = "mlperf_log_"
    out.copy_summary_to_stdout = False

    log_settings = lg.LogSettings()
    log_settings.enable_trace = bool(enable_trace)
    log_settings.log_output = out
    return log_settings


def _write_metadata(
    output_dir: Path,
    config: "HarnessConfig",
    result_summary: dict,
) -> Path:
    meta = {
        "harness_version": HARNESS_VERSION,
        "config": config.as_dict(),
        "result": result_summary,
        "env": {
            k: os.environ[k]
            for k in (
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "HOSTNAME",
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            )
            if k in os.environ
        },
    }
    path = output_dir / "harness_metadata.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_dataset(config: "HarnessConfig") -> "PromptDataset":
    """Load the QSL prompts, falling back to a synthetic set for the mock dry-run.

    The fallback exists so a fresh clone of the repo can exercise the dry-run
    on day one, before ``data/vbench_prompts.txt`` has been downloaded.
    """
    from .data.prompts import load_prompts, synthetic_prompts

    if config.prompts_path.exists():
        return load_prompts(config.prompts_path)

    if config.backend != "mock":
        raise FileNotFoundError(
            f"Real backend {config.backend!r} requires real prompts; "
            f"{config.prompts_path} is missing."
        )
    fallback_count = 32
    _log.warning(
        "prompts file %s not found; using %d synthetic prompts (mock dry-run only)",
        config.prompts_path,
        fallback_count,
    )
    return synthetic_prompts(fallback_count)


def _finalize_post_run_overhead(
    config: "HarnessConfig",
    *,
    rank: int,
) -> None:
    if not config.measure_post_run_overhead:
        return
    collector = get_collector()
    out = config.output_dir.resolve() / f"post_run_overhead_rank{rank}.json"
    collector.write_summary(out)
    collector.log_summary()
    _log.info("post_run_overhead rank=%d summary written to %s", rank, out)


def _run_worker(
    config: "HarnessConfig",
    *,
    rank: int,
    world_size: int,
) -> RunResult:
    """Entry point for non-rank-0 ranks in a multi-rank run.

    These ranks never construct LoadGen; they sit in the dispatcher's
    worker loop processing :class:`~wan_harness.wire.WorkUnit` messages
    until rank 0 broadcasts a :class:`~wan_harness.wire.Shutdown`.
    """
    get_collector().configure(
        enabled=config.measure_post_run_overhead,
        rank=rank,
    )
    backend: "Backend" = build_backend(config.backend, config)
    backend.setup(rank=rank, world_size=world_size)
    try:
        dispatcher = build_dispatcher(backend, rank=rank, world_size=world_size)
        _log.info(
            "worker rank %d/%d: entering dispatcher loop (backend=%s)",
            rank, world_size, backend.name,
        )
        dispatcher.run_worker_loop()
    finally:
        backend.teardown()
        _finalize_post_run_overhead(config, rank=rank)
    return RunResult(
        scenario=config.scenario,
        mode=config.mode,
        backend=config.backend,
        output_dir=config.output_dir.resolve(),
        issued=0,
        completed=0,
        artefacts_dir=None,
    )


def run(config: "HarnessConfig", *, rank: int = 0, world_size: int = 1) -> RunResult:
    """Execute one LoadGen test according to ``config``.

    Returns a :class:`RunResult` summary. LoadGen log files land in
    ``config.output_dir`` and are documented in
    ``config.output_dir/harness_metadata.json``.

    In multi-rank runs, only rank 0 runs the LoadGen flow below; the
    other ranks are routed to :func:`_run_worker` and never touch
    LoadGen.
    """
    if world_size > 1 and rank != 0:
        return _run_worker(config, rank=rank, world_size=world_size)

    get_collector().configure(
        enabled=config.measure_post_run_overhead,
        rank=rank,
    )

    import mlperf_loadgen as lg  # type: ignore[import-not-found]

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _log.info("output directory: %s", output_dir)

    dataset = _load_dataset(config)
    perf_count = (
        config.performance_sample_count
        if config.performance_sample_count is not None
        else min(len(dataset), 248)
    )
    qsl = WanQSL(dataset, performance_sample_count=perf_count)

    backend: "Backend" = build_backend(config.backend, config)
    backend.setup(rank=rank, world_size=world_size)
    try:
        dispatcher = build_dispatcher(backend, rank=rank, world_size=world_size)

        # Run pre-LoadGen warmup so torch.compile / kernel-build / CUDA-graph
        # capture happen BEFORE the measured test. Workers are already in
        # their dispatcher loop (see _run_worker) and pick up these units
        # the same way they pick up real LoadGen prompts.
        ws = backend.warmup_settings()
        if ws is not None:
            num_prompts_per_rank, warmup_prompt = ws
            _log.info(
                "running pre-LoadGen warmup: %d prompts/rank (world_size=%d)",
                num_prompts_per_rank, world_size,
            )
            dispatcher.warmup(num_prompts_per_rank, warmup_prompt)
        else:
            _log.info("backend %s: no warmup requested", backend.name)

        artefact_writer: ArtefactWriter | None = None
        artefacts_dir: Path | None = None
        if config.mode == "accuracy":
            artefacts_dir = output_dir / "artefacts"
            artefact_writer = ArtefactWriter(artefacts_dir)
            _log.info("accuracy mode: writing artefacts to %s", artefacts_dir)

        response_writer = LoadgenResponseWriter()

        sut = WanSUT(
            dispatcher=dispatcher,
            qsl=qsl,
            response_writer=response_writer,
            artefact_writer=artefact_writer,
        )

        # ---------- Plug LoadGen in. ----------
        settings = _build_test_settings(config)
        log_settings = _build_log_settings(output_dir, config.enable_loadgen_trace)

        lg_sut = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
        lg_qsl = lg.ConstructQSL(
            qsl.total_sample_count,
            qsl.performance_sample_count,
            qsl.load_query_samples,
            qsl.unload_query_samples,
        )

        audit_path = ""
        if config.audit_conf_path is not None and config.audit_conf_path.exists():
            audit_path = str(config.audit_conf_path)
            _log.info("compliance run: audit.config=%s (via StartTestWithLogSettings)", audit_path)
        _log.info(
            "starting LoadGen test: scenario=%s mode=%s backend=%s qsl=%d perf_count=%d",
            config.scenario,
            config.mode,
            config.backend,
            qsl.total_sample_count,
            qsl.performance_sample_count,
        )
        try:
            lg.StartTestWithLogSettings(lg_sut, lg_qsl, settings, log_settings, audit_path)
        finally:
            lg.DestroyQSL(lg_qsl)
            lg.DestroySUT(lg_sut)
            dispatcher.shutdown()
        _log.info("LoadGen test finished")

        # In accuracy mode, symlink mlperf_log_accuracy.json into artefacts/
        # so downstream tools can consume a single directory without duplicating
        # the (potentially multi-GB) log on disk, and flush the VBench
        # custom_input prompts.json mapping.
        if artefacts_dir is not None and artefact_writer is not None:
            artefact_writer.finalize()
            acc_log = output_dir / "mlperf_log_accuracy.json"
            if acc_log.exists():
                link = artefacts_dir / "mlperf_log_accuracy.json"
                link.unlink(missing_ok=True)
                link.symlink_to("../mlperf_log_accuracy.json")
        _finalize_post_run_overhead(config, rank=rank)
    finally:
        backend.teardown()

    summary = {
        "scenario": config.scenario,
        "mode": config.mode,
        "backend": config.backend,
        "issued": sut.issued_count,
        "completed": sut.completed_count,
        "output_dir": str(output_dir),
        "artefacts_dir": str(artefacts_dir) if artefacts_dir else None,
    }
    _write_metadata(output_dir, config, summary)
    return RunResult(
        scenario=config.scenario,
        mode=config.mode,
        backend=config.backend,
        output_dir=output_dir,
        issued=sut.issued_count,
        completed=sut.completed_count,
        artefacts_dir=artefacts_dir,
    )
