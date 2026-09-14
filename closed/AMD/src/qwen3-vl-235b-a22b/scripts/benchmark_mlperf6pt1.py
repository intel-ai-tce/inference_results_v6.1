#!/usr/bin/env python3
"""Benchmarking harness
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from orchestrators import (  # sibling module (script dir is on sys.path); Hydra-free orchestrators
    ServerHandle,
    resolve_orchestrator,
    tee,
    wait_until_healthy,
)
from parse_results import (  # sibling module (script dir is on sys.path)
    append_row,
    report_sample_counts,
    status_with_failures,
)
from utils import write_run_meta

import numa  # sibling module: NUMA-aware CPU pinning for the multi-replica topology

log = logging.getLogger(__name__)

_OOM_MARKERS = (
    "out of memory",
    "hsa_status_error_out_of_resources",
    "not enough gpu memory",
    "outofmemoryerror",
)

CONFIG_DIR = "../configs"  # benchmark_mlperf6pt1.py lives in scripts/; configs/ is one level up, in the qwen3-vl-235b-a22b dir
PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)  # project root = the qwen3-vl-235b-a22b dir (parent of scripts/, i.e. <submission>/src/qwen3-vl-235b-a22b/)
# submission root = <submission>/ (code lives at <submission>/src/qwen3-vl-235b-a22b/); runs + packages live at <submission>/outputs
OmegaConf.register_new_resolver("submission_root", lambda: str(PROJECT_ROOT.parents[1]), replace=True)


def _scenario_type(scenario_name: str) -> str:
    """Map a scenario file name to its type (offline/server/interactive) for the output path."""
    name = str(scenario_name).lower()
    return next(
        (t for t in ("offline", "server", "interactive") if name.startswith(t)), "other"
    )


# Lets hydra.run.dir/sweep.dir use the scenario *type* (e.g. ${scenario_type:${scenario}}).
OmegaConf.register_new_resolver("scenario_type", _scenario_type, replace=True)


def load_scenario(cfg: DictConfig) -> tuple[Path, DictConfig]:
    """Resolve the scenario YAML named by ``cfg.scenario``, flat under ``configs/``.

    The dev repo nests scenarios by mode (``configs/<mode>/<name>.yaml``); this frozen submission copy
    ships only the reference scenarios, flattened directly under ``configs/``. ``mode`` no longer selects a
    scenario subdir here (it still tags the output dir and force-disables profiling for reference)."""
    name = cfg.get("scenario")
    if not name:
        sys.exit("error: 'scenario' is not set (see benchmark.yaml)")
    path = Path(__file__).resolve().parent / CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        sys.exit(
            f"error: scenario config not found: '{path}' (check 'scenario' in benchmark.yaml)"
        )
    return path, OmegaConf.load(path)


def apply_scenario_overrides(scenario_cfg: DictConfig, cfg: DictConfig) -> bool:
    """Merge ``cfg.scenario_overrides`` (and the ``cfg.model`` shortcut) onto ``scenario_cfg`` in place.

    The scenario YAML is loaded outside the Hydra tree (``load_scenario`` -> ``OmegaConf.load``), so
    ``model_params.name=...`` on the CLI never reaches it. These two benchmark.yaml keys are the
    supported override path: a general ``scenario_overrides`` dict merged onto the scenario, plus a
    ``model`` shortcut that wins over both (sets ``model_params.name``).

    Returns True if anything was overridden -- the caller then materializes the resolved scenario to
    a file and hands *that* to the endpoints client, so vLLM and the client serve the same model.
    Returns False when nothing changed, so reference/submission runs pass the vendored scenario file
    through byte-for-byte (zero risk).
    """
    OmegaConf.set_struct(scenario_cfg, False)  # allow overrides to add keys (e.g. +scenario_overrides.x)
    applied = False
    overrides = cfg.get("scenario_overrides")
    if overrides and len(overrides) > 0:
        scenario_cfg.merge_with(overrides)
        applied = True
    model = cfg.get("model")
    if model:  # the model shortcut wins over scenario_overrides.model_params.name
        scenario_cfg.model_params.name = model
        applied = True
    return applied


def _apply_client_tuning(scenario_cfg: DictConfig, client) -> bool:
    """Merge the orchestrator's machine-selected ``client:`` block onto ``scenario.settings.client``.
    """
    if not client or len(client) == 0:
        return False
    OmegaConf.set_struct(scenario_cfg, False)
    settings = scenario_cfg.get("settings")
    if settings is None:
        return False
    base = settings.get("client")
    settings.client = OmegaConf.merge(base or OmegaConf.create({}), client)
    return True


def parse_endpoint(cfg: DictConfig) -> tuple[str, int]:
    """Derive (host, port) from the endpoints config's first endpoint URL."""
    url = cfg.endpoint_config.endpoints[0]
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname or "localhost", parsed.port or 8000


def run_status(run_dir: Path, started: bool, healthy: bool, rc: int) -> str:
    """Classify a run's outcome for the CSV: ok | oom | server_failed | client_failed | failed.

    ``failed`` = server came up and the client returned, but most samples errored.
    """
    if started and not healthy:
        # vllm backend writes vllm_server.log; the HAProxy backend writes one log per worker.
        logs = list(run_dir.glob("vllm_server.log")) + sorted(
            run_dir.glob("vllm_worker_*.log")
        )
        text = " ".join(
            p.read_text(errors="ignore").lower() for p in logs if p.is_file()
        )
        return "oom" if any(m in text for m in _OOM_MARKERS) else "server_failed"
    if rc != 0:
        return "client_failed"
    # Server was healthy and the client exited 0, but the engine may have died mid-run (mass
    # sample failures, e.g. a cudagraph/compile crash). Downgrade to "failed" in that case.
    completed, failed = report_sample_counts(run_dir)
    return status_with_failures("ok", completed, failed)


def link_shared_datasets(cwd: Path, datasets_dir: Path) -> None:
    """Point the client's cwd-relative dataset dirs at persistent shared dirs via symlinks.

    The endpoints client materializes TWO multi-GB, cwd-relative caches with no upstream
    env/config override, and we run the client with ``cwd=run_dir``:
      - ``datasets/``       -- the images->base64 converted parquet, and
      - ``dataset_cache/``  -- the raw HF-dataset download/convert cache (~6 GB parquet per
                               dataset; e.g. ``shopify_product_catalogue/.../train+test.parquet``).
    Left alone, BOTH land a full copy inside every ``outputs/`` run (dataset_cache alone was 261 GB
    / 95% of the home volume on 2026-07) and re-download + re-convert every run. Symlinking each to a
    shared location under the mounted HF cache means it's built once and reused across runs and team
    members (the loaders reuse an existing parquet via a ``dst_path.exists()``-style check).

    ``dataset_cache`` targets a sibling of ``datasets_dir`` (``<parent>/endpoints_dataset_cache``) so
    the two caches stay in distinct shared trees without needing a second config key.
    """
    links = {
        "datasets": datasets_dir,
        "dataset_cache": datasets_dir.parent / "endpoints_dataset_cache",
    }
    for name, target in links.items():
        target.mkdir(parents=True, exist_ok=True)
        link = cwd / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)


def run_endpoints_client(
    config: Path, cwd: Path, log_file, datasets_dir: Path | None = None,
    pin_prefix: list[str] | None = None,
) -> int:
    """Run the endpoints client (one scenario), tee'ing its output to console + ``log_file``."""
    if datasets_dir is not None:
        link_shared_datasets(cwd, datasets_dir)
    cmd = (pin_prefix or []) + ["inference-endpoint", "benchmark", "from-config", "-c", str(config)]
    log.info("running benchmark: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tee(proc.stdout, log_file)
    return proc.wait()


def run_warmup(
    scenario_cfg: DictConfig,
    run_dir: Path,
    n_samples: int,
    client_pin: list[str] | None,
    shared_datasets,
) -> int:
    """Untimed warmup pass before the timed run: issue a small batch so runtime-JIT kernels
    (flydsl MoE codegen, ``mha_varlen``, etc.) compile during SUT bring-up instead of during the
    measured run.
    """
    wcfg = OmegaConf.create(OmegaConf.to_container(scenario_cfg, resolve=False))
    OmegaConf.set_struct(wcfg, False)
    ds = wcfg.get("datasets")
    if ds:
        perf_only = [d for d in ds if str(d.get("type", "")).lower() == "performance"]
        if perf_only:
            wcfg.datasets = perf_only
    # Keep the scenario's OWN load pattern (max_throughput for offline, poisson/etc. for online).
    OmegaConf.update(
        wcfg, "settings.runtime.n_samples_to_issue", int(n_samples), force_add=True
    )
    wcfg.report_dir = "results_warmup/"  # keep warmup output out of the timed results/ dir
    warmup_file = run_dir / "scenario_warmup.yaml"
    OmegaConf.save(wcfg, warmup_file)
    with open(run_dir / "warmup_client.log", "w") as wlog:
        return run_endpoints_client(
            warmup_file,
            run_dir,
            wlog,
            Path(shared_datasets) if shared_datasets else None,
            pin_prefix=client_pin,
        )


def profiling_enabled(server_cfg: DictConfig, started: bool) -> bool:
    """Whether to run the torch profiler for this run. DEV/diagnostic only
    (``harness.profile``, off by default); never used by a submission run."""
    return bool(server_cfg.get("profile", False)) and started


def post_profile(host: str, port: int, action: str, timeout_s: int = 30) -> bool:
    """POST to a vLLM profiler control route (``start_profile``|``stop_profile``). Best-effort.

    Profiling is diagnostic and never required for a run, so any failure here only warns and is
    swallowed -- it must not affect the run outcome. vLLM flushes a torch trace only on
    ``/stop_profile``, so a caller that starts profiling must always stop it afterward.
    """
    url = f"http://{host}:{port}/{action}"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            log.info("%s -> HTTP %s", action, resp.status)
            return resp.status == 200
    except (
        urllib.error.URLError,
        OSError,
    ) as exc:  # noqa: BLE001 - diagnostic, non-critical
        log.warning("%s failed: %s", action, exc)
        return False


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="benchmark")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)  # tolerate optional/missing keys via .get()
    scenario_file, scenario_cfg = load_scenario(cfg)

    # Orchestrator config + client-tuning merge (AMD submission: single vendor)
    vendor = "amd"
    orch = cfg.get("orchestrator") or OmegaConf.create({})
    orch_name = str(orch.get("name", "haproxy"))
    orch_sec = OmegaConf.merge(OmegaConf.create({}), orch.get(vendor) or OmegaConf.create({}))
    OmegaConf.set_struct(orch_sec, False)
    orch_client = orch_sec.pop("client", None)  # split client-load tuning from orchestrator settings
    sc_applied = _apply_client_tuning(scenario_cfg, orch_client)

    overridden = apply_scenario_overrides(scenario_cfg, cfg) or sc_applied
    model = scenario_cfg.model_params.name
    server_cfg = OmegaConf.merge(
        cfg.get("server") or OmegaConf.create({}),
        cfg.get("harness") or OmegaConf.create({}),
        orch_sec,
    )
    OmegaConf.set_struct(server_cfg, False)
    nreplicas = int(server_cfg.get("nreplicas", 1))
    server_cfg.num_workers = nreplicas
    orchestrator = resolve_orchestrator(server_cfg, nreplicas, orch_name)
    log.info("orchestrator=%s vendor=%s", orch_name, vendor)

    host, port = parse_endpoint(scenario_cfg)  # scenario endpoint (:8000): HAProxy frontend / vLLM server

    # NUMA binding (default on)
    client_pin = numa.loadgen_prefix(
        numa.compute_plan(nreplicas, bool(server_cfg.get("enable_numa_binding", True)))
    )
    scenario_cfg.enable_cpu_affinity = not bool(client_pin)
    overridden = overridden or bool(client_pin)  # force a resolved scenario only when we pin

    # apply the server env vars (single consolidated block under `server.env`)
    env = os.environ.copy()
    for key, value in (server_cfg.get("env") or {}).items():
        env[str(key)] = str(value)

    # consolidate logs from hydra and endpoints client.
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info("run dir: %s", run_dir)
    log.info(
        "scenario=%s model=%s endpoint=%s:%s serving=%s nreplicas=%d",
        scenario_file.name,
        model,
        host,
        port,
        orchestrator.name,
        nreplicas,
    )

    # apply scenario overrides if any
    scenario_file_for_client = scenario_file
    if overridden:
        scenario_file_for_client = run_dir / "scenario_resolved.yaml"
        OmegaConf.save(scenario_cfg, scenario_file_for_client)
        log.info(
            "scenario overrides applied; wrote resolved scenario to %s", scenario_file_for_client
        )

    # Quantization is detected from the checkpoint's config.json; the submission uses no manual
    # override (the dev-only cfg.quant_override knob is dropped from this frozen submission copy).
    write_run_meta(run_dir, model, server_cfg, None)

    rc = 1
    started = bool(server_cfg.get("start_server", True))
    healthy = False
    handle: ServerHandle | None = None
    client_log = None

    profile_dir: Path | None = None
    profile_enabled = profiling_enabled(server_cfg, started)
    prof_host = "127.0.0.1"
    base_port = int(server_cfg.get("base_port", 8001))
    if orchestrator.name == "haproxy":
        prof_ports = [base_port]  # rank0 only (worker 0); re-add all workers when multi-GPU capture returns
    else:
        prof_ports = [int(port)]
    if profile_enabled:
        profile_dir = run_dir / (cfg.get("trace_dirname") or "traces")
        profile_dir.mkdir(parents=True, exist_ok=True)

    try:
        if started:
            # The orchestrator launches the server(s), tees their logs, gates per-worker readiness +
            # (HAProxy path) launches the proxy. It returns a handle whose .primary is the process
            # whose early exit signals failure, and whose .stop() tears everything down.
            handle = orchestrator.start(
                model, server_cfg, host, port, run_dir, env, run_dir, profile_dir
            )
        healthy = wait_until_healthy(
            host,
            port,
            server_cfg.get("health_timeout", 1800),
            handle.primary if handle else None,
        )
        if healthy:
            shared_datasets = cfg.get("endpoints_datasets_dir")
            # Untimed warmup so runtime-JIT kernels compile during SUT bring-up, not the timed run.
            warmup_cfg = server_cfg.get("warmup") or OmegaConf.create({})
            if started and bool(warmup_cfg.get("enabled", False)):
                wn = int(warmup_cfg.get("n_samples", 512))
                log.info(
                    "warmup: issuing %d samples (untimed) to compile JIT kernels before the timed run",
                    wn,
                )
                wrc = run_warmup(scenario_cfg, run_dir, wn, client_pin, shared_datasets)
                log.info("warmup complete (rc=%s); starting timed run", wrc)
            client_log = open(run_dir / "endpoints_client.log", "w")
            if profile_enabled:
                for pport in prof_ports:
                    post_profile(prof_host, pport, "start_profile")
            try:
                rc = run_endpoints_client(
                    scenario_file_for_client,
                    run_dir,
                    client_log,
                    Path(shared_datasets) if shared_datasets else None,
                    pin_prefix=client_pin,
                )
            finally:
                # vLLM flushes the trace to the trace dir only on stop; always stop if started.
                if profile_enabled:
                    for pport in prof_ports:
                        post_profile(prof_host, pport, "stop_profile")
        else:
            # rc stays 1; in a sweep the run below logs and continues to the next combination.
            log.error("server did not become healthy; marking this run failed")
    except Exception as exc:  # record-and-continue: never let one run abort a sweep
        log.error("run failed before completion: %s", exc)
    finally:
        if handle is not None and not server_cfg.get("keep_server", False):
            handle.stop()
        if client_log is not None:
            client_log.close()

    # Per-worker torch traces stay in place under <run_dir>/<trace_dirname>/rank<i>/ (each with its
    # own profiler_out_<i>.txt). We intentionally do NOT flatten/rename them, so each trace keeps its
    # rank folder + op table together.

    # Record the run outcome so failed runs (e.g. tp=1 OOM) leave a trace.
    status = run_status(run_dir, started, healthy, rc)
    try:
        (run_dir / "status.txt").write_text(status + "\n")
    except OSError as exc:  # noqa: BLE001
        log.warning("could not write status.txt: %s", exc)
    try:
        csv_path = append_row(run_dir)
        log.info("appended results row (status=%s) to %s", status, csv_path)
    except Exception as exc:  # noqa: BLE001 - tracking is non-critical
        log.warning("could not write results CSV: %s", exc)

    if rc != 0:
        # In a sweep, log and continue so one bad combination doesn't abort the rest.
        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.error("run failed with exit code %s (continuing sweep)", rc)
        else:
            sys.exit(rc)


if __name__ == "__main__":
    main()
