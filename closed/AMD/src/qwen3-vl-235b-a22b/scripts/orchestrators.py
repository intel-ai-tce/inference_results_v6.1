#!/usr/bin/env python3
"""Pluggable serving orchestrators for the Q3VL harness.

This module is **Hydra-free on purpose**, so the launch / teardown logic can be driven by the
benchmark harness (``benchmark_mlperf6pt1.py``).

The serving orchestrator:
  - ``HAProxyOrchestrator`` -- N TP1 ``vllm serve`` workers behind an HAProxy least-conn proxy
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from omegaconf import OmegaConf

import numa

from utils import (  # sibling module (script dir on sys.path)
    effective_mm_encoder_attn_backend,
    effective_quantization_flag,
)

log = logging.getLogger(__name__)


# --- low-level helpers (shared) --------------------------------------------------------------


def tee(stream, log_file, console=True) -> None:
    """Forward a subprocess's merged output to ``log_file`` (always) and the console (if ``console``)."""
    for line in iter(stream.readline, ""):
        if console:
            sys.stdout.write(line)
            sys.stdout.flush()
        try:
            log_file.write(line)
            log_file.flush()
        except ValueError:
            # The main thread may close log_file during shutdown while this daemon thread is still
            # draining buffered output. Drop the file copy rather than crashing the thread with
            # "I/O operation on closed file" at run end.
            pass
    stream.close()


def build_vllm_cmd(model, server_cfg, host, port, profile_dir=None):
    """Build the ``vllm serve`` argv from the ``server:``
    Booleans use vLLM's BooleanOptionalAction form (``--flag`` / ``--no-flag``).
    NOTE:``--no-enable-prefix-caching`` required for MLPerf compliance;
    """
    def boolflag(key: str) -> str:
        dashed = key.replace("_", "-")
        return f"--{dashed}" if server_cfg[key] else f"--no-{dashed}"

    # Eager profiling
    _comp = str(server_cfg.compilation_config)
    if server_cfg.get("enforce_eager"):
        _cc = json.loads(_comp)
        _cc["mode"] = 0
        _cc["cudagraph_mode"] = "NONE"
        _comp = json.dumps(_cc)

    cmd = [
        "vllm",
        "serve",
        model,
        "--gpu_memory_utilization",
        str(server_cfg.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(server_cfg.tensor_parallel_size),
        "--max-model-len",
        str(server_cfg.max_model_len),
        "--max-num-batched-tokens",
        str(server_cfg.max_number_of_batched_tokens),
        "--max-num-seqs",
        str(server_cfg.max_num_seqs),
        "--mm-encoder-tp-mode",
        str(server_cfg.mm_encoder_tp_mode),
        "--kv-cache-dtype",
        str(server_cfg.kv_cache_dtype),
        "--compilation-config",
        _comp,
        boolflag("async_scheduling"),
        boolflag("enable_chunked_prefill"),
        boolflag("enable_prefix_caching"),
        boolflag("enable_expert_parallel"),
        "--limit-mm-per-prompt.video",
        "0",
        "--host",
        host,
        "--port",
        str(port),
    ]
    attn = server_cfg.get("attention_backend")
    if attn not in (None, "", "auto"):
        cmd += ["--attention-backend", str(attn)]
    mm_attn = effective_mm_encoder_attn_backend(server_cfg)
    if mm_attn is not None:
        cmd += ["--mm-encoder-attn-backend", str(mm_attn)]
    mm_dtype = server_cfg.get("mm_encoder_attn_dtype")
    if mm_dtype not in (None, "", "auto"):
        cmd += ["--mm-encoder-attn-dtype", str(mm_dtype)]
    mm_scale_path = server_cfg.get("mm_encoder_fp8_scale_path")
    if mm_scale_path not in (None, "", "auto"):
        cmd += ["--mm-encoder-fp8-scale-path", str(mm_scale_path)]
    mm_scale_save = server_cfg.get("mm_encoder_fp8_scale_save_path")
    if mm_scale_save not in (None, "", "auto"):
        cmd += ["--mm-encoder-fp8-scale-save-path", str(mm_scale_save)]
    dp_size = server_cfg.get("data_parallel_size")
    if dp_size is not None and int(dp_size) > 1:
        cmd += ["--data-parallel-size", str(dp_size)]
    quant_flag = effective_quantization_flag(server_cfg)
    if quant_flag is not None:
        cmd += ["--quantization", quant_flag]
    gen_cfg = server_cfg.get("generation_config", "auto")
    if gen_cfg not in (None, "", "none"):
        cmd += ["--generation-config", str(gen_cfg)]
    seed = server_cfg.get("seed")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if profile_dir is not None:
        overrides = server_cfg.get("profiler")
        knobs = OmegaConf.to_container(overrides, resolve=True) if overrides else {}
        pcfg = {"profiler": "torch", "torch_profiler_dir": str(profile_dir), **knobs}
        cmd += ["--profiler-config", json.dumps(pcfg)]
    return cmd


def spawn(cmd, cwd, env, log_path, console=True, preexec_fn=None):
    """Popen ``cmd`` in its own process group, tee'ing output to ``log_path`` (+ console if ``console``).

    ``preexec_fn`` runs in the child after fork (in addition to the ``start_new_session`` setsid)."""
    log_file = open(log_path, "w")
    log.debug("starting: %s", " ".join(str(c) for c in cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            preexec_fn=preexec_fn,
        )
    except BaseException:
        log_file.close()  # don't leak the fd if the binary is missing / Popen fails
        raise
    t = threading.Thread(target=tee, args=(proc.stdout, log_file, console), daemon=True)
    t.start()
    return proc, t, log_file


def wait_until_healthy(host, port, timeout_s, proc=None) -> bool:
    """Poll ``http://host:port/health`` until it returns 200. Return True if healthy."""
    url = f"http://{host}:{port}/health"
    log.info("waiting for %s (timeout %ss)...", url, timeout_s)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            log.error("process exited early with code %s (%s)", proc.returncode, url)
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    log.info("%s is healthy.", url)
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(5)
    log.error("%s did not become healthy within %ss", url, timeout_s)
    return False


def _stop_proc(proc) -> None:
    """Terminate one process group (SIGTERM, then SIGKILL after 30s)."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass


# --- HAProxy config rendering ---------------------------------------------------------------


def render_haproxy_cfg(
    frontend_host,
    frontend_port,
    worker_host,
    worker_ports,
    balance="leastconn",
    timeout_s=600,
    maxconn=4096,
    stats_socket=None,
) -> str:
    """Render an HAProxy config: one frontend on the scenario endpoint -> N vLLM worker backends.

    ``balance leastconn`` = least-outstanding routing; ``option http-server-close`` keeps the backend
    connection count ~= in-flight generations so leastconn tracks real load. ``timeout_s`` must exceed
    the longest generation (SSE streams stay open for the whole response). ``stats_socket`` (a unix
    path) enables ``show stat`` so the orchestrator can poll per-worker request counts.
    """
    servers = "\n".join(
        f"    server w{i} {worker_host}:{p} check" for i, p in enumerate(worker_ports)
    )
    stats_line = (
        f"\n    stats socket {stats_socket} mode 660 level admin" if stats_socket else ""
    )
    return f"""# generated by orchestrators.py -- regenerated each run; do not hand-edit
global
    log stdout format raw local0
    maxconn {maxconn}{stats_line}

defaults
    mode http
    log global
    option httplog
    timeout connect 5s
    timeout client  {timeout_s}s
    timeout server  {timeout_s}s
    option http-server-close

frontend mlperf_in
    bind {frontend_host}:{frontend_port}
    default_backend vllm_pool

backend vllm_pool
    balance {balance}
    option httpchk GET /health
{servers}
"""


# --- live per-worker request counter (HAProxy stats socket) ---------------------------------


def _poll_haproxy_stats(sock_path, stop_event, interval) -> None:
    """Periodically log a compact per-worker request counter from HAProxy's stats socket.

    Connects to the unix stats socket, runs ``show stat`` (CSV), and logs total processed + per-worker
    totals + in-flight, but only when the processed count changes (so it stays quiet when idle).
    Best-effort: transient socket errors are ignored. Runs until ``stop_event`` is set.
    """
    prev = -1
    while not stop_event.wait(interval):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect(str(sock_path))
            s.sendall(b"show stat\n")
            buf = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
            s.close()
        except OSError:
            continue  # haproxy not up yet / socket busy -- try again next tick
        per, inflight, total = [], 0, None
        for ln in buf.decode(errors="ignore").splitlines():
            if not ln or ln.startswith("#"):
                continue
            c = ln.split(",")  # CSV cols: 0=pxname 1=svname 4=scur 7=stot
            if len(c) < 8 or c[0] != "vllm_pool":
                continue
            sv, scur, stot = c[1], c[4], c[7]
            if sv == "BACKEND":
                total = int(stot or 0)
            elif sv != "FRONTEND":
                per.append(f"{sv}={stot or 0}")
                inflight += int(scur or 0)
        if total is not None and total != prev:
            log.info("[haproxy] processed=%d in-flight=%d | %s", total, inflight, " ".join(per))
            prev = total


# --- orchestrators --------------------------------------------------------------------------


class ServerHandle:
    """Opaque handle over one or more launched processes: health-check via ``primary``, ``stop()``."""

    def __init__(self, procs=None, threads=None, log_files=None, primary=None, stop_event=None):
        self.procs = procs if procs is not None else []
        self.threads = threads if threads is not None else []
        self.log_files = log_files if log_files is not None else []
        self.primary = primary  # the process whose early exit signals failure to wait_until_healthy
        self.stop_event = stop_event  # signals background helpers (e.g. the stats poller) to exit

    def stop(self) -> None:
        """SIGTERM/SIGKILL every process group (reverse launch order: proxy before workers),
        signal background helpers, join threads, close log files. Idempotent."""
        if self.stop_event is not None:
            self.stop_event.set()
        for p in reversed(self.procs):
            _stop_proc(p)
        for t in self.threads:
            t.join(timeout=10)
        for f in self.log_files:
            try:
                f.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass


def _launch_workers(model, server_cfg, worker_host, worker_ports, cwd, env, log_dir, profile_dir=None):
    """Start len(worker_ports) NUMA-pinned TP1 vLLM workers and gate per-worker"""
    log_dir = Path(log_dir)
    # Compliance + topology guards: no prefix caching on any worker; one GPU per worker => TP1.
    if bool(server_cfg.get("enable_prefix_caching", False)):
        raise SystemExit(
            "multi-worker backends require server.enable_prefix_caching=false (MLPerf invariant)"
        )
    tp = int(server_cfg.get("tensor_parallel_size", 1))
    if tp != 1:
        raise SystemExit(
            f"multi-worker backends (nreplicas>1) require tensor_parallel_size=1, got {tp} -- the "
            "fan-out is one GPU per worker. Use a TP1 server config, or nreplicas=1 for a TP>1 single server."
        )
    n = len(worker_ports)
    # Worker i -> GPU server_cfg.gpu_ids[i] (explicit knob; e.g. [7] for a 1-worker run off GPU 0);
    # default GPUs 0..n-1.
    gids = server_cfg.get("gpu_ids")
    gpu_ids = [str(g) for g in gids] if gids else [str(j) for j in range(n)]
    # NUMA pinning plan (default on): each worker -> its GPU-local node cores; the load generator
    # (HAProxy / endpoints client) -> the reserved tail. Removes the load-generator-vs-worker CPU
    # contention that otherwise leaves GPUs idle. None (no-op) on a single-NUMA box.
    numa_plan = numa.compute_plan(
        n, bool(server_cfg.get("enable_numa_binding", True)), gpu_ids=[int(g) for g in gpu_ids]
    )
    if numa_plan:
        log.info(
            "NUMA binding ON: workers pinned to GPU-local cores; loadgen reserved cores=%s",
            numa_plan["loadgen_cores"],
        )
    timeout = int(server_cfg.get("health_timeout", 1800))
    stream = bool(server_cfg.get("stream_worker_logs", False))
    if profile_dir is not None:
        log.info("profiling RANK0 ONLY (worker 0) -> %s/rank0/ (multi-GPU capture TBD)", profile_dir)
    log.info(
        "launching %d TP1 workers (one GPU each); per-worker logs -> %s/vllm_worker_*.log", n, log_dir
    )
    if not stream:
        log.info("worker console output -> log files only (set harness.stream_worker_logs=true to stream)")
    handle = ServerHandle()
    try:
        for i, wport in enumerate(worker_ports):
            log.info("[%d/%d] starting worker on GPU %s -> %s:%d", i + 1, n, gpu_ids[i], worker_host, wport)
            # Pin each worker to one GPU (ROCm: HIP_VISIBLE_DEVICES).
            wenv = dict(env)
            wenv["HIP_VISIBLE_DEVICES"] = gpu_ids[i]
            wprofile = None
            if profile_dir is not None and i == 0:
                wprofile = Path(profile_dir) / "rank0"
                wprofile.mkdir(parents=True, exist_ok=True)
            cmd = build_vllm_cmd(model, server_cfg, worker_host, wport, wprofile)
            cmd = numa.worker_prefix(numa_plan, i) + cmd  # GPU-local NUMA pin (no-op if disabled)
            p, t, lf = spawn(cmd, cwd, wenv, log_dir / f"vllm_worker_{i}.log", console=stream)
            handle.procs.append(p)
            handle.threads.append(t)
            handle.log_files.append(lf)
        # Readiness gate: every worker must answer /health before we let load start.
        ready = 0
        for i, wport in enumerate(worker_ports):
            if wait_until_healthy(worker_host, wport, timeout, handle.procs[i]):
                ready += 1
                log.info("[%d/%d] worker ready (%s:%d)", ready, n, worker_host, wport)
            else:
                log.error("[%d/%d] worker FAILED to become healthy (%s:%d)", i + 1, n, worker_host, wport)
        # Fail fast on a degraded worker set. Proceeding with < n workers would silently run on
        # fewer GPUs (HAProxy just routes around the dead backend) yet still report status=ok --
        # invalidating throughput/latency. Raise so the run is marked failed; the except below tears
        # down the survivors. A GPU occupied at startup shows as "Free memory on device ..." in the
        # worker log. Override min_healthy_workers only for a deliberately degraded test.
        required = int(server_cfg.get("min_healthy_workers", n))
        if ready < required:
            raise RuntimeError(
                f"only {ready}/{n} workers became healthy (need >= {required}); aborting to avoid a "
                f"silently degraded run -- check {log_dir}/vllm_worker_*.log "
                f"('Free memory on device' = a GPU was occupied at startup)"
            )
        if ready == n:
            log.info("all %d workers ready", n)
        else:
            log.warning(
                "%d/%d workers ready (>= min_healthy_workers=%d); proceeding DEGRADED", ready, n, required
            )
        return handle, numa_plan
    except BaseException:
        handle.stop()  # never leak partially-launched workers
        raise


class HAProxyOrchestrator:
    """N TP1 ``vllm serve`` workers (one GPU each) behind an HAProxy least-conn proxy."""

    name = "haproxy"

    def start(self, model, server_cfg, host, port, cwd, env, log_dir, profile_dir=None):
        log_dir = Path(log_dir)
        n = int(server_cfg.get("num_workers", 8))
        base_port = int(server_cfg.get("base_port", 8001))
        balance = str(server_cfg.get("balance", "leastconn"))
        timeout = int(server_cfg.get("health_timeout", 1800))
        stream = bool(server_cfg.get("stream_worker_logs", False))
        status_interval = int(server_cfg.get("status_interval", 15))
        worker_host = "127.0.0.1"
        worker_ports = [base_port + i for i in range(n)]

        handle, numa_plan = _launch_workers(
            model, server_cfg, worker_host, worker_ports, cwd, env, log_dir, profile_dir
        )
        try:
            # Stats socket: UNIX socket paths have a ~108-char limit, but the run dir can be long.
            # Keep it short in /tmp, keyed by the frontend port; unlink any stale socket.
            sock_path = Path(f"/tmp/q3vl-haproxy-{port}.sock")
            sock_path.unlink(missing_ok=True)
            cfg_path = log_dir / "haproxy.cfg"
            cfg_path.write_text(
                render_haproxy_cfg(
                    host, port, worker_host, worker_ports, balance, timeout,
                    stats_socket=(sock_path if status_interval > 0 else None),
                )
            )
            hp, ht, hlf = spawn(
                numa.loadgen_prefix(numa_plan) + ["haproxy", "-f", str(cfg_path)],
                cwd, env, log_dir / "haproxy.log", console=stream,
            )
            handle.procs.append(hp)
            handle.threads.append(ht)
            handle.log_files.append(hlf)
            handle.primary = hp
            log.info("HAProxy up on %s:%d (balance=%s) fronting %d workers", host, port, balance, n)
            if status_interval > 0:
                handle.stop_event = threading.Event()
                poller = threading.Thread(
                    target=_poll_haproxy_stats,
                    args=(sock_path, handle.stop_event, status_interval),
                    daemon=True,
                )
                poller.start()
                handle.threads.append(poller)
            return handle
        except BaseException:
            handle.stop()  # never leak partially-launched workers / proxy
            raise


def resolve_orchestrator(server_cfg, nreplicas, name="haproxy"):
    """Return the serving orchestrator: N TP1 ``vllm serve`` workers behind an HAProxy least-conn
    proxy (the AMD submission path). Only TP1 is supported -- the fan-out is one GPU per worker."""
    tp = int(server_cfg.get("tensor_parallel_size", 1) or 1)
    dp = int(server_cfg.get("data_parallel_size", 1) or 1)
    if tp > 1 or dp > 1:
        raise SystemExit(
            f"this AMD submission harness serves NxTP1 workers behind HAProxy; "
            f"tensor_parallel_size/data_parallel_size must be 1 (got tp={tp}, dp={dp})."
        )
    router = str(name or "haproxy").lower()
    if router not in ("haproxy", "", "none"):
        raise SystemExit(f"unknown orchestrator={router!r}; expected 'haproxy'")
    return HAProxyOrchestrator()
