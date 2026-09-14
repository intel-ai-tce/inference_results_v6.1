"""
AMD-style multi-engine standalone backend, harness-side.

Workflow:
    start_server.sh --role standalone --hardware <hw> --model <m>
        (spawns N vLLM workers, one per GPU, listed in servers.standalone)
    run.sh <model> <scenario> <mode> --backend standalone
        (this SUT; connects to all N workers via ZMQ)

Each worker speaks the same ZMQ protocol as `src/workers/standalone.py`:
    PULL  (worker receive)  — request:  {"id": str, "prompt": [int, ...]}
    PUSH  (worker send)     — response: {"id": str, "first_token_ids": [...]}  (first token)
                                        {"id": str, "token_ids": [...]}        (complete)

The SUT:
- Opens N PUSH/PULL socket pairs (one per worker)
- Routes each LoadGen sample to the worker with the shortest queue + lowest
  recent-token weight (shortest_queue_with_tokens; matches AMD's default)
- Has one async collector task per worker that drains PULL and emits
  LoadGen first-token / complete responses
- Issues a harness-side warmup against every worker before signaling ready
  so all engines have hot CUDA graph caches before LoadGen starts

Concurrency / GIL notes:
- Engines run in *separate OS processes* (spawned by start_server.sh), so
  no GIL is shared between SUT and engines. The harness Python interpreter
  is dedicated to LoadGen + this SUT.
- Inside the SUT we keep exactly two OS threads: (a) the LoadGen caller
  thread (which only invokes ``issue_queries``) and (b) a daemon thread
  running an asyncio loop with all issuer + N collector coroutines.
- All hot-path C extensions (msgpack pack/unpack, zmq.asyncio.recv,
  lg.QuerySamplesComplete) release the GIL during their kernel work, so
  the LoadGen thread and the asyncio thread do not contend for it under
  normal load.
- Counters touched only by the asyncio loop are accessed without locks
  (single writer = single thread = no race). ``_n_issued`` is written
  only by the LoadGen thread; the asyncio loop only reads it, and a torn
  read is harmless (just one off in the progress log).
- uvloop is used when available (2-4x faster than the selector loop on
  socket-heavy workloads); we fall back to the stdlib asyncio loop
  otherwise.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import msgpack as _msgpack
    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False

try:
    import uvloop as _uvloop
    _HAS_UVLOOP = True
except ImportError:
    _HAS_UVLOOP = False


def _new_event_loop() -> asyncio.AbstractEventLoop:
    """Prefer uvloop when available (2-4x throughput vs selector loop)."""
    if _HAS_UVLOOP:
        return _uvloop.new_event_loop()
    return asyncio.new_event_loop()



_STOP = object()

import mlperf_loadgen as lg
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from sut.base import SUT, SUTConfig
from sut.utils import (
    create_response_and_send_complete,
    create_response_and_send_first_token,
)

log = logging.getLogger(__name__)

GC_INTERVAL = int(os.environ.get("SUT_GC_INTERVAL", "10000"))




_WARMUP_PROMPT = [
    1, 365, 3668, 23421, 22224, 7845, 27315, 29892, 5178, 312, 300,
    332, 594, 666, 275, 3277, 560, 277, 29889, 478, 342, 747, 352,
    398, 15937, 598, 2148, 2073, 398, 5065, 1056, 29892, 321, 657,
    9657, 398, 14172, 2497, 298, 355, 2872, 277, 263, 29889, 315,
    3417, 302, 747, 29882, 7866, 29877, 29892, 15937, 598, 7845,
    27315, 13081, 375, 7845, 27315, 29892, 782, 2801, 885, 7367,
    275, 802, 3737, 29889,
]


def _pack(obj: Any) -> bytes:
    if _HAS_MSGPACK:
        return _msgpack.packb(obj, use_bin_type=True)
    return json.dumps(obj, separators=(",", ":")).encode()


def _unpack(data: bytes) -> Any:
    if isinstance(data, (bytes, bytearray)) and data and data[0] not in (0x7B, 0x5B):
        if _HAS_MSGPACK:
            return _msgpack.unpackb(data, raw=False)
    return json.loads(data)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return int(value)


def _load_special_token_ids(tokenizer_path):
    if not tokenizer_path:
        return set()
    config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()

    added = cfg.get("added_tokens_decoder") or {}
    special_ids = set()
    for token_id, meta in added.items():
        if isinstance(meta, dict) and meta.get("special"):
            try:
                special_ids.add(int(token_id))
            except (TypeError, ValueError):
                continue
    return special_ids


def _parse_endpoints(servers_block) -> List[Tuple[str, int, int]]:
    """Return [(host, worker_pull_port, worker_push_port), ...].

    The endpoint string ``"host:port"`` is the worker's PULL port (where
    it receives requests, i.e. start_server.sh's ``ZMQ_PULL_PORT``). Its
    PUSH port (where it emits results, ``ZMQ_PUSH_PORT``) is
    conventionally ``port + 1000`` -- matches the fan-out in
    start_server.sh.
    """
    if servers_block is None:
        return []
    if isinstance(servers_block, str):
        items = servers_block.split()
    else:
        items = list(servers_block)
    out = []
    for raw in items:
        s = str(raw).strip()
        if not s:
            continue
        if ":" in s:
            host, port_str = s.rsplit(":", 1)
            pull = int(port_str)
        else:
            host, pull = s, 8200
        out.append((host, pull, pull + 1000))
    return out


@dataclass
class _Endpoint:
    """Connection state for one worker (named from the worker's POV).

    - ``worker_pull_port``: the worker binds PULL here; harness PUSHes to it.
    - ``worker_push_port``: the worker binds PUSH here; harness PULLs from it.

    Routing accounting (lock-free, single-writer-per-field):

    - ``sent`` / ``tokens_sent`` are written ONLY by the LoadGen thread
      (in ``issue_queries``).
    - ``finished`` / ``tokens_done`` are written ONLY by the asyncio
      collector (in ``_collector``).
    - The scheduler reads all four lock-free. The cross-thread read of
      ``finished`` / ``tokens_done`` may see a slightly stale value
      (CPython int rebinds are atomic, so we never see a torn value --
      just an older one). Off-by-a-few accounting is harmless: the
      scheduler is robust to small bias, and the difference self-heals
      as the next snapshot arrives.

    In-flight prompt-token load is computed at score time as
    ``tokens_sent - tokens_done``. This is the variable the
    ``shortest_queue_with_tokens`` algorithm wants: real load, not a
    sliding window of recent history.
    """

    host: str
    worker_pull_port: int
    worker_push_port: int
    push_sock: Any = None       
    pull_sock: Any = None       
    sent: int = 0
    finished: int = 0
    tokens_sent: int = 0        
    tokens_done: int = 0        

    @property
    def label(self) -> str:
        
        return f"{self.host}:{self.worker_pull_port}/{self.worker_push_port}"


class StandaloneMPServerSUT(SUT):
    """Server-scenario SUT for the ``standalone`` backend.

    Workers are spawned externally by ``start_server.sh --role standalone``.
    This SUT only connects, routes, and post-processes. The class name
    keeps the ``MP`` suffix for in-codebase continuity -- the public
    backend name is just ``standalone``.
    """

    def __init__(self, config, sampling_config):
        self._cfg_dc = getattr(config, "config", config)
        try:
            self.harness_config = OmegaConf.to_object(config["harness_config"])
        except (KeyError, AttributeError):
            self.harness_config = {}
        self.sampling_config = (
            OmegaConf.to_object(sampling_config) if sampling_config else {})
        self.scenario = str(getattr(config, "scenario", "")).lower()
        self.emit_first_token = self.scenario != "offline"
        try:
            self.llm_config = OmegaConf.to_object(config["llm_config"])
        except (KeyError, AttributeError):
            self.llm_config = {}
        try:
            self.pd_config = OmegaConf.to_object(config["pd_config"])
        except (KeyError, AttributeError):
            self.pd_config = {}

        super().__init__(SUTConfig(
            model=self.llm_config.get("model"),
            dataset_path=self.harness_config["dataset_path"],
            total_sample_count=self.harness_config.get(
                "total_sample_count", 24576),
            model_max_length=self.harness_config.get("model_max_length"),
            prompt_source=self.harness_config.get("prompt_source"),
            tokenizer_path=(
                self.pd_config.get("tokenizer_path")
                or self.llm_config.get("model")),
            add_special_tokens=bool(
                self.harness_config.get("prompt_add_special_tokens", True)),
        ))

        
        
        
        
        
        
        
        
        
        try:
            model_cfg = OmegaConf.to_object(self._cfg_dc)
        except Exception:
            model_cfg = {}
        servers = (model_cfg.get("servers") or {}) if isinstance(model_cfg, dict) else {}
        try:
            from config_helpers import resolve_standalone_endpoints
            hardware = model_cfg.get("hardware") if isinstance(model_cfg, dict) else None
            expanded = resolve_standalone_endpoints(
                model_cfg, hardware=hardware, scenario=self.scenario
            )
        except (ImportError, ValueError) as exc:
            log.warning("Endpoint auto-resolve failed (%s); "
                        "falling back to raw servers.standalone.", exc)
            expanded = None
        eps = _parse_endpoints(expanded if expanded is not None
                                else servers.get("standalone"))
        if not eps:
            eps = _parse_endpoints(servers.get("standalone_mp"))
            if eps:
                log.warning(
                    "servers.standalone_mp is deprecated; rename to "
                    "servers.standalone in the model YAML.")
        if not eps:
            raise RuntimeError(
                "servers.standalone is empty. Add e.g.\n"
                "  servers:\n"
                "    standalone:\n"
                "      - \"worker-host:8200\"\n"
                "      - \"worker-host:8201\"\n"
                "      - ...\n"
                "to the model YAML, and start workers with\n"
                "  ./start_server.sh --role standalone "
                "--hardware <hw> --model <model>")
        self.endpoints: List[_Endpoint] = [
            _Endpoint(host=h, worker_pull_port=p, worker_push_port=push)
            for (h, p, push) in eps
        ]

        
        
        self.tokenizer_path = (
            self.pd_config.get("tokenizer_path")
            or self.llm_config.get("model"))
        self.prompt_send_text = bool(
            self.harness_config.get("prompt_send_text", False))
        
        
        
        
        self.offline_batch_transport = (
            self.scenario == "offline"
            and bool(self.harness_config.get("offline_batch_transport", False))
        )
        
        
        
        
        
        self.offline_async_length_partition = (
            self.scenario == "offline"
            and bool(self.harness_config.get(
                "offline_async_length_partition", False))
        )
        if (self.offline_batch_transport
                and self.offline_async_length_partition):
            raise ValueError(
                "offline_batch_transport and offline_async_length_partition "
                "are mutually exclusive")
        self.offline_reference_global_batches = (
            self.offline_batch_transport
            and bool(self.harness_config.get(
                "offline_reference_global_batches", False))
        )
        self.offline_batch_buckets = tuple(
            self.harness_config.get("offline_batch_buckets", ()))
        self.offline_batch_max_requests = int(self.harness_config.get("offline_batch_max_requests", 0))
        if self.offline_batch_transport:
            if len(self.offline_batch_buckets) != len(self.endpoints):
                raise ValueError(
                    "offline_batch_transport requires one "
                    "offline_batch_buckets value per endpoint")
            if abs(sum(self.offline_batch_buckets) - 100.0) > 1e-6:
                raise ValueError(
                    "offline_batch_buckets must sum to 100")
            if self.offline_batch_max_requests < 0:
                raise ValueError("offline_batch_max_requests must be non-negative")
            batch_scope = (
                "reference-global" if self.offline_reference_global_batches
                else "QSL-local")
            log.info("Offline synchronous %s-batch transport enabled",
                     batch_scope)
        elif self.offline_async_length_partition:
            log.info("Offline async QSL-local length partition enabled")
        if self.prompt_send_text:
            log.info("Sending prompt_text to worker for prompt_source=%s",
                     self.harness_config.get("prompt_source"))

        self.strip_output_token_ids = set()
        if bool(self.harness_config.get("strip_output_special_tokens", False)):
            self.strip_output_token_ids = _load_special_token_ids(
                self.tokenizer_path)
            log.info("Stripping %d special token ids from final LoadGen "
                     "responses", len(self.strip_output_token_ids))

        
        algo = str(self.harness_config.get(
            "schedule_algo", "shortest_queue_with_tokens"))
        if algo not in (
                "shortest_queue_with_tokens", "shortest_queue", "round_robin"):
            raise ValueError(f"Unsupported schedule_algo: {algo}")
        self.schedule_algo = algo
        
        
        
        
        self.load_balance_window_size = int(self.harness_config.get(
            "load_balance_window_size", 10))
        self.load_balance_token_weight = float(self.harness_config.get(
            "load_balance_token_weight", 0.02))
        
        
        
        
        
        
        
        
        
        
        
        raw_weights = list(self.harness_config.get("endpoint_weights", ()) or ())
        if raw_weights:
            if len(raw_weights) != len(self.endpoints):
                raise ValueError(
                    "endpoint_weights must have one value per "
                    f"servers.standalone endpoint ({len(self.endpoints)}); "
                    f"got {len(raw_weights)}")
            weights = [float(w) for w in raw_weights]
            if any(w <= 0 for w in weights):
                raise ValueError("endpoint_weights must all be > 0")
        else:
            weights = [1.0] * len(self.endpoints)
        self.endpoint_weights = weights
        self._inv_endpoint_weights = [1.0 / w for w in weights]
        if any(w != 1.0 for w in weights):
            log.info("Weighted routing enabled; per-endpoint weights=[%s]",
                     ", ".join(f"{w:g}" for w in weights))
        
        
        
        
        
        
        
        
        
        
        self.sharded_issuer = bool(
            self.harness_config.get("sharded_issuer", False))
        if self.sharded_issuer:
            log.info("Sharded issuer enabled (%d per-endpoint issue queues)",
                     len(self.endpoints))
        
        
        
        
        
        
        
        
        
        
        
        self.transport = str(
            self.harness_config.get("transport", "asyncio")).lower()
        if self.transport not in ("asyncio", "threaded"):
            raise ValueError(
                "transport must be 'asyncio' or 'threaded'; got "
                f"{self.transport!r}")
        
        
        
        self.transport_threads = int(
            self.harness_config.get("transport_threads", 0))
        if "load_balance_window_size" in self.harness_config:
            log.info(
                "schedule_algo=%s uses live tokens_inflight; "
                "load_balance_window_size=%d is accepted for backward "
                "compat but is no longer applied.",
                algo, self.load_balance_window_size)

        self.enable_warmup = bool(self.harness_config.get("enable_warmup", True))
        
        
        self.warmup_count_per_worker = int(self.harness_config.get(
            "standalone_warmup_count",
            self.harness_config.get("standalone_mp_warmup_count", 2)))
        warmup_max_tokens = self.harness_config.get(
            "standalone_warmup_max_tokens",
            self.harness_config.get("warmup_max_tokens", 16))
        self.warmup_max_tokens = max(1, int(warmup_max_tokens))
        decode_max_tokens = self.sampling_config.get("max_tokens", None)
        self.decode_max_tokens = (
            None if decode_max_tokens is None
            else max(1, int(decode_max_tokens)))
        decode_min_tokens = _optional_int(
            self.sampling_config.get("min_tokens", None))
        self.decode_min_tokens = (
            None if decode_min_tokens is None
            else max(0, decode_min_tokens))
        warmup_min_tokens = self.harness_config.get(
            "standalone_warmup_min_tokens",
            self.harness_config.get("warmup_min_tokens", self.decode_min_tokens))
        warmup_min_tokens = _optional_int(warmup_min_tokens)
        self.warmup_min_tokens = (
            None if warmup_min_tokens is None
            else min(self.warmup_max_tokens, max(0, warmup_min_tokens)))
        decode_ignore_eos = self.sampling_config.get("ignore_eos", None)
        if decode_ignore_eos is None:
            self.decode_ignore_eos = None
        elif isinstance(decode_ignore_eos, str):
            self.decode_ignore_eos = (
                decode_ignore_eos.lower() in ("1", "true", "yes"))
        else:
            self.decode_ignore_eos = bool(decode_ignore_eos)

        self.tokenizer = None
        self._pending: Dict[str, Dict[str, Any]] = {}
        
        
        
        
        self._n_issued = 0
        self._n_completed = 0
        self._n_output_tokens = 0
        self._n_first = 0
        self._t0: Optional[float] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._issue_queue: Optional[asyncio.Queue] = None
        
        
        self._issue_queues: Optional[List[asyncio.Queue]] = None
        
        self._ctx = None                                  
        self._ep_thread: Optional[List[int]] = None        
        self._thread_eps: Optional[List[List[int]]] = None  
        self._out_queues: Optional[List["queue.Queue"]] = None
        self._worker_threads: List[threading.Thread] = []
        self._ready_barrier: Optional[threading.Barrier] = None
        self._ready = threading.Event()
        self._shutdown_flag = False
        self._rr_idx = 0
        
        self._req_seq = 0

    

    def _pick_device(self, prompt_len: int) -> int:
        
        
        
        
        
        eps = self.endpoints
        if self.schedule_algo == "round_robin":
            d = self._rr_idx
            self._rr_idx = (self._rr_idx + 1) % len(eps)
            return d
        inv = self._inv_endpoint_weights
        if self.schedule_algo == "shortest_queue":
            best, best_diff = 0, float("inf")
            for i, ep in enumerate(eps):
                diff = (ep.sent - ep.finished) * inv[i]
                if diff < best_diff:
                    best_diff, best = diff, i
            return best
        
        
        
        
        
        
        
        
        
        w = self.load_balance_token_weight
        best, best_score = 0, float("inf")
        for i, ep in enumerate(eps):
            score = ((ep.sent - ep.finished) + w * (
                ep.tokens_sent - ep.tokens_done)) * inv[i]
            if score < best_score:
                best_score, best = score, i
        return best

    

    def start(self) -> None:
        gc.collect()
        gc.disable()
        log.info("GC disabled (manual collect every %d completions)", GC_INTERVAL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        if self.transport == "threaded":
            self._start_threaded()
            return

        self._loop = _new_event_loop()
        if _HAS_UVLOOP:
            log.info("standalone SUT: using uvloop event loop")
        self._loop_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="standalone_loop")
        self._loop_thread.start()

        
        
        wait_s = int(self.harness_config.get(
            "warmup_poll_timeout_ms", 600_000)) // 1000
        if not self._ready.wait(timeout=max(wait_s, 60)):
            eps = ", ".join(ep.label for ep in self.endpoints)
            raise RuntimeError(
                f"standalone SUT did not reach ready state within "
                f"{wait_s}s. Harness warmup is still waiting on workers at "
                f"{eps}. Likely cause: workers not started or still loading "
                f"the model. Bring them up with `./start_server.sh --role "
                f"standalone --hardware <hw> --model <model>` and wait "
                f"for all to log 'ZMQ ready' before re-running.")

        log.info(
            "standalone ready -- %d worker(s): %s",
            len(self.endpoints),
            ", ".join(ep.label for ep in self.endpoints))

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except RuntimeError as exc:
            if (self._shutdown_flag
                    and "Event loop stopped before Future completed" in str(exc)):
                log.debug("standalone async loop stopped during shutdown")
            else:
                log.exception("standalone async loop crashed")
        except Exception:
            log.exception("standalone async loop crashed")

    async def _async_main(self) -> None:
        import zmq
        import zmq.asyncio  

        ctx = zmq.asyncio.Context()
        for i, ep in enumerate(self.endpoints):
            
            push = ctx.socket(zmq.PUSH)
            push.setsockopt(zmq.SNDHWM, 4096)
            push.connect(f"tcp://{ep.host}:{ep.worker_pull_port}")
            
            pull = ctx.socket(zmq.PULL)
            pull.setsockopt(zmq.RCVHWM, 8192)
            pull.connect(f"tcp://{ep.host}:{ep.worker_push_port}")
            ep.push_sock = push
            ep.pull_sock = pull
            log.info("Connected to worker %d at %s", i, ep.label)

        
        
        await asyncio.sleep(1.0)

        if self.enable_warmup and self.warmup_count_per_worker > 0:
            await self._harness_warmup()

        if self.sharded_issuer:
            self._issue_queues = [
                asyncio.Queue() for _ in self.endpoints]
            issuer_tasks = [
                self._issuer_sharded(i) for i in range(len(self.endpoints))]
        else:
            self._issue_queue = asyncio.Queue()
            issuer_tasks = [self._issuer()]
        self._t0 = time.time()
        self._ready.set()

        try:
            await asyncio.gather(
                *issuer_tasks,
                *[self._collector(i) for i in range(len(self.endpoints))],
                self._progress(),
            )
        finally:
            for ep in self.endpoints:
                try:
                    if ep.push_sock is not None:
                        ep.push_sock.close()
                    if ep.pull_sock is not None:
                        ep.pull_sock.close()
                except Exception:
                    pass
            ctx.term()

    async def _harness_warmup(self) -> None:
        """Send a few prompts to every worker to fully heat CUDA graphs.

        Each worker already runs its own internal warmup on startup, but
        sending real prompts through the ZMQ path also flushes the SUT
        -> worker socket and round-trips a response per endpoint, which
        is the most reliable signal that the worker is ready.
        """
        prompt = (_WARMUP_PROMPT * 4)[:256]
        per_ep = self.warmup_count_per_worker
        pending = per_ep * len(self.endpoints)
        log.info(
            "Harness warmup: %d requests (%d per worker x %d workers, "
            "max_tokens=%d)",
            pending, per_ep, len(self.endpoints), self.warmup_max_tokens)

        
        
        per_ep_pending: Dict[int, int] = {i: 0 for i in range(len(self.endpoints))}
        for i, ep in enumerate(self.endpoints):
            for j in range(per_ep):
                req_id = f"_harness_warmup_{i}_{j}"
                self._pending[req_id] = {"sample_id": None, "first_sent": False}
                msg = {
                    "id": req_id,
                    "prompt": prompt,
                    "max_tokens": self.warmup_max_tokens,
                    "ignore_eos": False,
                }
                if self.warmup_min_tokens is not None:
                    msg["min_tokens"] = self.warmup_min_tokens
                await ep.push_sock.send(_pack(msg))
                per_ep_pending[i] += 1

        warmup_poll_ms = int(self.harness_config.get(
            "warmup_poll_timeout_ms", 600_000))

        import zmq

        async def drain(ep_idx: int) -> None:
            ep = self.endpoints[ep_idx]
            while per_ep_pending[ep_idx] > 0:
                if not await ep.pull_sock.poll(timeout=warmup_poll_ms):
                    raise RuntimeError(
                        f"Harness warmup timed out after "
                        f"{warmup_poll_ms // 1000}s waiting for worker "
                        f"{ep.label}")
                while per_ep_pending[ep_idx] > 0:
                    try:
                        raw = await ep.pull_sock.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    result = _unpack(raw)
                    req_id = result.get("id", "")
                    if not req_id.startswith("_harness_warmup_"):
                        continue
                    if ("first_token_ids" in result
                            and "token_ids" not in result):
                        continue
                    self._pending.pop(req_id, None)
                    per_ep_pending[ep_idx] -= 1

        await asyncio.gather(*(drain(i) for i in range(len(self.endpoints))))
        log.info("Harness warmup complete (%d requests)", pending)

    def _encode_request(self, req_id, prompt_payload) -> bytes:
        if req_id is None:
            
            
            msg = dict(prompt_payload)
        else:
            msg = {"id": req_id}
            msg.update(prompt_payload)
        if self.decode_max_tokens is not None:
            msg["max_tokens"] = self.decode_max_tokens
        if self.decode_min_tokens is not None:
            msg["min_tokens"] = self.decode_min_tokens
        if self.decode_ignore_eos is not None:
            msg["ignore_eos"] = self.decode_ignore_eos
        return _pack(msg)

    async def _issuer(self) -> None:
        
        
        
        q = self._issue_queue
        while True:
            item = await q.get()
            if item is _STOP:
                return
            ep_idx, req_id, prompt_payload = item
            ep = self.endpoints[ep_idx]
            await ep.push_sock.send(self._encode_request(req_id, prompt_payload))

    async def _issuer_sharded(self, ep_idx: int) -> None:
        
        
        
        
        q = self._issue_queues[ep_idx]
        ep = self.endpoints[ep_idx]
        while True:
            item = await q.get()
            if item is _STOP:
                return
            _ep_idx, req_id, prompt_payload = item
            await ep.push_sock.send(self._encode_request(req_id, prompt_payload))

    def _handle_result(self, result, ep) -> None:
        
        
        
        
        
        
        req_id = result.get("id", "")
        entry = self._pending.get(req_id)
        if entry is None:
            return
        sample_id = entry.get("sample_id")
        if sample_id is None:
            
            self._pending.pop(req_id, None)
            return
        if "first_token_ids" in result and "token_ids" not in result:
            if not self.emit_first_token:
                return
            if not entry["first_sent"]:
                create_response_and_send_first_token(
                    sample_id, result["first_token_ids"])
                entry["first_sent"] = True
                self._n_first += 1
            return
        token_ids = result.get("token_ids", [])
        if self.strip_output_token_ids and token_ids:
            token_ids = [
                t for t in token_ids
                if int(t) not in self.strip_output_token_ids
            ]
        
        completed_entry = self._pending.pop(req_id, None)
        create_response_and_send_complete(sample_id, token_ids)
        self._n_completed += 1
        self._n_output_tokens += len(token_ids)
        ep.finished += 1
        
        
        if completed_entry is not None:
            ep.tokens_done += completed_entry.get("prompt_len", 0)

    def _dispatch_raw(self, raw, ep) -> int:
        
        
        envelope = _unpack(raw)
        if envelope.get("type") == "offline_batch_result":
            results = envelope.get("results", [])
        else:
            results = [envelope]
        for result in results:
            self._handle_result(result, ep)
        return len(results)

    async def _collector(self, ep_idx: int) -> None:
        import zmq
        ep = self.endpoints[ep_idx]
        pull_sock = ep.pull_sock
        gc_counter = 0
        while not self._shutdown_flag:
            
            
            if not await pull_sock.poll(timeout=1000):
                continue
            
            
            
            while True:
                try:
                    raw = await pull_sock.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                gc_counter += self._dispatch_raw(raw, ep)
                if GC_INTERVAL > 0 and gc_counter >= GC_INTERVAL:
                    gc.collect()
                    gc_counter = 0

    def _log_progress(self) -> None:
        elapsed = time.time() - self._t0
        issued = self._n_issued
        completed = self._n_completed
        inflight = issued - completed
        qps = completed / elapsed if elapsed > 0 else 0.0
        tok_s = self._n_output_tokens / elapsed if elapsed > 0 else 0.0
        queues = ",".join(
            str(ep.sent - ep.finished) for ep in self.endpoints)
        
        
        
        tok_q = ",".join(
            f"{(ep.tokens_sent - ep.tokens_done) // 1000}k"
            for ep in self.endpoints)
        log.info(
            "Progress [%.0fs] issued=%d first=%d done=%d inflight=%d "
            "%.1f qps %.0f tok/s queues=%s tok_q=%s",
            elapsed, issued, self._n_first, completed,
            inflight, qps, tok_s, queues, tok_q)

    async def _progress(self) -> None:
        while not self._shutdown_flag:
            await asyncio.sleep(10)
            if self._shutdown_flag or self._t0 is None:
                break
            self._log_progress()

    def _progress_thread(self) -> None:
        while not self._shutdown_flag:
            time.sleep(10)
            if self._shutdown_flag or self._t0 is None:
                break
            self._log_progress()

    

    def _start_threaded(self) -> None:
        import zmq

        n_ep = len(self.endpoints)
        n_threads = self.transport_threads or min(n_ep, os.cpu_count() or 8)
        n_threads = max(1, min(n_threads, n_ep))
        self._ctx = zmq.Context()
        
        try:
            self._ctx.set(zmq.IO_THREADS, min(4, n_threads))
        except Exception:
            pass
        
        
        self._ep_thread = [i % n_threads for i in range(n_ep)]
        self._thread_eps = [
            [i for i in range(n_ep) if self._ep_thread[i] == t]
            for t in range(n_threads)]
        self._out_queues = [queue.Queue() for _ in range(n_threads)]
        
        
        self._ready_barrier = threading.Barrier(n_threads + 1)
        log.info(
            "Threaded transport: %d worker thread(s) over %d endpoint(s) "
            "(shards=%s)", n_threads, n_ep,
            [len(s) for s in self._thread_eps])

        for t in range(n_threads):
            th = threading.Thread(
                target=self._transport_thread, args=(t,), daemon=True,
                name=f"standalone_tx{t}")
            th.start()
            self._worker_threads.append(th)

        ready_timeout = max(
            60, int(self.harness_config.get("warmup_poll_timeout_ms",
                                            600_000)) // 1000)
        try:
            self._ready_barrier.wait(timeout=ready_timeout)
        except threading.BrokenBarrierError:
            raise RuntimeError(
                f"threaded transport did not reach ready state within "
                f"{ready_timeout}s (a worker thread failed to connect or warm "
                f"up its engines). Ensure all engines logged 'ZMQ ready'.")
        self._t0 = time.time()
        prog = threading.Thread(
            target=self._progress_thread, daemon=True, name="standalone_prog")
        prog.start()
        self._worker_threads.append(prog)
        log.info(
            "standalone ready (threaded) -- %d worker(s): %s",
            n_ep, ", ".join(ep.label for ep in self.endpoints))

    def _transport_thread(self, t: int) -> None:
        import zmq

        eps = self._thread_eps[t]
        poller = zmq.Poller()
        pull_to_ep = {}
        try:
            for i in eps:
                ep = self.endpoints[i]
                push = self._ctx.socket(zmq.PUSH)
                push.setsockopt(zmq.SNDHWM, 4096)
                push.connect(f"tcp://{ep.host}:{ep.worker_pull_port}")
                pull = self._ctx.socket(zmq.PULL)
                pull.setsockopt(zmq.RCVHWM, 8192)
                pull.connect(f"tcp://{ep.host}:{ep.worker_push_port}")
                ep.push_sock = push
                ep.pull_sock = pull
                poller.register(pull, zmq.POLLIN)
                pull_to_ep[pull] = ep
                log.info("tx%d connected to worker %d at %s", t, i, ep.label)

            
            
            time.sleep(1.0)
            if self.enable_warmup and self.warmup_count_per_worker > 0:
                self._threaded_warmup(eps)
        except Exception:
            log.exception("tx%d failed during setup", t)
            
            if self._ready_barrier is not None:
                self._ready_barrier.abort()
            return

        try:
            self._ready_barrier.wait()
        except threading.BrokenBarrierError:
            return

        out_q = self._out_queues[t]
        gc_counter = 0
        try:
            while not self._shutdown_flag:
                
                
                
                drained = False
                while True:
                    try:
                        ep_idx, req_id, prompt_payload = out_q.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    ep = self.endpoints[ep_idx]
                    ep.push_sock.send(
                        self._encode_request(req_id, prompt_payload))
                
                
                
                
                
                events = dict(poller.poll(timeout=(0 if drained else 1)))
                for pull, ep in pull_to_ep.items():
                    if events.get(pull) != zmq.POLLIN:
                        continue
                    while True:
                        try:
                            raw = pull.recv(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        gc_counter += self._dispatch_raw(raw, ep)
                if GC_INTERVAL > 0 and gc_counter >= GC_INTERVAL:
                    gc.collect()
                    gc_counter = 0
        except Exception:
            log.exception("tx%d crashed", t)
        finally:
            for pull, ep in pull_to_ep.items():
                try:
                    pull.close()
                    ep.push_sock.close()
                except Exception:
                    pass

    def _threaded_warmup(self, eps: List[int]) -> None:
        import zmq

        prompt = (_WARMUP_PROMPT * 4)[:256]
        per_ep = self.warmup_count_per_worker
        pending = {}
        for i in eps:
            ep = self.endpoints[i]
            for j in range(per_ep):
                req_id = f"_harness_warmup_{i}_{j}"
                self._pending[req_id] = {"sample_id": None, "first_sent": False}
                msg = {
                    "id": req_id,
                    "prompt": prompt,
                    "max_tokens": self.warmup_max_tokens,
                    "ignore_eos": False,
                }
                if self.warmup_min_tokens is not None:
                    msg["min_tokens"] = self.warmup_min_tokens
                ep.push_sock.send(_pack(msg))
            pending[i] = per_ep

        poll_ms = int(self.harness_config.get("warmup_poll_timeout_ms", 600_000))
        poller = zmq.Poller()
        for i in eps:
            poller.register(self.endpoints[i].pull_sock, zmq.POLLIN)
        while any(v > 0 for v in pending.values()):
            socks = dict(poller.poll(timeout=poll_ms))
            if not socks:
                raise RuntimeError(
                    f"threaded warmup timed out after {poll_ms // 1000}s")
            for i in eps:
                ps = self.endpoints[i].pull_sock
                if socks.get(ps) != zmq.POLLIN:
                    continue
                while True:
                    try:
                        raw = ps.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    result = _unpack(raw)
                    req_id = result.get("id", "")
                    if not req_id.startswith("_harness_warmup_"):
                        continue
                    if ("first_token_ids" in result
                            and "token_ids" not in result):
                        continue
                    self._pending.pop(req_id, None)
                    pending[i] -= 1

    

    def _sample_prompt_payload(self, sample_index):
        prompt = self.data_object.input_ids[sample_index]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        elif not isinstance(prompt, list):
            prompt = list(prompt)
        plen = len(prompt)
        if self.prompt_send_text:
            text_prompts = getattr(self.data_object, "text_prompts", None)
            if text_prompts is None:
                raise RuntimeError(
                    "prompt_send_text requires a text prompt_source")
            return plen, {"prompt_text": text_prompts[sample_index]}
        return plen, {"prompt": prompt}

    def _sample_prompt_length(self, sample_index: int) -> int:
        """Return a prompt length without materializing its wire payload."""
        return len(self.data_object.input_ids[sample_index])

    def _offline_length_partition(self, query_samples):
        """Reproduce the reference Offline bucket partition."""
        samples = list(query_samples)
        endpoint_count = len(self.endpoints)
        qsl_size = len(self.data_object.input_ids)
        if endpoint_count <= 1 or not samples or qsl_size <= 0:
            return [(sample, 0) for sample in samples]

        
        
        
        if len(samples) % qsl_size:
            log.warning(
                "Offline length partition disabled: %d samples is not a "
                "multiple of the %d-sample QSL", len(samples), qsl_size)
            return [(sample, None) for sample in samples]

        assignments = []
        total_bucket_counts = [0] * endpoint_count

        if self.offline_reference_global_batches:
            current_start = 0
            for endpoint_idx, percentage in enumerate(self.offline_batch_buckets):
                if endpoint_idx == endpoint_count - 1:
                    end = len(samples)
                else:
                    end = current_start + int(
                        (percentage / 100.0) * len(samples))
                bucket = samples[current_start:end]
                assignments.extend((sample, endpoint_idx) for sample in bucket)
                total_bucket_counts[endpoint_idx] += len(bucket)
                current_start = end
            log.info(
                "Offline reference global bucket partition: copies=%d "
                "samples=%d counts=%s",
                len(samples) // qsl_size, len(samples), total_bucket_counts)
            return assignments

        for start in range(0, len(samples), qsl_size):
            qsl = samples[start:start + qsl_size]
            qsl_indices = [sample.index for sample in qsl]
            if (len(set(qsl_indices)) != qsl_size
                    or any(index < 0 or index >= qsl_size
                           for index in qsl_indices)):
                log.warning(
                    "Offline length partition disabled for QSL copy %d: "
                    "the callback chunk is not one complete canonical QSL",
                    start // qsl_size)
                assignments.extend((sample, None) for sample in qsl)
                continue

            
            
            
            
            current_start = 0
            for endpoint_idx, percentage in enumerate(self.offline_batch_buckets):
                if endpoint_idx == endpoint_count - 1:
                    end = qsl_size
                else:
                    end = current_start + int((percentage / 100.0) * qsl_size)
                bucket = qsl[current_start:end]
                assignments.extend((sample, endpoint_idx) for sample in bucket)
                total_bucket_counts[endpoint_idx] += len(bucket)
                current_start = end

        log.info(
            "Offline reference bucket partition: copies=%d samples=%d counts=%s",
            len(samples) // qsl_size, len(samples), total_bucket_counts)
        return assignments

    def _offline_async_length_partition(self, query_samples):
        """Assign sorted, token-balanced QSL ranges to async endpoints.

        Offline permits reordering. To keep the submitted workload unchanged,
        each complete canonical QSL copy is sorted and partitioned independently;
        no sample is omitted, duplicated, or moved across a QSL boundary. The
        workers still use normal one-request async transport.
        """
        samples = list(query_samples)
        endpoint_count = len(self.endpoints)
        qsl_size = len(self.data_object.input_ids)
        if endpoint_count <= 1 or not samples or qsl_size <= 0:
            return [(sample, 0) for sample in samples]
        if len(samples) % qsl_size:
            log.warning(
                "Offline async length partition disabled: %d samples is not "
                "a multiple of the %d-sample QSL", len(samples), qsl_size)
            return [(sample, None) for sample in samples]

        assignments = []
        total_bucket_counts = [0] * endpoint_count
        total_bucket_tokens = [0] * endpoint_count
        for start in range(0, len(samples), qsl_size):
            qsl = samples[start:start + qsl_size]
            qsl_indices = [sample.index for sample in qsl]
            if (len(set(qsl_indices)) != qsl_size
                    or any(index < 0 or index >= qsl_size
                           for index in qsl_indices)):
                log.warning(
                    "Offline async length partition disabled for QSL copy %d: "
                    "the callback chunk is not one complete canonical QSL",
                    start // qsl_size)
                assignments.extend((sample, None) for sample in qsl)
                continue

            
            
            
            ordered = sorted(
                ((self._sample_prompt_length(sample.index), sample)
                 for sample in qsl),
                key=lambda item: item[0])
            next_item = 0
            remaining_tokens = sum(length for length, _ in ordered)
            for endpoint_idx in range(endpoint_count):
                remaining_endpoints = endpoint_count - endpoint_idx
                bucket = []
                bucket_tokens = 0
                if endpoint_idx == endpoint_count - 1:
                    bucket = ordered[next_item:]
                    bucket_tokens = sum(length for length, _ in bucket)
                else:
                    target_tokens = remaining_tokens / remaining_endpoints
                    
                    
                    
                    while next_item < len(ordered) - (remaining_endpoints - 1):
                        length, sample = ordered[next_item]
                        if bucket and bucket_tokens + length > target_tokens:
                            previous_gap = target_tokens - bucket_tokens
                            next_gap = bucket_tokens + length - target_tokens
                            if next_gap < previous_gap:
                                bucket.append((length, sample))
                                bucket_tokens += length
                                next_item += 1
                            break
                        bucket.append((length, sample))
                        bucket_tokens += length
                        next_item += 1
                assignments.extend(
                    (sample, endpoint_idx) for _, sample in bucket)
                total_bucket_counts[endpoint_idx] += len(bucket)
                total_bucket_tokens[endpoint_idx] += bucket_tokens
                remaining_tokens -= bucket_tokens

        log.info(
            "Offline async length partition: copies=%d samples=%d "
            "counts=%s prompt_tokens=%s",
            len(samples) // qsl_size, len(samples), total_bucket_counts,
            total_bucket_tokens)
        return assignments

    def issue_queries(self, query_samples) -> None:
        if self.offline_batch_transport:
            assignments = self._offline_length_partition(query_samples)
        elif self.offline_async_length_partition:
            assignments = self._offline_async_length_partition(query_samples)
        else:
            assignments = ((sample, None) for sample in query_samples)

        if self.offline_batch_transport:
            assignments = list(assignments)
            qsl_size = len(self.data_object.input_ids)
            if not qsl_size or len(assignments) % qsl_size:
                raise RuntimeError(
                    "Offline batch transport requires complete canonical QSL "
                    "copies in each issue_queries callback")
            if self.offline_reference_global_batches:
                batch_ranges = ((0, len(assignments)),)
            else:
                batch_ranges = (
                    (start, start + qsl_size)
                    for start in range(0, len(assignments), qsl_size))
            for start, end in batch_ranges:
                endpoint_batches = [[] for _ in self.endpoints]
                for sample, assigned_endpoint in assignments[start:end]:
                    if assigned_endpoint is None:
                        raise RuntimeError(
                            "Offline batch transport requires a complete "
                            "canonical QSL before it can reorder requests")
                    plen, prompt_payload = self._sample_prompt_payload(sample.index)
                    ep = self.endpoints[assigned_endpoint]
                    self._req_seq += 1
                    req_id = f"r{self._req_seq}"
                    self._pending[req_id] = {
                        "sample_id": sample.id,
                        "first_sent": False,
                        "ep_idx": assigned_endpoint,
                        "prompt_len": plen,
                    }
                    ep.sent += 1
                    ep.tokens_sent += plen
                    self._n_issued += 1
                    endpoint_batches[assigned_endpoint].append({
                        "id": req_id,
                        **prompt_payload,
                    })
                for ep_idx, requests in enumerate(endpoint_batches):
                    if requests:
                        batch_size = (self.offline_batch_max_requests
                                      or len(requests))
                        for request_start in range(0, len(requests), batch_size):
                            self._enqueue_issue(ep_idx, None, {
                                "type": "offline_batch",
                                "requests": requests[
                                    request_start:request_start + batch_size],
                            })
            return

        for sample, assigned_endpoint in assignments:
            plen, prompt_payload = self._sample_prompt_payload(sample.index)
            ep_idx = (assigned_endpoint if assigned_endpoint is not None
                      else self._pick_device(plen))
            ep = self.endpoints[ep_idx]
            self._req_seq += 1
            req_id = f"r{self._req_seq}"
            
            
            
            self._pending[req_id] = {
                "sample_id": sample.id,
                "first_sent": False,
                "ep_idx": ep_idx,
                "prompt_len": plen,
            }
            ep.sent += 1
            ep.tokens_sent += plen
            self._n_issued += 1
            self._enqueue_issue(ep_idx, req_id, prompt_payload)

    def _enqueue_issue(self, ep_idx, req_id, prompt_payload) -> None:
        
        if self.transport == "threaded":
            
            
            self._out_queues[self._ep_thread[ep_idx]].put_nowait(
                (ep_idx, req_id, prompt_payload))
            return
        
        
        if self.sharded_issuer:
            q = self._issue_queues[ep_idx]
        else:
            q = self._issue_queue
        self._loop.call_soon_threadsafe(
            q.put_nowait, (ep_idx, req_id, prompt_payload))

    def flush_queries(self) -> None:
        pass

    def stop(self) -> None:
        self._shutdown_flag = True
        if self.transport == "threaded":
            
            for th in self._worker_threads:
                th.join(timeout=30)
            if self._ctx is not None:
                try:
                    self._ctx.term()
                except Exception:
                    pass
            log.info(
                "standalone stopped (threaded, issued=%d completed=%d "
                "first=%d tokens=%d)", self._n_issued, self._n_completed,
                self._n_first, self._n_output_tokens)
            return
        if self._loop is not None and self._loop.is_running():
            
            
            
            if self._issue_queues is not None:
                for q in self._issue_queues:
                    self._loop.call_soon_threadsafe(q.put_nowait, _STOP)
            elif self._issue_queue is not None:
                self._loop.call_soon_threadsafe(
                    self._issue_queue.put_nowait, _STOP)
            self._loop.call_soon_threadsafe(lambda: None)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=30)
            if (self._loop_thread.is_alive()
                    and self._loop is not None
                    and self._loop.is_running()):
                log.warning("standalone async loop did not stop cleanly; forcing")
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop_thread.join(timeout=5)
        log.info(
            "standalone stopped (issued=%d completed=%d first=%d tokens=%d)",
            self._n_issued, self._n_completed, self._n_first,
            self._n_output_tokens)
