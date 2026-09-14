from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
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
    if isinstance(data, (bytes, bytearray)) and data and data[0] not in (
            0x7B, 0x5B):
        if _HAS_MSGPACK:
            return _msgpack.unpackb(data, raw=False)
    return json.loads(data)


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
class _PrefillEndpoint:
    """One prefill engine. Harness has a PUSH (out) and a PULL (in) socket.

    - ``worker_pull_port``: the worker binds PULL here; harness PUSHes to it.
    - ``worker_push_port``: the worker binds PUSH here; harness PULLs from it
      (first-token notifications).

    Routing accounting (lock-free, single-writer-per-field):
    - ``sent`` / ``tokens_sent`` are written ONLY by the LoadGen thread.
    - ``finished`` / ``tokens_done`` are written ONLY by the asyncio
      decode collector (which credits back to the prefill engine that
      originally received the request -- see ``_decode_collector``).
    - In-flight load is ``tokens_sent - tokens_done``. See the standalone
      SUT docstring for the rationale: a live counter avoids the stale
      penalty a sliding window carried after engines drained.
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


@dataclass
class _DecodeEndpoint:
    """One decode engine. Harness only PULLs final-token messages.

    The prefill worker pushes decode requests directly to its peer
    decode engines; the harness never PUSHes to a decode worker.
    """

    host: str
    worker_pull_port: int     
    worker_push_port: int     
    pull_sock: Any = None     

    @property
    def label(self) -> str:
        return f"{self.host}:{self.worker_pull_port}/{self.worker_push_port}"


class PDServerSUT(SUT):
    """Server-scenario SUT for the ``pd`` (prefill-decode disagg) backend.

    Workers are spawned externally:
      - ``start_server.sh --role prefill`` on the prefill node
      - ``start_server.sh --role decode``  on the decode node
    This SUT only connects, routes prompts to prefill, and post-processes
    first-token + final-token responses.
    """

    def __init__(self, config, sampling_config):
        self._cfg_dc = getattr(config, "config", config)
        try:
            self.harness_config = OmegaConf.to_object(config["harness_config"])
        except (KeyError, AttributeError):
            self.harness_config = {}
        self.sampling_config = (
            OmegaConf.to_object(sampling_config) if sampling_config else {})
        try:
            self.llm_config = OmegaConf.to_object(config["llm_config"])
        except (KeyError, AttributeError):
            self.llm_config = {}
        try:
            self.pd_config = OmegaConf.to_object(config["pd_config"])
        except (KeyError, AttributeError):
            self.pd_config = {}
        try:
            self.vllm_env_config = OmegaConf.to_object(
                config["vllm_env_config"])
        except (KeyError, AttributeError):
            self.vllm_env_config = {}

        source = str(
            self.harness_config.get("pd_first_token_source")
            or self.pd_config.get("first_token_source")
            or self.vllm_env_config.get("PD_FIRST_TOKEN_SOURCE")
            or os.environ.get("PD_FIRST_TOKEN_SOURCE", "prefill")
        ).strip().lower()
        if source not in ("prefill", "decode"):
            raise ValueError(f"Unsupported PD_FIRST_TOKEN_SOURCE: {source}")
        self.first_token_source = source
        self.first_token_from_decode = source == "decode"

        super().__init__(SUTConfig(
            model=self.llm_config.get("model"),
            dataset_path=self.harness_config["dataset_path"],
            total_sample_count=self.harness_config.get(
                "total_sample_count", 24576),
            model_max_length=self.harness_config.get("model_max_length"),
        ))

        
        
        try:
            servers = OmegaConf.to_object(self._cfg_dc.get("servers", {}))
        except Exception:
            servers = {}
        prefill_eps = _parse_endpoints(servers.get("prefill"))
        decode_eps = _parse_endpoints(servers.get("decode"))
        if not prefill_eps or not decode_eps:
            raise RuntimeError(
                "servers.prefill and/or servers.decode is empty. Add e.g.\n"
                "  servers:\n"
                "    prefill:\n"
                "      - \"prefill-host:8200\"\n"
                "      - \"prefill-host:8201\"\n"
                "      - ...\n"
                "    decode:\n"
                "      - \"decode-host:8300\"\n"
                "      - \"decode-host:8301\"\n"
                "      - ...\n"
                "to the model YAML, and start workers with\n"
                "  ./start_server.sh --role prefill --hardware <hw> "
                "--model <model>   (on prefill node)\n"
                "  ./start_server.sh --role decode  --hardware <hw> "
                "--model <model>   (on decode node)")

        self.prefill_endpoints: List[_PrefillEndpoint] = [
            _PrefillEndpoint(host=h, worker_pull_port=p, worker_push_port=push)
            for (h, p, push) in prefill_eps
        ]
        self.decode_endpoints: List[_DecodeEndpoint] = [
            _DecodeEndpoint(host=h, worker_pull_port=p, worker_push_port=push)
            for (h, p, push) in decode_eps
        ]

        self.tokenizer_path = (
            self.pd_config.get("tokenizer_path")
            or self.llm_config.get("model"))

        
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
        if "load_balance_window_size" in self.harness_config:
            log.info(
                "pd schedule_algo=%s uses live tokens_inflight; "
                "load_balance_window_size=%d is accepted for backward "
                "compat but is no longer applied.",
                algo, self.load_balance_window_size)

        self.enable_warmup = bool(self.harness_config.get(
            "enable_warmup", True))
        self.warmup_count_per_worker = int(self.harness_config.get(
            "pd_warmup_count",
            self.harness_config.get("standalone_mp_warmup_count", 2)))
        warmup_decode_max = self.harness_config.get(
            "pd_warmup_decode_max_tokens", None)
        self.warmup_decode_max_tokens = (
            None if warmup_decode_max is None else int(warmup_decode_max))
        decode_max_tokens = self.sampling_config.get("max_tokens", None)
        self.decode_max_tokens = (
            None if decode_max_tokens is None
            else max(1, int(decode_max_tokens)))

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
        self._ready = threading.Event()
        self._shutdown_flag = False
        self._rr_idx = 0
        self._req_seq = 0
    

    def _pick_prefill(self, prompt_len: int) -> int:
        
        
        
        
        
        eps = self.prefill_endpoints
        if self.schedule_algo == "round_robin":
            d = self._rr_idx
            self._rr_idx = (self._rr_idx + 1) % len(eps)
            return d
        if self.schedule_algo == "shortest_queue":
            best, best_diff = 0, float("inf")
            for i, ep in enumerate(eps):
                diff = ep.sent - ep.finished
                if diff < best_diff:
                    best_diff, best = diff, i
            return best
        
        
        
        w = self.load_balance_token_weight
        best, best_score = 0, float("inf")
        for i, ep in enumerate(eps):
            score = (ep.sent - ep.finished) + w * (
                ep.tokens_sent - ep.tokens_done)
            if score < best_score:
                best_score, best = score, i
        return best

    

    def start(self) -> None:
        gc.collect()
        gc.disable()
        log.info("GC disabled (manual collect every %d completions)",
                 GC_INTERVAL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        self._loop = _new_event_loop()
        if _HAS_UVLOOP:
            log.info("pd SUT: using uvloop event loop")
        self._loop_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="pd_loop")
        self._loop_thread.start()

        wait_s = int(self.harness_config.get(
            "warmup_poll_timeout_ms", 600_000)) // 1000
        if not self._ready.wait(timeout=max(wait_s, 60)):
            peps = ", ".join(ep.label for ep in self.prefill_endpoints)
            deps = ", ".join(ep.label for ep in self.decode_endpoints)
            raise RuntimeError(
                f"pd SUT did not reach ready state within {wait_s}s. "
                f"Harness warmup is still waiting on prefill workers at "
                f"[{peps}] and/or decode workers at [{deps}]. Likely cause: "
                f"workers not started or still loading the model. Bring "
                f"them up with `./start_server.sh --role prefill|decode "
                f"--hardware <hw> --model <model>` and wait for all to log "
                f"'ZMQ ready' before re-running.")

        log.info(
            "pd ready -- %d prefill worker(s): %s | %d decode worker(s): %s",
            len(self.prefill_endpoints),
            ", ".join(ep.label for ep in self.prefill_endpoints),
            len(self.decode_endpoints),
            ", ".join(ep.label for ep in self.decode_endpoints))

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except RuntimeError as exc:
            if (self._shutdown_flag
                    and "Event loop stopped before Future completed" in str(exc)):
                log.debug("pd async loop stopped during shutdown")
            else:
                log.exception("pd async loop crashed")
        except Exception:
            log.exception("pd async loop crashed")

    async def _async_main(self) -> None:
        import zmq
        import zmq.asyncio  

        ctx = zmq.asyncio.Context()
        for i, ep in enumerate(self.prefill_endpoints):
            
            push = ctx.socket(zmq.PUSH)
            push.setsockopt(zmq.SNDHWM, 4096)
            push.connect(f"tcp://{ep.host}:{ep.worker_pull_port}")
            
            pull = ctx.socket(zmq.PULL)
            pull.setsockopt(zmq.RCVHWM, 8192)
            pull.connect(f"tcp://{ep.host}:{ep.worker_push_port}")
            ep.push_sock = push
            ep.pull_sock = pull
            log.info("Connected to prefill worker %d at %s", i, ep.label)

        for j, ep in enumerate(self.decode_endpoints):
            
            pull = ctx.socket(zmq.PULL)
            pull.setsockopt(zmq.RCVHWM, 8192)
            pull.connect(f"tcp://{ep.host}:{ep.worker_push_port}")
            ep.pull_sock = pull
            log.info("Connected to decode worker %d at %s", j, ep.label)

        await asyncio.sleep(1.0)

        if self.enable_warmup and self.warmup_count_per_worker > 0:
            await self._harness_warmup()

        self._issue_queue = asyncio.Queue()
        self._t0 = time.time()
        self._ready.set()

        try:
            await asyncio.gather(
                self._issuer(),
                *[self._prefill_collector(i)
                  for i in range(len(self.prefill_endpoints))],
                *[self._decode_collector(j)
                  for j in range(len(self.decode_endpoints))],
                self._progress(),
            )
        finally:
            for ep in self.prefill_endpoints:
                try:
                    if ep.push_sock is not None:
                        ep.push_sock.close()
                    if ep.pull_sock is not None:
                        ep.pull_sock.close()
                except Exception:
                    pass
            for ep in self.decode_endpoints:
                try:
                    if ep.pull_sock is not None:
                        ep.pull_sock.close()
                except Exception:
                    pass
            ctx.term()

    async def _harness_warmup(self) -> None:
        """Heat the SUT -> prefill -> decode -> SUT path.

        Send K prompts per prefill engine, drain first-token notifications
        from each prefill PULL, and drain final-token completes across all
        decode PULLs. We don't know up-front which decode engine will
        service which warmup request (prefill chooses via round-robin), so
        we drain decodes by total expected count, not per-engine.
        """
        prompt = (_WARMUP_PROMPT * 4)[:256]
        per_ep = self.warmup_count_per_worker
        total = per_ep * len(self.prefill_endpoints)
        log.info(
            "Harness warmup: %d requests (%d per prefill x %d prefill "
            "engines), draining across %d decode engines",
            total, per_ep, len(self.prefill_endpoints),
            len(self.decode_endpoints))

        first_pending: Dict[int, int] = (
            {} if self.first_token_from_decode else {
                i: 0 for i in range(len(self.prefill_endpoints))})
        
        
        warmup_ids = set()
        for i, ep in enumerate(self.prefill_endpoints):
            for j in range(per_ep):
                req_id = f"_harness_warmup_{i}_{j}"
                warmup_ids.add(req_id)
                self._pending[req_id] = {
                    "sample_id": None,
                    "first_sent": False,
                    "prefill_idx": i,
                }
                msg = {"id": req_id, "prompt": prompt}
                if self.first_token_from_decode:
                    msg["emit_first_token"] = True
                if self.warmup_decode_max_tokens is not None:
                    msg["decode_max_tokens"] = self.warmup_decode_max_tokens
                await ep.push_sock.send(_pack(msg))
                if not self.first_token_from_decode:
                    first_pending[i] += 1

        warmup_poll_ms = int(self.harness_config.get(
            "warmup_poll_timeout_ms", 600_000))

        async def drain_prefill(ep_idx: int) -> None:
            ep = self.prefill_endpoints[ep_idx]
            while first_pending[ep_idx] > 0:
                if not await ep.pull_sock.poll(timeout=warmup_poll_ms):
                    raise RuntimeError(
                        f"Harness warmup timed out after "
                        f"{warmup_poll_ms // 1000}s waiting for prefill "
                        f"worker {ep.label} (first_token).")
                raw = await ep.pull_sock.recv()
                result = _unpack(raw)
                req_id = result.get("id", "")
                if req_id not in warmup_ids:
                    continue
                
                first_pending[ep_idx] -= 1

        async def drain_decode_all() -> None:
            remaining = total
            polls = [ep.pull_sock for ep in self.decode_endpoints]
            while remaining > 0:
                drained_this_round = False
                for pull_sock in polls:
                    if not await pull_sock.poll(timeout=100):
                        continue
                    raw = await pull_sock.recv()
                    result = _unpack(raw)
                    req_id = result.get("id", "")
                    if req_id not in warmup_ids:
                        continue
                    if ("first_token_ids" in result
                            and "token_ids" not in result):
                        drained_this_round = True
                        continue
                    
                    self._pending.pop(req_id, None)
                    remaining -= 1
                    drained_this_round = True
                if not drained_this_round:
                    
                    
                    
                    
                    await asyncio.sleep(0.05)

        drain_tasks = []
        if not self.first_token_from_decode:
            drain_tasks.extend(
                asyncio.create_task(drain_prefill(i))
                for i in range(len(self.prefill_endpoints))
            )
        drain_tasks.append(asyncio.create_task(drain_decode_all()))

        try:
            await asyncio.wait_for(
                asyncio.gather(*drain_tasks),
                timeout=warmup_poll_ms / 1000,
            )
        except asyncio.TimeoutError as exc:
            raise asyncio.TimeoutError(
                f"Harness warmup decode drain exceeded "
                f"{warmup_poll_ms // 1000}s."
            ) from exc
        finally:
            for task in drain_tasks:
                if not task.done():
                    task.cancel()

        log.info("Harness warmup complete (%d requests)", total)

    async def _issuer(self) -> None:
        
        
        q = self._issue_queue
        while True:
            item = await q.get()
            if item is _STOP:
                return
            ep_idx, msg = item
            ep = self.prefill_endpoints[ep_idx]
            await ep.push_sock.send(_pack(msg))

    async def _prefill_collector(self, ep_idx: int) -> None:
        """Drain first-token notifications from one prefill engine."""
        import zmq
        ep = self.prefill_endpoints[ep_idx]
        pull_sock = ep.pull_sock
        while not self._shutdown_flag:
            if not await pull_sock.poll(timeout=1000):
                continue
            
            
            while True:
                try:
                    raw = await pull_sock.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                result = _unpack(raw)
                req_id = result.get("id", "")
                entry = self._pending.get(req_id)
                if entry is None:
                    continue
                sample_id = entry.get("sample_id")
                if sample_id is None:
                    
                    continue
                first_token_ids = result.get("first_token_ids", [])
                if first_token_ids and not entry["first_sent"]:
                    create_response_and_send_first_token(
                        sample_id, first_token_ids)
                    entry["first_sent"] = True
                    self._n_first += 1

    async def _decode_collector(self, ep_idx: int) -> None:
        """Drain final-token messages from one decode engine."""
        import zmq
        ep = self.decode_endpoints[ep_idx]
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
                result = _unpack(raw)
                req_id = result.get("id", "")
                entry = self._pending.get(req_id)
                if entry is None:
                    continue
                sample_id = entry.get("sample_id")
                if sample_id is None:
                    self._pending.pop(req_id, None)
                    continue
                first_token_ids = result.get("first_token_ids", [])
                if first_token_ids and "token_ids" not in result:
                    if not entry["first_sent"]:
                        create_response_and_send_first_token(
                            sample_id, first_token_ids)
                        entry["first_sent"] = True
                        self._n_first += 1
                    continue
                token_ids = result.get("token_ids", [])
                if token_ids and not entry["first_sent"]:
                    create_response_and_send_first_token(
                        sample_id, token_ids[:1])
                    entry["first_sent"] = True
                    self._n_first += 1
                entry = self._pending.pop(req_id)
                create_response_and_send_complete(sample_id, token_ids)
                self._n_completed += 1
                self._n_output_tokens += len(token_ids)
                
                
                
                i = entry.get("prefill_idx", 0)
                if 0 <= i < len(self.prefill_endpoints):
                    pe = self.prefill_endpoints[i]
                    pe.finished += 1
                    pe.tokens_done += entry.get("prompt_len", 0)
                gc_counter += 1
                if GC_INTERVAL > 0 and gc_counter >= GC_INTERVAL:
                    gc.collect()
                    gc_counter = 0

    async def _progress(self) -> None:
        while not self._shutdown_flag:
            await asyncio.sleep(10)
            if self._shutdown_flag or self._t0 is None:
                break
            elapsed = time.time() - self._t0
            issued = self._n_issued
            completed = self._n_completed
            inflight = issued - completed
            qps = completed / elapsed if elapsed > 0 else 0.0
            tok_s = self._n_output_tokens / elapsed if elapsed > 0 else 0.0
            queues = ",".join(
                str(ep.sent - ep.finished) for ep in self.prefill_endpoints)
            
            
            tok_q = ",".join(
                f"{(ep.tokens_sent - ep.tokens_done) // 1000}k"
                for ep in self.prefill_endpoints)
            log.info(
                "Progress [%.0fs] issued=%d first=%d done=%d inflight=%d "
                "%.1f qps %.0f tok/s prefill_queues=%s tok_q=%s",
                elapsed, issued, self._n_first, completed,
                inflight, qps, tok_s, queues, tok_q)

    

    def issue_queries(self, query_samples) -> None:
        for sample in query_samples:
            prompt = self.data_object.input_ids[sample.index]
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            elif not isinstance(prompt, list):
                prompt = list(prompt)
            plen = len(prompt)
            ep_idx = self._pick_prefill(plen)
            ep = self.prefill_endpoints[ep_idx]
            self._req_seq += 1
            req_id = f"r{self._req_seq}"
            
            
            self._pending[req_id] = {
                "sample_id": sample.id,
                "first_sent": False,
                "prefill_idx": ep_idx,
                "prompt_len": plen,
            }
            ep.sent += 1
            ep.tokens_sent += plen
            self._n_issued += 1
            msg = {"id": req_id, "prompt": prompt}
            if self.first_token_from_decode:
                msg["emit_first_token"] = True
            if self.decode_max_tokens is not None:
                msg["decode_max_tokens"] = self.decode_max_tokens
            self._loop.call_soon_threadsafe(
                self._issue_queue.put_nowait, (ep_idx, msg))

    def flush_queries(self) -> None:
        pass

    def stop(self) -> None:
        self._shutdown_flag = True
        if self._loop is not None and self._loop.is_running():
            
            
            
            if self._issue_queue is not None:
                self._loop.call_soon_threadsafe(
                    self._issue_queue.put_nowait, _STOP)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=30)
        log.info(
            "pd stopped (issued=%d completed=%d first=%d tokens=%d)",
            self._n_issued, self._n_completed, self._n_first,
            self._n_output_tokens)
