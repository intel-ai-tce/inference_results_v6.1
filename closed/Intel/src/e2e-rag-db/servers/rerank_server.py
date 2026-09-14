# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
ColBERT reranker as an aiohttp HTTP service backed by an internal worker pool.

The pipeline sees one service on one port; behind it, N pinned worker processes
share one queue (pull-based: an idle worker takes the next job). ColBERT rerank
does not batch — each (query, passages) group is a separate forward pass — so
throughput scales by worker count, not batch size. Each worker is NUMA-pinned to
its own physical-core block, off the LLM's cores.

    POST /rerank   {"query": "...", "passages": ["...", ...]}
      -> {"scored": [["passage", score], ...], "batch_size": <workers busy>}
    GET  /health   -> per-worker stats

Config (env): RERANK_NUM_WORKERS, RERANK_WORKER_CORES (";"-separated core lists,
one per worker, e.g. "79-84;122-127"), or legacy
INFERENCE_RERANKER_NUMA_NODE / INFERENCE_RERANKER_OMP_NUM_THREADS if CORES unset.
"""

import os
import time
import argparse
import asyncio
import threading
import multiprocessing as mp

from aiohttp import web   # safe at top: aiohttp does not import torch


def _parse_core_list(spec):
    """'79-84' or '79,80,81' -> [79,80,81,82,83,84]."""
    cores = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            cores.extend(range(int(a), int(b) + 1))
        elif part:
            cores.append(int(part))
    return cores


def _pinned_worker_main(model_name, device, cores, request_q, response_q, ready_q):
    """Worker process: pin to `cores`, load ColBERT, serve from the shared queue.

    Order matters: set affinity + OMP_NUM_THREADS BEFORE importing torch, else
    the thread pool is sized wrong. All workers share one request_q, so a free
    worker naturally pulls the next job (pull-based load balancing).
    """
    from servers.reranker_worker import _set_parent_death_signal, _do_rerank
    _set_parent_death_signal()

    if cores:
        try:
            os.sched_setaffinity(0, set(cores))
        except OSError as e:
            print(f"[rerank worker {os.getpid()}] sched_setaffinity failed: {e}")
        os.environ["OMP_NUM_THREADS"] = str(len(cores))
    if device == "cpu":
        from common.utils import apply_cpu_threading_env
        apply_cpu_threading_env()

    import torch  # noqa: F401 (after pinning)
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[rerank worker pid={os.getpid()}] ready on cores={cores[:1]}..{cores[-1:]} "
          f"({len(cores)} cores)")
    ready_q.put(os.getpid())

    while True:
        item = request_q.get()
        if item is None:
            break
        req_id, query, passages = item
        try:
            scored = _do_rerank(model, tokenizer, device, query, passages)
            response_q.put((req_id, scored, None))
        except Exception as e:  # pragma: no cover
            response_q.put((req_id, None, str(e)))


class RerankPool:
    """Front-end: owns the worker pool + async bridge over mp queues."""

    def __init__(self, model_name, device, worker_cores):
        self.model_name = model_name
        self.device = device
        self.worker_cores = worker_cores          # list[list[int]], one per worker
        self.n_workers = len(worker_cores)
        self.ctx = mp.get_context("spawn")
        self.request_q = self.ctx.Queue()
        self.response_q = self.ctx.Queue()
        self.ready_q = self.ctx.Queue()
        self.procs = []
        self._futures = {}                         # req_id -> asyncio.Future
        self._next_id = 0
        self._inflight = 0
        self.n_requests = 0
        self.loop = None

    async def start(self, app):
        self.loop = asyncio.get_running_loop()
        for cores in self.worker_cores:
            p = self.ctx.Process(
                target=_pinned_worker_main,
                args=(self.model_name, self.device, cores,
                      self.request_q, self.response_q, self.ready_q),
                daemon=True,
            )
            p.start()
            self.procs.append(p)
        # Wait for all workers to load the model (off the loop).
        for _ in range(self.n_workers):
            await self.loop.run_in_executor(None, self.ready_q.get)
        print(f"[rerank] pool ready: {self.n_workers} workers")
        # One background thread drains response_q and resolves futures on the loop.
        app["reader"] = threading.Thread(target=self._reader, daemon=True)
        app["reader"].start()

    async def stop(self, app):
        for _ in self.procs:
            self.request_q.put(None)

    def _reader(self):
        while True:
            req_id, scored, err = self.response_q.get()
            fut = self._futures.pop(req_id, None)
            if fut is None:
                continue
            if err is not None:
                self.loop.call_soon_threadsafe(fut.set_exception, RuntimeError(err))
            else:
                self.loop.call_soon_threadsafe(fut.set_result, scored)

    async def handle_rerank(self, request):
        payload = await request.json()
        query = payload["query"]
        passages = payload["passages"]
        req_id = self._next_id
        self._next_id += 1
        self.n_requests += 1
        fut = self.loop.create_future()
        self._futures[req_id] = fut
        self._inflight += 1
        busy = self._inflight            # approx workers busy at submit time
        try:
            self.request_q.put((req_id, query, passages))
            scored = await fut
        finally:
            self._inflight -= 1
        # batch_size kept for trace compatibility: how many were in flight (≈ pool load)
        return web.json_response({"scored": scored, "batch_size": min(busy, self.n_workers)})

    async def handle_health(self, request):
        alive = sum(1 for p in self.procs if p.is_alive())
        return web.json_response({
            "status": "ok",
            "workers": self.n_workers,
            "workers_alive": alive,
            "requests": self.n_requests,
            "inflight": self._inflight,
        })


def _resolve_worker_cores(n_workers):
    """Decide per-worker core lists from env, in priority order:
    1. RERANK_WORKER_CORES ("a-b;c-d;...")  — explicit, one list per worker
    2. INFERENCE_RERANKER_NUMA_NODE (+OMP)  — legacy single-node pinning
    3. unpinned                              — inherit (all cores)
    """
    spec = os.environ.get("RERANK_WORKER_CORES")
    if spec:
        lists = [_parse_core_list(s) for s in spec.split(";") if s.strip()]
        return lists
    numa = os.environ.get("INFERENCE_RERANKER_NUMA_NODE")
    omp = os.environ.get("INFERENCE_RERANKER_OMP_NUM_THREADS")
    if numa is not None:
        from common.utils import _physical_cores_for_node
        cores = _physical_cores_for_node(int(numa))
        if omp:
            cores = cores[: int(omp)]
        return [cores]
    # unpinned: n_workers with empty core lists (inherit affinity)
    return [[] for _ in range(n_workers)]


def main():
    p = argparse.ArgumentParser(description="ColBERT reranker pool service")
    p.add_argument("--reranker-model", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8101)
    args = p.parse_args()

    n_workers = int(os.environ.get("RERANK_NUM_WORKERS", "1"))
    worker_cores = _resolve_worker_cores(n_workers)
    # If CORES/NUMA gave fewer lists than requested workers, pad by repeating None.
    if len(worker_cores) < n_workers:
        worker_cores += [[] for _ in range(n_workers - len(worker_cores))]
    n_workers = len(worker_cores)
    print(f"[rerank] launching {n_workers} workers; cores per worker: "
          f"{[ (c[0], c[-1]) if c else 'unpinned' for c in worker_cores ]}")

    pool = RerankPool(args.reranker_model, args.device, worker_cores)
    app = web.Application()
    app.router.add_post("/rerank", pool.handle_rerank)
    app.router.add_get("/health", pool.handle_health)
    app.on_startup.append(pool.start)
    app.on_cleanup.append(pool.stop)
    web.run_app(app, host=args.host, port=args.port, print=lambda *a: None)


if __name__ == "__main__":
    main()
