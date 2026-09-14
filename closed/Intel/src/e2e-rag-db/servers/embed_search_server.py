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
Embedding + FAISS retrieval as an aiohttp HTTP service with server-side batching.

The e5 embedder and FAISS index are co-located here so the orchestrator POSTs a
query string and gets documents back, like it POSTs to the vLLM servers.

    POST /embed_search   {"query": "...", "k": 10}
      -> {"docs": [{"page_content": "...", "metadata": {...}}, ...]}
    GET  /health         -> 200 {"status": "ok"}

Concurrent requests are coalesced into one embed_documents([...]) forward pass.
Since embed_query(t) == embed_documents([t])[0] here (no query prefix), results
are numerically identical to the sequential baseline's rag_db.lookup().
"""

import os
import time
import hashlib
import argparse
import asyncio

from aiohttp import web

# Batcher tuning
MAX_BATCH = 32
MAX_WAIT_MS = 5


def _file_md5(path, _chunk=1 << 20):
    """md5 hex of a file, or '(missing)' if it can't be read."""
    try:
        # usedforsecurity=False: file-identity checksum, not a security digest
        # (silences bandit B324).
        h = hashlib.md5(usedforsecurity=False)
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(_chunk), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return "(missing)"


def build_vectordb(db_path, embedding_model, device):
    """Load the embedder + FAISS index (no reranker) the same way the baseline
    does, so retrieval results are identical."""
    from engine import VectorDB
    rag_db = VectorDB(
        embedding_model=embedding_model,
        reranker_model=None,          # this service does NOT rerank
        device=device,
        database=db_path.replace(".db", ""),
        num_embedding_devices=1,
        benchmark=False,
    )
    rag_db.from_serialized(db_path)
    return rag_db


class EmbedSearchService:
    def __init__(self, rag_db, db_path="", db_md5=""):
        self.rag_db = rag_db
        self.db_path = db_path      # DB this server loaded (reported via /health)
        self.db_md5 = db_md5
        self.in_q = None            # created inside the loop
        self.n_requests = 0
        self.n_batches = 0
        self.n_embeddings = 0

    async def start(self, app):
        self.in_q = asyncio.Queue()
        app["batcher"] = asyncio.create_task(self._batcher())

    async def stop(self, app):
        task = app.get("batcher")
        if task:
            task.cancel()

    async def _batcher(self):
        """Coalesce queued requests into one batched embed + FAISS search."""
        while True:
            query, k, fut = await self.in_q.get()
            items = [(query, k, fut)]
            # Drain what's waiting (up to MAX_BATCH) within a tiny window.
            deadline = time.perf_counter() + MAX_WAIT_MS / 1000.0
            while len(items) < MAX_BATCH:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.in_q.get(), timeout))
                except asyncio.TimeoutError:
                    break

            bsz = len(items)  # coalesced batch size shared by every request in it
            try:
                results = self._run_batch([(q, k) for q, k, _ in items])
                for (_, _, fut), docs in zip(items, results):
                    if not fut.done():
                        fut.set_result((docs, bsz))
            except Exception as e:  # pragma: no cover - surface to all awaiters
                for _, _, fut in items:
                    if not fut.done():
                        fut.set_exception(e)

    def _run_batch(self, batch):
        """batch: list of (query, k) -> list of doc-lists (JSON-ready)."""
        queries = [q for q, _ in batch]
        vs = self.rag_db._vector_store
        lock = self.rag_db._embedding_lock

        # Batched embedding (single forward pass over all queued queries).
        if lock:
            with lock:
                vectors = self.rag_db._embedding_model.embed_documents(queries)
        else:
            vectors = self.rag_db._embedding_model.embed_documents(queries)

        self.n_batches += 1
        self.n_embeddings += len(queries)

        out = []
        for (query, k), vec in zip(batch, vectors):
            docs = vs.similarity_search_by_vector(vec, k=k)
            out.append([
                {"page_content": d.page_content, "metadata": dict(d.metadata)}
                for d in docs
            ])
        return out

    async def handle_embed_search(self, request):
        payload = await request.json()
        query = payload["query"]
        k = int(payload.get("k", 10))
        self.n_requests += 1
        fut = asyncio.get_running_loop().create_future()
        await self.in_q.put((query, k, fut))
        docs, batch_size = await fut
        return web.json_response({"docs": docs, "batch_size": batch_size})

    async def handle_health(self, request):
        return web.json_response({
            "status": "ok",
            "db": self.db_path,
            "db_md5": self.db_md5,
            "requests": self.n_requests,
            "batches": self.n_batches,
            "embeddings": self.n_embeddings,
            "avg_batch": (self.n_embeddings / self.n_batches) if self.n_batches else 0,
        })


def main():
    p = argparse.ArgumentParser(description="Embedding + FAISS aiohttp service")
    p.add_argument("--db", required=True)
    p.add_argument("--embedding-model", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    args = p.parse_args()

    print(f"[embed_search] loading FAISS db {args.db} + model {args.embedding_model} ...")
    t0 = time.perf_counter()
    rag_db = build_vectordb(args.db, args.embedding_model, args.device)
    # Enable the embedding lock so batched access is thread-safe (harmless single-loop).
    rag_db.enable_threading()
    db_md5 = _file_md5(args.db)
    print(f"[embed_search] ready in {time.perf_counter()-t0:.1f}s (db_md5={db_md5})")

    service = EmbedSearchService(rag_db, db_path=args.db, db_md5=db_md5)
    app = web.Application()
    app.router.add_post("/embed_search", service.handle_embed_search)
    app.router.add_get("/health", service.handle_health)
    app.on_startup.append(service.start)
    app.on_cleanup.append(service.stop)

    web.run_app(app, host=args.host, port=args.port, print=lambda *a: None)


if __name__ == "__main__":
    main()
