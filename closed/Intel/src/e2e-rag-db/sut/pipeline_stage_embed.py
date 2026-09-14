# Copyright 2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# =============================================================================

"""Stage 2: embed passages via a vLLM /v1/embeddings pooling server.

FAISS + engine/vectordb.py's query side expect L2-normalized vectors, so
this client normalizes every response (idempotent on already-normalized
outputs).

On unrecoverable failure (server unreachable at boot, or first fatal
error mid-run), this worker signals via ready_queue ('stage2_error') and
os._exit(3)s rather than silently spinning."""

import asyncio
import logging
import os
import queue as pyqueue
import time
import urllib.error
import urllib.request
from multiprocessing import Queue

import aiohttp
import numpy as np

from sut.pipeline_config import PipelineConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Stage2Embed")

# One quick retry covers transient hiccups; anything worse means the server
# is really down and we want to fail fast rather than spam.
_RETRY_DELAYS = (0.5,)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


def _preflight(url: str, tries: int = 3, delay: float = 2.0) -> None:
    """Verify the embed vLLM server is reachable. Raises RuntimeError on
    failure with an actionable message."""
    last_exc = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=3) as resp:
                if 200 <= resp.status < 300:
                    log.info(f"[Stage2] embed server reachable at {url}")
                    return
                last_exc = RuntimeError(f"HTTP {resp.status}")
        except (urllib.error.URLError, OSError, ValueError) as e:
            last_exc = e
        if attempt < tries:
            time.sleep(delay)
    raise RuntimeError(
        f"Embed server at {url} unreachable ({last_exc}). "
        f"Start it with: bash scripts/servers/launch_server_embed_vllm.sh"
    )


def _log_task_exception(task: asyncio.Task) -> None:
    """Consume exceptions on fire-and-forget tasks so asyncio doesn't
    print 'Task exception was never retrieved' for every failure."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.debug(f"[Stage2] batch task swallowed: {exc}")


async def _embed_texts(
    session: aiohttp.ClientSession, url: str, model: str, texts: list,
) -> np.ndarray:
    # truncate_prompt_tokens=-1 asks the server to clamp each passage to its
    # own max_model_len (512 for e5-base-v2). Matches 256d40c's original
    # tokenize(..., truncation=True, max_length=256) guarantee: no over-limit
    # input ever reaches the model.
    payload = {"model": model, "input": texts, "truncate_prompt_tokens": -1}
    last_exc = None
    for attempt, delay in enumerate((0.0, *_RETRY_DELAYS)):
        if delay:
            await asyncio.sleep(delay)
        try:
            async with session.post(f"{url}/v1/embeddings", json=payload) as resp:
                resp.raise_for_status()
                body = await resp.json()
            vecs = np.asarray(
                [item["embedding"] for item in body["data"]], dtype=np.float32,
            )
            return _l2_normalize(vecs)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exc = e
    raise last_exc


async def _process_batch(
    batch: list, session: aiohttp.ClientSession, url: str, model: str,
    output_queue: Queue, timing_queue: Queue, sem: asyncio.Semaphore,
    fatal_event: asyncio.Event,
):
    passage_items = [it for it in batch if it.get('kind') == 'passage']
    texts = [it['text'] for it in passage_items]

    if texts:
        async with sem:
            t0 = time.time()
            try:
                emb = await _embed_texts(session, url, model, texts)
            except Exception as e:
                if not fatal_event.is_set():
                    log.error(f"[Stage2] embed batch failed, aborting: {e}")
                    fatal_event.set()
                return
            for it, e in zip(passage_items, emb):
                it['embedding'] = e
            if timing_queue is not None:
                timing_queue.put(('stage2', 'embed_batch', time.time() - t0,
                                  {'batch_size': len(texts)}))

    output_queue.put(batch)


async def _run_embed_loop(
    config: PipelineConfig, input_queue: Queue, output_queue: Queue,
    timing_queue: Queue, embed_url: str, model: str,
) -> bool:
    """Returns True on clean exit (None sentinel), False if fatal_event fired."""
    sem = asyncio.Semaphore(config.embed_concurrency)
    connector = aiohttp.TCPConnector(limit=config.embed_concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=300)
    inflight: set = set()
    fatal_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while not fatal_event.is_set():
            try:
                item = await loop.run_in_executor(
                    None, input_queue.get, True, 0.5)
            except pyqueue.Empty:
                continue
            if item is None:
                break
            task = asyncio.create_task(_process_batch(
                item, session, embed_url, model, output_queue, timing_queue,
                sem, fatal_event,
            ))
            inflight.add(task)
            task.add_done_callback(inflight.discard)
            task.add_done_callback(_log_task_exception)
            if len(inflight) >= config.embed_concurrency * 4:
                await asyncio.wait(
                    inflight, return_when=asyncio.FIRST_COMPLETED)

        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

    return not fatal_event.is_set()


def stage2_embed_worker(
    config: PipelineConfig,
    input_queue: Queue,
    output_queue: Queue,
    timing_queue: Queue,
    ready_queue: Queue,
    embed_url: str,
    model_name: str,
):
    log.info(f"[Stage2] embed worker starting; url={embed_url} "
             f"model={model_name} concurrency={config.embed_concurrency}")

    try:
        _preflight(embed_url)
    except RuntimeError as e:
        log.error(f"[Stage2] preflight failed: {e}")
        ready_queue.put(('stage2_error', str(e)))
        os._exit(3)

    ready_queue.put(('stage2', 0))

    ok = False
    try:
        ok = asyncio.run(_run_embed_loop(
            config, input_queue, output_queue, timing_queue,
            embed_url, model_name,
        ))
    except Exception as e:
        log.error(f"[Stage2] embed loop error: {e}")

    if not ok:
        log.error("[Stage2] exiting with code 3 after unrecoverable error")
        os._exit(3)

    output_queue.put(None)
    log.info("[Stage2] embed worker shutting down")
