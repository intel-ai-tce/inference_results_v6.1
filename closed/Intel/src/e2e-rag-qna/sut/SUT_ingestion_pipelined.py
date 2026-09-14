# Copyright 2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# =============================================================================

"""Pipelined MLPerf SUT for the RAG ingestion workload.

3-stage pipeline:
  Stage 1 (N parse processes)  → chunk HTML into passages
  Stage 2 (1  embed process)   → POST /v1/embeddings on a vLLM pooling server
  Stage 3 (1  index process)   → FAISS-add + save DB + MD5

Set env PIPELINE_TIMING=1 to enable per-stage timing collection.
"""

import array
import logging
import multiprocessing as mp
import os
import queue as pyqueue
import threading
import time
from collections import defaultdict
from typing import Any, Dict

import mlperf_loadgen as lg
import numpy as np

from sut.QSL_ingestion import DatasetupQSLInMemory
from sut.pipeline_config import PipelineConfig
from sut.pipeline_stage_embed import stage2_embed_worker
from sut.pipeline_stage_index import stage3_index_worker
from sut.pipeline_stage_parse import stage1_parse_chunk_worker
from common.utils import get_device_config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PipelinedDatasetupSUT")


class _TimingLogger:
    """Thread-safe collector for per-stage timing metrics."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.lock = threading.Lock()

    def log_timing(self, stage: str, operation: str, duration: float,
                   metadata: dict = None):
        entry = {'timestamp': time.time(), 'duration': duration}
        if metadata:
            entry.update(metadata)
        with self.lock:
            self.timings[f"{stage}.{operation}"].append(entry)

    def get_summary(self) -> Dict[str, Any]:
        summary = {}
        with self.lock:
            for key, entries in self.timings.items():
                if not entries:
                    continue
                d = [e['duration'] for e in entries]
                summary[key] = {
                    'count': len(d), 'total': sum(d),
                    'mean': float(np.mean(d)), 'median': float(np.median(d)),
                    'min': min(d), 'max': max(d),
                    'p95': float(np.percentile(d, 95)),
                    'p99': float(np.percentile(d, 99)),
                }
        return summary

    def print_summary(self):
        log.info("=" * 80)
        log.info("PIPELINE TIMING SUMMARY")
        log.info("=" * 80)
        for key, s in sorted(self.get_summary().items()):
            log.info(f"\n{key}: count={s['count']} total={s['total']:.2f}s "
                     f"mean={s['mean']:.4f}s median={s['median']:.4f}s "
                     f"p95={s['p95']:.4f}s p99={s['p99']:.4f}s")
        log.info("=" * 80)


class PipelinedDatasetupSUT:
    """Pipelined SUT with parse/embed/index in separate processes."""

    def __init__(
        self,
        documents_dir: str,
        database: str,
        chunk_size: int = 768,
        chunk_overlap: int = 32,
        text_boundary: str = "word",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        vector_index_method: str = "hnsw",
        embed_url: str = "http://127.0.0.1:8194",
    ):
        self.config = PipelineConfig()
        self._timing_enabled = os.environ.get("PIPELINE_TIMING") == "1"

        log.info("=" * 80)
        log.info("PIPELINED DATASETUP SUT INITIALIZATION")
        log.info(f"  Layout: {self.config.summary()}")
        log.info(f"  Embed server: {embed_url}")
        log.info(f"  Embedding model: {embedding_model}")
        log.info(f"  Device config: {get_device_config()}")
        log.info(f"  Timing enabled: {self._timing_enabled}")
        log.info("=" * 80)

        log.info("Initializing Datasetup Query Sample Library...")
        self.qsl = DatasetupQSLInMemory(documents_dir)
        log.info(f"QSL loaded: {len(self.qsl)} documents")

        ctx = mp.get_context('spawn')
        self.doc_queue = ctx.Queue()
        self.queue1 = ctx.Queue(maxsize=self.config.queue1_maxsize)
        self.queue2 = ctx.Queue(maxsize=self.config.queue2_maxsize)
        self.ready_queue = ctx.Queue()
        self.done_queue = ctx.Queue()
        self.results_queue = ctx.Queue()

        # Timing wiring is opt-in; when disabled the workers get None and skip
        # their timing_queue.put calls.
        self.timing_queue = ctx.Queue() if self._timing_enabled else None
        self.timing_logger = _TimingLogger() if self._timing_enabled else None

        self.processes = []
        self.parse_processes = []

        for i in range(self.config.num_parse_workers):
            p = ctx.Process(
                target=stage1_parse_chunk_worker,
                args=(
                    i, self.config, self.doc_queue, self.queue1,
                    self.timing_queue,
                    chunk_size, chunk_overlap, text_boundary,
                    self.ready_queue, self.done_queue,
                ),
            )
            p.start()
            self.processes.append(p)
            self.parse_processes.append(p)
            log.info(f"Started Stage1 worker {i} (PID {p.pid})")

        self.embed_process = ctx.Process(
            target=stage2_embed_worker,
            args=(
                self.config, self.queue1, self.queue2, self.timing_queue,
                self.ready_queue, embed_url, embedding_model,
            ),
        )
        self.embed_process.start()
        self.processes.append(self.embed_process)
        log.info(f"Started Stage2 embed worker (PID {self.embed_process.pid})")

        last_sample_id = len(self.qsl) - 1
        self.index_process = ctx.Process(
            target=stage3_index_worker,
            args=(
                self.config, self.queue2, self.timing_queue,
                self.results_queue, len(self.qsl), last_sample_id,
                database, embedding_model, reranker_model,
                vector_index_method, self.ready_queue,
            ),
        )
        self.index_process.start()
        self.processes.append(self.index_process)
        log.info(f"Started Stage3 index worker (PID {self.index_process.pid})")

        self._wait_for_workers_ready()

        if self._timing_enabled:
            self.timing_thread = threading.Thread(
                target=self._collect_timings, daemon=True)
            self.timing_thread.start()
        else:
            self.timing_thread = None

        self._result_pump = threading.Thread(
            target=self._pump_results, daemon=True)
        self._result_pump.start()

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        log.info("SUT construction complete")

    def _wait_for_workers_ready(self, timeout: float = 600.0):
        expected = self.config.num_parse_workers + 2  # 1 embed + 1 index
        received = 0
        deadline = time.monotonic() + timeout
        log.info(f"Waiting for {expected} workers to warm up...")
        while received < expected:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Only {received}/{expected} workers ready within {timeout}s")
            try:
                stage, payload = self.ready_queue.get(timeout=1.0)
            except pyqueue.Empty:
                continue
            if stage == 'stage2_error':
                for p in self.processes:
                    if p.is_alive():
                        p.terminate()
                raise RuntimeError(payload)
            received += 1
            log.info(f"Worker ready: {stage} ({payload}) [{received}/{expected}]")
        log.info("All pipeline workers ready")

    def _stage_died(self) -> str:
        """Return a description of the first stage that exited non-zero, or ''."""
        for name, procs in (('parse', self.parse_processes),
                            ('embed', [self.embed_process]),
                            ('index', [self.index_process])):
            for p in procs:
                if p.exitcode is not None and p.exitcode != 0:
                    return f"{name} process {p.pid} exited with code {p.exitcode}"
        return ''

    def _collect_timings(self):
        while True:
            try:
                td = self.timing_queue.get(timeout=0.5)
            except pyqueue.Empty:
                continue
            if td is None:
                break
            stage, op, dur, meta = td
            self.timing_logger.log_timing(stage, op, dur, meta)

    def _pump_results(self):
        while True:
            try:
                item = self.results_queue.get(timeout=0.5)
            except pyqueue.Empty:
                died = self._stage_died()
                if died:
                    log.error(f"[Pump] aborting: {died}")
                    break
                continue
            if item is None:
                break
            query_id, response_bytes = item
            arr = array.array('B', response_bytes)
            bi = arr.buffer_info()
            lg.QuerySamplesComplete([lg.QuerySampleResponse(
                query_id, bi[0], bi[1] * arr.itemsize, len(response_bytes),
            )])

    def issue_queries(self, query_samples):
        log.info(f"[Loadgen] Received {len(query_samples)} documents")
        for s in sorted(query_samples, key=lambda x: x.index):
            di = self.qsl[s.index]
            self.doc_queue.put(
                (s.index, s.id, di['file_path'], di['file_name']))
        log.info(f"[Loadgen] All {len(query_samples)} documents queued")

    def flush_queries(self):
        """Ordered shutdown: pill parse workers → join → pill embed → wait all.
        If any stage has already died, skip the graceful path and terminate."""
        flush_t0 = time.monotonic()
        try:
            q1_at_flush, q2_at_flush = self.queue1.qsize(), self.queue2.qsize()
        except Exception:
            q1_at_flush = q2_at_flush = -1
        log.info(
            f"[Loadgen] flush_queries (queue1={q1_at_flush}/{self.config.queue1_maxsize}, "
            f"queue2={q2_at_flush}/{self.config.queue2_maxsize})"
        )

        died = self._stage_died()
        if died:
            log.error(f"[Shutdown] aborting: {died}")
            for p in self.processes:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=10)
            self._result_pump.join(timeout=5)
            log.info(
                f"[Loadgen] flush complete (aborted, "
                f"flush_elapsed={time.monotonic() - flush_t0:.2f}s)"
            )
            return

        for _ in range(self.config.num_parse_workers):
            self.doc_queue.put(None)

        parse_done = 0
        deadline = time.monotonic() + 600
        while parse_done < self.config.num_parse_workers:
            try:
                self.done_queue.get(timeout=1.0)
            except pyqueue.Empty:
                if time.monotonic() > deadline:
                    log.warning("[Shutdown] parse-worker wait timed out")
                    break
                continue
            parse_done += 1
            log.info(f"[Shutdown] parse done {parse_done}/{self.config.num_parse_workers}")

        # Join parse processes so their Queue feeder threads flush queue1
        # before we tell the embed worker no more input is coming.
        for p in self.parse_processes:
            p.join(timeout=60)

        log.info("[Shutdown] queue1 sentinel to embed")
        self.queue1.put(None)

        for p in (self.embed_process, self.index_process):
            p.join(timeout=600)
            if p.is_alive():
                log.warning(f"Process {p.pid} did not complete, terminating")
                p.terminate()
                p.join()

        # Index writes its own None sentinel on results_queue after MD5; wait
        # for the pump to drain it before returning to loadgen.
        self._result_pump.join(timeout=30)

        if self._timing_enabled:
            self.timing_queue.put(None)
            self.timing_thread.join(timeout=5)
            self.timing_logger.print_summary()
        log.info(
            f"[Loadgen] flush complete "
            f"(flush_elapsed={time.monotonic() - flush_t0:.2f}s)"
        )

    def shutdown(self):
        """Explicit teardown before os._exit(0) — FAISS/OpenMP otherwise
        unwind in an undefined order at exit and can segfault."""
        for p in self.processes:
            try:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=10)
            except Exception:
                pass
        if self._timing_enabled:
            try:
                self.timing_queue.put(None)
                if self.timing_thread:
                    self.timing_thread.join(timeout=5)
            except Exception:
                pass
