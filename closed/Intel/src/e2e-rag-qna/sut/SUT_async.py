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
Async System Under Test for the RAG-QnA workload.

Drop-in replacement for E2ESUT that runs the async multi-shot retrieval instead
of the sequential multi_shot_retrieval. One asyncio loop on a background thread
runs one coroutine per sample; issue_queries submits all samples and blocks
until done, so the Offline batch is processed concurrently.
"""

import os
import json
import time
import array
import asyncio
import logging
import threading
from datetime import datetime

import mlperf_loadgen as lg

from sut.QSL_qna import E2EQSLInMemory
from common.utils import get_model_name_from_service, get_max_tokens_from_service
from rag import multi_shot_retrieval_async as pr
from common import tracer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("E2ESUTAsync")


class E2ESUTAsync:
    def __init__(self, dataset_path, db_path, args):
        self.dataset_path = dataset_path
        self.args = args
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Optional tracing (env-gated so a normal run is unaffected).
        # TRACE = comma-list of views (e.g. "batch-curve,query-timeline"); empty = off.
        # TRACE_DIR = directory each view file is written into.
        trace_views = os.environ.get("TRACE")
        trace_dir = os.environ.get("TRACE_DIR", os.path.join(self.output_dir, "results"))
        selected = tracer.enable(trace_views, trace_dir) if trace_views else []
        if selected:
            log.info(f"Tracing enabled -> views={selected} dir={trace_dir}")

        # QSL (same as sync SUT).
        log.info("Initializing QSL...")
        self.qsl = E2EQSLInMemory(dataset_path, args.perf_count)

        # Resolve models/endpoints (mirror the sequential config).
        llm_url = args.llm_service_url
        query_url = getattr(args, "query_service_url", None) or llm_url
        suff_url = getattr(args, "sufficiency_service_url", None) or query_url
        embed_url = os.environ.get("EMBED_URL", "http://127.0.0.1:8100")
        rerank_url = os.environ.get("RERANK_URL", "http://127.0.0.1:8101")

        # Resolve model names from each /v1/models: vLLM serves under the fs id.
        grader_model = get_model_name_from_service(llm_url)
        query_model = get_model_name_from_service(query_url)
        suff_model = get_model_name_from_service(suff_url)
        max_tokens = getattr(args, "max_tokens", "auto")
        max_tokens = int(get_max_tokens_from_service(llm_url)
                         if (isinstance(max_tokens, str) and max_tokens.lower() == "auto")
                         else max_tokens)

        self.cfg = {
            "grader_service_url": llm_url, "grader_model_name": grader_model,
            "query_service_url": query_url, "query_model_name": query_model,
            "sufficiency_service_url": suff_url, "sufficiency_model_name": suff_model,
            "max_tokens": max_tokens, "temperature": args.temperature,
            "max_retries": args.max_retries,
        }
        self.urls = {"embed": embed_url, "rerank": rerank_url}
        self.params = {
            "max_iterations": args.max_iterations,
            "max_sub_queries": args.max_sub_queries,
            "top_k_retriever": args.top_k_retriever,
            "top_k_reranking": args.top_k_reranking,
        }
        # Per-server host-side concurrency caps (SERVER_LIMITS or --server_limits).
        server_spec = getattr(args, "server_limits", None) or os.environ.get("SERVER_LIMITS")
        self.server_limits = pr.parse_server_limits(server_spec)

        # Perf mode (--perf-test-mode <recorded log>): replay recorded LLM
        # responses for determinism while still issuing the real calls, which
        # are what the run measures.
        self.perf_test_cache = None
        perf_log = getattr(args, "perf_test_mode", None)
        if perf_log:
            from common.llm_replay_cache import PerfTestCache
            log.info(f"Perf replay: loading recorded LLM responses from {perf_log}")
            self.perf_test_cache = PerfTestCache(perf_log)
            stats = self.perf_test_cache.get_cache_stats()
            log.info(f"  cached responses={stats['total_responses']} "
                     f"queries={stats['unique_queries']} "
                     f"components={stats['unique_components']}")

        log.info(f"Async SUT config: grader={grader_model} query={query_model} "
                 f"suff={suff_model} max_tokens={max_tokens}")
        log.info(f"  embed={embed_url} rerank={rerank_url}")
        log.info(f"  server_limits={self.server_limits or 'unlimited'}")

        self.results = {}
        self.results_lock = threading.Lock()

        # Start a dedicated event loop in a background thread.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        log.info("Async SUT initialized")

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _process_all(self, query_samples):
        """Build jobs for all samples and run them concurrently on the loop."""
        import aiohttp

        jobs = []
        for qs in query_samples:
            sample = self.qsl[qs.index]
            job = pr.QueryJob(qs.index, sample["query"],
                              sample["expected_urls"], sample["ground_truth"])
            job._loadgen_id = qs.id
            jobs.append(job)

        # query_index (str) -> loadgen query id, for completion callbacks.
        idx_to_lgid = {str(j.query_id): j._loadgen_id for j in jobs}

        total = len(jobs)
        t_batch0 = time.perf_counter()
        progress = {"done": 0}

        def on_done(result):
            qidx = str(result["query_index"])
            lg_id = idx_to_lgid.get(qidx)
            answer = result.get("llm_answer", "Unknown")

            with self.results_lock:
                self.results[lg_id] = {
                    "query": result["query"],
                    "answer": answer,
                    "ground_truth": result.get("ground_truth_answer", ""),
                    "retrieved_urls": result.get("retrieved_urls", []),
                    "expected_urls": result.get("expected_urls", []),
                    "metrics": result.get("metrics", {}),
                }
                progress["done"] += 1
                done = progress["done"]

            # Per-query progress line; ETA from the running completion rate.
            elapsed = time.perf_counter() - t_batch0
            rate = done / elapsed if elapsed > 0 else 0.0          # queries/sec
            eta = (total - done) / rate if rate > 0 else 0.0
            m = result.get("metrics", {})
            # Server-reported running-batch depth per LLM (from the /metrics poller).
            batch = tracer.vllm_batch_now()
            batch_str = ("  |  batch: " + "  ".join(f"{k}={v}" for k, v in sorted(batch.items()))) if batch else ""
            log.info(
                f"[{done:>4}/{total}] {100*done/total:5.1f}%  "
                f"q{qidx} it={result.get('num_iterations')} "
                f"P={m.get('precision',0):.2f} R={m.get('recall',0):.2f} F1={m.get('f1',0):.2f} "
                f"wall={result.get('wall_s',0):.0f}s | "
                f"elapsed={elapsed/60:.1f}min rate={rate*60:.1f} q/min ETA={eta/60:.1f}min"
                f"{batch_str}"
            )

            # Report this query's completion to loadgen immediately.
            # LoadGen's accuracy log decodes `data` as 4-byte int32 token IDs and
            # counts them for the TEST09 token-length check, so emit a blob of
            # int32s sized by the answer generation's real completion_tokens
            # (the readable answer is kept in results.json). Fall back to a
            # whitespace-token estimate when the server gave no usage count.
            osl = result.get("answer_output_tokens")
            if not osl or osl <= 0:
                osl = max(1, len(answer.split()))
            token_ids = array.array("i", [0] * osl)
            bi = token_ids.buffer_info()
            lg.QuerySamplesComplete([
                lg.QuerySampleResponse(lg_id, bi[0], bi[1] * token_ids.itemsize, osl)
            ])

        async def _run():
            # limit=0: connection count is already bounded per-server
            connector = aiohttp.TCPConnector(limit=0, enable_cleanup_closed=True)
            # trust_env=True: honor proxy env vars (aiohttp ignores them by default)
            async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
                await pr.run_batch(jobs, self.cfg, self.urls, self.params,
                                   server_limits=self.server_limits,
                                   on_done=on_done, session=session,
                                   perf_test_cache=self.perf_test_cache)

        fut = asyncio.run_coroutine_threadsafe(_run(), self._loop)
        fut.result()   # block until all queries done (Offline batch semantics)

    def issue_queries(self, query_samples):
        n = len(query_samples)
        log.info(f"issue_queries: {n} samples (Offline batch)")
        self._process_all(query_samples)
        log.info(f"issue_queries: all {n} completed")
        if self.perf_test_cache is not None:
            misses = pr.perf_cache_misses()
            if misses:
                log.warning(f"perf replay: {misses} call(s) had no recorded "
                            f"response and used the real LLM output -- this run "
                            f"is not fully deterministic")
            else:
                log.info("perf replay: every LLM call was served from the recording")

    def flush_queries(self):
        pass

    def save_results(self, output_path):
        with self.results_lock:
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=2)
        log.info(f"Results saved to {output_path} ({len(self.results)} queries)")

    def finalize(self):
        if tracer.is_enabled():
            paths = tracer.export()
            for p in paths:
                log.info(f"Trace written to {p}")
            s = tracer.summary()
            for cname, bd in s.get("batch_depth", {}).items():
                log.info(f"  {cname}: mean={bd.get('mean_over_active',0):.1f} peak={bd.get('peak',0)}")
        # stop the loop
        self._loop.call_soon_threadsafe(self._loop.stop)

    def __del__(self):
        if hasattr(self, "sut") and self.sut is not None:
            lg.DestroySUT(self.sut)
