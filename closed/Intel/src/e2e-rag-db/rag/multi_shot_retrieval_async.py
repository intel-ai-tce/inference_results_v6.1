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
Async, event-driven multi-shot retrieval.

One asyncio loop runs one coroutine per query. Every resource is an HTTP service
the loop awaits — LLM (vLLM), embedder+FAISS (embed_search_server), reranker
(rerank_server) — so N queries' waits overlap on one thread and the servers
batch concurrent requests.

Per-query control flow mirrors multi_shot_retrieval: rewriter -> retrieve ->
grader -> sufficiency -> {answer | next hop}. Prompts and JSON parsing are reused
from multi_shot_retrieval, so LLM behavior matches the baseline; only the
transport is async.

Usage (servers must already be running):
    python3 -m rag.multi_shot_retrieval_async --dataset data/frames_dataset.tsv \
        --embed-url http://127.0.0.1:8100 --rerank-url http://127.0.0.1:8101 \
        --llm-service-url http://127.0.0.1:8192/v1/chat/completions \
        --query-service-url http://127.0.0.1:8123/v1/chat/completions \
        --sufficiency-service-url http://127.0.0.1:8123/v1/chat/completions \
        --batch 8 --trace pipe.json
"""

import os
import re
import json
import time
import logging
import argparse
import asyncio

import aiohttp
import pandas as pd

# Reuse prompt templates + config/metrics from the sequential implementation.
from rag import multi_shot_retrieval as msr
from rag.multi_shot_retrieval import (
    RELEVANCE_CHECK_PROMPT, SUFFICIENCY_CHECK_PROMPT, QUERY_GENERATION_PROMPT,
    get_chat_completions_headers,
)
from evaluation.retrieval_metrics import calculate_retrieval_metrics
from common.utils import get_model_name_from_service, get_max_tokens_from_service

from common import tracer

log = logging.getLogger("E2ESUTAsync")

# --------------------------------------------------------------------------
# Per-server concurrency limits — a client-side semaphore capping host requests
# in flight to each server (independent of the server's own max_num_seqs). Names
# match tracer resource labels. Absent/<=0 = unlimited.
SERVER_NAMES = ("LLM-120B", "LLM-20B", "embedder", "reranker")
_SERVER_SEMS = {}   # server -> asyncio.Semaphore (only for limited servers)


def parse_server_limits(spec):
    """Parse a server-limit spec into {server: int}. Accepts a dict, or a
    comma-string like 'LLM-120B=256,reranker=24'. Unknown servers are ignored
    with a warning; non-positive limits mean unlimited (dropped)."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        items = spec.items()
    else:
        items = []
        for tok in str(spec).split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "=" not in tok:
                print(f"[server-limits] ignoring malformed token '{tok}' (want server=N)")
                continue
            k, v = tok.split("=", 1)
            items.append((k.strip(), v.strip()))
    out = {}
    for k, v in items:
        if k not in SERVER_NAMES:
            print(f"[server-limits] unknown server '{k}' (known: {', '.join(SERVER_NAMES)})")
            continue
        try:
            n = int(v)
        except (TypeError, ValueError):
            print(f"[server-limits] non-integer limit for '{k}': {v!r}")
            continue
        if n > 0:
            out[k] = n
    return out


def init_server_sems(limits):
    """(Re)build the per-server semaphore registry from a {server: int} map.
    Must be called inside the event loop (Semaphore binds to the running loop)."""
    _SERVER_SEMS.clear()
    for server, n in (limits or {}).items():
        if n and n > 0:
            _SERVER_SEMS[server] = asyncio.Semaphore(n)


class _NullCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


def server_slot(server):
    """Context manager capping concurrency to `server`; no-op if unlimited."""
    sem = _SERVER_SEMS.get(server)
    return sem if sem is not None else _NullCtx()


# --------------------------------------------------------------------------
# Perf-test replay cache — module-global like _SERVER_SEMS so the per-call
# lookup needs no extra plumbing through every component function.
#
# Semantics mirror multi_shot_retrieval.call_chat_completions: the real LLM
# call STILL happens (that is what perf mode measures), its output is thrown
# away, and the recorded response is returned so the retrieval trajectory is
# identical run to run. A failed call is fatal in this mode -- a run where some
# calls didn't happen is not a valid performance measurement.
_PERF_CACHE = None
_PERF_MISSES = 0


class PerfReplayError(RuntimeError):
    """An LLM call failed while replaying. Distinct type so the per-component
    `except Exception -> fallback` handlers can re-raise it instead of quietly
    degrading, which would leave an invalid perf measurement looking clean."""


def init_perf_cache(cache):
    """Install (or clear, with None) the perf-test replay cache."""
    global _PERF_CACHE, _PERF_MISSES
    _PERF_CACHE = cache
    _PERF_MISSES = 0


def perf_cache_misses():
    """Count of replay lookups that found no recorded response."""
    return _PERF_MISSES

# no_proxy for localhost (mirrors multi_shot_retrieval)
_np = os.environ.get("no_proxy", "")
os.environ["no_proxy"] = "127.0.0.1,localhost," + _np
os.environ["NO_PROXY"] = "127.0.0.1,localhost," + _np

_NO_RERANK = os.environ.get("NO_RERANK", "0") == "1"

# Reranker backend: 'colbert' = servers/rerank_server.py /rerank (default),
# 'score' = vLLM pooling server /v1/score. The vLLM server registers the model
# under its on-disk path, so /v1/score needs that name in the payload.
_RERANK_API = os.environ.get("RERANK_API", "colbert").strip().lower()
_RERANK_MODEL = (os.environ.get("RERANKER_MODEL_PATH")
                 or os.environ.get("INFERENCE_RERANKER_MODEL") or "")
# Must match the score server's --max-model-len (launch_server_rerank_vllm.sh).
_RERANK_MAX_LEN = int(os.environ.get("SERVER_RERANK_VLLM_MAX_LEN", "512"))
if _RERANK_API not in ("colbert", "score"):
    raise ValueError(f"RERANK_API must be 'colbert' or 'score', got {_RERANK_API!r}")


# --------------------------------------------------------------------------
# Async LLM call — aiohttp mirror of multi_shot_retrieval.call_chat_completions
# --------------------------------------------------------------------------
def _extract_output(result):
    """Same extraction as the sync path: content, else reasoning_content JSON."""
    message = result["choices"][0]["message"]
    out = (message.get("content") or "").strip()
    if not out:
        reasoning = message.get("reasoning_content") or ""
        if reasoning:
            m = re.search(r"\{.*\}", reasoning, re.DOTALL)
            if m:
                out = m.group(0)
    return out


async def call_llm_async(session, service_url, model_name, messages,
                         temperature=1.0, max_tokens=4096, max_retries=5,
                         component="unknown", hop_count=None, query_id=None,
                         return_usage=False):
    """Async POST to an OpenAI-compatible /v1/chat/completions endpoint.

    With return_usage=True, returns (text, completion_tokens) so callers can
    report the server's real output-token count (needed for TEST09) instead of
    estimating it from the answer text.
    """
    payload = {
        "model": model_name, "messages": messages,
        "temperature": temperature, "top_p": 1.0, "top_k": -1,
        "max_tokens": max_tokens, "reasoning_effort": "medium",
    }
    headers = get_chat_completions_headers(service_url)
    label = tracer._COMPONENT_LABEL.get(component, component)
    resource = tracer._resource_for_model(model_name)

    # Perf-test replay: look up the recorded response up front, but still issue
    # the real call below. A hit makes any failure fatal (see module comment).
    cached_response = None
    if _PERF_CACHE is not None and query_id is not None and hop_count is not None:
        cached_response = _PERF_CACHE.get_response(str(query_id), component, hop_count)
        if not cached_response:
            global _PERF_MISSES
            _PERF_MISSES += 1
            log.warning("[perf-replay] no recorded response for q%s %s hop %s; "
                        "using the real LLM output", query_id, component, hop_count)

    for attempt in range(max_retries):
        try:
            # Per-server cap, outside the span so queue-wait isn't timed as service.
            async with server_slot(resource):
                with tracer.span(label, resource, query_id=query_id,
                                 iteration=hop_count, model=model_name,
                                 max_tokens=max_tokens), tracer.inflight(f"inflight {resource}"):
                    async with session.post(service_url, json=payload, headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=1200)) as resp:
                        if resp.status == 429 or resp.status in (502, 503, 504):
                            await asyncio.sleep(2 ** attempt)
                            continue
                        resp.raise_for_status()
                        result = await resp.json()
            # Perf mode: discard the real output, return the recorded one. The
            # usage count stays from the real call -- it's the measurement.
            out = cached_response if cached_response else _extract_output(result)
            if return_usage:
                osl = (result.get("usage") or {}).get("completion_tokens")
                return out, osl
            return out
        except Exception as e:
            # Log the type: many aiohttp connection errors stringify to ''.
            msg = f"{type(e).__name__}: {e}".rstrip(": ")
            if attempt < max_retries - 1:
                log.warning("[%s] attempt %d/%d failed (%s); retrying",
                            component, attempt + 1, max_retries, msg)
                await asyncio.sleep(2 ** attempt)
                continue
            log.error("[%s] gave up after %d attempts: %s",
                      component, max_retries, msg)
            if cached_response:
                # Falling back to the recording would time a call that never
                # completed, so the run is not a valid perf measurement.
                raise PerfReplayError(
                    "perf replay requires every LLM call to succeed for "
                    f"run-to-run equivalency; {component} failed: {msg}") from e
            raise
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {component}")


# --------------------------------------------------------------------------
# Async component functions — same prompts/parsing as multi_shot_retrieval.
# --------------------------------------------------------------------------
def _extract_json(text):
    """Mirror the sync parsing: strip ``` fences, then grab {...}."""
    if "```" in text:
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else None


def _normalize_queries(raw, fallback):
    """Coerce the LLM's `queries` into a list of non-empty strings.

    Reasoning models sometimes emit queries as objects (e.g. {"query": "..."}
    or {"text": "..."}) or nested lists instead of plain strings. 
    """
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return [fallback]
    out = []
    for q in raw:
        if isinstance(q, str):
            s = q.strip()
        elif isinstance(q, dict):
            # common shapes: {"query": ...}, {"text": ...}, {"q": ...}
            v = q.get("query") or q.get("text") or q.get("q")
            s = v.strip() if isinstance(v, str) else ""
        else:
            # ints, None, nested lists, etc. are not usable queries -> drop
            s = ""
        if s:
            out.append(s)
    return out or [fallback]


async def gen_search_queries_async(session, cfg, question, kept_docs, max_queries,
                                   query_history, query_results, feedback_history,
                                   hop_count, query_id):
    kept_docs_text = ""
    if kept_docs:
        for i, doc in enumerate(kept_docs):
            content = doc[1] if len(doc) >= 2 else ""
            kept_docs_text += f"\n[DOC {i+1}] {content}\n"
    else:
        kept_docs_text = "None"
    history_text = ""
    if query_history and query_results:
        for q, count in zip(query_history, query_results):
            history_text += f"- Query: '{q}' → {count} docs\n"
    if not history_text:
        history_text = "No queries yet"
    feedback_text = "\n".join(feedback_history) if feedback_history else "Iteration 1 - Initial search"
    prompt = QUERY_GENERATION_PROMPT.format(
        question=question, kept_docs=kept_docs_text,
        history=history_text, feedback_history=feedback_text, max_queries=max_queries)

    try:
        out = await call_llm_async(
            session, cfg["query_service_url"], cfg["query_model_name"],
            [{"role": "user", "content": prompt}],
            temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
            max_retries=cfg["max_retries"], component="generate_search_queries",
            hop_count=hop_count, query_id=query_id)
        if not out:
            return {"queries": [question], "feedback": "LLM returned empty"}
        js = _extract_json(out)
        if not js:
            return {"queries": [question], "feedback": "No JSON in LLM response"}
        r = json.loads(js)
        return {"queries": _normalize_queries(r.get("queries", [question]), question),
                "feedback": r.get("feedback", "Generating queries")}
    except PerfReplayError:
        raise
    except Exception as e:
        return {"queries": [question], "feedback": f"Error: {e}"}


async def eval_relevance_async(session, cfg, question, new_docs, kept_docs,
                               hop_count, query_id):
    if not new_docs:
        return {"relevance": []}
    new_docs_text = ""
    for i, (url, content) in enumerate(new_docs):
        new_docs_text += f"\n[NEW {i+1}] {content}\n"
    kept_docs_text = ""
    if kept_docs:
        kept_docs_text = f"[{len(kept_docs)} documents already kept as relevant]\n"
        for i, doc in enumerate(kept_docs[:5]):
            content = doc[1] if len(doc) >= 2 else ""
            snippet = content[:300] if len(content) > 300 else content
            kept_docs_text += f"[KEPT {i+1}] {snippet}...\n"
    prompt = RELEVANCE_CHECK_PROMPT.format(
        question=question, new_docs=new_docs_text,
        kept_docs=kept_docs_text if kept_docs_text else "None", num_docs=len(new_docs))

    try:
        out = await call_llm_async(
            session, cfg["grader_service_url"], cfg["grader_model_name"],
            [{"role": "user", "content": prompt}],
            temperature=cfg["temperature"], max_tokens=4096,
            max_retries=cfg["max_retries"], component="evaluate_document_relevance",
            hop_count=hop_count, query_id=query_id)
        if not out:
            return {"relevance": [1] * len(new_docs)}
        js = _extract_json(out)
        if not js:
            return {"relevance": [1] * len(new_docs)}
        rel = json.loads(js).get("relevance", [])
        if len(rel) != len(new_docs):
            return {"relevance": [1] * len(new_docs)}
        return {"relevance": rel}
    except PerfReplayError:
        raise
    except Exception:
        return {"relevance": [1] * len(new_docs)}


async def check_sufficiency_async(session, cfg, question, kept_docs, iteration,
                                  max_iterations, hop_count, query_id):
    kept_docs_text = ""
    if kept_docs:
        for i, doc in enumerate(kept_docs):
            content = doc[1] if len(doc) >= 2 else ""
            kept_docs_text += f"\n[DOC {i+1}] {content}\n"
    else:
        kept_docs_text = "None"
    prompt = SUFFICIENCY_CHECK_PROMPT.format(
        question=question, kept_docs=kept_docs_text,
        iteration=iteration, max_iterations=max_iterations)

    try:
        out = await call_llm_async(
            session, cfg["sufficiency_service_url"], cfg["sufficiency_model_name"],
            [{"role": "user", "content": prompt}],
            temperature=cfg["temperature"], max_tokens=10240,
            max_retries=cfg["max_retries"], component="check_sufficiency",
            hop_count=hop_count, query_id=query_id)
        if not out:
            if iteration >= max_iterations:
                return {"sufficient": True, "reasoning": "Max iterations reached"}
            return {"sufficient": False, "reasoning": "LLM returned empty"}
        js = _extract_json(out)
        if not js:
            if iteration >= max_iterations:
                return {"sufficient": True, "reasoning": "Max iterations reached, no JSON"}
            return {"sufficient": False, "reasoning": "No JSON in response"}
        r = json.loads(js)
        sufficient = r.get("sufficient", False)
        reasoning = r.get("reasoning", "")
        if iteration >= max_iterations and not sufficient:
            sufficient = True
            reasoning = f"Max iterations reached. {reasoning}"
        return {"sufficient": sufficient, "reasoning": reasoning}
    except PerfReplayError:
        raise
    except Exception as e:
        if iteration >= max_iterations:
            return {"sufficient": True, "reasoning": f"Max iterations reached (error: {e})"}
        return {"sufficient": False, "reasoning": f"Error: {e}"}


async def generate_answer_async(session, cfg, question, kept_docs, hop_count, query_id):
    """Generate the final answer. Returns (answer, output_token_count), where
    output_token_count is the server's usage.completion_tokens (None if the
    answer was not generated by a real LLM call)."""
    kept_docs_text = ""
    if kept_docs:
        for i, doc in enumerate(kept_docs):
            content = doc[1] if len(doc) >= 2 else ""
            kept_docs_text += f"\n[DOC {i+1}] {content}\n"
    else:
        return "Unknown", None
    prompt = f"""Answer the question based ONLY on the provided documents.

QUESTION: {question}

DOCUMENTS:
{kept_docs_text}

INSTRUCTIONS:
- Provide a specific, concise answer
- Base your answer ONLY on facts from the documents
- If documents don't contain enough information, answer "Unknown"
- Do NOT guess or make assumptions

Answer:"""
    try:
        out, osl = await call_llm_async(
            session, cfg["sufficiency_service_url"], cfg["sufficiency_model_name"],
            [{"role": "user", "content": prompt}],
            temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
            max_retries=cfg["max_retries"], component="answer_generator",
            hop_count=hop_count, query_id=query_id, return_usage=True)
        return (out.strip() if out and out.strip() else "Unknown"), osl
    except PerfReplayError:
        raise
    except Exception:
        return "Unknown", None


# --------------------------------------------------------------------------
# Async retrieval clients (embed+FAISS server, rerank server)
# --------------------------------------------------------------------------
def _doc_to_url(metadata):
    """Same URL derivation as multi_shot_retrieval's retrieval loop."""
    url = None
    if metadata.get("original_url"):
        url = metadata["original_url"]
    elif metadata.get("source"):
        source = metadata["source"]
        if source.startswith("en.wikipedia.org_wiki_"):
            page = source.replace("en.wikipedia.org_wiki_", "").replace(".html", "")
            url = f"https://en.wikipedia.org/wiki/{page}"
    return url


async def embed_search_async(session, embed_url, query, k, query_id, iteration):
    async with server_slot("embedder"):
        with tracer.span("embed", "embedder", query_id=query_id, iteration=iteration,
                         k=k) as sp, tracer.inflight("inflight embedder"):
            async with session.post(f"{embed_url}/embed_search",
                                    json={"query": query, "k": k},
                                    timeout=aiohttp.ClientTimeout(total=60)) as r:
                r.raise_for_status()
                data = await r.json()
            if "batch_size" in data:
                sp["batch_size"] = data["batch_size"]
    return data["docs"]   # [{page_content, metadata}, ...]


async def rerank_async(session, rerank_url, query, passages, query_id, iteration):
    """Score passages, returning [[passage, score], ...] in input order.

    Two backends, selected by RERANK_API:
      colbert (default) — servers/rerank_server.py, POST /rerank
      score             — vLLM pooling server, POST /v1/score (same protocol as
                          engine/ragdb.py::rerank, which is the sequential path)
    """
    async with server_slot("reranker"):
        with tracer.span("rerank", "reranker", query_id=query_id, iteration=iteration,
                         n_passages=len(passages)) as sp, tracer.inflight("inflight reranker"):
            if _RERANK_API == "score":
                # truncate_prompt_tokens: the server is launched with
                # --max-model-len 512 and REJECTS longer input with HTTP 400,
                # while the ColBERT backend truncates to 512 silently
                # (reranker_worker.py: truncation=True, max_length=512). A
                # 768-char chunk can tokenize past 512, so without this one long
                # passage 400s and kills the whole batch. Match ColBERT instead.
                payload = {"model": _RERANK_MODEL, "text_1": query,
                           "text_2": passages,
                           "truncate_prompt_tokens": _RERANK_MAX_LEN}
                async with session.post(f"{rerank_url}/v1/score", json=payload,
                                        timeout=aiohttp.ClientTimeout(total=120)) as r:
                    r.raise_for_status()
                    body = await r.json()
                # /v1/score may return out of order; entry["index"] is authoritative.
                scores = [0.0] * len(passages)
                for entry in body["data"]:
                    scores[entry["index"]] = float(entry["score"])
                return [[p, s] for p, s in zip(passages, scores)]
            async with session.post(f"{rerank_url}/rerank",
                                    json={"query": query, "passages": passages},
                                    timeout=aiohttp.ClientTimeout(total=120)) as r:
                r.raise_for_status()
                data = await r.json()
            if "batch_size" in data:
                sp["batch_size"] = data["batch_size"]
    return data["scored"]   # [[passage, score], ...] aligned to input passage order


# --------------------------------------------------------------------------
# Per-query coroutine — mirrors multi_shot_retrieval routing
# --------------------------------------------------------------------------
class QueryJob:
    def __init__(self, query_id, question, expected_urls, ground_truth):
        self.query_id = str(query_id)
        self.question = question
        self.expected_urls = expected_urls
        self.ground_truth = ground_truth
        self.iteration = 0
        self.kept_docs = []          # (url, content, summary)
        self.query_history = []
        self.query_results = []
        self.feedback_history = []
        self.all_retrieved_urls = set()
        self.sufficient = False
        self.final_answer = ""
        # usage.completion_tokens of the final answer_generator call, reported
        # to LoadGen as the response token length (TEST09).
        self.answer_output_tokens = None
        self.num_iterations = 0
        self.t_start = None
        self.t_end = None


async def run_query(job, session, cfg, urls, params):
    """Run one query's full multi-shot loop asynchronously."""
    job.t_start = time.perf_counter()
    max_iterations = params["max_iterations"]
    max_sub_queries = params["max_sub_queries"]
    top_k_retriever = params["top_k_retriever"]
    top_k_reranking = params["top_k_reranking"]

    previous_feedback = ""

    while not job.sufficient and job.iteration < max_iterations:
        job.iteration += 1
        new_docs = []

        # ---- Step 1: rewriter (or, from iter 2+, grade->sufficiency first) ----
        if job.iteration == 1 and not job.kept_docs:
            qr = await gen_search_queries_async(
                session, cfg, job.question, [], max_sub_queries,
                None, None, None, job.iteration, job.query_id)
            sub_queries = qr.get("queries", [job.question]) or [job.question]
            current_feedback = qr.get("feedback", "Initial query decomposition")
        else:
            # grader on new docs from previous retrieve
            relevance = []
            if job._pending_new_docs:
                rr = await eval_relevance_async(
                    session, cfg, job.question, job._pending_new_docs, job.kept_docs,
                    job.iteration, job.query_id)
                relevance = rr.get("relevance", [1] * len(job._pending_new_docs))
                for i, (url, content) in enumerate(job._pending_new_docs):
                    if i < len(relevance) and relevance[i] == 1:
                        job.kept_docs.append((url, content, content[:1000]))
            # sufficiency
            if job.kept_docs:
                sr = await check_sufficiency_async(
                    session, cfg, job.question, job.kept_docs, job.iteration,
                    max_iterations, job.iteration, job.query_id)
                job.sufficient = sr.get("sufficient", False)
                sufficiency_reasoning = sr.get("reasoning", "")
            else:
                job.sufficient = False
                sufficiency_reasoning = "No relevant documents kept yet"

            if job.sufficient:
                job.final_answer, job.answer_output_tokens = await generate_answer_async(
                    session, cfg, job.question, job.kept_docs, job.iteration, job.query_id)
                current_feedback = sufficiency_reasoning
                sub_queries = []
            else:
                docs_for_gen = job.kept_docs[-12:] if len(job.kept_docs) > 12 else job.kept_docs
                qr = await gen_search_queries_async(
                    session, cfg, job.question, docs_for_gen, max_sub_queries,
                    job.query_history, job.query_results, job.feedback_history,
                    job.iteration, job.query_id)
                sub_queries = qr.get("queries", [])
                current_feedback = qr.get("feedback", "")

        if current_feedback and current_feedback.strip() and current_feedback != previous_feedback:
            job.feedback_history.append(current_feedback.strip())
        previous_feedback = current_feedback

        job._pending_new_docs = []

        if job.sufficient:
            break

        if not sub_queries:
            sub_queries = [job.question]

        # ---- Step 2: retrieve (embed+FAISS) + per-subquery rerank ----
        num_sub = len(sub_queries)
        docs_per_subquery = max(1, top_k_retriever)
        target_per_sub = max(3, top_k_retriever // num_sub)
        per_query_counts = []

        for sq in sub_queries:
            start_count = len(new_docs)
            docs = await embed_search_async(session, urls["embed"], sq,
                                            docs_per_subquery, job.query_id, job.iteration)
            if _NO_RERANK:
                # Skip reranking: keep the retriever's top docs in their original order.
                docs = docs[:target_per_sub]
            elif len(docs) > target_per_sub:
                passages = [d["page_content"] for d in docs]
                scored = await rerank_async(session, urls["rerank"], sq, passages,
                                            job.query_id, job.iteration)
                # scored[i] <-> docs[i]; sort docs by score index (dup-safe).
                order = sorted(range(len(docs)), key=lambda i: scored[i][1], reverse=True)
                docs = [docs[i] for i in order][:target_per_sub]
            for d in docs:
                url = _doc_to_url(d["metadata"])
                if url and url not in job.all_retrieved_urls:
                    job.all_retrieved_urls.add(url)
                    new_docs.append((url, d["page_content"]))
            per_query_counts.append(len(new_docs) - start_count)

        for sq, c in zip(sub_queries, per_query_counts):
            job.query_history.append(sq)
            job.query_results.append(c)

        job._pending_new_docs = new_docs

    job.num_iterations = job.iteration

    # ---- Metrics (same as baseline) ----
    retrieved_urls = [d[0] for d in job.kept_docs][:top_k_reranking]
    expected_set = set(u for u in job.expected_urls if u and u.strip())
    metrics = calculate_retrieval_metrics(list(expected_set), retrieved_urls)
    job.t_end = time.perf_counter()
    return {
        "query_index": job.query_id,
        "query": job.question,
        "ground_truth_answer": job.ground_truth,
        "expected_urls": job.expected_urls,
        "llm_answer": job.final_answer or "Unknown",
        "answer_output_tokens": job.answer_output_tokens,
        "retrieved_urls": retrieved_urls,
        "num_iterations": job.num_iterations,
        "sufficient": job.sufficient,
        "metrics": {
            "precision": metrics.get("precision@N", 0),
            "recall": metrics.get("recall@N", 0),
            "f1": metrics.get("f1@N", 0),
        },
        "wall_s": job.t_end - job.t_start,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def load_queries(dataset_path, indices):
    df = pd.read_csv(dataset_path, sep="\t")
    jobs = []
    for idx in indices:
        row = df.iloc[idx]
        urls = []
        for col in df.columns:
            if col.startswith("wikipedia_link_"):
                v = row[col]
                if pd.notna(v) and str(v).strip():
                    urls.append(str(v).strip())
        jobs.append(QueryJob(idx, row["Prompt"], urls, row["Answer"]))
    return jobs


def resolve(value, fn, *a):
    if isinstance(value, str) and value.lower() == "auto":
        return fn(*a)
    return value


async def run_batch(jobs, cfg, urls, params, server_limits=None,
                    on_done=None, session=None, perf_test_cache=None):
    """Run all jobs concurrently on the event loop.

    server_limits: {server: int} per-server host-side concurrency caps
        (LLM-120B/LLM-20B/embedder/reranker). The only host-side throttle;
        absent servers are unlimited.
    on_done(result): callback per completed query (SUT reports to LoadGen).
    perf_test_cache: PerfTestCache for perf mode. Real LLM calls still happen
        (that is the measurement); their output is replaced by the recorded
        response so the retrieval trajectory is deterministic.
    """
    # Build per-server semaphores on the running loop before any query starts.
    init_server_sems(server_limits)
    if _SERVER_SEMS:
        print(f"  per-server limits: {server_limits}")
    init_perf_cache(perf_test_cache)
    if perf_test_cache is not None:
        print("  perf replay: real LLM calls issued and timed, recorded "
              "responses returned to the pipeline")
    own_session = session is None
    if own_session:
        # enable_cleanup_closed reaps sockets the server closed on keep-alive
        # idle-timeout, so a pooled dead connection isn't handed out -> avoids
        # spurious ServerDisconnectedError (client-side race, no server error).
        connector = aiohttp.TCPConnector(limit=0, enable_cleanup_closed=True)
        # trust_env=True: honor proxy env vars (aiohttp ignores them by default)
        session = aiohttp.ClientSession(connector=connector, trust_env=True)

    # Background poller: vLLM running-batch per distinct server base. 
    # Only servers that expose /metrics are polled
    poller = None
    seen, servers = set(), []
    for label, key in (("LLM-120B", "query_service_url"),
                       ("LLM-20B", "grader_service_url")):
        u = cfg.get(key)
        if not u:
            continue
        base = u.rsplit("/v1/", 1)[0]
        if base in seen:
            continue
        seen.add(base)
        try:
            async with session.get(base.rstrip("/") + "/metrics",
                                   timeout=aiohttp.ClientTimeout(total=3)) as r:
                has_metrics = r.status < 300
        except Exception:
            has_metrics = False
        if has_metrics:
            servers.append((label, base))
        else:
            print(f"[poller] {label} ({base}) has no /metrics — batch polling disabled for it")
    if servers:
        poller = asyncio.ensure_future(tracer.poll_vllm_metrics(session, servers))

    try:
        # All jobs run concurrently; throttled only by the per-server semaphores.
        async def guarded(job):
            job._pending_new_docs = []
            try:
                r = await run_query(job, session, cfg, urls, params)
            except PerfReplayError:
                # Perf replay is all-or-nothing: a missed call invalidates the
                # measurement, so let this abort the batch.
                raise
            except Exception as e:
                # Isolate the failure to this query. Previously any exception
                # propagated out of the gather below and discarded every other
                # query's result, turning one bad call into a whole lost run.
                log.error("q%s failed (%s: %s); reporting Unknown",
                          job.query_id, type(e).__name__, e)
                # Same shape run_query returns, so on_done/LoadGen are unaffected.
                partial = [d[0] for d in job.kept_docs][:params["top_k_reranking"]]
                m = calculate_retrieval_metrics(
                    list({u for u in job.expected_urls if u and u.strip()}), partial)
                r = {
                    "query_index": job.query_id,
                    "query": job.question,
                    "ground_truth_answer": job.ground_truth,
                    "expected_urls": job.expected_urls,
                    "llm_answer": job.final_answer or "Unknown",
                    "answer_output_tokens": job.answer_output_tokens,
                    "retrieved_urls": partial,
                    "num_iterations": job.iteration,
                    "sufficient": False,
                    "metrics": {"precision": m.get("precision@N", 0),
                                "recall": m.get("recall@N", 0),
                                "f1": m.get("f1@N", 0)},
                    "wall_s": (time.perf_counter() - job.t_start) if job.t_start else 0.0,
                    "error": f"{type(e).__name__}: {e}",
                }
            if on_done is not None:
                on_done(r)
            return r

        return await asyncio.gather(*[guarded(j) for j in jobs])
    finally:
        if poller is not None:
            poller.cancel()
            try:
                await poller
            except (asyncio.CancelledError, Exception):
                pass
        if own_session:
            await session.close()


def main():
    p = argparse.ArgumentParser(description="Async multi-shot retrieval")
    p.add_argument("--dataset", required=True, help="path to frames_dataset.tsv")
    p.add_argument("--batch", type=int, default=8, help="number of queries (indices 0..batch-1)")
    p.add_argument("--indices", default=None, help="comma-separated query indices (overrides --batch)")
    p.add_argument("--server-limits", default=None,
                   help="per-server host-side concurrency caps, comma-list of "
                        "server=N (servers: LLM-120B,LLM-20B,embedder,reranker). "
                        "e.g. 'LLM-120B=256,reranker=24'. Absent servers unlimited. "
                        "This is the only host-side throttle.")
    p.add_argument("--embed-url", default="http://127.0.0.1:8100")
    p.add_argument("--rerank-url", default="http://127.0.0.1:8101")
    p.add_argument("--llm-service-url", default="http://127.0.0.1:8192/v1/chat/completions")
    p.add_argument("--query-service-url", default="http://127.0.0.1:8123/v1/chat/completions")
    p.add_argument("--sufficiency-service-url", default="http://127.0.0.1:8123/v1/chat/completions")
    p.add_argument("--llm-model", default="auto")
    p.add_argument("--query-model", default="auto")
    p.add_argument("--sufficiency-model", default="auto")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--max-sub-queries", type=int, default=3)
    p.add_argument("--top-k-retriever", type=int, default=10)
    p.add_argument("--top-k-reranking", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--max-tokens", default="auto")
    p.add_argument("--output", default="async_result.json")
    p.add_argument("--trace", default=None,
                   help="Comma-list of trace views to emit (e.g. 'batch-curve,query-timeline', "
                        "or 'all'). Empty/omitted disables tracing. See tracer.available_views().")
    p.add_argument("--trace-dir", default=".",
                   help="Directory each trace view file is written into (default: cwd).")
    args = p.parse_args()

    trace_views = tracer.enable(args.trace, args.trace_dir) if args.trace else []

    if args.indices:
        indices = [int(x) for x in args.indices.split(",")]
    else:
        indices = list(range(args.batch))

    grader_model = resolve(args.llm_model, get_model_name_from_service, args.llm_service_url)
    query_model = resolve(args.query_model, get_model_name_from_service, args.query_service_url)
    suff_model = resolve(args.sufficiency_model, get_model_name_from_service, args.sufficiency_service_url)
    max_tokens = resolve(args.max_tokens, get_max_tokens_from_service, args.llm_service_url)
    max_tokens = int(max_tokens)

    cfg = {
        "grader_service_url": args.llm_service_url, "grader_model_name": grader_model,
        "query_service_url": args.query_service_url, "query_model_name": query_model,
        "sufficiency_service_url": args.sufficiency_service_url, "sufficiency_model_name": suff_model,
        "max_tokens": max_tokens, "temperature": args.temperature, "max_retries": args.max_retries,
    }
    urls = {"embed": args.embed_url, "rerank": args.rerank_url}
    params = {
        "max_iterations": args.max_iterations, "max_sub_queries": args.max_sub_queries,
        "top_k_retriever": args.top_k_retriever, "top_k_reranking": args.top_k_reranking,
    }

    server_limits = parse_server_limits(args.server_limits)

    print("=" * 80)
    print(f"ASYNC MULTI-SHOT RETRIEVAL — {len(indices)} queries, "
          f"server_limits={server_limits or 'unlimited'}")
    print(f"  indices: {indices}")
    print(f"  models: grader={grader_model} query={query_model} suff={suff_model}")
    print("=" * 80)

    jobs = load_queries(args.dataset, indices)
    t0 = time.perf_counter()

    async def _go():
        return await run_batch(jobs, cfg, urls, params, server_limits=server_limits)
    results = asyncio.run(_go())
    wall = time.perf_counter() - t0

    results.sort(key=lambda r: int(r["query_index"]))
    print("\n" + "=" * 80)
    print("ASYNC RESULTS")
    print("=" * 80)
    for r in results:
        m = r["metrics"]
        print(f"  q{r['query_index']:>3}  it={r['num_iterations']}  "
              f"P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}  "
              f"wall={r['wall_s']:.1f}s  ans={r['llm_answer'][:50]!r}")
    mean_r = sum(r["metrics"]["recall"] for r in results) / len(results)
    mean_p = sum(r["metrics"]["precision"] for r in results) / len(results)
    mean_f = sum(r["metrics"]["f1"] for r in results) / len(results)
    print("-" * 80)
    print(f"  BATCH wall time:   {wall:.1f}s  ({wall/len(results):.1f}s/query amortized)")
    print(f"  mean P@N={mean_p:.3f}  R@N={mean_r:.3f}  F1@N={mean_f:.3f}")

    if args.trace:
        s = tracer.summary()
        print("  Component busy (from trace):")
        for name, v in sorted(s["components"].items(), key=lambda kv: -kv[1]["total_s"]):
            print(f"    {name:<14}{v['total_s']:>8.1f}s  {v['calls']:>4} calls")
        for res, t in sorted(s["resources"].items(), key=lambda kv: -kv[1]):
            print(f"    [{res}] {t:.1f}s")
        print("  Estimated batch depth (in-flight requests per server):")
        for cname, bd in s.get("batch_depth", {}).items():
            print(f"    {cname:<20} mean={bd.get('mean_over_active', 0):.1f}  peak={bd.get('peak', 0)}")
        if s.get("server_batch"):
            print("  Server-reported coalesced batch size (embed/rerank):")
            for comp, sb in s["server_batch"].items():
                print(f"    {comp:<20} mean={sb['mean_batch']:.2f}  peak={sb['peak_batch']}  "
                      f"hist={sb['histogram']}")
    print("=" * 80)

    with open(args.output, "w") as f:
        json.dump({"wall_s": wall, "server_limits": server_limits, "results": results}, f, indent=2)
    print(f"Results written to {args.output}")
    if trace_views:
        for path in tracer.export():
            print(f"Trace view written to {path}")


if __name__ == "__main__":
    main()
