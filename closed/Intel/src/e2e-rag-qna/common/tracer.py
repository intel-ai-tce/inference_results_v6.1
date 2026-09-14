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
Span tracer for the E2E RAG pipeline, with a Perfetto / Chrome Trace exporter.

Non-intrusive: monkey-patches a handful of functions at install() time. Call
tracer.enable(path) once during startup.
"""

import os
import json
import time
import threading


# --- Raw span store ---------------------------------------------------------
class _Tracer:
    """Process-wide singleton collecting raw spans."""

    def __init__(self):
        self.enabled = False
        self.installed = False
        self.views = []
        self.output_dir = None
        self._spans = []
        self._lock = threading.Lock()
        self._epoch_ns = None
        self._tid_map = {}
        self._tid_next = 0
        # Per-thread ambient context: set by the LLM hook, read by retrieval hooks.
        self._tls = threading.local()
        # {counter_name: [(ts_us, value), ...]} — estimated per-server batch depth.
        self._counters = {}
        self._counter_now = {}   # counter_name -> current value

    def enable(self, views, output_dir):
        """Turn on collection. views: list of view names to emit at export time;
        output_dir: directory each view file is written into."""
        self.enabled = True
        self.views = list(views)
        self.output_dir = output_dir
        self._epoch_ns = time.perf_counter_ns()

    # -- thread-local ambient context --
    def _ctx(self):
        tls = self._tls
        if not hasattr(tls, "query_id"):
            tls.query_id = None
            tls.iteration = None
            tls.subq = None
        return tls

    def set_query_iteration(self, query_id, iteration):
        """LLM hook: bind query/iteration; reset sub-query counter on a new hop."""
        tls = self._ctx()
        if query_id is not None:
            tls.query_id = str(query_id)
        if iteration is not None and iteration != tls.iteration:
            tls.iteration = iteration
            tls.subq = None
        elif iteration is not None:
            tls.iteration = iteration

    def next_subq(self):
        """Advance the sub-query index within the current iteration."""
        tls = self._ctx()
        tls.subq = 1 if tls.subq is None else tls.subq + 1
        return tls.subq

    def counter_add(self, name, delta):
        """Step a counter by delta, sampling a point (step-function track)."""
        if not self.enabled:
            return
        ts = (time.perf_counter_ns() - self._epoch_ns) / 1000.0
        with self._lock:
            v = self._counter_now.get(name, 0) + delta
            self._counter_now[name] = v
            self._counters.setdefault(name, []).append((ts, v))

    def counter_set(self, name, value):
        """Record an absolute counter value at the current time."""
        if not self.enabled:
            return
        ts = (time.perf_counter_ns() - self._epoch_ns) / 1000.0
        with self._lock:
            self._counter_now[name] = value
            self._counters.setdefault(name, []).append((ts, value))

    def _tid(self, ident):
        tid = self._tid_map.get(ident)
        if tid is None:
            tid = self._tid_next
            self._tid_map[ident] = tid
            self._tid_next += 1
        return tid

    def record(self, component, resource, start_ns, dur_ns, extra):
        tls = self._ctx()
        ident = threading.get_ident()
        with self._lock:
            tid = self._tid(ident)
            self._spans.append({
                "component": component,
                "resource": resource,
                "ts_us": (start_ns - self._epoch_ns) / 1000.0,
                "dur_us": dur_ns / 1000.0,
                "query_id": tls.query_id,
                "iteration": tls.iteration,
                "sub_query_idx": tls.subq,
                "thread_id": tid,
                "thread_name": threading.current_thread().name,
                "extra": extra or {},
            })

    def record_explicit(self, component, resource, start_ns, dur_ns,
                        query_id, iteration, sub_query_idx, extra):
        """Record with explicit indices (async path); exporter lanes by query_id."""
        with self._lock:
            self._spans.append({
                "component": component,
                "resource": resource,
                "ts_us": (start_ns - self._epoch_ns) / 1000.0,
                "dur_us": dur_ns / 1000.0,
                "query_id": str(query_id) if query_id is not None else None,
                "iteration": iteration,
                "sub_query_idx": sub_query_idx,
                "thread_id": None,   # lane by query_id
                "thread_name": None,
                "extra": extra or {},
            })

    @property
    def spans(self):
        with self._lock:
            return list(self._spans)


_TRACER = _Tracer()


def _emit(component, resource, start_ns, dur_ns, **extra):
    _TRACER.record(component, resource, start_ns, dur_ns, extra)


def reset_subq():
    """Clear the sub-query index (start of a new retrieval loop)."""
    _TRACER._ctx().subq = None


# --- Explicit span API (for the async orchestrator) -------------------------
# Async path passes query_id/iteration explicitly (coroutines share one thread).
import contextlib as _contextlib


@_contextlib.contextmanager
def span(component, resource, query_id=None, iteration=None,
         sub_query_idx=None, **extra):
    """Time a span; yields the mutable `extra` dict (recorded at exit). No-op if off."""
    if not _TRACER.enabled:
        yield {}
        return
    start_ns = time.perf_counter_ns()
    try:
        yield extra
    finally:
        dur_ns = time.perf_counter_ns() - start_ns
        _TRACER.record_explicit(component, resource, start_ns, dur_ns,
                                query_id, iteration, sub_query_idx, extra)


def _parse_vllm_gauge(text, metric):
    """Pull a single gauge value out of Prometheus text (sums across labels)."""
    total = 0.0
    found = False
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(metric):
            continue
        # line: 'metric{labels} value'  (guard against metric-name prefixes)
        head, _, val = line.rpartition(" ")
        if not head.startswith(metric):
            continue
        nxt = head[len(metric):len(metric) + 1]
        if nxt not in ("", "{", " "):   # e.g. metric vs metric_by_reason
            continue
        try:
            total += float(val)
            found = True
        except ValueError:
            pass
    return total if found else None


# LLM batch in progress line--independent of TRACE
_BATCH_NOW = {}


def vllm_batch_now():
    """Latest vLLM running-batch per label ({'LLM-120B': int, ...}); {} if unpolled."""
    return dict(_BATCH_NOW)


async def poll_vllm_metrics(session, servers, interval_s=1.0):
    """Sample each vLLM server's /metrics (running + waiting). Always updates the
    _BATCH_NOW snapshot (for the progress line); additionally records counter
    tracks when tracing is enabled. servers: list of (label, base_url).
    """
    while True:
        for label, base in servers:
            try:
                url = base.rstrip("/") + "/metrics"
                async with session.get(url, timeout=_aiohttp_timeout(3)) as r:
                    text = await r.text()
                run = _parse_vllm_gauge(text, "vllm:num_requests_running")
                wait = _parse_vllm_gauge(text, "vllm:num_requests_waiting")
                if run is not None:
                    _BATCH_NOW[label] = int(run)
                    if _TRACER.enabled:
                        _TRACER.counter_set(f"vllm running {label}", run)
                if wait is not None and _TRACER.enabled:
                    _TRACER.counter_set(f"vllm waiting {label}", wait)
            except Exception:
                pass  # a missed sample must never disturb the run
        try:
            import asyncio
            await asyncio.sleep(interval_s)
        except Exception:
            return


def _aiohttp_timeout(total):
    import aiohttp
    return aiohttp.ClientTimeout(total=total)


# --- Public API -------------------------------------------------------------
# Registered trace VIEWS. Each entry: name -> (filename, builder_fn). The builder
# takes the raw spans list and returns a Chrome-Trace/Perfetto dict. Views are
# selected via the TRACE env list; all render from the same collected spans.
_VIEWS = {}   # name -> (filename, fn)


def register_view(name, filename, fn):
    """Register a trace view: name (for TRACE list), output filename, builder(spans)->dict."""
    _VIEWS[name] = (filename, fn)


def available_views():
    return sorted(_VIEWS)


def _parse_views(spec):
    """Resolve a TRACE spec (list or comma-string) to concrete view names.
    'all' expands to every registered view; unknown names are dropped with a warning."""
    if spec is None:
        return []
    names = spec if isinstance(spec, (list, tuple)) else [s.strip() for s in str(spec).split(",")]
    names = [n for n in names if n]
    if "all" in names:
        return available_views()
    out = []
    for n in names:
        if n in _VIEWS:
            out.append(n)
        else:
            print(f"[tracer] unknown trace view '{n}' (available: {', '.join(available_views())})")
    return out


def enable(views, output_dir="."):
    """Turn tracing on and install hooks. views: list/comma-string of view names
    (see available_views()); output_dir: where each view file is written.
    A falsy/empty selection is a no-op (tracing stays off)."""
    selected = _parse_views(views)
    if not selected:
        return []
    _TRACER.enable(selected, output_dir)
    install()
    return selected


def is_enabled():
    return _TRACER.enabled


# Detailed batch-depth tracks (client-side inflight + server-side vLLM
# running/waiting) are collected ONLY when this view is selected — they add
# per-request counter churn and a background /metrics poller.
DETAIL_VIEW = "batch-curve-detail"


def detail_enabled():
    """True if the batch-curve-detail view is active (gates inflight + vLLM poll)."""
    return _TRACER.enabled and DETAIL_VIEW in _TRACER.views


@_contextlib.contextmanager
def inflight(counter_name):
    """Count in-flight requests on a resource (+1/-1) for the batch-curve-detail
    view. No-op unless that view is active."""
    if not detail_enabled():
        yield
        return
    _TRACER.counter_add(counter_name, +1)
    try:
        yield
    finally:
        _TRACER.counter_add(counter_name, -1)


# --- Hook installation ------------------------------------------------------
_COMPONENT_LABEL = {
    "generate_search_queries": "rewriter",
    "evaluate_document_relevance": "grader",
    "check_sufficiency": "sufficiency",
    "answer_generator": "answer",
}


def _resource_for_model(model_name):
    mn = (model_name or "").lower()
    if "120b" in mn:      # check 120b first ("120b" contains "20b")
        return "LLM-120B"
    if "20b" in mn:
        return "LLM-20B"
    return "LLM"


def install():
    """Monkey-patch the trace points. Idempotent; safe to call multiple times."""
    if _TRACER.installed:
        return
    try:
        from rag import multi_shot_retrieval as msr
        from engine.vectordb import VectorDB
        from engine.ragdb import RagDB
        from langchain_community.vectorstores import FAISS
    except Exception as e:  # pragma: no cover - import ordering safety
        print(f"[tracer] deferring install (imports not ready: {e})")
        return

    # 1) LLM calls. Extract metadata from the call's own kwargs.
    _orig_llm = msr.call_chat_completions

    def _llm_hook(*args, **kwargs):
        component = kwargs.get("component", "unknown")
        hop = kwargs.get("hop_count")
        qid = kwargs.get("query_id")
        model = kwargs.get("model_name")
        if model is None and len(args) >= 2:
            model = args[1]  # positional service_url, model_name
        _TRACER.set_query_iteration(qid, hop)
        label = _COMPONENT_LABEL.get(component, component)
        resource = _resource_for_model(model)
        t0 = time.perf_counter_ns()
        try:
            return _orig_llm(*args, **kwargs)
        finally:
            _emit(label, resource, t0, time.perf_counter_ns() - t0,
                  model=model, max_tokens=kwargs.get("max_tokens"),
                  raw_component=component)

    _llm_hook.__wrapped__ = _orig_llm
    msr.call_chat_completions = _llm_hook

    # 2) Embedding (first op of each retrieval -> advance sub-query counter).
    _orig_embed = VectorDB.embed_query

    def _embed_hook(self, query, *a, **kw):
        _TRACER.next_subq()
        t0 = time.perf_counter_ns()
        try:
            return _orig_embed(self, query, *a, **kw)
        finally:
            _emit("embed", "embedder", t0, time.perf_counter_ns() - t0)

    _embed_hook.__wrapped__ = _orig_embed
    VectorDB.embed_query = _embed_hook

    # 3) FAISS vector search.
    _orig_search = FAISS.similarity_search_by_vector

    def _search_hook(self, embedding, k=4, *a, **kw):
        t0 = time.perf_counter_ns()
        try:
            return _orig_search(self, embedding, k, *a, **kw)
        finally:
            _emit("retrieval", "FAISS", t0, time.perf_counter_ns() - t0, k=k)

    _search_hook.__wrapped__ = _orig_search
    FAISS.similarity_search_by_vector = _search_hook

    # 4) Reranking (parent side; blocks on the out-of-process queue).
    _orig_rerank = RagDB.rerank

    def _rerank_hook(self, query, passages, *a, **kw):
        t0 = time.perf_counter_ns()
        try:
            return _orig_rerank(self, query, passages, *a, **kw)
        finally:
            _emit("rerank", "reranker", t0, time.perf_counter_ns() - t0,
                  n_passages=len(passages))

    _rerank_hook.__wrapped__ = _orig_rerank
    RagDB.rerank = _rerank_hook

    _TRACER.installed = True
    print("[tracer] hooks installed (LLM, embed, FAISS, rerank)")


# --- Aggregation + export ---------------------------------------------------
def summary():
    """Per-component and per-resource aggregate timing (seconds)."""
    from collections import defaultdict
    comp = defaultdict(lambda: [0.0, 0])
    res = defaultdict(float)
    for s in _TRACER.spans:
        comp[s["component"]][0] += s["dur_us"] / 1e6
        comp[s["component"]][1] += 1
        res[s["resource"]] += s["dur_us"] / 1e6
    # Estimated batch depth per counter: time-weighted mean + peak of in-flight.
    batch = {}
    for cname, samples in _TRACER._counters.items():
        if len(samples) < 2:
            batch[cname] = {"mean": 0.0, "peak": 0}
            continue
        peak = max(v for _, v in samples)
        area = active = 0.0  # time-weighted mean over active (value>0) intervals
        for (t0, v0), (t1, _v1) in zip(samples, samples[1:]):
            area += v0 * (t1 - t0)
            if v0 > 0:
                active += (t1 - t0)
        batch[cname] = {
            "mean_over_active": (area / active) if active else 0.0,
            "peak": peak,
        }
    # Server-reported coalesced batch size (embed/rerank), from span extra.
    from collections import Counter
    srv_batch = {}
    hist = defaultdict(Counter)
    for s in _TRACER.spans:
        bs = s.get("extra", {}).get("batch_size")
        if bs is not None:
            hist[s["component"]][bs] += 1
    for comp_name, c in hist.items():
        n = sum(c.values())
        weighted = sum(k * v for k, v in c.items())
        srv_batch[comp_name] = {
            "calls": n,
            "mean_batch": (weighted / n) if n else 0.0,
            "peak_batch": max(c) if c else 0,
            "histogram": dict(sorted(c.items())),
        }
    return {
        "components": {k: {"total_s": v[0], "calls": v[1]} for k, v in comp.items()},
        "resources": dict(res),
        "batch_depth": batch,
        "server_batch": srv_batch,
    }


def _write_trace(trace, path):
    """Write a Chrome-Trace dict to path (.gz -> gzipped)."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = json.dumps(trace).encode("utf-8")
    if str(path).endswith(".gz"):
        import gzip
        with gzip.open(path, "wb") as f:
            f.write(data)
    else:
        with open(path, "wb") as f:
            f.write(data)
    return path


def _run_label(output_dir):
    """Derive a run label from the output dir so downloaded trace files are
    self-identifying. Trace files live in <run>/results, so the label is the
    parent dir name (e.g. .../perf-idle-check/results -> 'perf-idle-check');
    falls back to the dir's own name."""
    import os
    d = os.path.abspath(output_dir).rstrip("/")
    parent = os.path.basename(os.path.dirname(d))
    base = os.path.basename(d)
    label = parent if base in ("results", "") and parent else base
    return label or "trace"


def export():
    """Render every selected view from the collected spans into output_dir.
    Each file is prefixed with the run label (from output_dir) so downloaded
    trace files are identifiable. Returns list of written file paths."""
    if not _TRACER.enabled:
        return []
    import os
    spans = _TRACER.spans
    label = _run_label(_TRACER.output_dir)
    written = []
    for name in _TRACER.views:
        filename, fn = _VIEWS[name]
        try:
            trace = fn(spans)
        except Exception as e:  # a broken view must not sink the others / the run
            print(f"[tracer] view '{name}' failed to render: {e}")
            continue
        path = os.path.join(_TRACER.output_dir, f"{label}_{filename}")
        _write_trace(trace, path)
        written.append(path)
    return written


# --- Built-in views ---------------------------------------------------------
def _view_query_timeline(spans):
    """Per-query spans across resources + flow arrows (lane-packed Perfetto view)."""
    return to_perfetto(spans, group_by="resource", flows=True)


# Pipeline order for batch-curve tracks: rewriter/suff/answer (120B) -> embed ->
# rerank -> grade (20B). Resources not listed sort after, alphabetically.
_RESOURCE_ORDER = ("LLM-120B", "embedder", "reranker", "LLM-20B")


def _resource_sort_key(res):
    try:
        return (0, _RESOURCE_ORDER.index(res))
    except ValueError:
        return (1, res)


def _view_batch_curve(spans):
    """Per-resource curves swept from span start(+1)/end(-1) edges. Two tracks
    per resource: 'concurrency <res>' (in-flight now) and 'cumulative <res>'
    (running total of calls started). Tracks are ordered by pipeline stage
    (120B -> embedder -> reranker -> 20B). Sampled vLLM running/waiting counters
    are folded in after."""
    import collections
    edges = collections.defaultdict(list)   # resource -> [(ts_us, +/-1)]
    starts = collections.defaultdict(list)   # resource -> [ts_us of each call start]
    q_end = {}                               # query_id -> latest span end (ts_us)
    for e in spans:
        res = e.get("resource") or e.get("args", {}).get("resource", "?")
        ts = e.get("ts_us")
        dur = e.get("dur_us", 0)
        if ts is None:   # raw span dict uses ts_us/dur_us
            continue
        edges[res].append((ts, +1))
        edges[res].append((ts + dur, -1))
        starts[res].append(ts)
        qid = e.get("query_id")
        if qid is not None:
            end = ts + dur
            if end > q_end.get(qid, -1):
                q_end[qid] = end   # a query "finishes" when its last span ends

    events = []
    pid_names = {}
    # Deterministic pid blocks so Perfetto orders tracks by pipeline stage:
    # each resource gets a concurrency pid and a cumulative pid, grouped together.
    pid = 300

    # Finished-queries curve: cumulative count of queries whose last span has ended
    # (completion progress over time — the throughput ramp). Put it first (pid 300).
    if q_end:
        pid_names[pid] = "finished queries"
        done = 0
        for end_ts in sorted(q_end.values()):
            done += 1
            events.append({"ph": "C", "name": "finished queries", "pid": pid, "tid": 0,
                           "ts": round(end_ts, 3), "args": {"finished": done}})
        pid += 1
    for res in sorted(edges, key=_resource_sort_key):
        # concurrency (in-flight) track
        cname = f"concurrency {res}"
        pid_names[pid] = cname
        cur = 0
        for ts, delta in sorted(edges[res]):
            cur += delta
            events.append({"ph": "C", "name": cname, "pid": pid, "tid": 0,
                           "ts": round(ts, 3), "args": {res: cur}})
        pid += 1
        # cumulative-calls track (monotonic count of calls started)
        kname = f"cumulative {res}"
        pid_names[pid] = kname
        tot = 0
        for ts in sorted(starts[res]):
            tot += 1
            events.append({"ph": "C", "name": kname, "pid": pid, "tid": 0,
                           "ts": round(ts, 3), "args": {res: tot}})
        pid += 1

    # Detail tracks (only collected when batch-curve-detail is active): live
    # client-side inflight + sampled vLLM running/waiting counters.
    for cname, samples in _TRACER._counters.items():
        if not (cname.startswith("vllm ") or cname.startswith("inflight ")):
            continue
        pid_names[pid] = cname
        for ts, val in samples:
            events.append({"ph": "C", "name": cname, "pid": pid, "tid": 0,
                           "ts": round(ts, 3), "args": {cname: val}})
        pid += 1

    for p, key in pid_names.items():
        events.append({"ph": "M", "name": "process_name", "pid": p, "tid": 0, "args": {"name": key}})
        events.append({"ph": "M", "name": "process_sort_index", "pid": p, "tid": 0, "args": {"sort_index": p}})
    return {"traceEvents": events, "displayTimeUnit": "ms"}


register_view("query-timeline", "query_timeline.json.gz", _view_query_timeline)
register_view("batch-curve", "batch_curve.json.gz", _view_batch_curve)
# Same curves + live inflight/vLLM running/waiting tracks (see DETAIL_VIEW).
register_view(DETAIL_VIEW, "batch_curve_detail.json.gz", _view_batch_curve)


# Resource lanes get stable pids so track ordering is deterministic.
_RESOURCE_PID = {
    "LLM-120B": 10,
    "LLM-20B": 11,
    "embedder": 20,
    "FAISS": 21,
    "reranker": 22,
}
_UNKNOWN_PID_BASE = 90


def _pid_for(group_by, span, dynamic_pids):
    if group_by == "resource":
        key = span["resource"]
        if key in _RESOURCE_PID:
            return _RESOURCE_PID[key], key
    elif group_by == "query":
        key = f"query {span['query_id']}"
    elif group_by == "component":
        key = span["component"]
    else:
        key = span.get(group_by) or "other"

    pid = dynamic_pids.get(key)
    if pid is None:
        pid = _UNKNOWN_PID_BASE + len(dynamic_pids)
        dynamic_pids[key] = pid
    return pid, key


def _assign_lanes(spans):
    """Lane-pack spans within each pid so row count = peak concurrency, not query
    count (greedy interval-graph coloring). Sync spans keep their thread lane.
    """
    # Assign by start time so "lowest free lane" is meaningful.
    ordered = sorted(range(len(spans)), key=lambda i: spans[i]["ts_us"])
    lane_free_at = {}          # (pid, lane) -> end_ts of last slice in that lane
    pid_lane_count = {}        # pid -> how many lanes allocated so far
    assign = {}
    for i in ordered:
        s = spans[i]
        pid = s["_pid"]
        if s["thread_id"] is not None:
            assign[i] = ("thread", s["thread_id"])
            continue
        start, end = s["ts_us"], s["ts_us"] + s["dur_us"]
        chosen = None
        for lane in range(pid_lane_count.get(pid, 0)):
            if lane_free_at.get((pid, lane), -1) <= start:
                chosen = lane
                break
        if chosen is None:                       # need a new lane
            chosen = pid_lane_count.get(pid, 0)
            pid_lane_count[pid] = chosen + 1
        lane_free_at[(pid, chosen)] = end
        assign[i] = ("lane", chosen)
    return assign


def _flow_events(placed):
    """Emit flow-event arrows chaining each query's stages in time order.

    placed: list of (query_id, ts_us, dur_us, pid, tid).
    """
    from collections import defaultdict
    by_q = defaultdict(list)
    for qid, ts, dur, pid, tid in placed:
        by_q[qid].append((ts, dur, pid, tid))

    ev = []
    flow_id = 0
    for qid, segs in by_q.items():
        segs.sort(key=lambda x: x[0])
        for (ts0, dur0, pid0, tid0), (ts1, dur1, pid1, tid1) in zip(segs, segs[1:]):
            flow_id += 1
            cat = "flow"
            name = f"q{qid}"
            ev.append({   # start: end of source slice
                "ph": "s", "id": flow_id, "cat": cat, "name": name,
                "pid": pid0, "tid": tid0,
                "ts": round(ts0 + dur0, 3),
            })
            ev.append({   # finish: start of destination slice
                "ph": "f", "bp": "e", "id": flow_id, "cat": cat, "name": name,
                "pid": pid1, "tid": tid1,
                "ts": round(ts1, 3),
            })
    return ev


def to_perfetto(spans, group_by="resource", flows=True):
    """Convert raw spans to Chrome Trace Format events.

    group_by: track axis (pid) — "resource" / "query" / "component". Rows within
    a pid are lane-packed; flows=True draws arrows following each query's stages.
    """
    events = []
    dynamic_pids = {}
    pid_names = {}
    seen_threads = set()
    placed = []   # (query_id, ts_us, dur_us, pid, tid) per slice, for flow events

    # Pre-resolve pid for every span, then lane-pack.
    for s in spans:
        pid, pid_key = _pid_for(group_by, s, dynamic_pids)
        s["_pid"] = pid
        pid_names[pid] = pid_key
    lane_assign = _assign_lanes(spans)

    for i, s in enumerate(spans):
        pid = s["_pid"]
        kind, val = lane_assign[i]
        if kind == "thread":
            tid = val
            tname = f"{s['thread_name']} (t{tid})"
        else:
            tid = val
            tname = f"lane {tid}"

        if (pid, tid) not in seen_threads:
            seen_threads.add((pid, tid))
            events.append({
                "ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                "args": {"name": tname},
            })

        idx = []
        if s["query_id"] is not None:
            idx.append(f"q{s['query_id']}")
        if s["iteration"] is not None:
            idx.append(f"it{s['iteration']}")
        if s["sub_query_idx"] is not None:
            idx.append(f"sq{s['sub_query_idx']}")
        name = s["component"] + (" " + "/".join(idx) if idx else "")

        args = {
            "component": s["component"],
            "resource": s["resource"],
            "query_id": s["query_id"],
            "iteration": s["iteration"],
            "sub_query_idx": s["sub_query_idx"],
            "dur_ms": round(s["dur_us"] / 1000.0, 3),
        }
        args.update(s["extra"])

        events.append({
            "ph": "X",
            "name": name,
            "cat": s["component"],
            "pid": pid,
            "tid": tid,
            "ts": round(s["ts_us"], 3),
            "dur": round(s["dur_us"], 3),
            "args": args,
        })
        if s["query_id"] is not None:
            placed.append((s["query_id"], s["ts_us"], s["dur_us"], pid, tid))

    if flows:
        events.extend(_flow_events(placed))

    # Counter tracks: each gets its own pid so it renders as a dedicated track.
    counter_pid = 50
    for cname, samples in _TRACER._counters.items():
        pid_names[counter_pid] = cname
        for ts, val in samples:
            events.append({
                "ph": "C", "name": cname, "pid": counter_pid, "tid": 0,
                "ts": round(ts, 3), "args": {cname: val},
            })
        counter_pid += 1

    for pid, key in pid_names.items():
        events.append({
            "ph": "M", "name": "process_name", "pid": pid, "tid": 0,
            "args": {"name": key},
        })
        events.append({
            "ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0,
            "args": {"sort_index": pid},
        })

    return {"traceEvents": events, "displayTimeUnit": "ms"}
