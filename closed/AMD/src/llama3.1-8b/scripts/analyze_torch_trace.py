#!/usr/bin/env python3
"""Analyze a vLLM/torch-profiler chrome trace for GPU idle gaps.

vLLM (with VLLM_TORCH_PROFILER_DIR set) writes one chrome trace per worker,
typically named like ``<host>_<pid>.<ts>.pt.trace.json.gz``. This script finds
where the GPU goes idle during the captured window and what host-side activity
(CPU ops / cuda launch / sync) coincides with those gaps, so we can pinpoint the
mechanism behind the fluctuating utilization (cudagraph-replay misses, Triton
JIT, host scheduling bubbles, syncs, etc.).

Usage:
    python3 scripts/analyze_torch_trace.py traces/<file>.pt.trace.json.gz \
        [--gap-us 200] [--top 25] [--window-ms 50]
"""
import argparse
import gzip
import json
from collections import defaultdict, Counter


# Categories that represent actual work executing ON the GPU.
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
# Host-side categories (what the CPU/runtime is doing).
HOST_CATS = {"cpu_op", "user_annotation", "cuda_runtime", "cuda_driver", "python_function"}


def load_trace(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--gap-us", type=float, default=200.0,
                    help="min GPU idle gap (us) to count as a stall")
    ap.add_argument("--top", type=int, default=25, help="how many largest gaps to list")
    ap.add_argument("--window-ms", type=float, default=50.0,
                    help="time-bucket size for the busy/idle timeline")
    args = ap.parse_args()

    events = load_trace(args.trace)
    print(f"== Loaded {len(events)} trace events from {args.trace} ==\n")

    # Bucket complete events ('X') by whether they are GPU work or host work.
    gpu_ev = []   # (ts_us, dur_us, name)
    host_ev = []  # (ts_us, dur_us, name, cat)
    cat_counter = Counter()
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        cat_counter[cat] += 1
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None or dur is None:
            continue
        if cat in GPU_CATS:
            gpu_ev.append((float(ts), float(dur), e.get("name", "")))
        elif cat in HOST_CATS:
            host_ev.append((float(ts), float(dur), e.get("name", ""), cat))

    print("== Event categories (ph=X) ==")
    for c, n in cat_counter.most_common():
        print(f"  {c or '(none)':<22} {n}")
    print()

    if not gpu_ev:
        print("No GPU-side events (kernel/gpu_memcpy/...) found. "
              "Categories present above — adjust GPU_CATS if needed.")
        return

    gpu_ev.sort()
    t0 = gpu_ev[0][0]
    tN = max(ts + dur for ts, dur, _ in gpu_ev)
    span_us = tN - t0
    print(f"== GPU trace span: {span_us/1000.0:.1f} ms, {len(gpu_ev)} GPU ops ==")

    # Merge overlapping GPU intervals -> true busy time + gaps.
    merged = []
    cs, ce = gpu_ev[0][0], gpu_ev[0][0] + gpu_ev[0][1]
    for ts, dur, _ in gpu_ev[1:]:
        if ts <= ce:
            ce = max(ce, ts + dur)
        else:
            merged.append((cs, ce))
            cs, ce = ts, ts + dur
    merged.append((cs, ce))

    busy = sum(e - s for s, e in merged)
    util = 100.0 * busy / span_us if span_us > 0 else 0.0
    print(f"== GPU busy {busy/1000.0:.1f} ms  util {util:.1f}%  "
          f"({len(merged)} busy intervals) ==\n")

    gap_us = args.gap_us
    gaps = []  # (dur_us, gap_start, gap_end, prev_name, next_name)
    ev_by_start = gpu_ev
    for i in range(1, len(merged)):
        g = merged[i][0] - merged[i - 1][1]
        if g >= gap_us:
            gs, ge = merged[i - 1][1], merged[i][0]
            prev_nm = next((nm for ts, dur, nm in reversed(ev_by_start)
                            if ts + dur <= gs + 1), "")
            next_nm = next((nm for ts, dur, nm in ev_by_start if ts >= ge - 1), "")
            gaps.append((g, gs, ge, prev_nm, next_nm))

    idle = sum(g for g, *_ in gaps)
    print(f"== {len(gaps)} GPU idle gaps > {gap_us:.0f}us, "
          f"total idle {idle/1000.0:.1f} ms ({100.0*idle/span_us:.1f}% of span) ==")
    gaps.sort(reverse=True)
    print(f"\n== Top {args.top} idle gaps ==")
    print(f"{'dur_us':>10} {'t_ms':>10}  prev_gpu_op -> next_gpu_op")
    for g, gs, ge, pnm, nnm in gaps[:args.top]:
        print(f"{g:>10.1f} {(gs-t0)/1000.0:>10.1f}  {pnm[:38]!r} -> {nnm[:38]!r}")
    print()

    # What host-side ops overlap the biggest gaps?
    if host_ev:
        host_ev.sort()
        print("== Host-side ops overlapping the top idle gaps (cause hints) ==")
        for g, gs, ge, pnm, nnm in gaps[:min(args.top, 12)]:
            overlap = [(nm, cat, min(ge, ts + dur) - max(gs, ts))
                       for ts, dur, nm, cat in host_ev
                       if ts < ge and ts + dur > gs]
            agg = defaultdict(lambda: [0, 0.0])
            for nm, cat, ov in overlap:
                key = f"[{cat}] {nm[:42]}"
                agg[key][0] += 1
                agg[key][1] += max(0.0, ov)
            head = f"gap {g:.0f}us @t={(gs-t0)/1000.0:.1f}ms:"
            if not agg:
                print(f"  {head} (no host op overlaps -> pure host bubble / "
                      f"waiting for next launch)")
            else:
                print(f"  {head}")
                for key, (cnt, ov) in sorted(agg.items(),
                                             key=lambda kv: kv[1][1], reverse=True)[:6]:
                    print(f"      {key:<50} x{cnt:<4} ov={ov:>9.1f}us")
        print()

    # Busy/idle timeline.
    nb = max(1, int(span_us / (args.window_ms * 1000.0)) + 1)
    win = args.window_ms * 1000.0
    occ = [0.0] * nb
    for s, e in merged:
        b0 = int((s - t0) // win)
        b1 = int((e - t0) // win)
        for b in range(b0, min(b1 + 1, nb)):
            ws = t0 + b * win
            we = ws + win
            occ[b] += max(0.0, min(e, we) - max(s, ws))
    line = "".join("." if occ[b] / win > 0.5 else "#" for b in range(nb))
    print(f"== Busy/idle timeline ({args.window_ms:.0f}ms buckets; "
          f"'.' busy >50%, '#' mostly idle) ==")
    for i in range(0, len(line), 100):
        print(f"  {i*args.window_ms/1000.0:7.1f}s {line[i:i+100]}")


if __name__ == "__main__":
    main()
