#!/usr/bin/env python3
"""Analyze an RPD (rocmProfileData) SQLite trace for per-GPU idle gaps.

Goal: find WHERE the GPUs go idle during the interactive run and WHAT host-side
HIP API activity coincides with those gaps (sync, malloc/free, memcpy, module
load, kernel-launch stalls, etc.), per GPU, so we can pinpoint the mechanism
behind the rotating-idle utilization.

Usage:
    python3 scripts/analyze_rpd_gaps.py traces/<file>.rpd [--gap-us 200] [--top 25]
"""
import argparse
import sqlite3
from collections import defaultdict


def table_exists(cur, name):
    cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?", (name,))
    return cur.fetchone() is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rpd")
    ap.add_argument("--gap-us", type=float, default=200.0,
                    help="min GPU idle gap (microseconds) to count as a stall")
    ap.add_argument("--top", type=int, default=25, help="how many largest gaps to list")
    ap.add_argument("--window-ms", type=float, default=100.0,
                    help="time-bucket size for the rotating-idle timeline")
    args = ap.parse_args()

    con = sqlite3.connect(args.rpd)
    cur = con.cursor()

    # --- discover schema ----------------------------------------------------
    print(f"== Tables/views ==")
    cur.execute("SELECT name,type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")
    for n, t in cur.fetchall():
        print(f"  {t:5} {n}")
    print()

    if not table_exists(cur, "rocpd_op"):
        print("No rocpd_op table; is this a valid RPD trace?")
        return

    # --- pull GPU ops -------------------------------------------------------
    # rocpd_op: gpuId, start, end (ns). opType/description via rocpd_string.
    cur.execute("PRAGMA table_info(rocpd_op)")
    op_cols = [r[1] for r in cur.fetchall()]
    name_join = ""
    name_sel = "'' AS opname"
    if "description_id" in op_cols:
        name_sel = "COALESCE(s.string,'') AS opname"
        name_join = "LEFT JOIN rocpd_string s ON op.description_id = s.id"

    cur.execute(f"""
        SELECT op.gpuId, op.start, op.end, {name_sel}
        FROM rocpd_op op {name_join}
        ORDER BY op.gpuId, op.start
    """)
    rows = cur.fetchall()
    if not rows:
        print("rocpd_op empty.")
        return

    t0 = min(r[1] for r in rows)
    tN = max(r[2] for r in rows)
    span_ms = (tN - t0) / 1e6
    print(f"== Trace span: {span_ms:.1f} ms across "
          f"{len({r[0] for r in rows})} GPUs, {len(rows)} GPU ops ==\n")

    by_gpu = defaultdict(list)
    for gpu, s, e, nm in rows:
        by_gpu[gpu].append((s, e, nm))

    gap_ns = args.gap_us * 1000.0
    all_gaps = []  # (dur_ns, gpu, gap_start, gap_end, prev_op, next_op)

    print("== Per-GPU utilization (within trace span) ==")
    print(f"{'gpu':>4} {'busy_ms':>10} {'util%':>7} {'ops':>8} "
          f"{'gaps>thr':>9} {'idle_in_gaps_ms':>15} {'max_gap_us':>11}")
    for gpu in sorted(by_gpu):
        ev = sorted(by_gpu[gpu])
        # merge overlapping ops to compute true busy time
        busy = 0
        cur_s, cur_e = ev[0][0], ev[0][1]
        merged = []
        for s, e, nm in ev[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        for s, e in merged:
            busy += (e - s)

        # gaps between consecutive (merged) busy intervals
        gpu_gaps = []
        for i in range(1, len(merged)):
            g = merged[i][0] - merged[i - 1][1]
            if g >= gap_ns:
                gpu_gaps.append((g, merged[i - 1][1], merged[i][0]))
        idle_in_gaps = sum(g for g, _, _ in gpu_gaps)
        maxgap = max((g for g, _, _ in gpu_gaps), default=0)
        # attribute prev/next op names
        for g, gs, ge in gpu_gaps:
            prev_nm = next((nm for s, e, nm in reversed(ev) if e <= gs), "")
            next_nm = next((nm for s, e, nm in ev if s >= ge), "")
            all_gaps.append((g, gpu, gs, ge, prev_nm, next_nm))

        util = 100.0 * busy / (tN - t0) if tN > t0 else 0
        print(f"{gpu:>4} {busy/1e6:>10.1f} {util:>7.1f} {len(ev):>8} "
              f"{len(gpu_gaps):>9} {idle_in_gaps/1e6:>15.1f} {maxgap/1e3:>11.1f}")
    print()

    # --- largest gaps and what brackets them --------------------------------
    all_gaps.sort(reverse=True)
    print(f"== Top {args.top} GPU idle gaps (>{args.gap_us}us) ==")
    print(f"{'dur_us':>10} {'gpu':>4} {'t_start_ms':>11}  prev_op -> next_op")
    for g, gpu, gs, ge, pnm, nnm in all_gaps[:args.top]:
        print(f"{g/1e3:>10.1f} {gpu:>4} {(gs-t0)/1e6:>11.1f}  "
              f"{pnm[:40]!r} -> {nnm[:40]!r}")
    print()

    # --- HIP API activity during the gaps (host-side cause) -----------------
    if table_exists(cur, "rocpd_api"):
        cur.execute("PRAGMA table_info(rocpd_api)")
        api_cols = [r[1] for r in cur.fetchall()]
        if "apiName_id" in api_cols:
            print("== HIP API calls overlapping the top idle gaps (host-side cause) ==")
            for g, gpu, gs, ge, pnm, nnm in all_gaps[:min(args.top, 12)]:
                cur.execute("""
                    SELECT s.string, COUNT(*), SUM(a.end-a.start)/1000.0
                    FROM rocpd_api a JOIN rocpd_string s ON a.apiName_id = s.id
                    WHERE a.start < ? AND a.end > ?
                    GROUP BY s.string ORDER BY 3 DESC LIMIT 6
                """, (ge, gs))
                apis = cur.fetchall()
                head = f"gap {g/1e3:.0f}us @gpu{gpu} t={(gs-t0)/1e6:.1f}ms:"
                if not apis:
                    print(f"  {head} (no API call spans the whole gap)")
                else:
                    print(f"  {head}")
                    for nm, cnt, us in apis:
                        print(f"      {nm[:45]:<45} x{cnt:<4} {us:>10.1f}us")
            print()

    # --- rotating-idle timeline: which GPUs are idle per time bucket ---------
    print(f"== Rotating-idle timeline ({args.window_ms:.0f}ms buckets; "
          f"'.' busy, '#' mostly idle) ==")
    nb = max(1, int(span_ms / args.window_ms) + 1)
    win_ns = args.window_ms * 1e6
    busy_per = {gpu: [0.0] * nb for gpu in by_gpu}
    for gpu in by_gpu:
        for s, e, nm in by_gpu[gpu]:
            b0 = int((s - t0) // win_ns)
            b1 = int((e - t0) // win_ns)
            for b in range(b0, min(b1 + 1, nb)):
                ws = t0 + b * win_ns
                we = ws + win_ns
                busy_per[gpu][b] += max(0, min(e, we) - max(s, ws))
    for gpu in sorted(busy_per):
        line = "".join("." if busy_per[gpu][b] / win_ns > 0.5 else "#"
                       for b in range(nb))
        print(f"  gpu{gpu}: {line}")
    print("\n  (vertical stripes of '#' across rows = synchronized stalls; "
          "diagonal = rotating idle)")

    con.close()


if __name__ == "__main__":
    main()
