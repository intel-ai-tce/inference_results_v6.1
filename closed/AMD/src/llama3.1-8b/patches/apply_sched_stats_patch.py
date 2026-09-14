#!/usr/bin/env python3
"""Idempotent, READ-ONLY instrumentation of the vLLM v1 scheduler.

When VLLM_SCHED_STATS=1, records per-step scheduler composition so we can see
*why* TTFT tails while TPOT stays flat: the distribution of prefill tokens per
step (mega-steps?), the waiting-queue depth, and how many new prefills are
admitted per step. Purely observational -- it never changes scheduling, so it is
safe to leave applied (default off -> byte-for-byte stock behavior).

Because the probe harness kills engines with SIGKILL (atexit won't run), stats
are flushed to /tmp/sched_stats_<pid>.txt every VLLM_SCHED_STATS_EVERY steps.

Buckets (prefill tokens/step): 0 | 1..2k | 2k..4k | 4k..8k | 8k..16k |
16k..32k | 32k..64k | 64k+   (edges chosen around the 73728 token budget).
"""
import ast
import os
import shutil
import sys

MARKER = "MLPERF: sched-stats instrumentation"


def patch(path: str) -> int:
    src = open(path).read()
    if MARKER in src:
        print("[apply_sched_stats_patch] already applied -> no-op.")
        return 0

    # --- 1) init fields ---
    init_anchor = "        self.running: list[Request] = []\n"
    if init_anchor not in src:
        sys.stderr.write("[apply_sched_stats_patch] ERROR: init anchor not found.\n")
        return 2
    init_inject = init_anchor + (
        "        # --- {marker} (env-driven, default off) ---\n"
        "        import os as _os_ss\n"
        "        self._ss_on = _os_ss.getenv('VLLM_SCHED_STATS', '0') == '1'\n"
        "        self._ss_every = int(_os_ss.getenv('VLLM_SCHED_STATS_EVERY', '2000'))\n"
        "        self._ss_path = '/tmp/sched_stats_%d.txt' % _os_ss.getpid()\n"
        "        self._ss_edges = [0, 2048, 4096, 8192, 16384, 32768, 65536]\n"
        "        self._ss_pf_hist = [0] * (len(self._ss_edges) + 1)\n"
        "        self._ss_wedges = [0, 1, 4, 16, 64, 256, 1024]\n"
        "        self._ss_wait_hist = [0] * (len(self._ss_wedges) + 1)\n"
        "        self._ss_steps = 0\n"
        "        self._ss_busy_steps = 0\n"
        "        self._ss_pf_sum = 0\n"
        "        self._ss_pf_max = 0\n"
        "        self._ss_wait_sum = 0\n"
        "        self._ss_wait_max = 0\n"
        "        self._ss_new_max = 0\n"
        "        self._ss_new_sum = 0\n"
    ).format(marker=MARKER)
    src = src.replace(init_anchor, init_inject, 1)

    # --- 2) per-step record + periodic flush ---
    stats_anchor = (
        "        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n"
        "        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens\n"
    )
    if stats_anchor not in src:
        sys.stderr.write("[apply_sched_stats_patch] ERROR: stats anchor not found.\n")
        return 2
    stats_inject = stats_anchor + (
        "        # --- {marker}: observe per-step composition (no scheduling change) ---\n"
        "        if self._ss_on:\n"
        "            import bisect as _bs_ss\n"
        "            _ss_dec = len(scheduled_running_reqs)\n"
        "            _ss_new = len(scheduled_new_reqs) + len(scheduled_resumed_reqs)\n"
        "            _ss_pf = total_num_scheduled_tokens - _ss_dec\n"
        "            if _ss_pf < 0:\n"
        "                _ss_pf = 0\n"
        "            _ss_w = len(self.waiting)\n"
        "            self._ss_steps += 1\n"
        "            if total_num_scheduled_tokens > 0:\n"
        "                self._ss_busy_steps += 1\n"
        "            self._ss_pf_sum += _ss_pf\n"
        "            self._ss_new_sum += _ss_new\n"
        "            self._ss_wait_sum += _ss_w\n"
        "            if _ss_pf > self._ss_pf_max:\n"
        "                self._ss_pf_max = _ss_pf\n"
        "            if _ss_w > self._ss_wait_max:\n"
        "                self._ss_wait_max = _ss_w\n"
        "            if _ss_new > self._ss_new_max:\n"
        "                self._ss_new_max = _ss_new\n"
        "            self._ss_pf_hist[_bs_ss.bisect_right(self._ss_edges, _ss_pf)] += 1\n"
        "            self._ss_wait_hist[_bs_ss.bisect_right(self._ss_wedges, _ss_w)] += 1\n"
        "            if self._ss_steps % self._ss_every == 0:\n"
        "                try:\n"
        "                    _n = max(1, self._ss_steps)\n"
        "                    with open(self._ss_path, 'w') as _f_ss:\n"
        "                        _f_ss.write('steps=%d busy=%d\\n' % (self._ss_steps, self._ss_busy_steps))\n"
        "                        _f_ss.write('prefill_tokens/step: mean=%.0f max=%d\\n' % (self._ss_pf_sum / _n, self._ss_pf_max))\n"
        "                        _f_ss.write('new_prefills/step: mean=%.2f max=%d\\n' % (self._ss_new_sum / _n, self._ss_new_max))\n"
        "                        _f_ss.write('waiting: mean=%.2f max=%d\\n' % (self._ss_wait_sum / _n, self._ss_wait_max))\n"
        "                        _f_ss.write('pf_hist(0,<2k,<4k,<8k,<16k,<32k,<64k,64k+)=%s\\n' % (self._ss_pf_hist,))\n"
        "                        _f_ss.write('wait_hist(0,1,<4,<16,<64,<256,<1024,1024+)=%s\\n' % (self._ss_wait_hist,))\n"
        "                except Exception:\n"
        "                    pass\n"
    ).format(marker=MARKER)
    src = src.replace(stats_anchor, stats_inject, 1)

    ast.parse(src)
    shutil.copy(path, path + ".bak_schedstats")
    open(path, "w").write(src)
    print("[apply_sched_stats_patch] applied ->", path)
    return 0


def main() -> int:
    try:
        import vllm
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[apply_sched_stats_patch] vllm import failed: {e}\n")
        return 0
    target = os.path.join(os.path.dirname(vllm.__file__), "v1", "core", "sched", "scheduler.py")
    if not os.path.isfile(target):
        sys.stderr.write(f"[apply_sched_stats_patch] target not found: {target}\n")
        return 0
    return patch(target)


if __name__ == "__main__":
    raise SystemExit(main())
