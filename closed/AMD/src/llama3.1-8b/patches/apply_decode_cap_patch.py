#!/usr/bin/env python3
"""Idempotent in-container patcher: add a *decode-batch cap under prefill backlog*
to the vLLM v1 scheduler. This is the INVERSE of the prefill cadence: instead of
throttling prefill to protect decode/TPOT, it throttles DECODE to accelerate
prefill/TTFT -- the useful direction for the TTFT-bound, TPOT-headroom MLPerf
llama3.1-8b Interactive/Server workloads.

Mechanism
---------
When the waiting (prefill) backlog is high, cap the number of RUNNING requests
that decode this step to `VLLM_DECODE_CAP_BATCH`. The per-step forward pass then
spends less on decode and (because a backlog exists) backfills that budget with
prefill -> higher prefill throughput -> lower TTFT, at the cost of slightly higher
TPOT for the decode requests deferred to a later step (spends TPOT headroom).

Fairness: `self.running` is rotated by the cap each step so every request takes
its turn decoding (no tail starvation).

Env knobs (both default 0 -> mechanism OFF -> byte-for-byte stock behavior):
  VLLM_DECODE_CAP_BATCH        max running reqs to decode/step when backlog high.
  VLLM_DECODE_CAP_WAITING_HWM  waiting-queue backlog that activates the cap.

Safe to run repeatedly (idempotent via marker). Applies on top of the cadence
patch (apply_scheduler_patch.py); order-independent.
"""
import ast
import os
import shutil
import sys

MARKER = "MLPERF: decode-cap under backlog"


def patch(path: str) -> int:
    src = open(path).read()
    if MARKER in src:
        print("[apply_decode_cap_patch] already applied -> no-op.")
        return 0

    # --- 1) init fields ---
    init_anchor = "        self.running: list[Request] = []\n"
    if init_anchor not in src:
        sys.stderr.write("[apply_decode_cap_patch] ERROR: init anchor not found.\n")
        return 2
    init_inject = init_anchor + (
        "        # --- {marker} (env-driven, default off) ---\n"
        "        import os as _os_dc\n"
        "        self.decode_cap = int(_os_dc.getenv('VLLM_DECODE_CAP_BATCH', '0'))\n"
        "        self.decode_cap_hwm = int(_os_dc.getenv('VLLM_DECODE_CAP_WAITING_HWM', '0'))\n"
        "        self._decode_cap_active = False\n"
    ).format(marker=MARKER)
    src = src.replace(init_anchor, init_inject, 1)

    # --- 2) pre-loop: decide activation + rotate running for fairness ---
    preloop_anchor = (
        "        # First, schedule the RUNNING requests.\n"
        "        req_index = 0\n"
    )
    if preloop_anchor not in src:
        sys.stderr.write("[apply_decode_cap_patch] ERROR: pre-loop anchor not found.\n")
        return 2
    preloop_inject = (
        "        # First, schedule the RUNNING requests.\n"
        "        # {marker}: activate when a prefill backlog exists and the running\n"
        "        # set exceeds the cap; rotate running so decoding is fair across steps.\n"
        "        self._decode_cap_active = bool(\n"
        "            self.decode_cap and self.decode_cap_hwm\n"
        "            and len(self.waiting) >= self.decode_cap_hwm\n"
        "            and len(self.running) > self.decode_cap)\n"
        "        if self._decode_cap_active:\n"
        "            _k = self.decode_cap % len(self.running)\n"
        "            if _k:\n"
        "                self.running = self.running[_k:] + self.running[:_k]\n"
        "        req_index = 0\n"
    ).format(marker=MARKER)
    src = src.replace(preloop_anchor, preloop_inject, 1)

    # --- 3) loop-top: break once the decode cap is reached ---
    loop_anchor = (
        "        while req_index < len(self.running) and token_budget > 0:\n"
        "            request = self.running[req_index]\n"
    )
    if loop_anchor not in src:
        sys.stderr.write("[apply_decode_cap_patch] ERROR: running-loop anchor not found.\n")
        return 2
    loop_inject = (
        "        while req_index < len(self.running) and token_budget > 0:\n"
        "            # {marker}: stop scheduling decode once the cap is hit so the\n"
        "            # step's compute/budget is reallocated to prefill (lowers TTFT).\n"
        "            if (self._decode_cap_active\n"
        "                    and len(scheduled_running_reqs) >= self.decode_cap):\n"
        "                break\n"
        "            request = self.running[req_index]\n"
    ).format(marker=MARKER)
    src = src.replace(loop_anchor, loop_inject, 1)

    ast.parse(src)
    shutil.copy(path, path + ".bak_decodecap")
    open(path, "w").write(src)
    print("[apply_decode_cap_patch] applied ->", path)
    return 0


def main() -> int:
    try:
        import vllm
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[apply_decode_cap_patch] vllm import failed: {e}\n")
        return 0
    target = os.path.join(os.path.dirname(vllm.__file__), "v1", "core", "sched", "scheduler.py")
    if not os.path.isfile(target):
        sys.stderr.write(f"[apply_decode_cap_patch] target not found: {target}\n")
        return 0
    return patch(target)


if __name__ == "__main__":
    raise SystemExit(main())
