#!/usr/bin/env python3
"""Idempotent in-container patcher: make the MLPerf prefill-admission scheduler
knobs LIVE in the installed (stock) vLLM v1 scheduler, and add an adaptive
waiting-queue bypass.

Background
----------
The AMD build-time patch (setup/llama3.1-8b/patches/llama3-8b_vllm.patch) adds a
prefill-admission *cadence* to vllm/v1/core/sched/scheduler.py driven by
VLLM_SUBSEQUENT_DECODE_STEPS / VLLM_MIN_REQUEST_DECODE_STEP. That patch is NOT
present in the vllm/vllm-openai-rocm:v0.22.0 image (verified: no such symbols in
site-packages), so those env vars set in the YAMLs are silently ignored. This
script injects the logic at run time so the knobs actually work, and adds:

  VLLM_SUBSEQUENT_DECODE_STEPS   (int, default 0=off): pure-decode steps between
                                 prefill-admission steps (the cadence period).
  VLLM_MIN_REQUEST_DECODE_STEP   (int, default 0): decode-batch floor; below it,
                                 prefill is always admitted (ramp protection).
  VLLM_ADAPTIVE_PREFILL_WAITING_HWM (int, default 0=off): if the waiting queue
                                 backlog reaches this high-watermark, bypass the
                                 cadence and admit prefill THIS step, so TTFT
                                 backlog drains fast (closed-loop on load).

All default to 0 -> byte-for-byte stock behavior (admit prefill every step).
Safe to run repeatedly (idempotent via marker).
"""
import ast
import os
import shutil
import sys

MARKER = "MLPERF: live prefill-admission control"


def patch(path: str) -> int:
    src = open(path).read()
    if MARKER in src:
        print("[apply_scheduler_patch] already applied -> no-op.")
        return 0

    # --- 1) init fields (anchor: self.running list init, unique) ---
    init_anchor = "        self.running: list[Request] = []\n"
    if init_anchor not in src:
        sys.stderr.write("[apply_scheduler_patch] ERROR: init anchor not found.\n")
        return 2
    init_inject = init_anchor + (
        "        # --- {marker} (env-driven, default off) ---\n"
        "        import os as _os  # scheduler.py has no top-level os import\n"
        "        self.max_decode_step = int(_os.getenv('VLLM_SUBSEQUENT_DECODE_STEPS', '0'))\n"
        "        self.min_request_for_decode = int(_os.getenv('VLLM_MIN_REQUEST_DECODE_STEP', '0'))\n"
        "        self.adaptive_prefill_hwm = int(_os.getenv('VLLM_ADAPTIVE_PREFILL_WAITING_HWM', '0'))\n"
        "        self.current_step = 0\n"
    ).format(marker=MARKER)
    src = src.replace(init_anchor, init_inject, 1)

    # --- 2) prefill-admission gate inside the WAITING loop ---
    loop_anchor = (
        "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
        "                if len(self.running) == self.max_num_running_reqs:\n"
        "                    break\n"
    )
    if loop_anchor not in src:
        sys.stderr.write("[apply_scheduler_patch] ERROR: waiting-loop anchor not found.\n")
        return 2
    gate = loop_anchor + (
        "                # {marker}: cadence-gate new prefills to protect decode/TPOT,\n"
        "                # with adaptive bypass when the waiting backlog is high (TTFT).\n"
        "                if (self.max_decode_step\n"
        "                        and self.current_step != 0\n"
        "                        and len(self.running) >= self.min_request_for_decode\n"
        "                        and (self.adaptive_prefill_hwm <= 0\n"
        "                             or len(self.waiting) < self.adaptive_prefill_hwm)):\n"
        "                    break\n"
    ).format(marker=MARKER)
    src = src.replace(loop_anchor, gate, 1)

    # --- 3) advance the cadence counter at the end of schedule() ---
    ret_anchor = "        return scheduler_output\n"
    if src.count(ret_anchor) != 1:
        sys.stderr.write(
            f"[apply_scheduler_patch] ERROR: expected 1 'return scheduler_output', "
            f"found {src.count(ret_anchor)}.\n")
        return 2
    step_inject = (
        "        # {marker}: advance cadence counter (0..max_decode_step)\n"
        "        if self.max_decode_step:\n"
        "            self.current_step = (0 if self.current_step >= self.max_decode_step\n"
        "                                 else self.current_step + 1)\n"
        "        return scheduler_output\n"
    ).format(marker=MARKER)
    src = src.replace(ret_anchor, step_inject, 1)

    ast.parse(src)  # fail loudly on syntax break
    shutil.copy(path, path + ".bak_sched")
    open(path, "w").write(src)
    print("[apply_scheduler_patch] applied ->", path)
    return 0


def main() -> int:
    try:
        import vllm
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[apply_scheduler_patch] vllm import failed: {e}\n")
        return 0
    target = os.path.join(os.path.dirname(vllm.__file__), "v1", "core", "sched", "scheduler.py")
    if not os.path.isfile(target):
        sys.stderr.write(f"[apply_scheduler_patch] target not found: {target}\n")
        return 0
    return patch(target)


if __name__ == "__main__":
    raise SystemExit(main())
