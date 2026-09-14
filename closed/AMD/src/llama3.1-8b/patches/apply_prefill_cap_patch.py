#!/usr/bin/env python3
"""Idempotent in-container patcher: cap NEW prefill tokens admitted per step in
the vLLM v1 scheduler, to split "monster" prefill steps.

Motivation (measured, llama3.1-8b Interactive, MI355X, 8 data-parallel engines)
-------------------------------------------------------------------------------
Per-step instrumentation showed the TTFT tail is caused by rare bursts where the
scheduler admits up to ~1366 requests / ~73728 prefill tokens in a SINGLE step
(~50-80ms). Requests that arrive mid-monster must wait for that whole indivisible
step before they can even be scheduled -> TTFT p99 ~570ms vs the 500ms limit,
while TPOT p99 ~16ms sits far under its 30ms limit (2x headroom).

Decode is negligible on those steps (~174 tokens vs ~73728), so throttling decode
cannot help. The lever that touches the tail-causing steps is bounding prefill
tokens per step: a burst is then spread over several short steps, so a mid-burst
arrival waits ~one small step instead of a full monster -> lower TTFT p99. Because
smaller steps let decode run more frequently, TPOT is not harmed (often improved).

Mechanism
---------
While admitting WAITING (prefill) requests, stop once the cumulative new-prefill
tokens scheduled this step reach VLLM_MAX_PREFILL_TOKENS_PER_STEP; the remaining
waiting requests are simply scheduled next step. Decode (running) scheduling and
the global token_budget are untouched. Overshoot per step is bounded by one
request's chunk (<= max_model_len), so effective step size ~= cap.

Env knob (default 0 -> mechanism OFF -> byte-for-byte stock behavior):
  VLLM_MAX_PREFILL_TOKENS_PER_STEP   e.g. 8192 / 12288 / 16384

Safe to run repeatedly (idempotent via marker). Order-independent w.r.t. the
cadence / decode-cap / sched-stats patches.
"""
import ast
import os
import shutil
import sys

MARKER = "MLPERF: prefill-tokens-per-step cap"


def patch(path: str) -> int:
    src = open(path).read()
    if MARKER in src:
        print("[apply_prefill_cap_patch] already applied -> no-op.")
        return 0

    # --- 1) init field ---
    init_anchor = "        self.running: list[Request] = []\n"
    if init_anchor not in src:
        sys.stderr.write("[apply_prefill_cap_patch] ERROR: init anchor not found.\n")
        return 2
    init_inject = init_anchor + (
        "        # --- {marker} (env-driven, default off) ---\n"
        "        import os as _os_pf\n"
        "        self.max_prefill_tokens_per_step = int(_os_pf.getenv('VLLM_MAX_PREFILL_TOKENS_PER_STEP', '0'))\n"
        "        self._step_prefill_tokens = 0\n"
    ).format(marker=MARKER)
    src = src.replace(init_anchor, init_inject, 1)

    # --- 2) reset accumulator right before the WAITING loop ---
    reset_anchor = (
        "        # Next, schedule the WAITING requests.\n"
        "        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:\n"
    )
    if reset_anchor not in src:
        sys.stderr.write("[apply_prefill_cap_patch] ERROR: reset anchor not found.\n")
        return 2
    reset_inject = (
        "        # {marker}: reset per-step new-prefill accumulator.\n"
        "        self._step_prefill_tokens = 0\n"
        "        # Next, schedule the WAITING requests.\n"
        "        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:\n"
    ).format(marker=MARKER)
    src = src.replace(reset_anchor, reset_inject, 1)

    # --- 3) break the waiting loop once the per-step prefill cap is hit ---
    brk_anchor = (
        "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
        "                if len(self.running) == self.max_num_running_reqs:\n"
        "                    break\n"
    )
    if brk_anchor not in src:
        sys.stderr.write("[apply_prefill_cap_patch] ERROR: waiting-loop anchor not found.\n")
        return 2
    brk_inject = brk_anchor + (
        "                # {marker}: stop admitting prefill once this step's cap is\n"
        "                # reached; remaining waiting reqs go next step (splits monster\n"
        "                # steps -> lower TTFT p99, spends the TPOT headroom).\n"
        "                if (self.max_prefill_tokens_per_step\n"
        "                        and self._step_prefill_tokens >= self.max_prefill_tokens_per_step):\n"
        "                    break\n"
    ).format(marker=MARKER)
    src = src.replace(brk_anchor, brk_inject, 1)

    # --- 4) accumulate admitted new-prefill tokens ---
    acc_anchor = (
        "                self.running.append(request)\n"
        "                if self.log_stats:\n"
    )
    if acc_anchor not in src:
        sys.stderr.write("[apply_prefill_cap_patch] ERROR: admit anchor not found.\n")
        return 2
    acc_inject = (
        "                self.running.append(request)\n"
        "                # {marker}: count new-prefill tokens admitted this step.\n"
        "                self._step_prefill_tokens += num_new_tokens\n"
        "                if self.log_stats:\n"
    ).format(marker=MARKER)
    src = src.replace(acc_anchor, acc_inject, 1)

    ast.parse(src)
    shutil.copy(path, path + ".bak_prefillcap")
    open(path, "w").write(src)
    print("[apply_prefill_cap_patch] applied ->", path)
    return 0


def main() -> int:
    try:
        import vllm
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[apply_prefill_cap_patch] vllm import failed: {e}\n")
        return 0
    target = os.path.join(os.path.dirname(vllm.__file__), "v1", "core", "sched", "scheduler.py")
    if not os.path.isfile(target):
        sys.stderr.write(f"[apply_prefill_cap_patch] target not found: {target}\n")
        return 0
    return patch(target)


if __name__ == "__main__":
    raise SystemExit(main())
