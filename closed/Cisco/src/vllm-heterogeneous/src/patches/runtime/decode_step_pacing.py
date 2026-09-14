
"""
Port AMD MLPerf v6.0 decode-step pacing patch into our vLLM build.

Source: mlcommons<submission-root>_results_v6.0/closed/AMD/setup/gpt-oss-120b/patches/
        gpt-oss-120b_vllm_v0.14.0_amd_dev.patch

Adds two scheduler knobs:

  VLLM_MIN_REQUEST_DECODE_STEP   (int, default 0)
  VLLM_SUBSEQUENT_DECODE_STEPS   (int, default 0)

Behaviour (per AMD patch):
  - Every schedule() iteration increments a step counter.
  - When the counter reaches VLLM_SUBSEQUENT_DECODE_STEPS it resets to 0.
    On the reset iteration, waiting requests are admitted normally (prefill).
  - On non-reset iterations, if >= VLLM_MIN_REQUEST_DECODE_STEP requests are
    already running, the WAITING admission loop is short-circuited so the
    next step stays decode-only. This keeps decode batches dense and prevents
    prefill bursts from preempting them.

Also registers two diagnostic vars from the same patch (VLLM_LOG_PREEMPTIONS,
ENABLE_TRACING_RPD) in envs.py so they no longer produce "unknown env var"
warnings, but their *behaviour* is not wired in (RPD profiler hook + preempt
counter print are not needed for steady-state throughput tuning).

Usage: invoked from start_server.sh via runtime_patches.
"""

import os
import shutil
import sys


def find_file(candidates, fallback_subpath=None):
    for c in candidates:
        if os.path.exists(c):
            return c
    if fallback_subpath:
        try:
            import vllm
            p = os.path.join(os.path.dirname(vllm.__file__), *fallback_subpath)
            if os.path.exists(p):
                return p
        except ImportError:
            pass
    return None


ENVS_FILE = find_file(
    [
    ],
    fallback_subpath=["envs.py"],
)

SCHED_FILE = find_file(
    [
    ],
    fallback_subpath=["v1", "core", "sched", "scheduler.py"],
)

if ENVS_FILE is None or SCHED_FILE is None:
    print("[DECODE-PACING] ERROR: could not locate envs.py or scheduler.py")
    sys.exit(1)

print("=" * 60)
print("Decode-Step Pacing Patch (AMD MLPerf v6.0)")
print("=" * 60)
print(f"  envs.py:      {ENVS_FILE}")
print(f"  scheduler.py: {SCHED_FILE}")






def patch_file(path, edits, sentinel, backup_suffix):
    """Apply a list of (anchor, replacement) edits to a file with safety.

    Returns True if any change was written, False if already patched.
    Raises if any anchor is missing.
    """
    with open(path) as f:
        original = f.read()

    if sentinel in original:
        print(f"  [skip] {os.path.basename(path)} already patched")
        return False

    backup = path + backup_suffix
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"  backup: {backup}")

    new = original
    for anchor, replacement in edits:
        if anchor not in new:
            raise RuntimeError(
                f"anchor not found in {path}:\n----\n{anchor}\n----"
            )
        if new.count(anchor) > 1:
            raise RuntimeError(
                f"anchor not unique in {path} ({new.count(anchor)}x):\n"
                f"----\n{anchor}\n----"
            )
        new = new.replace(anchor, replacement, 1)

    with open(path, "w") as f:
        f.write(new)

    try:
        compile(new, path, "exec")
    except SyntaxError as e:
        print(f"  SYNTAX ERROR in {path}: {e}")
        shutil.copy2(backup, path)
        print("  restored from backup")
        raise

    return True






ENVS_TYPE_ANCHOR = "    VLLM_USE_AOT_COMPILE: bool = False"
ENVS_TYPE_REPLACEMENT = (
    "    # [DECODE-PACING] env vars (AMD MLPerf v6.0)\n"
    "    VLLM_SUBSEQUENT_DECODE_STEPS: int = 0\n"
    "    VLLM_MIN_REQUEST_DECODE_STEP: int = 0\n"
    "    VLLM_LOG_PREEMPTIONS: bool = False\n"
    "    ENABLE_TRACING_RPD: bool = False\n"
    "    VLLM_USE_AOT_COMPILE: bool = False"
)

ENVS_DICT_ANCHOR = (
    "environment_variables: dict[str, Callable[[], Any]] = {\n"
)
ENVS_DICT_REPLACEMENT = (
    "environment_variables: dict[str, Callable[[], Any]] = {\n"
    "    # [DECODE-PACING] env vars (AMD MLPerf v6.0)\n"
    "    \"VLLM_SUBSEQUENT_DECODE_STEPS\": lambda: int(\n"
    "        os.getenv(\"VLLM_SUBSEQUENT_DECODE_STEPS\", \"0\")\n"
    "    ),\n"
    "    \"VLLM_MIN_REQUEST_DECODE_STEP\": lambda: int(\n"
    "        os.getenv(\"VLLM_MIN_REQUEST_DECODE_STEP\", \"0\")\n"
    "    ),\n"
    "    \"VLLM_LOG_PREEMPTIONS\": lambda: bool(\n"
    "        int(os.getenv(\"VLLM_LOG_PREEMPTIONS\", \"0\"))\n"
    "    ),\n"
    "    \"ENABLE_TRACING_RPD\": lambda: bool(\n"
    "        int(os.getenv(\"ENABLE_TRACING_RPD\", \"0\"))\n"
    "    ),\n"
)

envs_changed = patch_file(
    ENVS_FILE,
    edits=[
        (ENVS_TYPE_ANCHOR, ENVS_TYPE_REPLACEMENT),
        (ENVS_DICT_ANCHOR, ENVS_DICT_REPLACEMENT),
    ],
    sentinel="[DECODE-PACING] env vars",
    backup_suffix=".decode_pacing_bak",
)
print(f"  envs.py: {'patched' if envs_changed else 'unchanged'}")







SCHED_IMPORT_ANCHOR = "from typing import Any\n\n"
SCHED_IMPORT_REPLACEMENT = (
    "from typing import Any\n\n"
    "from vllm import envs  # [DECODE-PACING] MLPerf scheduler knobs\n"
)




SCHED_INIT_VARIANTS = [
    (
        (
            "        self.use_v2_model_runner = vllm_config.use_v2_model_runner\n"
            "        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle\n"
            "        # cadence (`next_decode_eligible_step`).\n"
            "        self.current_step = 0\n"
        ),
        (
            "        self.use_v2_model_runner = vllm_config.use_v2_model_runner\n"
            "        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle\n"
            "        # cadence (`next_decode_eligible_step`).\n"
            "        self.current_step = 0\n"
            "        # [DECODE-PACING] reserve N decode-only steps between prefill bursts\n"
            "        self.max_decode_step = envs.VLLM_SUBSEQUENT_DECODE_STEPS\n"
            "        self.min_request_for_decode = envs.VLLM_MIN_REQUEST_DECODE_STEP\n"
            "        self.decode_pacing_current_step = 0\n"
            "        self.preemption_count = 0\n"
        ),
    ),
    (
        (
            "        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER\n"
        ),
        (
            "        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER\n"
            "        # [DECODE-PACING] reserve N decode-only steps between prefill bursts\n"
            "        self.max_decode_step = envs.VLLM_SUBSEQUENT_DECODE_STEPS\n"
            "        self.min_request_for_decode = envs.VLLM_MIN_REQUEST_DECODE_STEP\n"
            "        self.decode_pacing_current_step = 0\n"
            "        self.preemption_count = 0\n"
        ),
    ),
]





SCHED_LOOP_VARIANTS = [
    (
        (
            "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
            "                if len(self.running) == self.max_num_running_reqs:\n"
            "                    break\n"
        ),
        (
            "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
            "                # [DECODE-PACING] keep decode batches dense between prefill bursts\n"
            "                if (self.max_decode_step\n"
            "                        and self.decode_pacing_current_step != 0\n"
            "                        and (len(self.running) >= self.min_request_for_decode)):\n"
            "                    break\n"
            "                if len(self.running) == self.max_num_running_reqs:\n"
            "                    break\n"
        ),
    ),
    (
        (
            "            while self.waiting and token_budget > 0:\n"
            "                if len(self.running) == self.max_num_running_reqs:\n"
            "                    break\n"
        ),
        (
            "            while self.waiting and token_budget > 0:\n"
            "                # [DECODE-PACING] keep decode batches dense between prefill bursts\n"
            "                if (self.max_decode_step\n"
            "                        and self.decode_pacing_current_step != 0\n"
            "                        and (len(self.running) >= self.min_request_for_decode)):\n"
            "                    break\n"
            "                if len(self.running) == self.max_num_running_reqs:\n"
            "                    break\n"
        ),
    ),
]

with open(SCHED_FILE) as f:
    _sched_source = f.read()

for SCHED_INIT_ANCHOR, SCHED_INIT_REPLACEMENT in SCHED_INIT_VARIANTS:
    if SCHED_INIT_ANCHOR in _sched_source:
        break
else:
    raise RuntimeError(
        "no supported scheduler init anchor found in "
        f"{SCHED_FILE}; update decode_step_pacing.py"
    )

for SCHED_LOOP_ANCHOR, SCHED_LOOP_REPLACEMENT in SCHED_LOOP_VARIANTS:
    if SCHED_LOOP_ANCHOR in _sched_source:
        break
else:
    raise RuntimeError(
        "no supported WAITING loop anchor found in "
        f"{SCHED_FILE}; update decode_step_pacing.py"
    )


SCHED_END_ANCHOR = (
    "            self._update_after_schedule(scheduler_output)\n"
    "        return scheduler_output\n"
)
SCHED_END_REPLACEMENT = (
    "            self._update_after_schedule(scheduler_output)\n"
    "\n"
    "        # [DECODE-PACING] tick / reset the MLPerf decode-step counter\n"
    "        if self.max_decode_step:\n"
    "            if self.decode_pacing_current_step >= self.max_decode_step:\n"
    "                self.decode_pacing_current_step = 0\n"
    "            else:\n"
    "                self.decode_pacing_current_step = self.decode_pacing_current_step + 1\n"
    "        return scheduler_output\n"
)

sched_changed = patch_file(
    SCHED_FILE,
    edits=[
        (SCHED_IMPORT_ANCHOR, SCHED_IMPORT_REPLACEMENT),
        (SCHED_INIT_ANCHOR, SCHED_INIT_REPLACEMENT),
        (SCHED_LOOP_ANCHOR, SCHED_LOOP_REPLACEMENT),
        (SCHED_END_ANCHOR, SCHED_END_REPLACEMENT),
    ],
    sentinel="[DECODE-PACING]",
    backup_suffix=".decode_pacing_bak",
)
print(f"  scheduler.py: {'patched' if sched_changed else 'unchanged'}")

print()
print("  [DECODE-PACING] patch applied successfully.")
print("  Active when both env vars are set:")
print("    VLLM_SUBSEQUENT_DECODE_STEPS  (steps between prefill admissions)")
print("    VLLM_MIN_REQUEST_DECODE_STEP  (running-req threshold to throttle)")
