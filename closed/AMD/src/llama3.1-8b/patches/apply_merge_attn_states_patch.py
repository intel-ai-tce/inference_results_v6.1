#!/usr/bin/env python3
"""Patch 0007 - stop per-value Triton recompiles of merge_attn_states_kernel.

!!! DISABLED - DO NOT RE-ENABLE WITHOUT AN ACCURACY (ROUGE) PASS !!!
    This patch fixed the Offline recompile storm and throughput (cold-cache
    1020 -> 1242 samples/s, 100% stable GPU util) but BROKE ACCURACY: ROUGE
    collapsed (rouge1 ~10.2 / rouge2 ~0.8 / rougeL ~7.9 vs ~38 / ~16 / ~24)
    with runaway gen_len. Although prefill_tokens_with_context is only used as a
    runtime threshold (so the change *looks* numerically identical), lowering it
    from tl.constexpr to a runtime arg triggers a ROCm/Triton codegen path that
    produces wrong attention merges on this stack. It is therefore NOT safe.
    Kept in-tree only for reference; the call site in apply_vllm_patches.sh is
    commented out. Eliminate the recompiles via warmup precompilation (which
    keeps the constexpr / exact numerics) instead.


Root cause (measured on Llama3.1-8B Offline, MI355X):
    vllm/v1/attention/ops/triton_merge_attn_states.py declares
        prefill_tokens_with_context: tl.constexpr
    on the @triton.jit kernel. That value is the number of prefill tokens with
    context and VARIES per scheduler step (chunked-prefill-with-context across
    large batches). Because it is a tl.constexpr, Triton compiles a brand-new
    kernel specialization for EVERY distinct value. In Offline this produced
    ~422 distinct specializations PER GPU that JIT-compile *during the timed run*
    (invisible to vLLM's jit_monitor, which dedups warnings by kernel name) ->
    per-engine stalls -> a rotating straggler (one GPU at ~63% util) that gates
    the whole run (Offline time = slowest engine). Server/Interactive hit only a
    few distinct values so they never showed the problem.

    The constexpr buys NOTHING here: it is used only as a runtime threshold,
        prefix_mask = token_idx < prefill_tokens_with_context
    where token_idx = tl.program_id(0) is already a runtime value, so the branch
    cannot be folded at compile time regardless. Making it a normal runtime
    scalar arg yields identical results and collapses ~422 specializations down
    to the handful Triton derives from int divisibility (==1 / %16), eliminating
    hot-path compilation.

Effect: Offline no longer needs a fully pre-warmed Triton cache to hit peak
throughput; the merge-kernel recompiles that caused the cold-cache gap disappear
by construction. Data-independent kernel-level fix (MLPerf-legal).

Idempotent + version-tolerant: matches the constexpr annotation textually and is
a no-op once applied. Safe if vLLM layout changes (warns, leaves file intact).
"""
import ast
import os
import shutil
import sys

MARKER = "PATCH 0007: de-constexpr (stop per-value Triton recompiles)"
OLD = "    prefill_tokens_with_context: tl.constexpr,\n"
NEW = (
    "    prefill_tokens_with_context,  # " + MARKER + "\n"
)


def main() -> int:
    try:
        import vllm  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[apply_vllm_patches] WARNING: cannot import vllm; skipping merge_attn_states patch ({e}).")
        return 0

    vllm_dir = os.path.dirname(vllm.__file__)
    target = os.path.join(vllm_dir, "v1", "attention", "ops", "triton_merge_attn_states.py")
    if not os.path.isfile(target):
        print(f"[apply_vllm_patches] WARNING: {target} not found; skipping merge_attn_states patch.")
        return 0

    src = open(target).read()
    if MARKER in src:
        print("[apply_vllm_patches] merge_attn_states de-constexpr already applied -> no-op.")
        return 0

    if OLD not in src:
        sys.stderr.write(
            "[apply_vllm_patches] ERROR: expected 'prefill_tokens_with_context: tl.constexpr,' "
            "not found in triton_merge_attn_states.py; vLLM layout may have changed. "
            "Apply manually.\n"
        )
        return 2

    new_src = src.replace(OLD, NEW, 1)
    ast.parse(new_src)  # fail loudly if we broke syntax
    shutil.copy(target, target + ".bak_mergeattn")
    open(target, "w").write(new_src)
    print(f"[apply_vllm_patches] applied merge_attn_states de-constexpr to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
