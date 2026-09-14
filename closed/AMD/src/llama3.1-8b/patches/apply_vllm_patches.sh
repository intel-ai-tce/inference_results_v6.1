#!/usr/bin/env bash
# Idempotent in-container patcher for the installed vLLM package.
#
# Why this exists:
#   vLLM is baked into the docker image (site-packages), so any edit made inside
#   a running container is lost when a fresh container is launched. The repo's
#   code/ directory IS bind-mounted into every container, so we keep the patch
#   here and re-apply it at run time.
#
# What it does:
#   Forces the ROCm shuffle-KV ASM paged-attention path to use the non-high-
#   precision kernel variant by passing high_precision=0 into
#   rocm_aiter_ops.paged_attention_common -> aiter.paged_attention_common.
#
#     high_precision=0 -> pa_bf16_pertokenFp8_gqa8_2tg_4w.co      (selected here)
#     high_precision=1 -> pa_bf16_pertokenFp8_gqa8_2tg_4w_hp.co   (vLLM default)
#     high_precision=2 -> pa_bf16_pertokenFp8_gqa8_2tg_4w_uhp.co
#
#   Validated on Llama3.1-8B Offline (MI355X):
#     perf  : 1061.19 vs 1058.25 samples/s (hp=1 baseline)
#     ROUGE : rougeL 24.4541 / rouge1 38.5278 / rouge2 16.0577  (passes 99% floor)
#
# Safe to run repeatedly: it is a no-op once the package is already patched.
set -euo pipefail

MARKER="TUNE: non-hp ASM PA kernel"

# ---------------------------------------------------------------------------
# 0001 - vLLM ASM paged-attention high_precision=0
# Wrapped in a function so an early return for this patch does NOT skip the
# GEMM patch below.
# ---------------------------------------------------------------------------
apply_pa_patch() {
    local VLLM_DIR TARGET
    VLLM_DIR="$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${VLLM_DIR}" ] || [ ! -d "${VLLM_DIR}" ]; then
        echo "[apply_vllm_patches] WARNING: could not locate installed vllm package; skipping PA patch." >&2
        return 0
    fi

    TARGET="${VLLM_DIR}/_aiter_ops.py"
    if [ ! -f "${TARGET}" ]; then
        echo "[apply_vllm_patches] WARNING: ${TARGET} not found; skipping PA patch." >&2
        return 0
    fi

    if grep -q "${MARKER}" "${TARGET}"; then
        echo "[apply_vllm_patches] vLLM ASM-PA high_precision=0 already applied -> no-op."
        return 0
    fi

    # Robust, version-tolerant in-place edit (does not rely on exact line numbers).
    python3 - "${TARGET}" <<'PY'
import re, shutil, sys

path = sys.argv[1]
src = open(path).read()

# Anchor on the unique paged_attention_common(...) call inside the
# rocm_aiter_ops wrapper, identified by its K_QScale_asm/out_ argument block.
block = (
    "            K_QScale_asm=K_QScale_asm,\n"
    "            V_QScale_asm=V_QScale_asm,\n"
    "            out_=out_,\n"
    "            kv_cache_dtype=kv_cache_dtype,\n"
    "        )"
)
if "high_precision=" in src and "paged_attention_common(" in src.split("high_precision=")[0][-400:]:
    print("[apply_vllm_patches] high_precision already passed; no-op.")
    sys.exit(0)
if block not in src:
    sys.stderr.write(
        "[apply_vllm_patches] ERROR: expected paged_attention_common call block "
        "not found; vLLM layout may have changed. Apply manually.\n")
    sys.exit(2)

new = block.replace(
    "            kv_cache_dtype=kv_cache_dtype,\n        )",
    "            kv_cache_dtype=kv_cache_dtype,\n"
    "            high_precision=0,  # TUNE: non-hp ASM PA kernel (pa_..._2tg_4w), fastest + ROUGE-validated MLPerf-legal\n"
    "        )",
)
shutil.copy(path, path + ".bak_hp")
open(path, "w").write(src.replace(block, new, 1))

import ast
ast.parse(open(path).read())  # fail loudly if we broke syntax
print("[apply_vllm_patches] applied high_precision=0 to", path)
PY
}

apply_pa_patch

# ---------------------------------------------------------------------------
# 0002 - aiter a4w4 (MXFP4) GEMM tuned config
# Installs the retuned a4w4_blockscale_tuned_gemm.csv (adds tuned kernels for the
# actual decode batch sizes; 13-37% faster on mid-M qkv/o_proj/down_proj GEMMs).
# Strictly >= shipped per shape (only better kernels were written). See README.
# ---------------------------------------------------------------------------
SELF_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GEMM_CSV_SRC="${SELF_DIR}/a4w4_blockscale_tuned_gemm.csv"

AITER_DIR="$(python3 -c 'import os, aiter; print(os.path.dirname(aiter.__file__))' 2>/dev/null || true)"
if [ -n "${AITER_DIR}" ] && [ -f "${GEMM_CSV_SRC}" ]; then
    GEMM_CSV_DST="${AITER_DIR}/configs/a4w4_blockscale_tuned_gemm.csv"
    if [ -f "${GEMM_CSV_DST}" ] && cmp -s "${GEMM_CSV_SRC}" "${GEMM_CSV_DST}"; then
        echo "[apply_vllm_patches] a4w4 tuned GEMM CSV already installed -> no-op."
    else
        [ -f "${GEMM_CSV_DST}" ] && [ ! -f "${GEMM_CSV_DST}.bak_gemmtune" ] \
            && cp "${GEMM_CSV_DST}" "${GEMM_CSV_DST}.bak_gemmtune"
        cp "${GEMM_CSV_SRC}" "${GEMM_CSV_DST}"
        rm -rf /tmp/aiter_configs   # force re-merge of tuned configs on next import
        echo "[apply_vllm_patches] installed a4w4 tuned GEMM CSV -> ${GEMM_CSV_DST}"
    fi
else
    echo "[apply_vllm_patches] WARNING: aiter or tuned GEMM CSV not found; skipping GEMM patch." >&2
fi

# ---------------------------------------------------------------------------
# 0003 - vLLM v1 scheduler: make the prefill-admission cadence knobs LIVE
# (VLLM_SUBSEQUENT_DECODE_STEPS / VLLM_MIN_REQUEST_DECODE_STEP) + adaptive
# VLLM_ADAPTIVE_PREFILL_WAITING_HWM. All default to 0 => stock behavior, so
# this is a no-op unless a scenario YAML sets a non-zero cadence. Idempotent.
# See code/patches/README.md (section 0003) for measured per-scenario effects.
# ---------------------------------------------------------------------------
SELF_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SELF_DIR}/apply_scheduler_patch.py" ]; then
    python3 "${SELF_DIR}/apply_scheduler_patch.py" || \
        echo "[apply_vllm_patches] WARNING: scheduler patch failed; continuing with stock scheduler." >&2
fi

# 0004 - decode-batch cap under prefill backlog (INVERSE of cadence: trades TPOT
# headroom for lower TTFT). Env-gated, default off. See README.md section 0004.
if [ -f "${SELF_DIR}/apply_decode_cap_patch.py" ]; then
    python3 "${SELF_DIR}/apply_decode_cap_patch.py" || \
        echo "[apply_vllm_patches] WARNING: decode-cap patch failed; continuing without it." >&2
fi

# 0005 - READ-ONLY scheduler instrumentation (per-step prefill-token / waiting
# histograms). Enabled only when VLLM_SCHED_STATS=1; never changes scheduling.
# Used to diagnose the TTFT tail (mega prefill steps) vs flat TPOT. See README.
if [ -f "${SELF_DIR}/apply_sched_stats_patch.py" ]; then
    python3 "${SELF_DIR}/apply_sched_stats_patch.py" || \
        echo "[apply_vllm_patches] WARNING: sched-stats patch failed; continuing without it." >&2
fi

# 0006 - prefill-tokens-per-step cap: splits "monster" prefill steps to cut the
# TTFT p99 tail (head-of-line blocking of mid-burst arrivals), spending the TPOT
# headroom. Env-gated VLLM_MAX_PREFILL_TOKENS_PER_STEP, default 0=off. See README.
if [ -f "${SELF_DIR}/apply_prefill_cap_patch.py" ]; then
    python3 "${SELF_DIR}/apply_prefill_cap_patch.py" || \
        echo "[apply_vllm_patches] WARNING: prefill-cap patch failed; continuing without it." >&2
fi

# 0007 - DISABLED: de-constexpr of merge_attn_states prefill_tokens_with_context.
# It eliminated the Offline recompile storm (cold-cache 1020 -> 1242 samples/s,
# 100% stable util) BUT broke accuracy (ROUGE collapsed: rouge1 ~10 / rougeL ~8,
# runaway gen_len) -- the runtime-arg lowering hits a ROCm/Triton codegen path
# that produces wrong attention merges here, so it is NOT numerically equivalent
# in practice. Kept in-tree for reference; do NOT re-enable without an accuracy
# (ROUGE) pass. Pursue recompile elimination via warmup precompilation instead.
# if [ -f "${SELF_DIR}/apply_merge_attn_states_patch.py" ]; then
#     python3 "${SELF_DIR}/apply_merge_attn_states_patch.py" || \
#         echo "[apply_vllm_patches] WARNING: merge_attn_states patch failed; continuing without it." >&2
# fi

echo "[apply_vllm_patches] done."
