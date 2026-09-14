#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# OFFICIAL MLCommons compliance TEST08 for DLRM-v3 (ROCm), with all determinism
# fixes ON. Ground-truth PASS/FAIL via the official run_verification.py.
#
# Flow:
#   (a) Offline AccuracyOnly REFERENCE  -> full mlperf_log_accuracy.json
#       (NO audit.config present; LoadGen runs true AccuracyOnly)
#   (b) Server PerformanceOnly + TEST08 audit.config (mode=2, sample 4096)
#       -> sampled mlperf_log_accuracy.json + perf summary (VALID/INVALID)
#   (c) run_verification.py -r <ref acc> -t <audit acc>  (PASS iff ne_mismatch==0
#       and unmatched==0, rel-NE tol 0.1%)
#
# Determinism fixes applied to BOTH runs (so ref and test share numerics):
#   DLRM_FP8_SCALE_STORE_SHARED=1   (unified per-rank FP8 activation scale)
#   DLRM_ATTN_DETERM_MAXLEN=16384   (attention 1/MAX_SEQ_LEN pinned to model max)
#   DLRM_FLUSH_TRIM_RESULTS=1       (end-of-stream flush candidate_size fix)
#   DLRM_ACCURACY_RESPONSE_CANDIDATE_SIZE=2048 (ref emits Server 2048 width)
#
# Reference is Offline at BATCH=64 to isolate scenario consistency from batch
# size (our fixes make numerics batch-invariant); Server test is the GOLD cert
# point selected by CONF (default b64 q12200 PROD10min).
#
# OPTIMIZED_EMBED_COMPARE is a one-batch debug/equivalence guard for the
# optimized embedding lookup. When enabled, it runs both the optimized lookup and
# the baseline lookup, compares their metadata/tensors, and disables the
# optimized path after any mismatch. Keep optimized embedding ON and its compare
# guard OFF for TEST08: AccuracyOnly uses 32-candidate eval samples while the
# TEST08 fix emits Server-shaped 2048-candidate responses, and the optimized
# path handles that shape contract. The compare guard probes the baseline
# 32-candidate path, sees 32 != 2048, and disables the optimized path, which is
# not the passing TEST08 configuration.
# ------------------------------------------------------------------------------
set -u
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SELF}/../.." && pwd)"
WORKSPACE_HOST="${WORKSPACE_HOST:-$(cd "${ROOT}/.." && pwd)}"
cd "${ROOT}"

CONTAINER="${CONTAINER:-dlrmv3-e2e723}"
CONF="${CONF:-user_mi355x8_nve_b64_qps12200_PROD10min.conf}"
TEST08_TAG_PREFIX="${TEST08_TAG_PREFIX:-test08}"
TEST08_EXTRA_ENV="${TEST08_EXTRA_ENV:-}"
BENCH=/work/dlrm-v3-harness-rocm/benchmarks
AUDIT_SRC=/work/mlcommons-inference/compliance/TEST08/dlrm-v3/audit.config
VERIFY="${WORKSPACE_HOST}/mlcommons-inference/compliance/TEST08/run_verification.py"
SCALE="${ROOT}/artifacts/fp8_shared_scales_server"
FIX_ENV="$TEST08_EXTRA_ENV -e DLRM_ACCURACY_RESPONSE_CANDIDATE_SIZE=2048 -e TRITON_CACHE_AUTOTUNING=1 -e DLRM_FP8_SCALE_STORE=$SCALE -e DLRM_FP8_SCALE_STORE_SHARED=1 -e DLRM_ATTN_DETERM_MAXLEN=16384 -e DLRM_FLUSH_TRIM_RESULTS=1"
REF_FIX_ENV="$FIX_ENV -e DLRM_ACCURACY_USE_INFERENCE_DATASET=1"

if docker exec "$CONTAINER" pgrep -f run_benchmark.py >/dev/null 2>&1; then
  echo "ABORT: benchmark already running"; exit 1
fi

# Always leave benchmarks/ clean: no audit.config lingering for future runs.
cleanup_audit(){ docker exec "$CONTAINER" rm -f "$BENCH/audit.config" 2>/dev/null; }
trap cleanup_audit EXIT
cleanup_audit  # remove any stale copy before the reference run

export CPU_GUARD_ASSUME_YES=1
export WAIT="${WAIT:-14400}"
# Pin the exact TEST08-passing GOLD path instead of inheriting debug shell overrides.
export OPTIMIZED_EMBED_LOOKUP=1
export OPTIMIZED_EMBED_COMPARE=0
export HSTU_UNIFORM_TARGETS_METADATA=1
export WARMUP_STEPS=60
export BATCHING_WARMUP_STEPS=15

drain(){
  for i in $(seq 1 120); do
    if docker exec "$CONTAINER" python -c "import subprocess,sys,re; o=subprocess.run(['rocm-smi','--showmeminfo','vram'],capture_output=True,text=True).stdout; u=[int(re.sub(r'[^0-9]','',l.split(':')[-1])) for l in o.splitlines() if 'Used Memory' in l]; sys.exit(0 if (u and max(u)<5000000000) else 1)" 2>/dev/null; then
      echo "[drain] GPUs free after $((i*5))s"; break
    fi
    sleep 5
  done
}

echo "[test08] START $(date -u +%H:%M:%S)"
echo "[test08] CONF=$CONF"
echo "[test08] TEST08_TAG_PREFIX=$TEST08_TAG_PREFIX"

# ---------- (a) Offline AccuracyOnly reference ----------
echo "[test08][a] Offline AccuracyOnly reference $(date -u +%H:%M:%S)"
HSTU_UNIFORM_TARGETS_METADATA=0 EXTRA_ENV="$REF_FIX_ENV" TAG="${TEST08_TAG_PREFIX}_ref_offline_acc" \
  CONF="$CONF" SCENARIO=Offline MODE=accuracy BATCH=64 \
  bash scripts/run/run_gold.sh || { echo "[test08] reference FAILED rc=$?"; exit 2; }
REF=$(ls -dt "artifacts/gold_${TEST08_TAG_PREFIX}_ref_offline_acc_"*/ | head -1)
echo "[test08][a] REF=$REF"
REF_ACC="${REF}mlperf_log_accuracy.json"
if [ ! -s "$REF_ACC" ]; then echo "[test08] reference accuracy log missing"; exit 2; fi
echo "[test08][a] ref acc entries: $(grep -c '"data"' "$REF_ACC" 2>/dev/null)"
drain

# ---------- (b) Server PerformanceOnly + audit.config ----------
echo "[test08][b] install audit.config + Server PerformanceOnly $(date -u +%H:%M:%S)"
docker exec "$CONTAINER" cp "$AUDIT_SRC" "$BENCH/audit.config"
docker exec "$CONTAINER" cat "$BENCH/audit.config" | sed 's/^/[audit] /'
EXTRA_ENV="$FIX_ENV" TAG="${TEST08_TAG_PREFIX}_srv_perf_audit" \
  CONF="$CONF" SCENARIO=Server MODE=performance BATCH=64 \
  bash scripts/run/run_gold.sh; RC=$?
cleanup_audit
[ $RC -ne 0 ] && { echo "[test08] audit perf FAILED rc=$RC"; exit 3; }
TEST=$(ls -dt "artifacts/gold_${TEST08_TAG_PREFIX}_srv_perf_audit_"*/ | head -1)
echo "[test08][b] TEST=$TEST"
TEST_ACC="${TEST}mlperf_log_accuracy.json"

echo "[test08][b] --- audit sanity ---"
grep -haE "Found Audit Config|accuracy_log_sampling_target" "${TEST}mlperf_log_detail.txt" 2>/dev/null | head -3
echo "[test08][b] audit acc entries: $(grep -c '"data"' "$TEST_ACC" 2>/dev/null) (expect ~4096)"
echo "[test08][b] --- Server perf validity ---"
grep -haE "Result is|Performance constraints|Completed samples per second|Scheduled samples per second|99.00 percentile|target_qps|target_latency \(ns\)|INVALID|VALID" "${TEST}mlperf_log_summary.txt" 2>/dev/null | head -20

# ---------- (c) official verification ----------
echo "[test08][c] run_verification.py $(date -u +%H:%M:%S)"
REF_ACC_ABS="$(readlink -f "$REF_ACC")"
TEST_ACC_ABS="$(readlink -f "$TEST_ACC")"
docker exec "$CONTAINER" python3 "$VERIFY" -r "$REF_ACC_ABS" -t "$TEST_ACC_ABS" --tolerance 0.001 2>&1 | tee "${TEST}verify_accuracy.txt"
VERIFY_RC=${PIPESTATUS[0]}
echo "[test08][c] verifier exit=$VERIFY_RC"
[ "$VERIFY_RC" -ne 0 ] && { echo "[test08] verifier FAILED rc=$VERIFY_RC"; exit 4; }

echo "[test08] DONE $(date -u +%H:%M:%S)"
