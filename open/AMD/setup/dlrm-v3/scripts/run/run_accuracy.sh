#!/usr/bin/env bash
# run_accuracy.sh — Offline AccuracyOnly certification on the SAME GOLD stack as the perf cert
# (fp8 A-FUSE + C1-off full-causal + b64 Win-B). It REUSES run_gold.sh's env/launch with
# MODE=accuracy + SCENARIO=Offline (so the precision stack is identical and the perf cert's
# accuracy carries), then scores the resulting mlperf_log_accuracy.json with score_accuracy.py.
#
# The MLPerf DLRM-v3 accuracy bar is RELATIVE: GAUC >= 99.9% of the fp16 reference. The
# last local-small-table recert scores GAUC 0.7858977608 = 99.9505% of reference (PASS).
# Plan 61 P3 preprocessor LN-add fold recert scores GAUC 0.7858978083 (PASS).
# Plan 62 output-LN fast-inference recert scores GAUC 0.7858985604 (PASS).
# Plan 62 P3 bf16 no-op cast guard is bit-exact/no-op when tensors are already bf16 and is
# now inherited from run_gold.sh's GOLD default.
# Plan 63 vectorized response-buffer handling is PerformanceOnly-only; AccuracyOnly stays on
# the existing scored-output path and inherits no arithmetic change.
# Plan 64 P6/c40 + P8 clock determinism + P0/P2/P1 host hygiene are bit-exact/system/host-path
# changes; no arithmetic or scored-output change, so the existing accuracy cert carries.
# q12,200 GOLD promotes the production degree-5 gate: Offline AccuracyOnly GAUC
# 0.7862875110 (PASS), then TEST08 passed with zero unmatched / zero NE mismatches.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/run/run_accuracy.sh
#   CONF=user_mi355x8_nve_b64_qps12200_PROD10min.conf bash scripts/run/run_accuracy.sh
#   SCORE_ONLY=<artifact-dir> bash scripts/run/run_accuracy.sh   # (re)score an existing log only
#
# ── Env (plus anything run_gold.sh accepts) ──────────────────────────────────
#   CONTAINER  docker container                                   [dlrmv3-e2e723]
#   CONF       USER_CONF (its Offline section drives AccuracyOnly)[b64_qps12200_PROD10min]
#   SCENARIO   Offline | Server (accuracy cert convention=Offline)[Offline]
#   WAIT       safety cap (s) for the accuracy pass                [3600]
#   GR_DLRM    in-container GR tree on PYTHONPATH for the scorer   [/work/mlcommons-inference/recommendation/dlrm_v3]
#   SCORE_ONLY skip the run; just (re)score the log in this artifact dir
#   OPTIMIZED_EMBED_LOOKUP / HSTU_UNIFORM_TARGETS_METADATA default to 0 for AccuracyOnly
#              (both are Server perf shortcuts for the 2048-candidate inference shape)
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${CONTAINER:-dlrmv3-e2e723}"
CONF="${CONF:-user_mi355x8_nve_b64_qps12200_PROD10min.conf}"
SCENARIO="${SCENARIO:-Offline}"
WAIT="${WAIT:-3600}"
GR_DLRM="${GR_DLRM:-/work/mlcommons-inference/recommendation/dlrm_v3}"

if [[ -n "${SCORE_ONLY:-}" ]]; then
  out="${SCORE_ONLY}"
  echo "[acc] SCORE_ONLY: re-scoring ${out}"
else
  # Reuse run_gold.sh's GOLD env block + launch + wait (single source of truth for the
  # precision stack); only MODE/SCENARIO differ. run_gold prints "ARTIFACT=<dir>" last.
  runlog="$(mktemp)"
  MODE=accuracy SCENARIO="${SCENARIO}" CONF="${CONF}" CONTAINER="${CONTAINER}" WAIT="${WAIT}" \
    OPTIMIZED_EMBED_LOOKUP="${OPTIMIZED_EMBED_LOOKUP:-0}" \
    OPTIMIZED_EMBED_COMPARE="${OPTIMIZED_EMBED_COMPARE:-0}" \
    HSTU_UNIFORM_TARGETS_METADATA="${HSTU_UNIFORM_TARGETS_METADATA:-0}" \
    TAG="${TAG:-acc_$(echo "${CONF}" | sed 's/^user_mi355x8_nve_//; s/\.conf$//')}" \
    bash "${SELF}/run_gold.sh" | tee "${runlog}"
  out="$(sed -n 's/^ARTIFACT=//p' "${runlog}" | tail -1)"
  rm -f "${runlog}"
fi
[[ -n "${out}" ]] || { echo "[acc] ERROR: could not determine the artifact dir"; exit 1; }
# The scorer runs inside the container and reads the log by its HOST path (valid in-container
# via the 1:1 MOUNT_ROOT bind), so the path must be absolute (SCORE_ONLY may be relative).
[[ -d "${out}" ]] || { echo "[acc] ERROR: artifact dir not found: ${out}"; exit 1; }
out="$(cd "${out}" && pwd)"
log="${out}/mlperf_log_accuracy.json"
[[ -s "${log}" ]] || { echo "[acc] ERROR: accuracy log missing/empty: ${log}"; exit 1; }
echo "[acc] accuracy log: ${log}"

# Score INSIDE the container: the host artifact path is valid in-container via the 1:1
# MOUNT_ROOT bind, and the scorer needs the GR tree (configs.py/utils.py) on PYTHONPATH.
docker cp "${SELF}/score_accuracy.py" "${CONTAINER}:/tmp/score_accuracy.py" >/dev/null
echo "[acc] scoring (streaming; the accuracy log can be ~17 GB — this takes several minutes) ..."
docker exec -e PYTHONPATH="${GR_DLRM}" "${CONTAINER}" \
  python3 /tmp/score_accuracy.py --path "${log}" 2>&1 | tee "${out}/accuracy_metrics.txt"

echo ""
echo "[acc] DONE — metrics saved to ${out}/accuracy_metrics.txt"
echo "[acc] PASS criterion: relative GAUC >= 99.9% of the fp16 reference."
echo "[acc] last local-small-table recert: GAUC 0.7858977608 = 99.9505% of reference (PASS)."
echo "[acc] Plan 61 P3 lnaddfold recert: GAUC 0.7858978083 (PASS)."
echo "[acc] Plan 62 output-LN fast-inference recert: GAUC 0.7858985604 (PASS)."
echo "[acc] Plan 62 P3 bf16 no-op cast guard is inherited from run_gold.sh's GOLD default."
echo "[acc] Plan 63 response-vectorization is PerformanceOnly-only; AccuracyOnly scored-output path unchanged."
echo "[acc] Plan 64 c40/P6/P8/P0/P2/P1 are bit-exact/system/host-path changes; no GAUC surface."
echo "[acc] q12,200 GOLD degree-5 gate recert: GAUC 0.7862875110 (PASS); TEST08 zero mismatch."
