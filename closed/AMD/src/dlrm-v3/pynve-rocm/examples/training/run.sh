#!/usr/bin/env bash
# run.sh — run the NVE training smoke test inside the ROCm container, capturing the
# [NVE] C++ trace lines and the per-check JSONL trace into this directory.
#
#   bash run.sh
#
# Env overrides:
#   CONTAINER      docker container with the built pynve  [dlrmv3-e2e723]
#   REPO_C         pynve-rocm path INSIDE the container    [/work/pynve-rocm]
#   NVE_LOG_LEVEL  native log verbosity                    [INFO]
set -euo pipefail

CONTAINER="${CONTAINER:-dlrmv3-e2e723}"
REPO_C="${REPO_C:-/work/pynve-rocm}"
NVE_LOG_LEVEL="${NVE_LOG_LEVEL:-INFO}"

SELF_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C_DIR="${REPO_C}/examples/training"
LOG="${SELF_HOST}/train_smoke.log"

echo "[run-train] container=${CONTAINER} repo=${REPO_C} NVE_LOG_LEVEL=${NVE_LOG_LEVEL}"
echo "[run-train] log -> ${LOG}"

docker exec \
  -e PYTHONPATH="${REPO_C}/python" \
  -e LD_LIBRARY_PATH="${REPO_C}/build_rocm/lib" \
  -e NVE_LOG_LEVEL="${NVE_LOG_LEVEL}" \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
  -e NVE_TRACE_OUT="${C_DIR}/train_trace.jsonl" \
  -w "${C_DIR}" \
  "${CONTAINER}" \
  python3 "${C_DIR}/train_nve_smoke.py" 2>&1 | tee "${LOG}"
