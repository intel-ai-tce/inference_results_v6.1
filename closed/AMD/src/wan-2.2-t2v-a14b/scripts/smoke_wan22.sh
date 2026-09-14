#!/usr/bin/env bash
# In-container smoke test for the real WanBackend.
#
# Runs a tiny LoadGen test (a couple of synthetic prompts at reduced
# resolution / steps) with whichever scenario the user picks. Designed
# to stay well under 5 min wall time on an 8x MI300X / 8x H100 node, so
# a CI step or a local verification can be exercised without burning a
# full Offline run and without the official VBench prompt list.
#
# Usage:
#   ./scripts/smoke_wan22.sh [Offline|SingleStream]
#
# Optional env vars:
#   NPROC_PER_NODE   override --nproc-per-node (default: 8)
#   OUTPUT_DIR       override the logs directory
#   PROMPTS          path to prompts file (default: data/synthetic_prompts.txt)
#   SAMPLE_COUNT     number of prompts to feed LoadGen (default: 2)
#   MIN_QUERY_COUNT  shrink LoadGen's min_query_count (default: SAMPLE_COUNT)

set -euo pipefail

SCENARIO="${1:-Offline}"
NPROC="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/smoke/wan22/${SCENARIO}/performance/run_1}"
PROMPTS="${PROMPTS:-data/synthetic_prompts.txt}"
SAMPLE_COUNT="${SAMPLE_COUNT:-2}"
MIN_QUERY_COUNT="${MIN_QUERY_COUNT:-${SAMPLE_COUNT}}"

case "${SCENARIO}" in
    Offline|SingleStream) ;;
    *) echo "[smoke_wan22] unknown scenario: ${SCENARIO}" >&2; exit 1 ;;
esac

if [[ ! -f "${PROMPTS}" ]]; then
    echo "[smoke_wan22] prompts file not found: ${PROMPTS}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

BACKEND_CONFIG="configs/wan22/${SCENARIO}.yaml"

echo "[smoke_wan22] scenario=${SCENARIO} nproc=${NPROC} prompts=${PROMPTS} samples=${SAMPLE_COUNT}"
echo "[smoke_wan22] backend_config=${BACKEND_CONFIG}"
echo "[smoke_wan22] out=${OUTPUT_DIR}"

exec torchrun --standalone --nproc-per-node="${NPROC}" -m wan_harness.cli run \
    --backend wan22 \
    --scenario "${SCENARIO}" \
    --mode performance \
    --output-dir "${OUTPUT_DIR}" \
    --backend-config "${BACKEND_CONFIG}" \
    --prompts "${PROMPTS}" \
    --set height=480 \
    --set width=832 \
    --set num_frames=33 \
    --set sample_steps=4 \
    --set performance_sample_count="${SAMPLE_COUNT}" \
    --set min_query_count="${MIN_QUERY_COUNT}" \
    --set min_duration_ms=1
