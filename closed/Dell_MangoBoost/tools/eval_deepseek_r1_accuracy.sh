#!/bin/bash
# Score a DeepSeek-R1 accuracy run by delegating to the MLCommons reference eval.
#
# We do NOT vendor the reference scorer (it pulls in git submodules — LiveCodeBench,
# prm800k — and math/code eval deps). This wrapper runs the reference
# `language/deepseek-r1/eval_accuracy.py` against our loadgen accuracy log. It reads
# `mlperf_log_accuracy.json` (int32 token ids — the format our harness emits) plus
# the ground-truth dataset pickle, decodes, and grades per subset (AIME / MATH500 /
# GPQA / MMLU-Pro / LiveCodeBench).
#
# Prerequisites:
#   - The MLCommons inference repo checked out with its submodules + eval deps
#     installed (see that repo's language/deepseek-r1/README.md). Point at it via
#     MLPERF_INFERENCE_ROOT.
#   - The reference eval loads the `deepseek-ai/DeepSeek-R1` tokenizer from the HF
#     hub (checkpoint path is hardcoded there) — ensure it is cached / reachable.
#
# Usage:
#   MLPERF_INFERENCE_ROOT=/path/to/inference \
#     PYTHON3_PATH=/workspace/.venv/bin/python \
#     tools/eval_deepseek_r1_accuracy.sh <mlperf_log_accuracy.json> <dataset.pkl>

set -e

PYTHON3_PATH=${PYTHON3_PATH:-python3}  # override to the container venv
ACCURACY_JSON=${1}
DATASET_FILE=${2}

if [ -z "${MLPERF_INFERENCE_ROOT}" ]; then
    echo "set MLPERF_INFERENCE_ROOT to the MLCommons inference repo checkout"
    exit 1
fi

EVAL_SCRIPT_PATH="${MLPERF_INFERENCE_ROOT}/language/deepseek-r1/eval_accuracy.py"

if [ ! -f "${EVAL_SCRIPT_PATH}" ]; then
    echo "reference eval not found at ${EVAL_SCRIPT_PATH}; check MLPERF_INFERENCE_ROOT"
    exit 1
fi

if [ -z "${ACCURACY_JSON}" ] || [ ! -f "${ACCURACY_JSON}" ]; then
    echo "usage: ${0} <mlperf_log_accuracy.json> <dataset.pkl>"
    exit 1
fi

if [ -z "${DATASET_FILE}" ] || [ ! -f "${DATASET_FILE}" ]; then
    echo "dataset pickle not found; pass it as the 2nd arg (see README.md to download)"
    exit 1
fi

"${PYTHON3_PATH}" "${EVAL_SCRIPT_PATH}" \
    --input-file "${ACCURACY_JSON}" \
    --dataset-file "${DATASET_FILE}"
