#!/bin/bash
# Score a gpt-oss-120b accuracy run by delegating to the MLCommons reference eval.
#
# We do NOT vendor the reference scorer (it pulls in git submodules — LiveCodeBench —
# and math/code eval deps). This wrapper runs the reference
# `language/gpt-oss-120b/eval_mlperf_accuracy.py` against our loadgen accuracy log. It
# reads `mlperf_log_accuracy.json` (int32 token ids — the format our harness emits)
# plus the reference dataset (parquet/pickle with dataset + ground_truth), detokenizes,
# and grades per subset (AIME25 / GPQA-Diamond / LiveCodeBench-v6). The final
# exact_match is weighted by DATASET_REPEATS {aime25:8, gpqa_diamond:5,
# livecodebench_v6:3}. Reference score 83.13%; pass threshold is 99% of it = 82.30%.
#
# Prerequisites:
#   - The MLCommons inference repo checked out WITH its submodules + eval deps installed
#     (LiveCodeBench execution runs candidate code; see that repo's
#     language/gpt-oss-120b/README.md). Point at it via MLPERF_INFERENCE_ROOT.
#   - The gpt-oss (harmony) tokenizer, reachable at TOKENIZER (default: the local
#     openai/gpt-oss-120b checkpoint) so detokenization does not hit the HF hub.
#
# Usage:
#   MLPERF_INFERENCE_ROOT=/path/to/inference \
#     PYTHON3_PATH=/workspace/.venv/bin/python \
#     tools/eval_gpt_oss_accuracy.sh <mlperf_log_accuracy.json> <acc_eval_ref.parquet>

set -e

PYTHON3_PATH=${PYTHON3_PATH:-python3}  # override to the container venv
TOKENIZER=${TOKENIZER:-/models/models/openai--gpt-oss-120b}
ACCURACY_JSON=${1}
DATASET_FILE=${2}

if [ -z "${MLPERF_INFERENCE_ROOT}" ]; then
    echo "set MLPERF_INFERENCE_ROOT to the MLCommons inference repo checkout"
    exit 1
fi

EVAL_SCRIPT_PATH="${MLPERF_INFERENCE_ROOT}/language/gpt-oss-120b/eval_mlperf_accuracy.py"

if [ ! -f "${EVAL_SCRIPT_PATH}" ]; then
    echo "reference eval not found at ${EVAL_SCRIPT_PATH}; check MLPERF_INFERENCE_ROOT"
    exit 1
fi

if [ -z "${ACCURACY_JSON}" ] || [ ! -f "${ACCURACY_JSON}" ]; then
    echo "usage: ${0} <mlperf_log_accuracy.json> <acc_eval_ref.parquet>"
    exit 1
fi

if [ -z "${DATASET_FILE}" ] || [ ! -f "${DATASET_FILE}" ]; then
    echo "reference dataset not found; pass acc_eval_ref.parquet as the 2nd arg"
    exit 1
fi

"${PYTHON3_PATH}" "${EVAL_SCRIPT_PATH}" \
    --mlperf-log "${ACCURACY_JSON}" \
    --reference-data "${DATASET_FILE}" \
    --tokenizer "${TOKENIZER}"
