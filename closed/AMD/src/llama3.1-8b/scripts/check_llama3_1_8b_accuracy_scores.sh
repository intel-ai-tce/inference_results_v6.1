#!/bin/bash

# set -x
set -e

# Accuracy checker for llama3.1-8b (CNN/DailyMail summarization, ROUGE metrics).
# Mirrors the interface of the other check_*_accuracy_scores.sh scripts: it takes
# the path to an mlperf_log_accuracy.json as $1 and writes accuracy.txt next to it.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EVAL_DIR=${SCRIPT_DIR}/llama3.1-8b

DATASET_FILE_PATH=${DATASET_FILE_PATH:-/data/cnn_eval.json}
# run_model.sh passes MODEL_OVERRIDE_PATH when a custom model path is used.
MODEL_PATH=${MODEL_OVERRIDE_PATH:-${MODEL_PATH:-/model/}}
EVAL_SCRIPT_PATH=${EVAL_DIR}/evaluation.py

PYTHON3_PATH=$(which python3 2>/dev/null || which python 2>/dev/null)
if [ -z "$PYTHON3_PATH" ]; then
    echo "Error: No Python interpreter found."
    exit 1
fi

if [ ! -f ${DATASET_FILE_PATH} ]; then
    echo "dataset not found at ${DATASET_FILE_PATH}, check the README.md how to download it"
    exit 1
fi

if [ ! -d ${MODEL_PATH} ] || [ -z "$(ls -A ${MODEL_PATH})" ]; then
    echo "model not found (or empty) at ${MODEL_PATH}, check the README.md how to download it"
    exit 1
fi

if [ ! -f ${EVAL_SCRIPT_PATH} ]; then
    echo "evaluation.py not found at ${EVAL_SCRIPT_PATH}"
    exit 1
fi

ACCURACY_JSON=${1}
if [ -z "${ACCURACY_JSON}" ] || [ ! -f "${ACCURACY_JSON}" ]; then
    echo "incorrect accuracy path, set it with ${0} <path/to/mlperf_log_accuracy.json>"
    exit 1
fi

ACCURACY_JSON=$(readlink -f "${ACCURACY_JSON}")
OUTPUT_DIR=$(dirname "${ACCURACY_JSON}")
RESULT_TXT=${OUTPUT_DIR}/accuracy.txt

# evaluation.py imports the sibling dataset.py, so run from its own directory.
cd "${EVAL_DIR}"
${PYTHON3_PATH} -u "${EVAL_SCRIPT_PATH}" \
    --mlperf-accuracy-file "${ACCURACY_JSON}" \
    --dataset-file "${DATASET_FILE_PATH}" \
    --model-name "${MODEL_PATH}" \
    --dtype int32 | tee "${RESULT_TXT}"

echo "Check $RESULT_TXT for the accuracy scores"
