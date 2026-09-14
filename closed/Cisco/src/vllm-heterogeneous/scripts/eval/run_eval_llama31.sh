#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
DEPLOYMENT_ENV="${ROOT_DIR}/config/deployment.env"

if [[ ! -f "${DEPLOYMENT_ENV}" ]]; then
    echo "ERROR: configure ${DEPLOYMENT_ENV} from config/deployment.env.example" >&2
    exit 1
fi
source "${DEPLOYMENT_ENV}"

for required in MODEL_ROOT DATA_ROOT NLTK_DATA; do
    if [[ -z "${!required:-}" ]]; then
        echo "ERROR: ${required} is required in config/deployment.env" >&2
        exit 1
    fi
done

LOG="${1:-${ROOT_DIR}/results/llama3.1_8b/offline/accuracy/mlperf_log_accuracy.json}"
OUT_DIR="$(dirname "${LOG}")"
OUT_FILE="${OUT_DIR}/accuracy.txt"
SCORER="${SCRIPT_DIR}/evaluate_llama31_accuracy_local.py"
DATASET="${DATA_ROOT}/llama3.1-8b/cnn_eval.json"
MODEL="${MODEL_ROOT}/llama3.1-8b"

for required in "${LOG}" "${SCORER}" "${DATASET}"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: required file not found: ${required}" >&2
        exit 1
    fi
done
if [[ ! -d "${MODEL}" ]]; then
    echo "ERROR: tokenizer/model directory not found: ${MODEL}" >&2
    exit 1
fi

python3 "${SCORER}" \
    --mlperf-accuracy-file "${LOG}" \
    --dataset-file "${DATASET}" \
    --model-name "${MODEL}" \
    --nltk-data "${NLTK_DATA}" 2>&1 | tee "${OUT_FILE}"
