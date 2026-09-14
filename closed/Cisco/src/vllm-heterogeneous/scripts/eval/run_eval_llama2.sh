#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
DEPLOYMENT_ENV="${ROOT_DIR}/config/deployment.env"

if [ ! -f "${DEPLOYMENT_ENV}" ]; then
    echo "ERROR: configure ${DEPLOYMENT_ENV} from config/deployment.env.example" >&2
    exit 1
fi
. "${DEPLOYMENT_ENV}"

for required in MLPERF_INFERENCE_DIR DATA_ROOT MODEL_ROOT; do
    if [ -z "${!required:-}" ]; then
        echo "ERROR: ${required} is required in config/deployment.env" >&2
        exit 1
    fi
done

LOG="${1:-${ROOT_DIR}/results/llama2_70b/offline/accuracy/mlperf_log_accuracy.json}"
OUT_DIR="$(dirname "${LOG}")"
OUT_FILE="${OUT_DIR}/accuracy.txt"
EVAL_SCRIPT="${MLPERF_INFERENCE_DIR}/language/llama2-70b/evaluate-accuracy.py"
DATASET="${DATA_ROOT}/llama2-70b/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
TOKENIZER="${MODEL_ROOT}/llama2-70b/orig"

if [ ! -f "${LOG}" ]; then
    echo "ERROR: accuracy log not found: ${LOG}" >&2
    exit 1
fi
if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "ERROR: official evaluator not found: ${EVAL_SCRIPT}" >&2
    exit 1
fi

python3 "${EVAL_SCRIPT}"     --checkpoint-path "${TOKENIZER}"     --mlperf-accuracy-file "${LOG}"     --dataset-file "${DATASET}"     --dtype int32     2>&1 | tee "${OUT_FILE}"
