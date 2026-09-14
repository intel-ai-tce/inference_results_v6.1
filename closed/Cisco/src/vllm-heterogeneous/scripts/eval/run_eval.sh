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

LOG="${1:-${ROOT_DIR}/results/gptoss_120b/server/accuracy/mlperf_log_accuracy.json}"
OUT_DIR="$(dirname "${LOG}")"
OUT_FILE="${OUT_DIR}/accuracy.txt"
SOURCE_DIR="${MLPERF_INFERENCE_DIR}/language/gpt-oss-120b"
SCORER="${SOURCE_DIR}/eval_mlperf_accuracy.py"
REFERENCE="${DATA_ROOT}/gpt-oss-120b/acc/acc_eval_ref.parquet"
TOKENIZER="${MODEL_ROOT}/gpt-oss-120b/model"

if [ ! -f "${LOG}" ]; then
    echo "ERROR: accuracy log not found: ${LOG}" >&2
    exit 1
fi
if [ ! -f "${SCORER}" ]; then
    echo "ERROR: official scorer not found: ${SCORER}" >&2
    exit 1
fi
if [ ! -d "${SOURCE_DIR}/submodules/LiveCodeBench/lcb_runner" ]; then
    echo "ERROR: initialize the GPT-OSS LiveCodeBench submodule in MLPERF_INFERENCE_DIR" >&2
    exit 1
fi

PYTHONPATH="${SOURCE_DIR}:${PYTHONPATH:-}" timeout --signal=KILL 900 python3 "${SCORER}"     --mlperf-log "${LOG}"     --reference-data "${REFERENCE}"     --tokenizer "${TOKENIZER}"     2>&1 | tee "${OUT_FILE}"
