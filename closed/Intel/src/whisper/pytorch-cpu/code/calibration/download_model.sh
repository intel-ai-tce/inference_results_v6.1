#!/bin/bash

set -euo pipefail

CALIBRATION_PYTHON="${CALIBRATION_PYTHON:-/opt/calibration-venv/bin/python}"
HF_BIN="${HF_BIN:-$(dirname "${CALIBRATION_PYTHON}")/hf}"
MODEL_ID="${MODEL_ID:-openai/whisper-large-v3}"
MODEL_PATH="${MODEL_PATH:-/model/whisper-large-v3}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${MODEL_PATH}/.hf-cache}"

if [ ! -x "${HF_BIN}" ]; then
    echo "Expected Hugging Face CLI at ${HF_BIN}" >&2
    exit 1
fi

mkdir -p "$(dirname "${MODEL_PATH}")"

"${HF_BIN}" download \
    --repo-type model \
    --local-dir "${MODEL_PATH}" \
    --cache-dir "${HF_CACHE_DIR}" \
    "${MODEL_ID}"
