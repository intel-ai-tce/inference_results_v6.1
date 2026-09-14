#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CALIBRATION_PYTHON="${CALIBRATION_PYTHON:-/opt/calibration-venv/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/model/whisper-large-v3}"

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    bash "${SCRIPT_DIR}/download_model.sh"
fi

"${CALIBRATION_PYTHON}" "${SCRIPT_DIR}/quantize_model.py"
