#!/bin/bash

set -e

export MODEL_DIR="${MODEL_DIR:-/mad/model}"
export HF_MODEL_ID="${HF_MODEL_ID:-MISSING}"
export MODEL_PATH="${MODEL_PATH:-MISSING}"

hf download --token ${HUGGINGFACE_ACCESS_TOKEN} --local-dir "${MODEL_DIR}/${MODEL_PATH}" ${HF_MODEL_ID}
