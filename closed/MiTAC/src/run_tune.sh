#!/bin/bash

set -xeu

MODEL=${MODEL:-llama2-70b}
CONFIG_NAME=${CONFIG_NAME:-"offline_mi355x"}
CONFIG_PATH="/lab-mlperf-inference/code/llama2-70b-99/"
BACKEND=${BACKEND:-"vllm"}
NUM_TRIALS=${NUM_TRIALS:-50}
START_TIME=$(date +%m%d-%H%M%S)
BASE_LOG_DIR="${BASE_LOG_DIR:-${LAB_CLOG}/optuna/${START_TIME}}"
LOG_DIR=${BASE_LOG_DIR}

mkdir -p $LOG_DIR

LOG_FILE=${LOG_DIR}/perf_${MODEL}_${CONFIG_NAME}_optuna_hpt.log
STUDY_NAME=${MODEL}-${CONFIG_NAME}-${START_TIME}
STORAGE_NAME=${STORAGE_NAME:-"mlperf-inference"}

env | sort >> ${LOG_DIR}/ct-env.txt

python3 -u /lab-mlperf-inference/code/tune.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    --backend ${BACKEND} \
    --log-dir ${LOG_DIR} \
    --storage-name ${STORAGE_NAME} \
    --study-name ${STUDY_NAME} \
    --num-trials ${NUM_TRIALS} \
    2>&1 | tee ${LOG_FILE}
