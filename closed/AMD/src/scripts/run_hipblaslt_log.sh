#!/bin/bash

set -xeu

MODEL=${MODEL:-llama2-70b}
CONFIG_NAME=${CONFIG_NAME:-"offline_mi355x"}
NUM_SAMPLES=${NUM_SAMPLES:-2000}
DEVICE_COUNT=${DEVICE_COUNT:-1}
CONFIG_PATH="/lab-mlperf-inference/code/llama2-70b-99/"
DURATION_SEC=${DURATION_SEC:-600}
BACKEND=${BACKEND:-'vllm'}
START_TIME=$(date +%m%d-%H%M%S)
LOG_FILE_NAME=${LOG_FILE_NAME:-"hipblaslt_${MODEL}_${BACKEND}_${CONFIG_NAME}_${NUM_SAMPLES}_${START_TIME}"}

export HIPBLASLT_LOG_MASK=32

LOG_DIR=/lab-mlperf-inference/code/hipblaslt_logs
mkdir -p ${LOG_DIR}
export HIPBLASLT_LOG_FILE=${LOG_DIR}/${LOG_FILE_NAME}.log

python3 -u /lab-mlperf-inference/code/main.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    --backend ${BACKEND} \
    harness_config.total_sample_count=${NUM_SAMPLES} \
    harness_config.device_count=${DEVICE_COUNT} \
    harness_config.duration_sec=${DURATION_SEC}

sort ${HIPBLASLT_LOG_FILE} | uniq -c | sort -bgr > ${LOG_DIR}/processed_${LOG_FILE_NAME}.log
