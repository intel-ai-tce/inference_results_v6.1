#!/bin/bash

set -xeu

MODEL=${MODEL:-llama2-70b}
CONFIG_NAME=${CONFIG_NAME:-"offline_mi355x"}
export NUM_SAMPLES=20000
export DURATION_SEC=600
export DEVICE_COUNT=8
export CONFIG_NAME=$CONFIG_NAME
export MODEL=$MODEL
START_TIME=$(date +%m%d-%H%M%S)
BACKEND=${BACKEND:-'vllm'}

# run trace to get trace.rpd
export TRACE_FILE_NAME="trace_${MODEL}_${BACKEND}_${CONFIG_NAME}_${NUM_SAMPLES}_${START_TIME}"
export CREATE_JSON=0
bash /lab-mlperf-inference/code/scripts/run_trace.sh

# get hipblaslt log file
export LOG_FILE_NAME="hipblaslt_${MODEL}_${BACKEND}_${CONFIG_NAME}_${NUM_SAMPLES}_${START_TIME}"
bash /lab-mlperf-inference/code/scripts/run_hipblaslt_log.sh

python3 /lab-mlperf-inference/code/scripts/process_hipblaslt_log.py --file /lab-mlperf-inference/code/hipblaslt_logs/processed_${LOG_FILE_NAME}.log --db /lab-mlperf-inference/code/traces/${TRACE_FILE_NAME}.rpd --hipblaslt-bench /lab-mlperf-inference/code/scripts/bin/hipblaslt-bench --output /lab-mlperf-inference/code/comparison/${START_TIME}