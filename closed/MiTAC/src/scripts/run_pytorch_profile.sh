#!/bin/bash

set -xeu

SCENARIO_LOWER=$(echo "$SCENARIO" | tr '[:upper:]' '[:lower:]')
CONFIG_NAME="${SCENARIO_LOWER}_${GPU_NAME}"

MODEL=${MODEL:-deepseek-r1}
NUM_SAMPLES=${NUM_SAMPLES:-200}
DEVICE_COUNT=${DEVICE_COUNT:-8}
CONFIG_PATH="/lab-mlperf-inference/code/${MODEL}/"
DURATION_SEC=${DURATION_SEC:-1}
STEP_NUMS=${STEP_NUMS:-200}
STEP_OFFSET=${STEP_OFFSET:-200}
PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-1}
RECORD_SHAPE=${RECORD_SHAPE:-0}
START_TIME=$(date +%m%d-%H%M%S)
BACKEND=${BACKEND:-'sglang'}
TRACE_FILE_NAME=${TRACE_FILE_NAME:-"trace_${MODEL}_${BACKEND}_${CONFIG_NAME}_${NUM_SAMPLES}_${START_TIME}"}
CUSTOM_ARGS=${CUSTOM_ARGS:-''}

if [ "$BACKEND" = "sglang" ]; then
    WATCHDOG_TIMOUT="sglang_engine_config.watchdog_timeout=1500"
    if [ -n "$CUSTOM_ARGS" ]; then
        CUSTOM_ARGS="${CUSTOM_ARGS} ${WATCHDOG_TIMOUT}"
    else
        CUSTOM_ARGS="${WATCHDOG_TIMOUT}"
    fi
fi

export ENABLE_TRACING_PYTORCH=1
export SGLANG_TORCH_PROFILER_STEPS_NUM=${STEP_NUMS}
export SGLANG_TORCH_PROFILER_START_OFFSET=${STEP_OFFSET}
export SGLANG_TORCH_PROFILER_WITH_STACK=${PROFILE_WITH_STACK}
export SGLANG_TORCH_PROFILER_RECORD_SHAPE=${RECORD_SHAPE}
export SGLANG_TORCH_PROFILER_STAGE_PROFILE=0
export SGLANG_TORCH_PROFILER_PROFILE_ID="pytorch_profile_script"

TRACE_DIR=/lab-mlperf-inference/code/traces
mkdir -p ${TRACE_DIR}
export SGLANG_TORCH_PROFILER_DIR="$TRACE_DIR"
export TRITON_ENABLE_PRELOAD_KERNELS=0


bash /lab-mlperf-inference/code/run_harness.sh \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    --backend ${BACKEND} \
    harness_config.total_sample_count=${NUM_SAMPLES} \
    harness_config.device_count=${DEVICE_COUNT} \
    harness_config.duration_sec=${DURATION_SEC} \
    ${CUSTOM_ARGS}
