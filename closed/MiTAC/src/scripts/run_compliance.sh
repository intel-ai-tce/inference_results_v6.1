#!/bin/bash
set -xeu


MODEL=${MODEL:-llama3.1-405b}
BACKEND=${BACKEND:-'vllm'}
CODE_DIR=/lab-mlperf-inference/code
GPU_COUNT=${GPU_COUNT:-8}
GPU=$(bash $CODE_DIR/scripts/determine_accelerator.sh)
TEST=${TEST_VERSION:-"TEST06"}
TEST_DIR=/lab-mlperf-inference/mlperf_inference/compliance/$TEST
USER_CONF_PATH="$CODE_DIR/$MODEL/user_${GPU}.conf"

run_compliance() {
    local scenario="$1"

    python $CODE_DIR/main.py \
        --config-path $CODE_DIR/$MODEL/ \
        --config-name ${scenario}_${GPU} \
        --backend ${BACKEND} \
        test_mode=performance \
        harness_config.device_count=${GPU_COUNT} \
        harness_config.user_conf_path=${USER_CONF_PATH} \
        harness_config.output_log_dir=$CODE_DIR/results/$MODEL/${scenario^}/audit/compliance/$TEST

    python $TEST_DIR/run_verification.py \
        -c $CODE_DIR/results/$MODEL/${scenario^}/audit/compliance/$TEST \
        -o $CODE_DIR/results/$MODEL/${scenario^}/audit/compliance \
        -s ${scenario^}
}

cp $TEST_DIR/audit.config ./

run_compliance "offline"
run_compliance "server"

rm audit.config
