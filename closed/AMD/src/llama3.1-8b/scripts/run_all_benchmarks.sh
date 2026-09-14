#!/bin/bash
set -x

CODE_DIR=/lab-mlperf-inference/code
SCRIPTS_DIR=${CODE_DIR}/scripts
GPU_NAME=${GPU_NAME:-'mi325x'}
RESULTS=${RESULTS:-'results'}
BACKEND=${BACKEND:-'vllm'}
MODELS=("llama2-70b;llama2;llama2-70b-99" "llama2-70b-interactive;llama2;llama2-70b-99" "mixtral-8x7b;mixtral;mixtral-8x7b" "llama3_1-405b;llama3;llama3.1-405b")

# Setup up the environment for Mixtral accuracy check
bash $SCRIPTS_DIR/setup_mixtral_accuracy_env.sh

rm -rf "$RESULTS"

for CONFIG in ${MODELS[@]};
do
    MODEL="${CONFIG%;*}"
    SHORT_MODEL=$(echo "$CONFIG" | cut -d';' -f2)
    BENCHMARK=$(echo "$CONFIG" | cut -d';' -f3)

    RESULTS_MODEL=${RESULTS}/${MODEL}
    USER_CONF_PATH="${CODE_DIR}/${BENCHMARK}/user_${GPU_NAME}.conf"
    HARNESS_ARGS="harness_config.user_conf_path=${USER_CONF_PATH} --backend=${BACKEND}"

    if [ ! -z "$NUM_SAMPLES" ]; then
        HARNESS_ARGS="${HARNESS_ARGS} harness_config.total_sample_count=$NUM_SAMPLES harness_config.duration_sec=1"
    fi

    # Offline
    ## Perf
    python $CODE_DIR/main.py \
        --config-path $CODE_DIR/${BENCHMARK}/ \
        --config-name offline_$GPU_NAME \
        test_mode=performance \
        $HARNESS_ARGS \
        harness_config.output_log_dir=${RESULTS_MODEL}/Offline/performance

    ## Accuracy
    python $CODE_DIR/main.py \
        --config-path $CODE_DIR/${BENCHMARK}/ \
        --config-name offline_$GPU_NAME \
        test_mode=accuracy \
        $HARNESS_ARGS \
        harness_config.output_log_dir=${RESULTS_MODEL}/Offline/accuracy

    bash $CODE_DIR/scripts/check_${SHORT_MODEL}_accuracy_scores.sh \
    $RESULTS/${MODEL}/Offline/accuracy/mlperf_log_accuracy.json

    # Server
    ## Perf
    python $CODE_DIR/main.py \
        --config-path $CODE_DIR/${BENCHMARK}/ \
        --config-name server_$GPU_NAME \
        test_mode=performance \
        $HARNESS_ARGS \
        harness_config.output_log_dir=${RESULTS_MODEL}/Server/performance

    ## Accuracy
    python $CODE_DIR/main.py \
        --config-path $CODE_DIR/${BENCHMARK}/ \
        --config-name server_$GPU_NAME \
        test_mode=accuracy \
        $HARNESS_ARGS \
        harness_config.output_log_dir=${RESULTS_MODEL}/Server/accuracy

    bash $CODE_DIR/scripts/check_${SHORT_MODEL}_accuracy_scores.sh \
    $RESULTS/${MODEL}/Server/accuracy/mlperf_log_accuracy.json

done
