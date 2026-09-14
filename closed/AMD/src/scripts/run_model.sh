#!/bin/bash
set -u

CODE_DIR=/lab-mlperf-inference/code
SCRIPTS_DIR=${CODE_DIR}/scripts
GPU_NAME=${GPU_NAME:-'mi355x'}
RESULTS=${RESULTS:-'results'}
SCENARIO=${SCENARIO:-'Offline'}
PERFORMANCE=${PERFORMANCE:-'1'}
ACCURACY=${ACCURACY:-'1'}
PERFORMANCE_ITERATION=${PERFORMANCE_ITERATION:-1}
ACCURACY_ITERATION=${ACCURACY_ITERATION:-1}
BACKEND=${BACKEND:-'vllm'}
RUN_TYPE=${RUN_TYPE:-'test'}
MOE_ACCURACY_ENV=${CODE_DIR}/moe_accuracy_venv
LLAMA2_ACCURACY_ENV=${CODE_DIR}/llama2_accuracy_venv
RUN_HEALTHCHECK=${RUN_HEALTHCHECK:-0}
CUSTOM_ARGS=${CUSTOM_ARGS:-''}
CLEAR_RESULTS=${CLEAR_RESULTS:-'1'}

# Names of the accepted models
LLAMA2=llama2-70b-99
GPTOSS=gpt-oss-120b
MODELS=("$LLAMA2" "$GPTOSS")
ACCEPTED_MODELS=",$LLAMA2,$GPTOSS,"

# Check cmdline argument
if [ "$#" -eq 0 ]; then
    echo "Must provide at least one model name. Options are: ${MODELS[*]}"
    exit 1
fi

MODEL=$1

if [[ ,$ACCEPTED_MODELS, != *,$MODEL,* ]]; then
    echo "Benchmark must be one of ${MODELS[*]}"
    exit 1
fi

SHORT_MODEL=$(echo "$MODEL" | cut -d'-' -f1)
USER_CONF_PATH=${CODE_DIR}/${MODEL}/user_${GPU_NAME}.conf
RESULTS_MODEL=${RESULTS}/${MODEL}

# Remove existing results
if [[ "$CLEAR_RESULTS" == "1" ]]; then
    rm -rf ${RESULTS_MODEL}
fi

# Print info about the environment
dump_environment_info() {
    echo "Collect environment info"
    SYSTEM_INFO_DIR=${RESULTS}/system_info
    mkdir -p ${SYSTEM_INFO_DIR}
    HIPBLASLT_VERSION=$(apt show hipblaslt | awk 'NR==2' | awk '{print $2}')
    echo "$HIPBLASLT_VERSION" > ${SYSTEM_INFO_DIR}/hipblaslt_commit.txt
    echo "$RUN_TYPE" > ${SYSTEM_INFO_DIR}/run_type.txt

    if [[ "$BACKEND" == "vllm" ]]; then
        VLLM_VERSION=$(pip show vllm | grep Version | awk 'NR==1' | awk '{print $2}')
        echo "$VLLM_VERSION" > ${SYSTEM_INFO_DIR}/vllm_commit.txt
    elif [[ "$BACKEND" == "sglang" ]]; then
        SGLANG_VERSION=$(pip show sglang | grep Version | awk '{print $2}')
        echo "$SGLANG_VERSION" > ${SYSTEM_INFO_DIR}/sglang_commit.txt
    fi
}

HARNESS_ARGS="harness_config.user_conf_path=${USER_CONF_PATH} harness_config.resource_checker_abort_on_failure=True"

if [ ! -z "$CUSTOM_ARGS" ]; then
    HARNESS_ARGS="${HARNESS_ARGS} ${CUSTOM_ARGS}"
fi

# This should catch both "sglang_engine_config.model_path=..." and "vllm_engine_config.model=..."
is_model_changed=$(echo "$HARNESS_ARGS" | grep -oP 'engine_config\.model(?:_path)?=\K[^\s]+')
if [ -n "$is_model_changed" ]; then
    echo "Skipping resource checker abort on failure due to custom model path."
    HARNESS_ARGS="${HARNESS_ARGS/harness_config.resource_checker_abort_on_failure=True/harness_config.resource_checker_abort_on_failure=False}"
fi

SCENARIO_LOWER=$(echo "$SCENARIO" | tr '[:upper:]' '[:lower:]')

# Custom parameters for gpt-oss-120b
if [[ "$MODEL" == "$GPTOSS" ]]; then
    SHORT_MODEL="gptoss"
fi

if [[ "$PERFORMANCE" == "1" ]]; then
    for ((i=1; i<=PERFORMANCE_ITERATION; i++)); do
        echo "Performance run $i/$PERFORMANCE_ITERATION"
        bash $CODE_DIR/run_harness.sh \
            --config-path $CODE_DIR/${MODEL}/ \
            --config-name ${SCENARIO_LOWER}_${GPU_NAME} \
            --backend ${BACKEND} \
            test_mode=performance \
            $HARNESS_ARGS \
            harness_config.output_log_dir=${RESULTS_MODEL}/${SCENARIO}/performance_${i} || exit $?
    done
fi # PERFORMANCE

if [[ "$ACCURACY" == "1" ]]; then
    is_sample_count_changed=$(echo "$HARNESS_ARGS" | grep -oP '(?<=harness_config\.total_sample_count=)[^\s]+')
    is_duration_changed=$(echo "$HARNESS_ARGS" | grep -oP '(?<=harness_config\.duration_sec=)[^\s]+')

    if [ -n "$is_sample_count_changed" ] || [ -n "$is_duration_changed" ]; then
        echo "Sample or duration is overwritten, skipping accuracy..."
    else
        # Setup up the environment for model-specific accuracy computations
        if [[ "$MODEL" == "$LLAMA2" ]] && [ ! -d ${LLAMA2_ACCURACY_ENV} ]; then
            bash $SCRIPTS_DIR/setup_llama2_accuracy_env.sh || exit $?
        fi

        for ((i=1; i<=ACCURACY_ITERATION; i++)); do
            echo "Accuracy run $i/$ACCURACY_ITERATION"
            bash $CODE_DIR/run_harness.sh \
                --config-path $CODE_DIR/${MODEL}/ \
                --config-name ${SCENARIO_LOWER}_${GPU_NAME} \
                --backend ${BACKEND} \
                test_mode=accuracy \
                $HARNESS_ARGS \
                harness_config.output_log_dir=${RESULTS_MODEL}/${SCENARIO}/accuracy_${i} || exit $?

            MODEL_OVERRIDE_PATH=$is_model_changed bash $SCRIPTS_DIR/check_${SHORT_MODEL}_accuracy_scores.sh \
                ${RESULTS_MODEL}/${SCENARIO}/accuracy_${i}/mlperf_log_accuracy.json || exit $?
        done
    fi
fi # ACCURACY

dump_environment_info
