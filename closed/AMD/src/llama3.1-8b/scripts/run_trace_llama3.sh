#!/bin/bash

set -xeu

SCENARIO=${SCENARIO:-"server"}
GPU_NAME=${GPU_NAME:-"mi355x"}
SCENARIO_LOWER=$(echo "$SCENARIO" | tr '[:upper:]' '[:lower:]')
CONFIG_NAME="${SCENARIO_LOWER}_${GPU_NAME}"

MODEL=${MODEL:-llama3.1-8b}
NUM_SAMPLES=${NUM_SAMPLES:-13368}
DEVICE_COUNT=${DEVICE_COUNT:-8}
CONFIG_PATH="/lab-mlperf-inference/code/${MODEL}/"
DURATION_SEC=${DURATION_SEC:-1}
START_TIME=$(date +%m%d-%H%M%S)
BACKEND=${BACKEND:-'vllm'}
TRACE_FILE_NAME=${TRACE_FILE_NAME:-"trace_${MODEL}_${BACKEND}_${CONFIG_NAME}_${NUM_SAMPLES}_${START_TIME}"}
CREATE_JSON=${CREATE_JSON:-1}
ZIP_JSON=${ZIP_JSON:-1}
ENABLE_TRACING_RPD_NON_TIMED_VAR=${ENABLE_TRACING_RPD_NON_TIMED_VAR:-0}
RPDT_AUTOFLUSH_VAR=${RPDT_AUTOFLUSH_VAR:-1}
CUSTOM_ARGS=${CUSTOM_ARGS:-''}
TRACE_RANGE=${TRACE_RANGE:-''}

export ENABLE_TRACING_RPD=1
export ENABLE_TRACING_RPD_NON_TIMED=${ENABLE_TRACING_RPD_NON_TIMED_VAR}
export RPDT_FILENAME=/lab-mlperf-inference/code/trace.rpd
export RPDT_AUTOFLUSH=${RPDT_AUTOFLUSH_VAR}
LD_PRELOAD=librpd_tracer.so "$@"

TRACE_DIR=/lab-mlperf-inference/code/traces
mkdir -p ${TRACE_DIR}

rm -f ${RPDT_FILENAME}
python3 -m rocpd.schema --create ${RPDT_FILENAME}

bash /lab-mlperf-inference/code/run_harness.sh \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    --backend ${BACKEND} \
    harness_config.total_sample_count=${NUM_SAMPLES} \
    harness_config.device_count=${DEVICE_COUNT} \
    harness_config.duration_sec=${DURATION_SEC} \
    ${CUSTOM_ARGS}

mv ${RPDT_FILENAME} ${TRACE_DIR}/${TRACE_FILE_NAME}.rpd

TRACE_FILE=${TRACE_DIR}/${TRACE_FILE_NAME}.rpd

#if [[ "$CREATE_JSON" == 1 ]]; then
#    echo "Creating JSON file from trace"
    # if the json is too large, use --start --end with some percentage like --start 70% --end 75% to reduce its size
#    python3 /lab-mlperf-inference/rocm_profile_data/tools/rpd2tracing.py ${TRACE_RANGE} ${TRACE_FILE} ${TRACE_DIR}/${TRACE_FILE_NAME}.json
#    if [[ "$ZIP_JSON" == 1 ]]; then
#        zip -j ${TRACE_DIR}/${TRACE_FILE_NAME}.zip ${TRACE_DIR}/${TRACE_FILE_NAME}.json
#    fi
#fi

#ROI_START=$(sqlite3 ${TRACE_FILE} "select (start-(select MIN(rocpd_api.start) from rocpd_api)) / 1000000 from rocpd_api inner join rocpd_string on rocpd_api.apiName_id = rocpd_string.id where string='rpd_trace_mark_benchmark_start';")
#ROI_END=$(sqlite3 ${TRACE_FILE} "select (start-(select MIN(rocpd_api.start) from rocpd_api)) / 1000000 from rocpd_api inner join rocpd_string on rocpd_api.apiName_id = rocpd_string.id where string='rpd_trace_mark_benchmark_end';")
#python /lab-mlperf-inference/rocm_profile_data/raptor/raptor.py -c --roi-start ${ROI_START} --roi-end ${ROI_END} ${TRACE_FILE}

#HIP_MODULE_LOADS=$(sqlite3 ${TRACE_FILE} "select count(*) from rocpd_api inner join rocpd_string on rocpd_api.apiName_id = rocpd_string.id where string like '%HIPModuleLoad%' and start > "${ROI_START}"000000;")
#echo "Number of hipModuleLoads during benchmark: ${HIP_MODULE_LOADS}"
