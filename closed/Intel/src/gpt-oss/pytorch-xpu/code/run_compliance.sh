#!/bin/bash

# Runs gpt-oss-120b compliance tests TEST07 (accuracy-in-perf) and TEST09
# (output-token-length). The released run_benchmark.sh harness only wires up
# TEST01/TEST06, so this drop-in reuses its helpers with a compliance config.

export SCENARIO="${SCENARIO:-all}"
export TESTS="${TESTS:-TEST07 TEST09}"
export DEBUG="${DEBUG:-False}"
export MODE="Compliance"

export WORKSPACE_DIR=/workspace
export DATA_DIR=/data
export MODEL_DIR=/model
export LOG_DIR=/logs

cd ${WORKSPACE_DIR}
source code/configure_workload.sh
source code/mlperf_functions.sh

##########     SUPPORT FUNCTIONS BEGIN HERE     ##########

# Compliance mirrors the PERF run config (10240 tokens, perf batch sizes), but
# TEST07 uses the GPQA set (990) and TEST09 the perf set (6396). Overrides the
# harness's workload_specific_run, which otherwise selects the accuracy config.
workload_specific_run () {
  unset GPU_MEMORY_UTILIZATION
  export MODEL_NAME="/model/gpt-oss_model/gpt-oss-120b"
  export SAMPLING_TEMPERATURE=1.0 SAMPLING_TOP_P=1.0 SAMPLING_TOP_K=-1 SAMPLING_MAX_TOKENS=10240
  export KV_CACHE_DTYPE="fp8"
  export VLLM_XPU_FP8_ALLREDUCE=0 VLLM_XPU_FP8_ALLREDUCE_SCALE=128
  export VLLM_ENABLE_DIST_SAMPLE=1 VLLM_ENABLE_DIST_SAMPLE_FP32_REDUCE=1
  export VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TP=1 PP=1 MAX_MODEL_LEN=131072
  export OUTPUT_DIR=${RUN_LOGS}

  if [ "${TEST}" == "TEST07" ]; then
    export DATASET_PATH=${DATA_DIR}/gpt-oss_data/acc/acc_eval_compliance_gpqa.parquet TOTAL_SAMPLE_COUNT=990
  else
    export DATASET_PATH=${DATA_DIR}/gpt-oss_data/perf/perf_eval_ref.parquet TOTAL_SAMPLE_COUNT=6396
  fi

  export XPU_SUPPORTED_DEVICE_IDS=(57873 57891)
  if   (( XPU_DEVICE_ID == 57891 )); then export VLLM_XPU_FP8_ALLREDUCE_MIN_TOKENS=513 TRITON_INTEL_DEVICE_ARCH=bmg  # 4xB70
  elif (( XPU_DEVICE_ID == 57873 )); then export VLLM_XPU_FP8_ALLREDUCE_MIN_TOKENS=257                               # 8xB60
  fi

  if [ "${SCENARIO}" == "Server" ]; then source code/init_env_server; else source code/init_env_offline; fi
  bash code/run_local.sh
}

# Runs one compliance test for the current SCENARIO, then verifies it.
run_compliance_test () {
  TEST=$1
  SUITE=${COMPLIANCE_SUITE_DIR}/${TEST}
  AUDIT_CONFIG=${SUITE}/${MODEL}/audit.config
  export RUN_LOGS=${WORKSPACE_DIR}/run_output

  # Match the perf sample pool to this test's dataset. TEST07's GPQA set has only
  # 990 samples; with performance_issue_unique=1, an override > dataset size makes
  # LoadGen run out of unique queries and abort before logging anything.
  SAMPLES=$([ "${TEST}" == "TEST07" ] && echo 990 || echo 6396)
  sed -i "s/^\(${MODEL}\.\*\.performance_sample_count_override\) = .*/\1 = ${SAMPLES}/" ${WORKSPACE_DIR}/user.conf

  cp ${AUDIT_CONFIG} ${WORKSPACE_DIR}/audit.config   # LoadGen auto-reads this -> compliance mode
  if [ "${DEBUG}" == "False" ] ; then bash code/run_clean.sh; fi
  rm -rf ${RUN_LOGS}; mkdir -p ${RUN_LOGS}
  workload_specific_run
  rm -f ${WORKSPACE_DIR}/audit.config

  if [ "${TEST}" == "TEST07" ]; then
    python ${SUITE}/run_verification.py -c ${RUN_LOGS} -o ${RESULTS_DIR} --audit-config ${AUDIT_CONFIG} \
      --accuracy-script "python ${WORKSPACE_DIR}/code/eval_mlperf_accuracy.py --mlperf-log {accuracy_log} \
        --reference-data ${DATASET_PATH} --tokenizer openai/gpt-oss-120b"
  else
    python ${SUITE}/run_verification.py -c ${RUN_LOGS} -o ${RESULTS_DIR} --audit-config ${AUDIT_CONFIG}
  fi
}

##########     RUN BEGINS HERE     ##########

configure_system
if [[ "$SYSTEM" == "UNSUPPORTED" ]]; then
  (IFS=','; echo "ERROR: Hardware autodetection failed or system not supported. Supported systems: ${SUPPORTED_HW[*]}")
  exit
else
  echo "AUTODETECTED: SYSTEM=${SYSTEM}"
fi

export USER_CONF=user.conf
configure_userconf

[ "${SCENARIO}" == "all" ] && SCENARIOS="Offline Server" || SCENARIOS="${SCENARIO}"
for SCENARIO in ${SCENARIOS}; do
  export SCENARIO
  export RESULTS_DIR=${LOG_DIR}/results/${SYSTEM}/${WORKLOAD}/${SCENARIO}
  mkdir -p ${RESULTS_DIR}
  if [ "${DEBUG}" == "False" ] ; then prepare_suplements; fi

  for TEST in ${TESTS}; do
    echo "Running compliance ${TEST} (${SCENARIO}) ..."
    run_compliance_test ${TEST}
  done
done
