#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
export WORKLOAD="gpt-oss-120b"
export MODEL="gpt-oss-120b"
export IMPL="pytorch-xpu"
export COMPLIANCE_TESTS="TEST06"
export COMPLIANCE_SUITE_DIR=${WORKSPACE_DIR}/third_party/mlperf-inference/compliance

# Hardware support varification
export SUPPORTED_HW=("1-node-4x-BMG-B60" "1-node-4x-BMG-B70")

configure_system () {
  export XPU_COUNT=$(python -c "import torch; count = 0; count = torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0; print(count)")
  export XPU_DEVICE_ID=$(python -c "import torch; print(torch.xpu.get_device_properties(0).device_id)")

  if (( XPU_DEVICE_ID == 57891 )); then   export SYSTEM="1-node-${XPU_COUNT}x-BMG-B70"
  elif (( XPU_DEVICE_ID == 57873 )); then export SYSTEM="1-node-${XPU_COUNT}x-BMG-B60"
  elif (( XPU_DEVICE_ID == 57874 )); then export SYSTEM="1-node-${XPU_COUNT}x-BMG-B50"
  else export SYSTEM="UNSUPPORTED"
  fi

  if [[ ! " ${SUPPORTED_HW[*]} " =~ " ${SYSTEM} " ]]; then
      export SYSTEM="UNSUPPORTED"
  fi
}

# Creates the default user.conf file, either auto-selected, modified, or newly generated.
configure_userconf () {
  cd ${WORKSPACE_DIR}
  # Ensure no left-over user.conf files from previous runs, and use pre-configured SYSTEM file if available.
  if [ -f "${USER_CONF}" ]; then rm ${USER_CONF}; fi

  if [ "${MODEL}" == "gpt-oss-120b" ]; then
      if [ "${SYSTEM}" == "1-node-4x-BMG-B60" ]; then
          declare -A user_conf=( [offline-target_qps]="1.00" [server-target_qps]="0.33" )
      elif [ "${SYSTEM}" == "1-node-4x-BMG-B70" ]; then
          declare -A user_conf=( [offline-target_qps]="1.00" [server-target_qps]="0.95" )
      elif [ "${SYSTEM}" == "1-node-8x-BMG-B60" ]; then
          declare -A user_conf=( [offline-target_qps]="1.00" [server-target_qps]="0.90" )
      fi
  fi
  echo "${MODEL}.*.performance_sample_count_override = 6396" >> ${USER_CONF}
  echo "${MODEL}.*.accuracy_sample_count_override = 4395" >> ${USER_CONF}

  if [[ "$MODE" == "Compliance" ]]; then
      echo "${MODEL}.Offline.min_query_count = 990" >> ${USER_CONF}
  else
      echo "${MODEL}.Offline.min_query_count = 6396" >> ${USER_CONF}
  fi
  
  echo "${MODEL}.Offline.target_qps = ${user_conf[offline-target_qps]}" >> ${USER_CONF}
  echo "${MODEL}.Server.target_qps = ${user_conf[server-target_qps]}"   >> ${USER_CONF}

#TODO: Add fallback case
}

workload_specific_run () {
    unset GPU_MEMORY_UTILIZATION
    export XPU_COUNT=$(python -c "import torch; count = 0; count = torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0; print(count)")
    export MODEL_NAME="/model/gpt-oss_model/gpt-oss-120b"

    export SAMPLING_TEMPERATURE=1.0
    export SAMPLING_TOP_P=1.0
    export SAMPLING_TOP_K=-1
    export KV_CACHE_DTYPE="fp8"

    export VLLM_XPU_FP8_ALLREDUCE=0
    export VLLM_XPU_FP8_ALLREDUCE_SCALE=128

    export VLLM_ENABLE_DIST_SAMPLE=1              # vocab-parallel Gumbel-max (default off)
    export VLLM_ENABLE_DIST_SAMPLE_FP32_REDUCE=1  # fp32 SUM reduce (3.5x faster than int64 MAX)

    export OUTPUT_DIR=${RUN_LOGS}

    if [[ "$MODE" == "Performance" ]]; then
        export DATASET_PATH="/data/gpt-oss_data/perf/perf_eval_ref.parquet"
        export TOTAL_SAMPLE_COUNT=6396
        export SAMPLING_MAX_TOKENS=10240
    else
        export DATASET_PATH="/data/gpt-oss_data/acc/acc_eval_ref.parquet"
        export TOTAL_SAMPLE_COUNT=4395
        export SAMPLING_MAX_TOKENS=32768
    fi

    export VLLM_USE_V1=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export TP=1
    export PP=1

    export MAX_MODEL_LEN=131072
    
    # B60, B70 supported
    export XPU_SUPPORTED_DEVICE_IDS=(57873 57891)
   
    # Per-device settings
    if (( XPU_DEVICE_ID == 57891 )); then
        # 4xB70
        export VLLM_XPU_FP8_ALLREDUCE_MIN_TOKENS=513
        export TRITON_INTEL_DEVICE_ARCH=bmg  # triton-xpu 3.6.0
    elif (( XPU_DEVICE_ID == 57873 )); then
        # 8xB60
        export VLLM_XPU_FP8_ALLREDUCE_MIN_TOKENS=257
    fi

    if [ "${SCENARIO}" == "Server" ]; then
        source /workspace/code/init_env_server
    else
        source /workspace/code/init_env_offline
    fi

    bash code/run_local.sh
}
