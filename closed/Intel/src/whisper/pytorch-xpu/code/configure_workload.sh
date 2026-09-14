#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
export WORKLOAD="whisper"
export MODEL="whisper"
export IMPL="pytorch-xpu"
export COMPLIANCE_TESTS="TEST01"
export COMPLIANCE_SUITE_DIR=${WORKSPACE_DIR}/third_party/mlperf-inference/compliance

# Hardware support varification
export SUPPORTED_HW=("1-node-4x-BMG-B50" "1-node-4x-BMG-B60" "1-node-8x-BMG-B60" "1-node-4x-BMG-B70")

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

  if [ "${SYSTEM}" == "1-node-4x-BMG-B70" ]; then
      declare -A user_conf=( [min_query_count]="195960" )
  elif [ "${SYSTEM}" == "1-node-4x-BMG-B60" ]; then
      declare -A user_conf=( [min_query_count]="130640" )
  elif [ "${SYSTEM}" == "1-node-4x-BMG-B50" ]; then
      declare -A user_conf=( [min_query_count]="52256" )
  elif [ "${SYSTEM}" == "1-node-8x-BMG-B60" ]; then
      declare -A user_conf=( [min_query_count]="261280" )
  fi

  echo "${MODEL}.Offline.min_query_count = ${user_conf[min_query_count]}" >> ${USER_CONF}
}

# This function should handle each combination of the following parameters:
# - SCENARIO: Offline
# - MODE: Performance, Accuracy, and Compliance
workload_specific_run () {
  export MODEL_DIR=${MODEL_DIR}/whisper-large-v3_calibrated-xpu
  export MANIFEST_FILE=${DATA_DIR}/dev-all-repack.json
  export NUM_CORES=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
  export VLLM_USE_V1=1
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=2
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export ONEDNN_VERBOSE=0
  export USE_PRIMITIVE_CACHE=ON
  export VLLM_XPU_USE_W4A8=1
  export VLLM_FUSE_QUANT=1
  export VLLM_USE_SPLIT_XPU_ATTN=1
  export VLLM_USE_BATCHED_ENCODE=1
  if [ "${SYSTEM}" == "1-node-${XPU_COUNT}x-BMG-B70" ]; then
      export BATCH_SIZE=192
      export MAX_NUM_BATCHED_TOKENS=175488
      export ENCODER_BATCH_SIZE=16
  elif [ "${SYSTEM}" == "1-node-${XPU_COUNT}x-BMG-B60" ]; then
      export BATCH_SIZE=144
      export MAX_NUM_BATCHED_TOKENS=128960
      export ENCODER_BATCH_SIZE=12
  elif [ "${SYSTEM}" == "1-node-${XPU_COUNT}x-BMG-B50" ]; then
      export BATCH_SIZE=92
      export MAX_NUM_BATCHED_TOKENS=83520
      export ENCODER_BATCH_SIZE=8
  else
      export BATCH_SIZE=104
      export MAX_NUM_BATCHED_TOKENS=94768
      export ENCODER_BATCH_SIZE=8
  fi
  # export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
  # export CCL_ZE_IPC_EXCHANGE=drmfd
  if [ "${MODE}" == "Accuracy" ]; then
      export EXTRA_ARGS="--accuracy"
  else
      export EXTRA_ARGS=""
  fi
  export PBAR=1
  python code/main.py \
      --dataset_dir ${DATA_DIR} \
      --model_path ${MODEL_DIR} \
      --manifest ${MANIFEST_FILE} \
      --scenario Offline \
      --log_dir ${RUN_LOGS} \
      --num_workers ${XPU_COUNT} \
      ${EXTRA_ARGS}
}
