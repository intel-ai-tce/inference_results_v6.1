#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
export WORKLOAD="llama3_1-8b"
export MODEL="llama3_1-8b"
export IMPL="pytorch-xpu"
export COMPLIANCE_TESTS="TEST06"
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

  if [ "${SYSTEM}" == "1-node-4x-BMG-B50" ]; then
      declare -A user_conf=( [offline-target_qps]="13.5" [server-target_qps]="9.2" [min_query_count]="13368" )
  elif [ "${SYSTEM}" == "1-node-4x-BMG-B60" ]; then
      declare -A user_conf=( [offline-target_qps]="26.0" [server-target_qps]="24.5" [min_query_count]="20000" )
  elif [ "${SYSTEM}" == "1-node-4x-BMG-B70" ]; then
      declare -A user_conf=( [offline-target_qps]="35.0" [server-target_qps]="32.4" [min_query_count]="20000" )
  elif [ "${SYSTEM}" == "1-node-8x-BMG-B60" ]; then
      declare -A user_conf=( [offline-target_qps]="50.0" [server-target_qps]="49.0" [min_query_count]="30000" )
  fi
  echo "${MODEL}.Offline.target_qps = ${user_conf[offline-target_qps]}" >> ${USER_CONF}
  echo "${MODEL}.Server.target_qps = ${user_conf[server-target_qps]}"   >> ${USER_CONF}
  echo "${MODEL}.*.min_query_count = ${user_conf[min_query_count]}"     >> ${USER_CONF}

#TODO: Add fallback case
}

workload_specific_run () {
    unset GPU_MEMORY_UTILIZATION
    SCENARIO=${SCENARIO} MODE=${MODE} RUN_LOGS=${RUN_LOGS} bash code/${MODEL}/run_llama3.1-8b.sh
}
