#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
export WORKLOAD="llama3_1-8b"
export MODEL="llama3_1-8b"
export IMPL="pytorch-cpu"
export COMPLIANCE_TESTS="TEST06"
export COMPLIANCE_SUITE_DIR=${WORKSPACE_DIR}/third_party/mlperf-inference/compliance

# Hardware support varification
export SUPPORTED_HW=("1-node-4S-GNR_86C" "1-node-2S-GNR_128C" "1-node-2S-GNR_96C" "1-node-2S-GNR_86C" "1-node-1S-CWF_288C" "1-node-2S-CWF_288C")

configure_system () {
  export NUM_CORES=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
 
  if   [ "${NUM_CORES}" == "344" ]; then export SYSTEM="1-node-4S-GNR_86C"
  elif [ "${NUM_CORES}" == "256" ]; then export SYSTEM="1-node-2S-GNR_128C"
  elif [ "${NUM_CORES}" == "240" ]; then export SYSTEM="1-node-2S-GNR_120C"
  elif [ "${NUM_CORES}" == "192" ]; then export SYSTEM="1-node-2S-GNR_96C"
  elif [ "${NUM_CORES}" == "172" ]; then export SYSTEM="1-node-2S-GNR_86C"
  elif [ "${NUM_CORES}" == "120" ]; then export SYSTEM="1-node-1S-GNR_120C"
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

  if [ "${SYSTEM}" == "1-node-4S-GNR_86C" ]; then
      declare -A user_conf=( [server-target_qps]="10.75" [min_query_count]="13368" )
  elif [ "${SYSTEM}" == "1-node-2S-GNR_128C" ]; then
      declare -A user_conf=( [server-target_qps]="8.8" [min_query_count]="13368" )
  elif [ "${SYSTEM}" == "1-node-2S-GNR_120C" ]; then
      declare -A user_conf=( [server-target_qps]="7.5" [min_query_count]="13368" )
  elif [ "${SYSTEM}" == "1-node-2S-GNR_96C" ]; then
      declare -A user_conf=( [server-target_qps]="4.5" [min_query_count]="13368" )
  elif [ "${SYSTEM}" == "1-node-2S-GNR_86C" ]; then
      declare -A user_conf=( [server-target_qps]="4.0" [min_query_count]="13368" )
  elif [ "${SYSTEM}" == "1-node-1S-GNR_120C" ]; then
      declare -A user_conf=( [server-target_qps]="3.7" [min_query_count]="13368" )
  fi
  echo "${MODEL}.Server.target_qps = ${user_conf[server-target_qps]}"   >> ${USER_CONF}
  echo "${MODEL}.*.min_query_count = ${user_conf[min_query_count]}"     >> ${USER_CONF}

#TODO: Add fallback case
}

# This function should handle each combination of the following parameters:
# - SCENARIO: Offline or Server
# - MODE: Performance, Accuracy, and Compliance
workload_specific_run () {
  if [ "${MODE}" == "Compliance" ]; then
    export MODE="Performance"
  fi

  echo "Beginning run: SCENARIO=${SCENARIO} MODE=${MODE}"
  SCENARIO=${SCENARIO} MODE=${MODE} bash code/run_local.sh
}
