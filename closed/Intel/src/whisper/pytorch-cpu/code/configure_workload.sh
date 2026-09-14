#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
export WORKLOAD="whisper"
export MODEL="whisper"
export IMPL="pytorch-cpu"
export COMPLIANCE_TESTS="TEST01"
export COMPLIANCE_PART3="True"
export COMPLIANCE_SUITE_DIR=${WORKSPACE_DIR}/third_party/mlperf-inference/compliance

# Hardware support varification
export SUPPORTED_HW=("1-node-4S-GNR_86C" "1-node-2S-GNR_128C" "1-node-2S-GNR_96C" "1-node-2S-GNR_86C" "1-node-1S-CWF_288C")

configure_system () {
  export NUM_CORES=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`

  if   [ "${NUM_CORES}" == "344" ]; then export SYSTEM="1-node-4S-GNR_86C"
  elif [ "${NUM_CORES}" == "256" ]; then export SYSTEM="1-node-2S-GNR_128C"
  elif [ "${NUM_CORES}" == "240" ]; then export SYSTEM="1-node-2S-GNR_120C"
  elif [ "${NUM_CORES}" == "192" ]; then export SYSTEM="1-node-2S-GNR_96C"
  elif [ "${NUM_CORES}" == "172" ]; then export SYSTEM="1-node-2S-GNR_86C"
  elif [ "${NUM_CORES}" == "120" ]; then export SYSTEM="1-node-1S-GNR_120C"
  elif [ "${NUM_CORES}" == "288" ]; then export SYSTEM="1-node-1S-CWF_288C"
  elif [ "${NUM_CORES}" == "576" ]; then export SYSTEM="1-node-2S-CWF_288C"
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
      declare -A user_conf=( [min_query_count]="24000" )
  elif [ "${SYSTEM}" == "1-node-2S-GNR_128C" ]; then
      declare -A user_conf=( [min_query_count]="18000" )
  elif [ "${SYSTEM}" == "1-node-2S-GNR_120C" ]; then
      declare -A user_conf=( [min_query_count]="17000" )
  elif [ "${SYSTEM}" == "1-node-2S-GNR_96C" ]; then
      declare -A user_conf=( [min_query_count]="14000" )
  elif [ "${SYSTEM}" == "1-node-2S-GNR_86C" ]; then
      declare -A user_conf=( [min_query_count]="12000" )
  elif [ "${SYSTEM}" == "1-node-1S-GNR_120C" ]; then
      declare -A user_conf=( [min_query_count]="9000" )
  elif [ "${SYSTEM}" == "1-node-1S-CWF_288C" ]; then
      declare -A user_conf=( [min_query_count]="4000" )
  elif [ "${SYSTEM}" == "1-node-2S-CWF_288C" ]; then
      declare -A user_conf=( [min_query_count]="10000" )
  fi
  echo "${MODEL}.*.min_query_count = ${user_conf[min_query_count]}" >> ${USER_CONF}

#TODO: Add fallback case
}

# This function should handle each combination of the following parameters:
# - SCENARIO: Offline or Server
# - MODE: Performance, Accuracy, and Compliance
workload_specific_run () {
  if [ "${MODE}" == "Compliance" ]; then
    export MODE="Performance"
  fi

  export MODEL_PATH=${MODEL_DIR}/whisper-large-v3_calibrated-cpu
  export MANIFEST_FILE="${DATA_DIR}/dev-all-repack.json"

  #export HF_HOME=${DATA_DIR}/huggingface

  export NUM_NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $NF}')
  export NUM_NODES=NUM_NUMA_NODES

  export NUM_CORES=$(($(lscpu | grep "Socket(s):" | awk '{print $2}') * $(lscpu | grep "Core(s) per socket:" | awk '{print $4}')))
  export CORES_PER_NODE=$(($NUM_CORES / $NUM_NODES))

  if [[ "${SYSTEM}" == *"CWF"* ]]; then
      export CORES_PER_INST=24
      if [ $(pip list | grep vllm | rev | cut -d' ' -f1 | rev) != "0.1.dev16146+gc1306ff.cpu" ]; then 
          source /workspace/code/prepare_cwf.sh
      fi
  elif [[ "${SYSTEM}" == *"GNR_120C"* ]]; then
      export CORES_PER_INST=8
  else
      export CORES_PER_INST=6
  fi
  export VLLM_CPU_KVCACHE_SPACE=14

  echo "CORES_PER_INST: ${CORES_PER_INST}"
  echo "VLLM_CPU_KVCACHE_SPACE: ${VLLM_CPU_KVCACHE_SPACE}"
  
  # Using NUMA nodes here to not confuse SUT
  export INSTS_PER_NODE=$(($NUM_CORES / $NUM_NUMA_NODES / CORES_PER_INST))
  export NUM_INSTS=$((${INSTS_PER_NODE} * ${NUM_NUMA_NODES}))

  export EXTRA_ARGS=""
  if [ "${MODE}" == "Accuracy" ]; then
      export EXTRA_ARGS="--accuracy"
  fi

  python code/run.py \
      --dataset_dir ${DATA_DIR} \
      --model_path ${MODEL_PATH} \
      --manifest ${MANIFEST_FILE} \
      --user_conf ${USER_CONF} \
      --scenario ${SCENARIO} \
      --num_workers ${NUM_INSTS} \
      --log_dir ${RUN_LOGS} \
      ${EXTRA_ARGS}
}
