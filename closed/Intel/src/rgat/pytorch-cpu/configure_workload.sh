#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
export WORKLOAD="rgat"
export MODEL="rgat"
export IMPL="pytorch-cpu"
export COMPLIANCE_TESTS="TEST01"
export COMPLIANCE_SUITE_DIR=${WORKSPACE_DIR}/third_party/mlperf-inference/compliance
export MAX_LATENCY=10000000000

# This function should handle each combination of the following parameters:
# - SCENARIO: Offline or Server
# - MODE: Performance, Accuracy, and Compliance
workload_specific_run () {
  export SCENARIO=${SCENARIO}
  export MODE=${MODE}

  # Standard ENV settings (potentially redundant)
  export MODEL_DIR=${MODEL_DIR}
  export DATA_DIR=${DATA_DIR}
  export USER_CONF=${USER_CONF}
  export RUN_LOGS=${RUN_LOGS}

  NUM_CORES=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
  NUM_NUMA=$(numactl --hardware|grep available|awk -F' ' '{ print $2 }')
  NUM_SOCKETS=$(lscpu | grep "Socket(s)" | rev | cut -d' ' -f1 | rev)
  MEM_AVAILABLE=$(free -g | awk '/Mem:/ {print $2}')

  if [ $MEM_AVAILABLE -lt 1500 ]; then
      echo "Memory is less than 1500GB, setting NUMA balance"
      export NUM_PROC=1
      export CPUS_PER_PROC=${NUM_CORES}
      export WORKERS_PER_PROC=${NUM_SOCKETS}
      export CORE_OFFSET="[0,$((NUM_CORES / 2))]" # first core of each socket
  else
      echo "Memory is greater than 1500GB"

      echo "Detecting first core for each NUMA node..."
      # Get list of NUMA nodes (using sort -un to get unique values)

      # Create an array to store first cores
      declare -a first_cores

      # For each NUMA node, find its first core
      NUMA_NODES=$(numactl --hardware | grep -oP "node \K\d+" | sort -un)
      for node in ${NUMA_NODES}; do
      # Get first physical core ID for this NUMA node
      first_core=$(lscpu -p=CPU,NODE | grep -v '^#' | awk -v node="$node" -F',' '$2 == node {print $1; exit}')
      first_cores+=($first_core)
      echo "NUMA node $node: first core is $first_core"
      done

      # Format the array for use in your script
      core_offset="["
      for i in "${!first_cores[@]}"; do
      if [ $i -gt 0 ]; then
          core_offset+=","
      fi
      core_offset+="${first_cores[$i]}"
      done
      core_offset+="]"
      echo "Core offset array: $core_offset"

      # Set configuration variables
      export NUM_PROC=${NUM_SOCKETS} # 1 process per socket
      export CPUS_PER_PROC=$((NUM_CORES / NUM_SOCKETS)) # #cores per socket
      export WORKERS_PER_PROC=$((NUM_NUMA / NUM_SOCKETS)) # 1 worker per NUMA node
      export CORE_OFFSET=$core_offset # first core of each NUMA node
  fi

  echo "CORE_OFFSET: ${CORE_OFFSET}"
  echo "NUM_PROC: ${NUM_PROC}"
  echo "WORKERS_PER_PROC: ${WORKERS_PER_PROC}"

  export TMP_DIR=${RUN_LOGS}
  if [ "${MODE}" == "Accuracy" ]; then
      echo "Run ${MODEL} (${SCENARIO} Accuracy)."
      bash run_accuracy.sh
  else
      echo "Run ${MODEL} (${SCENARIO} Performance)."
      bash run_offline.sh
  fi
}
