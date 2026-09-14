#!/bin/bash

# Controls workload mode
export SCENARIO="${SCENARIO:-Offline}"
export MODE="${MODE:-Performance}"
export OFFLINE_QPS="${OFFLINE_QPS:-0}"
export SERVER_QPS="${SERVER_QPS:-0}"
export DEBUG="${DEBUG:-False}"
export MLPERF_STAGE="${MLPERF_STAGE:-False}"

# Setting standard environmental paths
export WORKSPACE_DIR=/workspace
export DATA_DIR=/data
export MODEL_DIR=/model
export LOG_DIR=/logs

##########     SUPPORT FUNCTIONS BEGIN HERE     ##########

# Initializes the system for an MLPerf run, then launches the run.
run_workload () {
  cd ${WORKSPACE_DIR}
  if [ "${DEBUG}" == "False" ] ; then bash code/run_clean.sh; fi
  if [ -f "${RUN_LOGS}" ]; then rm -r ${RUN_LOGS}; fi
  mkdir -p ${RUN_LOGS}
  workload_specific_run
}

##########     RUN BEGINS HERE     ##########

# Using workload-specific parameters from 'code/configure_workload.sh', create the submission dir structure.
source code/configure_workload.sh
configure_system
if [[ "$SYSTEM" == "UNSUPPORTED" ]]; then
  (IFS=','; echo "ERROR: Hardware autodetection failed or system not supported. Supported systems: ${SUPPORTED_HW[*]}")
  exit
else
  echo "AUTODETECTED: SYSTEM=${SYSTEM}"
fi
sleep 2

# Ensuring the user.conf file is created if auto is enabled. If disabled, checks for existing one.
export USER_CONF=user.conf
configure_userconf
if [ -f "${USER_CONF}" ]; then
  echo "LOG:::: Contents of user.conf:"
  cat ${USER_CONF}
else
  echo "ERROR::: No user.conf file found."
fi

# Begining workload runs, with Mode of: Performance, Accuracy, OR Compliance
export RUN_LOGS=${WORKSPACE_DIR}/run_output
if [[ "$MLPERF_STAGE" == "False" ]]; then
  run_workload
else
  source code/mlperf_functions.sh
  export RESULTS_DIR=${LOG_DIR}/results/${SYSTEM}/${MODEL}/${SCENARIO}
  mkdir -p ${RESULTS_DIR}

  # Creates the non-runtime submission content (src, systems, documents)
  if [ "${DEBUG}" == "False" ] ; then prepare_suplements; fi

  run_mlperf
fi
