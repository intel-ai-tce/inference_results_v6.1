# Places the standard MLPerf run log outputs to the specified final dir.
stage_logs () {
  OUTPUT_PATH=$1
  cd ${RUN_LOGS}
  mkdir -p ${OUTPUT_PATH}
  mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_PATH}/
  if [ -f accuracy.txt ]; then mv accuracy.txt ${OUTPUT_PATH}/; fi
}

prepare_suplements () {
  DOCUMENTATION_DIR=${LOG_DIR}/documentation
  SRC_DIR=${LOG_DIR}/src/${MODEL}/${IMPL}
  SYSTEMS_DIR=${LOG_DIR}/systems

  # Populate /logs/documentation directory
  mkdir -p ${DOCUMENTATION_DIR}
  cp ${WORKSPACE_DIR}/code/calibration.md ${DOCUMENTATION_DIR}/

  # Populate /logs/src directory
  mkdir -p ${SRC_DIR}
  cp -r ${WORKSPACE_DIR}/README.md ${SRC_DIR}/

  mkdir -p ${SYSTEMS_DIR}
  prepare_system_json ${SYSTEMS_DIR}

  mkdir -p ${RESULTS_DIR}
  cp ${WORKSPACE_DIR}/README.md ${RESULTS_DIR}/
  cp ${WORKSPACE_DIR}/user.conf ${RESULTS_DIR}/
  cp ${WORKSPACE_DIR}/code/measurements.json ${RESULTS_DIR}/
}

prepare_system_json () {
  OUTPUT_DIR=$1
  OUTPUT_FILE=${OUTPUT_DIR}/${SYSTEM}.json

  [ -f ${OUTPUT_FILE} ] && rm ${OUTPUT_FILE}

  HOST_PROCESSOR_MODEL_NAME=$(lscpu | grep "Model name:" | grep -v "BIOS" | cut -d':' -f2 | xargs)
  HOST_PROCESSORS_PER_NODE=$(lscpu | grep "Socket(s):" | awk '{print $2}')
  HOST_PROCESSOR_CORE_COUNT=$(lscpu | grep "Core(s) per socket:" | cut -d':' -f2 | xargs)
  HOST_MEMORY_CAPACITY=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) / 1024 / 1024 ))
  HOST_MEMORY_CONFIGURATION=$(echo "$(dmidecode --type 17 | grep 'Type:' | head -1 | awk '{print $NF}') \
          $(dmidecode --type 17 | grep 'MT/s' | head -1 | awk '{print $(NF-1), $NF}')" | xargs)
  HOST_NETWORKING=$(lspci | grep -i "Ethernet controller:" | rev | cut -d':' -f1 | rev | xargs)
  OPERATING_SYSTEM=$(grep "PRETTY_NAME" /etc/os-release | sed -n 's/.*"\([^"]*\)".*/\1/p')

  if   [ $(python -c "import torch; print(torch.xpu.is_available())") == "True" ]; then
      ACCELERATORS_PER_NODE=$(python -c "import torch; count = 0; count = torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0; print(count)")
      XPU_DEVICE_ID=$(python -c "import torch; print(torch.xpu.get_device_properties(0).device_id)")
      if (( XPU_DEVICE_ID == 57891 )); then   ACCELERATOR_MODEL_NAME="Intel Arc Pro B70"; ACCELERATOR_FREQUENCY="2800 MHz"; ACC_MEMORY_CAPACITY="32GB"; ACC_MEMORY_CONFIGURATION="GDDR6";
      elif (( XPU_DEVICE_ID == 57873 )); then ACCELERATOR_MODEL_NAME="Intel Arc Pro B60"; ACCELERATOR_FREQUENCY="2400 MHz"; ACC_MEMORY_CAPACITY="24GB"; ACC_MEMORY_CONFIGURATION="GDDR6";
      elif (( XPU_DEVICE_ID == 57874 )); then ACCELERATOR_MODEL_NAME="Intel Arc Pro B70"; ACCELERATOR_FREQUENCY="2600 MHz"; ACC_MEMORY_CAPACITY="16GB"; ACC_MEMORY_CONFIGURATION="GDDR6";
      else                                    ACCELERATOR_MODEL_NAME="N/A";               ACCELERATOR_FREQUENCY="N/A";      ACC_MEMORY_CAPACITY="N/A";  ACC_MEMORY_CONFIGURATION="N/A";
      fi
  else
      ACCELERATORS_PER_NODE="0"; ACCELERATOR_MODEL_NAME="N/A"; ACCELERATOR_FREQUENCY="N/A"; ACC_MEMORY_CAPACITY="N/A"; ACC_MEMORY_CONFIGURATION="N/A"
  fi

  echo "{"                                                                        >> ${OUTPUT_FILE}
  echo "  \"division\": \"closed\","                                              >> ${OUTPUT_FILE}
  echo "  \"submitter\": \"OEM\","                                                >> ${OUTPUT_FILE}
  echo "  \"status\": \"available\","                                             >> ${OUTPUT_FILE}
  echo "  \"system_type\":\"datacenter\","                                        >> ${OUTPUT_FILE}
  echo "  \"system_type_detail\":\"\","                                           >> ${OUTPUT_FILE}
  echo "  \"system_name\": \"${SYSTEM}\","                                        >> ${OUTPUT_FILE}
  echo "  \"number_of_nodes\": \"1\","                                            >> ${OUTPUT_FILE}
  echo "  \"host_processor_model_name\": \"${HOST_PROCESSOR_MODEL_NAME}\","       >> ${OUTPUT_FILE}
  echo "  \"host_processors_per_node\": \"${HOST_PROCESSORS_PER_NODE}\","         >> ${OUTPUT_FILE}
  echo "  \"host_processor_core_count\": \"${HOST_PROCESSOR_CORE_COUNT}\","       >> ${OUTPUT_FILE}
  echo "  \"host_processor_frequency\": \"\","                                    >> ${OUTPUT_FILE}
  echo "  \"host_processor_caches\": \"\","                                       >> ${OUTPUT_FILE}
  echo "  \"host_memory_configuration\": \"${HOST_MEMORY_CONFIGURATION}\","       >> ${OUTPUT_FILE}
  echo "  \"host_memory_capacity\": \"${HOST_MEMORY_CAPACITY} GB\","              >> ${OUTPUT_FILE}
  echo "  \"host_storage_capacity\": \"N/A\","                                    >> ${OUTPUT_FILE}
  echo "  \"host_storage_type\": \"SSD\","                                        >> ${OUTPUT_FILE}
  echo "  \"host_processor_interconnect\": \"\","                                 >> ${OUTPUT_FILE}
  echo "  \"host_networking\": \"${HOST_NETWORKING}\","                           >> ${OUTPUT_FILE}
  echo "  \"host_networking_topology\": \"N/A\","                                 >> ${OUTPUT_FILE}
  echo "  \"host_network_card_count\": \"1\","                                    >> ${OUTPUT_FILE}
  echo "  \"accelerators_per_node\": \"${ACCELERATORS_PER_NODE}\","               >> ${OUTPUT_FILE}
  echo "  \"accelerator_model_name\": \"${ACCELERATOR_MODEL_NAME}\","             >> ${OUTPUT_FILE}
  echo "  \"accelerator_frequency\": \"${ACCELERATOR_FREQUENCY}\","               >> ${OUTPUT_FILE}
  echo "  \"accelerator_host_interconnect\": \"N/A\","                            >> ${OUTPUT_FILE}
  echo "  \"accelerator_interconnect\": \"N/A\","                                 >> ${OUTPUT_FILE}
  echo "  \"accelerator_interconnect_topology\": \"\","                           >> ${OUTPUT_FILE}
  echo "  \"accelerator_memory_capacity\": \"${ACC_MEMORY_CAPACITY}\","           >> ${OUTPUT_FILE}
  echo "  \"accelerator_memory_configuration\": \"${ACC_MEMORY_CONFIGURATION}\"," >> ${OUTPUT_FILE}
  echo "  \"accelerator_on-chip_memories\": \"\","                                >> ${OUTPUT_FILE}
  echo "  \"cooling\": \"Air\","                                                  >> ${OUTPUT_FILE}
  echo "  \"hw_notes\": \"\","                                                    >> ${OUTPUT_FILE}
  echo "  \"framework\": \"PyTorch\","                                            >> ${OUTPUT_FILE}
  echo "  \"operating_system\": \"${OPERATING_SYSTEM}\","                         >> ${OUTPUT_FILE}
  echo "  \"other_software_stack\": \"Docker\","                                  >> ${OUTPUT_FILE}
  echo "  \"sw_notes\": \"N/A\""                                                  >> ${OUTPUT_FILE}
  echo "}"                                                                        >> ${OUTPUT_FILE}
}

# Runs benchmark using the appropriate MODE path 
run_mlperf () {
  # Begining workload runs, with Mode of: Performance, Accuracy, OR Compliance
  if [ "${MODE}" == "Performance" ]; then
      run_workload
      stage_logs "${RESULTS_DIR}/performance/run_1"
  elif [ "${MODE}" == "Accuracy" ]; then
      run_workload
      stage_logs "${RESULTS_DIR}/accuracy"
  elif [ "${MODE}" == "Compliance" ]; then
      for TEST in ${COMPLIANCE_TESTS}; do
          echo "Running compliance ${TEST} ..."

          if [ -f ${WORKSPACE_DIR}/audit.config ]; then rm ${WORKSPACE_DIR}/audit.config; fi
	  if ! [ -d ${RESULTS_DIR} ]; then
              echo "[ERROR] Compliance run could not be verified due to unspecified or non-existant RESULTS_DIR: ${RESULTS_DIR}"
              exit
          fi
          OUTPUT_PATH=${RUN_LOGS}

          if [ "${TEST}" == "TEST01" ]; then
              cp ${COMPLIANCE_SUITE_DIR}/${TEST}/${MODEL}/audit.config .
	      run_workload
	      python ${COMPLIANCE_SUITE_DIR}/${TEST}/run_verification.py -r ${RESULTS_DIR} -c ${OUTPUT_PATH} -o ${RESULTS_DIR} --dtype int64
          elif [ "${TEST}" == "TEST06" ]; then
              cp ${COMPLIANCE_SUITE_DIR}/${TEST}/audit.config .
	      run_workload
	      python ${COMPLIANCE_SUITE_DIR}/${TEST}/run_verification.py -s ${SCENARIO} -c ${OUTPUT_PATH} -o ${RESULTS_DIR} -d int64
          else
              echo "[ERROR] Compliance test specified is not valid: ${TEST}"
              exit
	  fi

	  if [ "${COMPLIANCE_PART3}" == "True" ]; then
              cd ${OUTPUT_PATH}
              bash ${COMPLIANCE_SUITE_DIR}/${TEST}/create_accuracy_baseline.sh ${RESULTS_DIR}/accuracy/mlperf_log_accuracy.json ${OUTPUT_PATH}/mlperf_log_accuracy.json
              python ${WORKSPACE_DIR}/code/accuracy_eval.py --log_dir ${OUTPUT_PATH} --manifest /data/dev-all-repack.json --acc_log mlperf_log_accuracy.json > ${RESULTS_DIR}/${TEST}/accuracy/accuracy.txt
              python ${WORKSPACE_DIR}/code/accuracy_eval.py --log_dir ${OUTPUT_PATH} --manifest /data/dev-all-repack.json --acc_log mlperf_log_accuracy.json > ${RESULTS_DIR}/${TEST}/accuracy/compliance_accuracy.txt
              python ${WORKSPACE_DIR}/code/accuracy_eval.py --log_dir ${OUTPUT_PATH} --manifest /data/dev-all-repack.json --acc_log mlperf_log_accuracy_baseline.json > ${RESULTS_DIR}/${TEST}/accuracy/baseline_accuracy.txt
	  fi
      done
  else
      echo "[ERROR] Missing value for MODE. Options: Performance, Accuracy, Compliance"
  fi
}
