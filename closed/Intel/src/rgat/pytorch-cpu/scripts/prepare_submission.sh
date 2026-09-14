#!/bin/bash

export LOG_DIR=/logs
export SUBMISSION_DIR=${LOG_DIR}/submission-$(date +%s)
export SUBMISSION_ORIGINAL=${SUBMISSION_DIR}/original
export SUBMISSION_PROCESSED=${SUBMISSION_DIR}/processed

RESULTS_DIR=${LOG_DIR}/results
SYSTEMS_DIR=${LOG_DIR}/systems
echo "Ensuring correct system directories and files match system ${SYSTEM}."
echo "The fillowing are expected:"
echo "- RESULTS: ${RESULTS_DIR}/${SYSTEM}"
echo "- SYSTEM FILE: ${SYSTEMS_DIR}/${SYSTEM}.json"
if ! [ -d ${RESULTS_DIR}/${SYSTEM} ];      then echo "[ERROR] RESULTS_DIR not found: ${RESULTS_DIR}/${SYSTEM}";           exit; fi
if ! [ -f ${SYSTEMS_DIR}/${SYSTEM}.json ]; then echo "[ERROR] SYSTEM file not found: ${SYSTEMS_DIR}/${SYSTEM}.json";      exit; fi

echo "Ensuring all scenarios have complete content."
for SCENARIO in $(find ${RESULTS_DIR} -mindepth 3 -maxdepth 3 -type d); do
   if ! [ -f ${SCENARIO}/measurements.json ]; then echo "[ERROR] File not found: ${SCENARIO}/measurements.json"; exit; fi
   if ! [ -f ${SCENARIO}/user.conf ];         then echo "[ERROR] File not found: ${SCENARIO}/user.conf";         exit; fi
   if ! [ -f ${SCENARIO}/README.md ];         then echo "[ERROR] File not found: ${SCENARIO}/README.md";         exit; fi
   if ! [ -d ${SCENARIO}/performance ];       then echo "[ERROR] No performance dir found.";                     exit; fi
   if ! [ -d ${SCENARIO}/accuracy ];          then echo "[ERROR] No accuracy dir found.";                        exit; fi
   if ! [ -d ${SCENARIO}/TEST* ];             then echo "[ERROR] No compliance tests found.";                    exit; fi
done

echo "Verifying correct 'submitter' and 'system_name' fields in: ${SYSTEMS_DIR}/${SYSTEM}.json. These should match 'config/workload.conf'."
if ! (( $(grep -r "\"submitter\": \"${VENDOR}\"" ${SYSTEMS_DIR}/${SYSTEM}.json | wc -l) > 0 ));   then echo "[ERROR] Field 'submitter' does not match 'VENDOR'.";   exit; fi
if ! (( $(grep -r "\"system_name\": \"${SYSTEM}\"" ${SYSTEMS_DIR}/${SYSTEM}.json | wc -l) > 0 )); then echo "[ERROR] Field 'system_name' does not match 'SYSTEM'."; exit; fi

mkdir -p ${SUBMISSION_ORIGINAL}/closed/${VENDOR}
cp -r ${LOG_DIR}/src \
      ${LOG_DIR}/documentation \
      ${LOG_DIR}/results \
      ${LOG_DIR}/systems \
      ${SUBMISSION_ORIGINAL}/closed/${VENDOR}/

echo "Truncating the logs: ${SUBMISSION_ORIGINAL} --> ${SUBMISSION_PROCESSED}"
cd /workspace/third_party
python -m mlperf-inference.tools.submission.truncate_accuracy_log --input ${SUBMISSION_ORIGINAL} --submitter ${VENDOR} --output ${SUBMISSION_PROCESSED}

echo "Running submission checker: ${SUBMISSION_PROCESSED}"
cd /workspace/third_party
python -m mlperf-inference.tools.submission.submission_checker.main --input ${SUBMISSION_PROCESSED} --submitter=${VENDOR} --version=v6.1
