#!/bin/bash

# Do the following inside the container...
# 
# Step 1: copy results from build/logs to build/submission-staging
#   $ python -m src.nv_mlpinf.common.mlcommons.results
# 
# Step 2: copy performance results from from build/logs.performance
#   $ ./scripts/copy_performance.sh
# 
# Step 3: copy loadgen-configs, src, configs and documentation, etc.
#   $ ./scripts/copy_rest_staging.sh
# 
# Step 4: run this script to generate build/submission
#   $ ./scripts/generate_submission.sh

set -ex

STAGING_DIR=build/submission-staging
SUBMISSION_DIR=build/submission
DIVISION=closed
SUBMITTER=Inventec
VERSION=v6.1

MLC_TOOLS_DIR=3rdparty/mlc-inference/tools/submission

# Generate the actual submission directory from the staging directory

rm -rf ${SUBMISSION_DIR}
python ${MLC_TOOLS_DIR}/truncate_accuracy_log.py \
  --input ${STAGING_DIR} --submitter ${SUBMITTER} --output ${SUBMISSION_DIR}
python ${MLC_TOOLS_DIR}/submission_checker/main.py \
  --input ${SUBMISSION_DIR} --version ${VERSION} \
  --submitter ${SUBMITTER} 2>&1 | tee submission_checker_log.txt
mv submission_checker_log.txt ${SUBMISSION_DIR}/${DIVISION}/${SUBMITTER}/
mv summary.csv ${SUBMISSION_DIR}/${DIVISION}/${SUBMITTER}/
