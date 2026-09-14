#!/bin/bash

set -ex

PERFORMANCE_DIR=build/logs.performance/default
STAGING_DIR=build/submission-staging
DIVISION=closed
SUBMITTER=Inventec

[[ $(ls ${STAGING_DIR}) != ${DIVISION} ]] && { echo Bad DIVISION; exit 1; }
[[ $(ls ${STAGING_DIR}/${DIVISION}) != ${SUBMITTER} ]] && { echo Bad SUBMITTER; exit 1; }

STAGING_SUBMITTER_DIR=${STAGING_DIR}/${DIVISION}/${SUBMITTER}
STAGING_RESULTS_DIR=${STAGING_SUBMITTER_DIR}/results
SYSTEM_IDS=$(ls ${STAGING_RESULTS_DIR})

for system_id in ${SYSTEM_IDS}; do
  for benchmark in $(ls ${STAGING_RESULTS_DIR}/${system_id}); do
    for scenario in $(ls ${STAGING_RESULTS_DIR}/${system_id}/${benchmark}); do
      sbs=${system_id}/${benchmark}/${scenario}
      srcdir=${PERFORMANCE_DIR}/${sbs}
      [[ -f ${srcdir}/metadata.json            ]] || { echo No metadata.json for ${sbs};            exit 1; }
      [[ -f ${srcdir}/mlperf_log_accuracy.json ]] || { echo No mlperf_log_accuracy.json for ${sbs}; exit 1; }
      [[ -f ${srcdir}/mlperf_log_detail.txt    ]] || { echo No mlperf_log_detail.txt for ${sbs};    exit 1; }
      [[ -f ${srcdir}/mlperf_log_summary.txt   ]] || { echo No mlperf_log_summary.txt for ${sbs};   exit 1; }
      dstdir=${STAGING_RESULTS_DIR}/${sbs}/performance/run_1
      mkdir -p ${dstdir}
      cp ${srcdir}/metadata.json            ${dstdir}
      cp ${srcdir}/mlperf_log_accuracy.json ${dstdir}
      cp ${srcdir}/mlperf_log_detail.txt    ${dstdir}
      cp ${srcdir}/mlperf_log_summary.txt   ${dstdir}
    done
  done
done
