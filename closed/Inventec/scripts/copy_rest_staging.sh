#!/bin/bash

set -ex

STAGING_DIR=build/submission-staging
DIVISION=closed
SUBMITTER=Inventec

[[ $(ls ${STAGING_DIR}) != ${DIVISION} ]] && { echo Bad DIVISION; exit 1; }
[[ $(ls ${STAGING_DIR}/${DIVISION}) != ${SUBMITTER} ]] && { echo Bad SUBMITTER; exit 1; }

STAGING_SUBMITTER_DIR=${STAGING_DIR}/${DIVISION}/${SUBMITTER}
STAGING_RESULTS_DIR=${STAGING_SUBMITTER_DIR}/results
LOADGEN_CONFIGS_DIR=build/loadgen-configs
SYSTEM_IDS=$(ls ${STAGING_RESULTS_DIR})

# Copy loadgen-configs and README.md
for system_id in ${SYSTEM_IDS}; do
  for benchmark in $(ls ${STAGING_RESULTS_DIR}/${system_id}); do
    case ${benchmark} in
      gpt-oss-120b)
        benchmark_src=gpt_oss_120b
        ;;
      deepseek-r1)
        benchmark_src=deepseek_r1
        ;;
      llama2-70b*)
        benchmark_src=llama2_70b
        ;;
      *)
        benchmark_src=${benchmark}
        ;;
    esac
    for scenario in $(ls ${STAGING_RESULTS_DIR}/${system_id}/${benchmark}); do
      sbs=${system_id}/${benchmark}/${scenario}
      srcdir1=build/loadgen-configs/${sbs}
      srcdir2=src/nv_mlpinf/benchmarks/${benchmark_src}
      [[ -f ${srcdir1}/measurements.json      ]] || { echo No measurements.json for ${sbs};      exit 1; }
      [[ -f ${srcdir1}/mlperf.conf            ]] || { echo No mlperf.json for ${sbs};            exit 1; }
      [[ -f ${srcdir1}/user.conf              ]] || { echo No user.json for ${sbs};              exit 1; }
      [[ -f ${srcdir2}/README_${SUBMITTER}.md ]] || { echo No README_${SUBMITTER}.md for ${sbs}; exit 1; }
      dstdir=${STAGING_RESULTS_DIR}/${system_id}/${benchmark}/${scenario}
      mkdir -p ${dstdir}
      cp ${srcdir1}/measurements.json      ${dstdir}
      cp ${srcdir1}/mlperf.conf            ${dstdir}
      cp ${srcdir1}/user.conf              ${dstdir}
      cp ${srcdir2}/README_${SUBMITTER}.md ${dstdir}/README.md
    done
  done
done

# Copy other necessary folders/files
mkdir -p ${STAGING_SUBMITTER_DIR}/src
rsync -av --exclude='__pycache__' src/nv_mlpinf ${STAGING_SUBMITTER_DIR}/src
rsync -av --exclude='__pycache__' configs docker docs documentation scaleout scripts systems ${STAGING_SUBMITTER_DIR}/
cp *.md Makefile* pyproject.toml attributions.txt ${STAGING_SUBMITTER_DIR}/
