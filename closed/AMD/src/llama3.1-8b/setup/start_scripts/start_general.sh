#!/bin/bash

set -e

export LAB_TS=`date +%m%d-%H%M`

export LAB_MLPINF=$(dirname $(dirname $SCRIPT_DIR))
if [ -d "${LAB_MLPINF}/code" ]; then
  export LAB_MLPINF_CODE="${LAB_MLPINF}/code"
else
  export LAB_MLPINF_CODE="${LAB_MLPINF}/src"
fi
export LAB_MLPINF_SETUP=${LAB_MLPINF}/setup
export LAB_MLPINF_SUBMISSION=${LAB_MLPINF}/submission
export LAB_MODEL="${LAB_MODEL:-/data/inference/model/}"
export LAB_DATASET="${LAB_DATASET:-/data/inference/data/}"
export LAB_VLLM_CACHE_ROOT="${LAB_VLLM_CACHE_ROOT:-/data/inference/vllm-cache}"

DKR_CONFIG_NAME=${DKR_CONFIG_NAME:-.$( basename $SCRIPT_DIR ).${CONFIG_FILE%%.*}}
export LAB_DKR_CTNAME_BASE=mlperf${DKR_CONFIG_NAME}.$(whoami)
export LAB_DKR_CTNAME=${OVERRIDE_CTNAME:-${LAB_DKR_CTNAME_BASE}.${LAB_TS}}

EXTRA_ARGS="${EXTRA_ARGS:---rm}"

if [[ -t 1 ]]; then
  DOCKER_FLAGS="-it"
  DOCKER_CMD=""
else
  DOCKER_FLAGS="--init"
  DOCKER_CMD="sleep infinity"
fi

docker run ${EXTRA_ARGS} ${DOCKER_FLAGS} --ipc=host --network=host --privileged \
        --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --name=${LAB_DKR_CTNAME} \
        -v ${LAB_MODEL}:/model/ \
        -v ${LAB_DATASET}:/data/ \
        -v ${LAB_VLLM_CACHE_ROOT}:/root/.cache \
        -v ${LAB_MLPINF_CODE}:/lab-mlperf-inference/code \
        -v ${LAB_MLPINF_SETUP}:/lab-mlperf-inference/setup \
        -v ${LAB_MLPINF_SUBMISSION}:/lab-mlperf-inference/submission \
        -v ${HOME}:/workdir \
        ${DOCKER_RESULT_IMAGE} ${DOCKER_CMD}
