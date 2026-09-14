#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

set -e

MODEL_DIR=$1
shift
RUN_SCRIPT=$*

if [ -z "$MODEL_DIR" ]; then
  echo -e "${RED}Error: No model folder specified.${NC}"
  exit 1
fi

if [ -z "$RUN_SCRIPT" ]; then
  echo -e "${RED}Error: No run script specified.${NC}"
  exit 1
fi

export LAB_TS=`date +%m%d-%H%M`

export LAB_MODEL="${LAB_MODEL:-/data/inference/model/}"
export LAB_DATASET="${LAB_DATASET:-/data/inference/data/}"

export LAB_DKR_CTNAME_BASE=mlperf.$( basename $SCRIPT_DIR ).${CONFIG_FILE%%.*}.$(whoami)
export LAB_DKR_CTNAME=${LAB_DKR_CTNAME_BASE}.${LAB_TS}

EXTRA_ARGS="--rm"


docker run ${EXTRA_ARGS} --init --ipc=host --network=host --privileged \
        --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --name=${LAB_DKR_CTNAME} \
        -v ${LAB_MODEL}:/model/ \
        -v ${LAB_DATASET}:/data/ \
        -v ${MODEL_DIR}:/lab-mlperf-inference/setup \
        -v ${HOME}:/workdir \
        ${DOCKER_RESULT_IMAGE} \
        bash setup/$RUN_SCRIPT
