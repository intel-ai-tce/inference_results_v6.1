#!/bin/bash

set -e

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export CONFIG_FILE="mad_llama2_70b_99_gfx950.config"

if [[ -n "$1" ]]; then
  CONFIG_FILE="$1"
fi

source "$SCRIPT_DIR/$CONFIG_FILE"

export LAB_TS=`date +%m%d-%H%M`
export LAB_MLPINF=$(dirname $(dirname $SCRIPT_DIR))
export LAB_MODEL="${LAB_MODEL:-/data/inference/model/}"
export LAB_DKR_CTNAME_BASE=mlperf.$( basename $SCRIPT_DIR ).${CONFIG_FILE%%.*}.$(whoami)
export LAB_DKR_CTNAME=${LAB_DKR_CTNAME_BASE}.${LAB_TS}

EXTRA_ARGS="--rm"

docker run ${EXTRA_ARGS} -it --ipc=host --network=host --privileged \
        --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --name=${LAB_DKR_CTNAME} \
        -v ${SCRIPT_DIR}/scripts:/lab-mlperf-inference/mad/ \
        -v ${HOME}:/workdir \
        -v ${LAB_MODEL}:/model \
        ${DOCKER_RESULT_IMAGE}
