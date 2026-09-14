#!/bin/bash
RED='\033[0;31m'
NC='\033[0m' # No Color

DOCKER_IMAGE=$1
shift
RUN_SCRIPT=$*

if [ -z "$DOCKER_IMAGE" ]; then
  echo -e "${RED}Error: No docker image specified.${NC}"
  exit 1
fi

if [ -z "$RUN_SCRIPT" ]; then
  echo -e "${RED}Error: No run script specified.${NC}"
  exit 1
fi

export LAB_TS=`date +%m%d-%H%M`

export LAB_MLPINF=$(dirname $(dirname $(dirname $(readlink -fm -- $0))))
export LAB_MLPINF_CODE=${LAB_MLPINF}/code
export LAB_MLPINF_SETUP=${LAB_MLPINF}/setup
export LAB_MLPINF_SUBMISSION=${LAB_MLPINF}/submission
export LAB_MLPINF_RESULTS=${LAB_MLPINF}/results
export LAB_MODEL="${LAB_MODEL:-/data/inference/model/}"
export LAB_DATASET="${LAB_DATASET:-/data/inference/data/}"

export LAB_DKR_CTNAME_BASE=mlperf.ci.${MODEL}.${SCENARIO}.$(whoami)
export LAB_DKR_CTNAME=${LAB_DKR_CTNAME_BASE}.${LAB_TS}

EXTRA_ARGS="--rm"
ENV_FILE="${ENV_FILE:-.env}"

if [[ -f "$ENV_FILE" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --env-file $ENV_FILE"
fi

cleanup_docker() {
    docker container rm -f "${LAB_DKR_CTNAME}" || true
}
trap 'set -eux; cleanup_docker' EXIT

docker run ${EXTRA_ARGS} --init --ipc=host --network=host --privileged \
        --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --name=${LAB_DKR_CTNAME} \
        -v ${LAB_MODEL}:/model/ \
        -v ${LAB_DATASET}:/data/ \
        -v ${LAB_MLPINF_CODE}:/lab-mlperf-inference/code \
        -v ${LAB_MLPINF_SETUP}:/lab-mlperf-inference/setup \
        -v ${LAB_MLPINF_SUBMISSION}:/lab-mlperf-inference/submission \
        -v ${LAB_MLPINF_RESULTS}:/lab-mlperf-inference/results \
        $DOCKER_IMAGE bash $RUN_SCRIPT
