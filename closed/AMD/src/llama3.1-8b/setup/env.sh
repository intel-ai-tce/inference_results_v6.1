# Timestamp
export LAB_TS=`date +%m%d-%H%M`

# Host side
export LAB_MLPINF=$(dirname $(dirname $(readlink -fm -- $0)))
export LAB_MLPINF_CODE=${LAB_MLPINF}/code
export LAB_MLPINF_SETUP=${LAB_MLPINF}/setup
export LAB_MLPINF_SUBMISSION=${LAB_MLPINF}/submission
export LAB_MLPINF_RESULTS=${LAB_MLPINF}/results
export LAB_HIST=${LAB_MLPINF}/lab-hist/
export LAB_BUILD=${LAB_HIST}/build
export LAB_LOG=${LAB_HIST}/log

export LAB_CWD=$(pwd)
export LAB_CLOG=${LAB_LOG}/${LAB_TS}
export LAB_MODEL="${LAB_MODEL:-/data/inference/model/}"
export LAB_DATASET="${LAB_DATASET:-/data/inference/data/}"

export LAB_MLCINF=${LAB_BUILD}/inference

export LAB_XDOCKER=${LAB_CLOG}/xdocker

# Docker
export LAB_DKR_CTNAME_BASE=mlperf${MLPINF_DOCKER_NAME_ABBREV:-''}.$(whoami)
export LAB_DKR_CTNAME=${LAB_DKR_CTNAME_BASE}.${LAB_TS}
