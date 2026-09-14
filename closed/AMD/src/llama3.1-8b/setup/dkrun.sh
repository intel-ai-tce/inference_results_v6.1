#!/bin/bash
set -xeu

DOCKER_IMNAME=$1
EXTRA_ARGS=""

#LAB_MODEL=/data/inference/model/llama3.1-8b/fp4_quantized/
#LAB_MODEL=/data/inference/model/Llama-3.1-8B-Instruct-FP8-KV/
#LAB_DATASET=/data/inference/data/llama3.1-8b/
#LAB_MODEL=/data/mlperf-endpoints/models/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ
LAB_DATASET=/data/mlperf-endpoints/datasets/
LAB_MODEL=/data/mlperf-endpoints/models/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ
#LAB_MODEL=/data/mlperf-endpoints/llama3-1-8b-redline-q1-tp1

docker run ${EXTRA_ARGS} -it -d --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --entrypoint /bin/bash \
        --name=${LAB_DKR_CTNAME} \
        -v ${LAB_MODEL}:/model/ \
        -v ${LAB_DATASET}:/data/ \
        -v ${LAB_HIST}:/lab-hist \
        -v ${LAB_MLPINF_CODE}:/lab-mlperf-inference/code \
        -v ${LAB_MLPINF_SETUP}:/lab-mlperf-inference/setup \
        -v ${LAB_MLPINF_SUBMISSION}:/lab-mlperf-inference/submission \
        -v ${LAB_XDOCKER}:/xdocker \
        -v ${HOME}:/workdir \
        -e LAB_CLOG=/lab-hist/log/${LAB_TS} \
        ${DOCKER_IMNAME}
