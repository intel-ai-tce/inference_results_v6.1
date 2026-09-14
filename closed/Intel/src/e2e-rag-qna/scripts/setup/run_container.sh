TARGET_DIR=/data
export DATA_DIR=${TARGET_DIR}
export MODEL_PATH=${TARGET_DIR}/models
export CODE_DIR=${PWD}
export LOGGING_DIR=${HOME}/log

# Usage: bash run_container.sh [cpu|gpu]   (default: cpu)
FLAVOR=${1:-cpu}
case "$FLAVOR" in
    cpu) VERSION=e2e-rag-cpu; DOCKER_IMAGE=${DOCKER_IMAGE:-tiyengar:vllm_cpu} ;;
    gpu) VERSION=e2e-rag-gpu; DOCKER_IMAGE=${DOCKER_IMAGE:-vllm_xpu:gpt-oss-ww25-rc0} ;;
    *)   echo "usage: $0 [cpu|gpu]"; exit 1 ;;
esac
docker run --privileged -it \
         --name hans-${VERSION}  \
         -u root \
        --ipc=host --net=host --cap-add=ALL \
        --device /dev/dri:/dev/dri \
        -v /dev/dri/by-path:/dev/dri/by-path \
        -v /lib/modules:/lib/modules \
        -v $DATA_DIR:/data \
        -v $MODEL_PATH:/models \
        -v $LOGGING_DIR:/logs \
         -v $CODE_DIR:/workspace/code \
         -v $HOME:/host \
        --workdir /workspace  \
        --entrypoint /bin/bash \
        ${DOCKER_IMAGE}
