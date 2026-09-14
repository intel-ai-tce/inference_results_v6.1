#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-${SCRIPT_DIR}/config/deployment.env}"
CONTAINER_ENV="${CONTAINER_ENV:-${SCRIPT_DIR}/config/container.env}"

if [[ ! -f "${DEPLOYMENT_ENV}" ]]; then
    echo "Missing deployment environment: ${DEPLOYMENT_ENV}" >&2
    exit 1
fi
if [[ ! -f "${CONTAINER_ENV}" ]]; then
    echo "Missing container environment: ${CONTAINER_ENV}" >&2
    exit 1
fi

source "${DEPLOYMENT_ENV}"
source "${CONTAINER_ENV}"

HARDWARE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hardware)
            HARDWARE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "${HARDWARE}" != "h200" && "${HARDWARE}" != "mi350x" ]]; then
    echo "Usage: $0 --hardware {h200|mi350x}" >&2
    exit 2
fi

if [[ -z "${WORK_DIR:-}" || -z "${MODEL_ROOT:-}" || -z "${DATA_ROOT:-}" ]]; then
    echo "WORK_DIR, MODEL_ROOT, and DATA_ROOT must be set in config/deployment.env" >&2
    exit 1
fi

if [[ "${HARDWARE}" == "h200" ]]; then
    DOCKER_IMAGE="${DOCKER_IMAGE:-${H200_IMAGE:-}}"
else
    DOCKER_IMAGE="${DOCKER_IMAGE:-${MI350X_IMAGE:-}}"
fi
if [[ -z "${DOCKER_IMAGE}" ]]; then
    echo "Set the image for ${HARDWARE} in config/container.env or DOCKER_IMAGE" >&2
    exit 1
fi

CONTAINER_NAME="${CONTAINER_NAME:-vllm-${HARDWARE}}"
DOCKER_ARGS=(--rm -it --name "${CONTAINER_NAME}" --network host --ipc host --shm-size "${SHM_SIZE:-16g}" --ulimit memlock=-1 --ulimit stack=67108864)
MOUNTS=(-v "${SCRIPT_DIR}:${WORK_DIR}:rw")

mount_pair() {
    local host_name="$1"
    local container_name="$2"
    local mode="$3"
    local host_path="${!host_name:-}"
    local container_path="${!container_name:-}"
    if [[ -z "${host_path}" && -z "${container_path}" ]]; then
        return
    fi
    if [[ -z "${host_path}" || -z "${container_path}" ]]; then
        echo "Set both ${host_name} and ${container_name}, or neither" >&2
        exit 1
    fi
    if [[ ! -e "${host_path}" ]]; then
        echo "Host path does not exist: ${host_path}" >&2
        exit 1
    fi
    MOUNTS+=(-v "${host_path}:${container_path}:${mode}")
}

mount_pair HOST_MLPERF_INFERENCE_DIR MLPERF_INFERENCE_DIR ro
mount_pair HOST_MODEL_ROOT MODEL_ROOT ro
mount_pair HOST_DATA_ROOT DATA_ROOT ro
mount_pair HOST_HF_CACHE_DIR HF_CACHE_MOUNT rw

if [[ "${HARDWARE}" == "h200" ]]; then
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q 'nvidia'; then
        DOCKER_ARGS+=(--runtime nvidia)
    else
        DOCKER_ARGS+=(--gpus all)
    fi
else
    DOCKER_ARGS+=(--device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined)
fi

exec docker run "${DOCKER_ARGS[@]}" "${MOUNTS[@]}" -w "${WORK_DIR}" "${DOCKER_IMAGE}" /bin/bash -i
