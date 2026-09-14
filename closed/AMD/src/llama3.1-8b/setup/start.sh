#!/bin/bash
set -e

script_dir=$(dirname -- $0)
dk_image_name=$1

if [ -z "$dk_image_name" ]; then
    echo "Docker image not specified, usage: $0 <docker image>"
    exit 1
fi

function shorten_docker_image_name() {
    # Convert it to lowercase
    DOCKER_IMAGE_NAME=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    case "$DOCKER_IMAGE_NAME" in
    *sglang*|*sgl*)
        echo ".sglang"
        ;;
    *vllm*)
        echo ".vllm"
        ;;
    esac
}

export MLPINF_DOCKER_NAME_ABBREV=$(shorten_docker_image_name ${dk_image_name})

source ${script_dir}/env.sh
${script_dir}/setup.sh
${script_dir}/dkrun.sh $dk_image_name
