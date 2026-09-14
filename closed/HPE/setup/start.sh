#!/bin/bash
set -e

RED='\033[0;31m'
NC='\033[0m'

if [ -z "$1" ]; then
  echo -e "${RED}Error: Please provide image name.${NC}"
  exit 1
fi

function shorten_docker_image_name() {
    # Convert it to lowercase
    DKR_IMAGE_NAME=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    case "$DKR_IMAGE_NAME" in
    *sglang*|*sgl*)
        echo ".sglang"
        ;;
    *vllm*)
        echo ".vllm"
        ;;
    *)
        echo ".dev"
        ;;
    esac
}

export DKR_CONFIG_NAME=$(shorten_docker_image_name $1)

SETUP_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SCRIPT_DIR=$(dirname $SETUP_DIR)/code/scripts DOCKER_RESULT_IMAGE=$1 bash $SETUP_DIR/start_scripts/start_general.sh
