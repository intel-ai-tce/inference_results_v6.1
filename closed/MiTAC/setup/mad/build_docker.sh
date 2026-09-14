#!/bin/bash

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../build_scripts/build_docker_general.sh"
export CONFIG_FILE="mad_llama2_70b_99_gfx950.config"

if [[ -n "$1" ]]; then
  CONFIG_FILE="$1"
fi

source "$SCRIPT_DIR/$CONFIG_FILE"

build_mad
