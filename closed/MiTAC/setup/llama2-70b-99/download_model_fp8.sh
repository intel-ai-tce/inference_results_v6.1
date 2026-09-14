#!/bin/bash

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../build_scripts/build_docker_general.sh"

export CONFIG_FILE="../build_scripts/dataset_and_model.config"
source "$SCRIPT_DIR/$CONFIG_FILE"

source "$SCRIPT_DIR/../build_scripts/bash_util/argparse.sh"
declare -A ARG_SPECS=(
    ["--token"]="value optional HF_TOKEN default:dummy"
    ["--quant-algo"]="value optional QUANT_ALGO choices:gptq default:gptq"
    ["--skip-download"]="flag"
    ["--download-prequantized"]="flag"
)

parse_args "$@"

build_dataset_and_model

bash "$SCRIPT_DIR/../start_scripts/run_model_dataset.sh" $SCRIPT_DIR dataset_and_model/prepare_model.sh "${ARGS[--token]}" "FP8" "${ARGS[--quant-algo]}" $(get_flag "--skip-download") $(get_flag "--download-prequantized")
