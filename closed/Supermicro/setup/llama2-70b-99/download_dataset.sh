#!/bin/bash

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../build_scripts/build_docker_general.sh"

export CONFIG_FILE="../build_scripts/dataset_and_model.config"
source "$SCRIPT_DIR/$CONFIG_FILE"

build_dataset_and_model

bash "$SCRIPT_DIR/../start_scripts/run_model_dataset.sh" $SCRIPT_DIR dataset_and_model/prepare_dataset.sh