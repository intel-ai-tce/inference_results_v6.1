#!/bin/bash

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../build_scripts/build_docker_general.sh"

export CONFIG_FILE="../build_scripts/dataset_and_model.config"
source "$SCRIPT_DIR/$CONFIG_FILE"

HF_TOKEN=$1
if [ -z "$HF_TOKEN" ]; then
  echo -e "${RED}Error: Hugging Face token is missing. Please provide it as the first parameter.${NC}"
  exit 1
fi

build_dataset_and_model

MODEL="amd/Llama-2-70b-chat-hf_FP8_MLPerf_V2"
MODEL_PATH="llama2-70b-chat-hf/fp8_quantized_v2"

bash "$SCRIPT_DIR/../start_scripts/run_model_dataset.sh" $SCRIPT_DIR dataset_and_model/prepare_model.sh $MODEL $MODEL_PATH $HF_TOKEN
