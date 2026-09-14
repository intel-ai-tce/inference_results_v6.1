#!/bin/bash

set -e
set -x


if [[ "$MAD_SYSTEM_GPU_ARCHITECTURE" != *"gfx950"* ]]; then 
    echo $MAD_SYSTEM_GPU_ARCHITECTURE
    echo "Unsupported GPU arch detected, please use supported MI35*X GPUs \n"
    exit 1
fi


export MODEL_DIR=/mad/model
export HF_MODEL_ID=amd/Llama-2-70b-chat-hf-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-MLPerf-GPTQ
export MODEL_PATH=llama2-70b-chat-hf/fp4_quantized
RESULT_CSV=pyt_mlperf_inf_mi355_llama2_70b_99.csv
DATASET_PATH=/mad/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
CONFIG_PATH=/lab-mlperf-inference/code/llama2-70b-99/
RESULTS_DIR=/lab-mlperf-inference/code/results
if [[ -z "$MAD_SECRETS_HFTOKEN" ]]; then
        echo "Mlperf quantized models are gated and require MAD_SECRETS_HFTOKEN=<your-huggingface-token> to be set as an environment variable."
        exit 1
fi
export HUGGINGFACE_ACCESS_TOKEN=$MAD_SECRETS_HFTOKEN


bash /lab-mlperf-inference/mad/download_model.sh


bash /lab-mlperf-inference/code/run_harness.sh \
        --config-path ${CONFIG_PATH} \
        --config-name offline_mi355x \
        --backend vllm \
        test_mode=performance \
        harness_config.output_log_dir=${RESULTS_DIR}/llama2_offline_performance \
        vllm_engine_config.model=${MODEL_DIR}/${MODEL_PATH} \
        harness_config.dataset_path=${DATASET_PATH}


bash /lab-mlperf-inference/code/run_harness.sh \
        --config-path ${CONFIG_PATH} \
        --config-name server_mi355x \
        --backend vllm \
        test_mode=performance \
        harness_config.output_log_dir=${RESULTS_DIR}/llama2_server_performance \
        vllm_engine_config.model=${MODEL_DIR}/${MODEL_PATH} \
        harness_config.dataset_path=${DATASET_PATH}


python3 /lab-mlperf-inference/mad/print_results.py \
        --output-file-name ${RESULT_CSV} \
        --offline-results ${RESULTS_DIR}/llama2_offline_performance \
        --server-results ${RESULTS_DIR}/llama2_server_performance

mv ${RESULT_CSV} $(pwd)/../${RESULT_CSV}
