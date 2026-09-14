#!/bin/bash

HF_TOKEN=${1:-dummy}
SKIP_DOWNLOAD=${2:-false}
# DOWNLOAD_PREQUANTIZED=${3:-false}

# if [[ "${DOWNLOAD_PREQUANTIZED}" == "true" ]]; then
#     MODEL=amd/gpt-oss-120b-w-mxfp4-a-fp8-Mlperf
#     OUTPUT_DIR="/model/gpt-oss-120b/fp4_quantized"
#     hf download $MODEL --token $HF_TOKEN --local-dir $OUTPUT_DIR
#     exit 0
# fi

MODEL=openai/gpt-oss-120b
MODEL_PATH="/model/gpt-oss-120b"
REVISION=b5c939d
if [[ "${SKIP_DOWNLOAD}" != "true" ]]; then
    hf download $MODEL --token $HF_TOKEN --local-dir $MODEL_PATH --revision $REVISION
fi

# DATASET="/data/gpt-oss-120b/calibration_unique_sampled1024.parquet"

# pushd "amd_quark-0.11/examples/torch/language_modeling/llm_ptq" > /dev/null

# OUTPUT_DIR="/model/gpt-oss-120b/fp4_quantized"
# EXCLUDE_LAYERS="*lm_head *self_attn* *router*"

# python3 quantize_quark.py \
#     --model_dir  "${MODEL_PATH}" \
#     --dataset "${DATASET}" \
#     --quant_scheme mxfp4_fp8 \
#     --exclude_layers ${EXCLUDE_LAYERS} \
#     --num_calib_data 1024 \
#     --output_dir "${OUTPUT_DIR}" \
#     --model_export hf_format \
#     --multi_gpu

