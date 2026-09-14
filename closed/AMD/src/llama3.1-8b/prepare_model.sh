#!/bin/bash

#HF_TOKEN=$1
QUANT_FORMAT=$1
QUANT_ALGO=${2:-autosmoothquant}
#QUANT_ALGO=${2:-awq}
#QUANT_ALGO=${2:-smoothquant}

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH="/model/llama3.1-8b/orig"
#hf download $MODEL --token $HF_TOKEN --local-dir $MODEL_PATH
# use local downloaded model and dataset

#DATASET="/data/cnn_dailymail_calibration.json"
#--dataset "${DATASET}" \

pushd "Quark/examples/torch/language_modeling/llm_ptq" > /dev/null
if [[ "$QUANT_FORMAT" == "FP8" ]]; then

    OUTPUT_DIR="/model/llama3.1-8b/fp8_quantized"
    python3 quantize_quark.py --model_dir "${MODEL_PATH}" \
                            --output_dir "${OUTPUT_DIR}" \
                            --multi_gpu \
                            --data_type auto \
                            --model_attn_implementation "sdpa" \
                            --quant_algo autosmoothquant \
                            --quant_scheme w_fp8_a_fp8 \
                            --kv_cache_dtype fp8 \
                            --min_kv_scale 1.0 \
                            --num_calib_data 512 \
                            --seq_len 8192 \
                            --model_export hf_format \
                            --custom_mode fp8 \
                            --exclude_layers "lm_head"

elif [[ "$QUANT_FORMAT" == "FP4" ]]; then

    OUTPUT_DIR="/model/llama3.1-8b/fp4_quantized"
    OUTPUT_ALGO_DIR="/model/llama3.1-8b/fp4_quantized_${QUANT_ALGO}"
    python3 quantize_quark.py --model_dir "${MODEL_PATH}" \
                          --output_dir "${OUTPUT_ALGO_DIR}" \
                          --model_attn_implementation "sdpa" \
                          --quant_algo "${QUANT_ALGO}" \
                          --quant_scheme mxfp4 \
                          --data_type bfloat16 \
                          --kv_cache_dtype fp8 \
                          --min_kv_scale 1.0 \
                          --exclude_layers "lm_head" \
                          --model_export hf_format 
    if [ ! -e $OUTPUT_DIR ]; then
        ln -s $OUTPUT_ALGO_DIR $OUTPUT_DIR
    fi
fi
popd > /dev/null
