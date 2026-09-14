#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/../.." && pwd)"
deployment_env="${DEPLOYMENT_ENV:-${root_dir}/config/deployment.env}"
prep_env="${MODEL_PREP_ENV:-${root_dir}/config/model_prep.env}"
quark_image="cisco-vllm-heterogeneous-quark:0.10"

for env_file in "${deployment_env}" "${prep_env}"; do
    if [[ ! -f "${env_file}" ]]; then
        echo "Missing environment file: ${env_file}" >&2
        exit 1
    fi
    source "${env_file}"
done

model_name="${1:-}"
format="${2:-}"
if [[ "${model_name}" != "llama2-70b" && "${model_name}" != "llama3.1-8b" ]]; then
    echo "Model must be llama2-70b or llama3.1-8b" >&2
    exit 1
fi
if [[ "${format}" != "fp8" && "${format}" != "fp4" ]]; then
    echo "Format must be fp8 or fp4" >&2
    exit 1
fi

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "${name} must be set in ${deployment_env}" >&2
        exit 1
    fi
}

require_var MODEL_ROOT
require_var DATA_ROOT

case "${model_name}" in
    llama2-70b)
        model_dir="${MODEL_ROOT}/llama2-70b"
        calibration_file="${DATA_ROOT}/llama2-70b/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
        if [[ "${format}" == "fp8" ]]; then
            artifact="fp8_dynamic"
            sequence_length="${LLAMA2_FP8_SEQ_LEN:-2048}"
            samples="${LLAMA2_FP8_CALIBRATION_SAMPLES:-512}"
            quant_args=(--quant_scheme w_fp8_a_fp8 --kv_cache_dtype fp8 --data_type float16 --custom_mode fp8)
        else
            artifact="fp4_quantized_gptq"
            sequence_length="${LLAMA2_FP4_SEQ_LEN:-2048}"
            samples="${LLAMA2_FP4_CALIBRATION_SAMPLES:-512}"
            quant_args=(--quant_scheme w_mxfp4_a_mxfp4 --group_size 32 --kv_cache_dtype fp8 --data_type bfloat16 --quant_algo gptq)
        fi
        ;;
    llama3.1-8b)
        model_dir="${MODEL_ROOT}/llama3.1-8b"
        calibration_file="${DATA_ROOT}/llama3.1-8b/preprocessed/cnn_dailymail_calibration.pkl"
        if [[ "${format}" == "fp8" ]]; then
            artifact="fp8_dynamic"
            sequence_length="${LLAMA31_FP8_SEQ_LEN:-1024}"
            samples="${LLAMA31_FP8_CALIBRATION_SAMPLES:-512}"
            quant_args=(--quant_scheme w_fp8_a_fp8 --kv_cache_dtype fp8 --data_type float16 --custom_mode fp8)
        else
            artifact="fp4_quantized_awq_gptq_seq2560_bf16"
            sequence_length="${LLAMA31_FP4_SEQ_LEN:-2560}"
            samples="${LLAMA31_FP4_CALIBRATION_SAMPLES:-512}"
            quant_args=(--quant_scheme w_mxfp4_a_mxfp4 --group_size 32 --kv_cache_dtype fp8 --data_type bfloat16 --quant_algo gptq)
        fi
        ;;
esac

if [[ ! -f "${model_dir}/config.json" ]]; then
    echo "Missing model directory: ${model_dir}" >&2
    exit 1
fi
if [[ ! -f "${calibration_file}" ]]; then
    echo "Missing calibration data: ${calibration_file}" >&2
    exit 1
fi

docker run --rm --init --ipc=host --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri \
    -v "${MODEL_ROOT}:${MODEL_ROOT}" \
    -v "${DATA_ROOT}:${DATA_ROOT}:ro" \
    "${quark_image}" \
    quantize_quark.py \
    --model_dir "${model_dir}" \
    --output_dir "${model_dir}/${artifact}" \
    --dataset "${calibration_file}" \
    --multi_gpu \
    --seq_len "${sequence_length}" \
    --num_calib_data "${samples}" \
    --model_export hf_format \
    --exclude_layers lm_head \
    --skip_evaluation \
    "${quant_args[@]}"
