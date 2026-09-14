#!/usr/bin/env bash
# Reproduce AMD Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ with Quark,
# using the LOCAL MLPerf CNN/DailyMail calibration set (dataset=mlperf_cnn, added
# by code/patches/apply_quark_mlperf_cnn.py). Run INSIDE the container.
#
# Usage: run_quark_quant.sh <num_calib> <seq_len> <output_dir> [extra args...]
set -u

NUM="${1:?need num_calib}"; shift
SEQ="${1:?need seq_len}"; shift
OUT="${1:?need output_dir}"; shift
EXTRA=("$@")

MODEL_DIR=/lab-mlperf-inference/code/Meta-Llama-3.1-8B-Instruct
SQ_CFG=/lab-mlperf-inference/code/patches/smoothquant_a0.62.json
# GPTQ override: damp_percent=0.1 (Quark default 0.01 yields a non-positive-definite
# Hessian -> torch.linalg.cholesky failure on the full 1000x2048 calibration run).
GPTQ_CFG=/lab-mlperf-inference/code/patches/gptq_damp0.1.json
export MLPERF_CALIB_JSON=/data/cnn_dailymail_calibration.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # everything is local; don't hit the network

cd /lab-mlperf-inference/Quark/examples/torch/language_modeling/llm_ptq/ || exit 3

echo "[run] num=$NUM seq=$SEQ out=$OUT extra=${EXTRA[*]:-none}"
python3 quantize_quark.py \
  --model_dir "$MODEL_DIR" \
  --model_attn_implementation sdpa \
  --quant_scheme mxfp4 \
  --quant_algo smoothquant,gptq \
  --quant_algo_config_file smoothquant "$SQ_CFG" \
  --quant_algo_config_file gptq "$GPTQ_CFG" \
  --dataset mlperf_cnn \
  --num_calib_data "$NUM" \
  --seq_len "$SEQ" \
  --kv_cache_dtype fp8 --min_kv_scale 1.0 \
  --model_export hf_format \
  --export_weight_format real_quantized \
  --skip_evaluation \
  --output_dir "$OUT" "${EXTRA[@]}"
echo "[run] exit=$?"
