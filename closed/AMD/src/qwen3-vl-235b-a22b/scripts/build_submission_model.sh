#!/usr/bin/env bash
# =====================================================================================
# Reproduce the MLPerf v6.1 Q3VL submission checkpoint in ONE deterministic PTQ stage:
#   A single AMD Quark pass: MXFP4 W4A4 (text decoder + MoE experts) + SmoothQuant alpha=0.35 over the
#   20 official mlcommons/inference#2600 calibration samples, with the vision tower quantized to
#   FP8-e4m3 in the SAME pass (--vit-fp8). No retraining; calibration-data-only.
#
# This is the recipe documented in submission/documentation/calibration.md, behind the pushed model
# sahirema/Qwen3-VL-235B-A22B-Instruct-MXFP4. It is verified to pass the F1 >= 0.7824 accuracy gate at
# the compliant sampler. Reference F1/latency: see the img-R / img-S reference rows in
# qwen3-vl/docs/results-tracker.md (do NOT hardcode numbers here — they drift with the image).
# The build is deterministic: a mathematical transform over a fixed calibration set, byte-stable
# across rebuilds.
#
# RUN INSIDE the IMG-S container (has amd-quark + base model access). Needs GPU (SmoothQuant runs a
# calibration forward over the 20 official mlcommons/inference#2600 samples).
#
# SERVE: IMG-S bakes VLLM_Q3VL_FUSE_VIT_GELU_FC1=0 (docker/vllm-rocm-vllm025-imgS.Dockerfile:172),
# so no runtime flag is needed for the fp8-ViT MLP; just point benchmark_mlperf6pt1.py at the output dir.
# =====================================================================================
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-235B-A22B-Instruct}"   # BF16 base
OUT_DIR="${OUT_DIR:-/root/.cache/huggingface}"             # HF cache root (host HF_CACHE mount)
OUT_NAME="${OUT_NAME:-Qwen3-VL-235B-A22B-Instruct-MXFP4-mlperf6.1-closed}"
MAX_GPU_MEM="${MAX_GPU_MEM:-250GiB}"

cd "$(dirname "$0")/.."   # -> qwen3-vl/ (repo) or src/qwen3-vl-235b-a22b/ (submission)

# Single Quark pass: MXFP4 W4A4 LLM + SmoothQuant (TEXT-ONLY calib) + FP8 vision tower (--vit-fp8).
python3 scripts/quantize_mxfp4_quark.py \
  --scheme mxfp4 \
  --smoothquant \
  --sq-alpha 0.35 \
  --num-calibration-samples 20 \
  --vit-fp8 \
  --model-id "$MODEL_ID" \
  --output-dir "$OUT_DIR" \
  --output-name "$OUT_NAME" \
  --max-gpu-mem "$MAX_GPU_MEM"

echo "Built submission checkpoint: $OUT_DIR/$OUT_NAME"
echo "Serve on IMG-S (VLLM_Q3VL_FUSE_VIT_GELU_FC1 defaults 0 -> no runtime flag needed)."
