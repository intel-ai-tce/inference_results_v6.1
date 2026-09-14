#!/bin/bash
###############################################################################
# Model preparation for deepseek-r1 (experiment S3_sq_a05_v2).
#
# Either downloads AMD's pre-quantized MLPerf checkpoint from HuggingFace, or
# reproduces it from the original DeepSeek-R1 FP8 release:
#   DeepSeek-R1 (FP8) -> BF16 (dequant) -> MXFP4 + SmoothQuant(alpha=0.5)
#
# Recipe:
#   quant_scheme   : w_mxfp4_a_mxfp4   (4-bit weights AND activations)
#   group_size     : 32
#   quant_algo     : smoothquant       (alpha = 0.5)
#   exclude_layers : self_attn / mlp.gate / lm_head   (kept BF16)
#   calibration    : mlperf deepseek-r1 500-sample fp8_eval pkl
#   export         : HuggingFace safetensors, multi-GPU
#
# Runs inside the model/dataset prep docker image (Quark 0.10).
###############################################################################
set -euo pipefail

HF_TOKEN=${1:-dummy}
SKIP_DOWNLOAD=${2:-false}
DOWNLOAD_PREQUANTIZED=${3:-false}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --------------------------- fast path: prequantized -------------------------
if [[ "${DOWNLOAD_PREQUANTIZED}" == "true" ]]; then
    MODEL=amd/Deepseek-S3_sq_a05_v2_mlperf6_1
    OUTPUT_DIR="/model/deepseek-r1/S3_sq_a05_v2"
    hf download $MODEL --token $HF_TOKEN --local-dir $OUTPUT_DIR
    exit 0
fi

# --------------------------- reproduce from FP8 ------------------------------
MODEL=deepseek-ai/DeepSeek-R1
ORIG_FP8="/model/deepseek-r1/orig"
BF16_DIR="/model/deepseek-r1/bf16"
OUTPUT_DIR="/model/deepseek-r1/S3_sq_a05_v2"
CALIB="/data/deepseek-r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl"

SCHEME=w_mxfp4_a_mxfp4
GROUP_SIZE=32
QUANT_ALGO=smoothquant
ALPHA=0.5
NUM_CALIB=500
EXCLUDE='*self_attn.kv* *self_attn.q* *mlp.gate.* *lm_head'

if [[ "${SKIP_DOWNLOAD}" != "true" ]]; then
    hf download $MODEL --token $HF_TOKEN --local-dir $ORIG_FP8
fi

# sanity checks
[ -f "$ORIG_FP8/config.json" ] || { echo "ERROR: original FP8 model not found at $ORIG_FP8"; exit 1; }
[ -f "$CALIB" ]                || { echo "ERROR: calibration pkl not found at $CALIB"; exit 1; }
[ -f "$SCRIPT_DIR/fp8_cast_bf16.py" ] || { echo "ERROR: fp8_cast_bf16.py not found in $SCRIPT_DIR"; exit 1; }

mkdir -p "$BF16_DIR" "$OUTPUT_DIR"

# ---- write the SmoothQuant-alpha injection patch (idempotent) ----
PATCH_DIR="$(dirname "$OUTPUT_DIR")/_patch"; mkdir -p "$PATCH_DIR"
cat > "$PATCH_DIR/sq_alpha_patch.py" <<'PYEOF'
# Injects: read QUARK_SQ_ALPHA env and override SmoothQuant alpha (idempotent).
import sys
f = sys.argv[1] if len(sys.argv) > 1 else "quantize_quark.py"
s = open(f).read()
MARK = "QUARK_SQ_ALPHA_INJECT"
if MARK in s:
    print("[patch] already applied"); sys.exit(0)
inj = (
"        import os as _os  # " + MARK + "\n"
"        _a = _os.environ.get('QUARK_SQ_ALPHA')\n"
"        if _a and getattr(quant_config, 'algo_config', None):\n"
"            for _ac in quant_config.algo_config:\n"
"                if hasattr(_ac, 'alpha'):\n"
"                    _ac.alpha = float(_a); print('[recipe] SmoothQuant alpha ->', _ac.alpha)\n"
)
anchor = "        quantizer = ModelQuantizer(quant_config"
assert anchor in s, "anchor not found in quantize_quark.py"
open(f, "w").write(s.replace(anchor, inj + anchor, 1))
print("[patch] SmoothQuant-alpha injection applied")
PYEOF

# =========================== STEP 1: DEQUANT FP8 -> BF16 =====================
if [ -f "$BF16_DIR/model.safetensors.index.json" ]; then
    echo "[step1] BF16 already present at $BF16_DIR -> skipping dequant"
else
    echo "[step1] Dequantizing FP8 -> BF16 (this writes ~1.3TB, ~15 min)..."
    python3 "$SCRIPT_DIR/fp8_cast_bf16.py" \
        --input-fp8-hf-path "$ORIG_FP8" --output-bf16-hf-path "$BF16_DIR"
    cp "$ORIG_FP8/tokenizer.json" "$ORIG_FP8/tokenizer_config.json" \
       "$ORIG_FP8/modeling_deepseek.py" "$ORIG_FP8/configuration_deepseek.py" \
       "$ORIG_FP8/config.json" "$BF16_DIR/"
    echo "[step1] Dequant done."
fi

# =========================== STEP 2: QUANTIZE (Quark) ========================
echo "[step2] Quantizing BF16 -> MXFP4 + SmoothQuant(alpha=$ALPHA) ..."
QDIR=/lab-mlperf-inference/amd_quark-0.10/examples/torch/language_modeling/llm_ptq
pushd "$QDIR" > /dev/null
python3 "$PATCH_DIR/sq_alpha_patch.py" quantize_quark.py
QUARK_SQ_ALPHA="$ALPHA" python3 quantize_quark.py \
    --model_dir "$BF16_DIR" \
    --quant_scheme $SCHEME \
    --group_size $GROUP_SIZE \
    --num_calib_data $NUM_CALIB \
    --dataset "$CALIB" \
    --exclude_layers $EXCLUDE \
    --quant_algo $QUANT_ALGO \
    --skip_evaluation --multi_gpu --model_export hf_format \
    --output_dir "$OUTPUT_DIR"
popd > /dev/null
echo "[step2] Quantization + export done."

# =========================== STEP 3: POST-PROCESS ===========================
echo "[step3] Post-processing config + copying tokenizer/aux files for serving..."
python3 - "$OUTPUT_DIR" <<'PYK'
import json, sys
p = sys.argv[1] + "/config.json"
c = json.load(open(p))
# sglang quark loader expects an (empty) kv_cache_group list in the export block
c.setdefault("quantization_config", {}).setdefault("export", {})["kv_cache_group"] = []
json.dump(c, open(p, "w"), indent=2)
print("[post] set quantization_config.export.kv_cache_group = []")
PYK
cp "$BF16_DIR/tokenizer.json" "$BF16_DIR/tokenizer_config.json" \
   "$BF16_DIR/modeling_deepseek.py" "$BF16_DIR/configuration_deepseek.py" \
   "$OUTPUT_DIR/" 2>/dev/null || true

echo "==================================================================="
echo " DONE. Quantized model written to:"
echo "   $OUTPUT_DIR"
du -sh "$OUTPUT_DIR" 2>/dev/null || true
echo "==================================================================="

