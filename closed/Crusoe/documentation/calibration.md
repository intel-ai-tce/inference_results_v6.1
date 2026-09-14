## Crusoe MLPerf Inference v6.1 — Calibration and Quantization

Crusoe's closed-division submissions for **gpt-oss-120b** and **DeepSeek-R1** use
publicly released, pre-quantized model checkpoints. No additional calibration or weight
retraining was performed by Crusoe; the checkpoints were used as published. Quantization
was produced with [AMD Quark](https://quark.docs.amd.com/latest/), a publicly available
model-optimization library.

## Quantization Strategy

For calibration, the full calibration dataset provided by
[mlcommons/inference](https://mlcommons.org/benchmarks/inference-datacenter/) was used for
each model. Inputs were tokenized and serialized into fixed-length sequences using dynamic
padding and truncation during preprocessing. Weights and activations of all `nn.Linear`
modules were quantized to OCP MXFP4 (weights) / OCP FP8-e4m3 (as applicable), and KV caches
to OCP FP8-e4m3. Post-quantization algorithms AutoSmoothQuant and GPTQ were applied for
MXFP4.

## OCP FP8-e4m3 Quantization

Per-tensor symmetric static quantization of weights and activations:

    x_q = rounding( clip( x / scale * 448, -448, 448 ) )

where `scale` is the absmax of the tensor and 448 is the numerical range of OCP FP8-e4m3.
Scaled values are rounded half-even after clipping.

## OCP MXFP4 Quantization

Static quantization for weights, dynamic for activations:

    x_q^MXFP4 = Encode_E2M1( clip( x / scale, -6, 6 ) ),  scale = 2^floor(log2(round(max_abs(x))) - 2)

MXFP4 encodes 32 values per micro-block; each block shares one 8-bit E8M0 (power-of-two)
scale and stores each element as a 4-bit E2M1 float, scaled into the representable FP4
range [-6, 6]. Even rounding is used when computing scales.

## Per-model summary

#### gpt-oss-120b
* Native MXFP4 checkpoint `openai/gpt-oss-120b` (commit `b5c939d`), used as published.
* Weights/activations MXFP4; KV cache FP8-e4m3. No Crusoe re-quantization.

#### DeepSeek-R1
* `amd/Deepseek-S3_sq_a05_v2_mlperf6_1` (AMD Quark).
* MXFP4 experts / FP8-e4m3 KV cache / BF16 MLA-absorb.
