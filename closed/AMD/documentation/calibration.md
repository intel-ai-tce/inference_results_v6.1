## AMD MLPerf Inference Calibration and Quantization Details

This section outlines our use of [AMD Quark](https://quark.docs.amd.com/latest/) for quantizing models submitted to MLPerf Inference. AMD Quark is a publicly available model optimization library that includes extensive documentation and usage examples. Our overall quantization strategy is described below. 

## Quantization Strategy

For calibration, we used the full calibration dataset provided by [mlcommons/inference](https://mlcommons.org/benchmarks/inference-datacenter/) for each model. Inputs from the dataset were tokenized and serialized into fixed-length sequences using dynamic padding and truncation as part of preprocessing. 

We quantized weights and activations of all nn.Linear modules (from PyTorch) to OCP FP8-e4m3 or OCP MXFP4 formats. Additionally, KV caches were quantized to OCP FP8-e4m3. We apply specific [post-quantization algorithmic techniques](https://quark.docs.amd.com/latest/pytorch/quark_torch_best_practices.html#apply-quantization-algorithms), namely AutoSmoothQuant and GPTQ, for MXFP4 quantization.

## OCP FP8-e4m3 Quantization
We applied per-tensor symmetric static quantization weights and activations of nn.Linear modules—using the following formula: 

x_q = rounding( clip (x / scale * 448, -448, 448))

where x_q is the quantized form of value x, scale is the maximum absolute value (absmax) of the tensor, the constant 448 represents the numerical range of values in OCP FP8-e4m3. The scaled value is rounded using the half-even method after clipping.  

## OCP MXFP4 Quantization 

For OCP MXFP4, we used static quantization for weights and dynamic quantization for activations. The quantization formula is: 

x_q^MXFP4 = Encode_E2M1( clip (x / scale, -6,6)), scale = 2^floor(log2^rounding(max_abs(x)) - 2)

MXFP4 encodes 32 values per micro‑block, with each block sharing one 8‑bit E8M0 (power‑of‑two) scale factor and each element stored as a 4‑bit E2M1 floating‑point number. x_q^MXFP4 is the quantized form of value x, scale is a power‑of‑two value, stored once per block in 8‑bit E8M0 format. All values x are scaled such that x / scale falls within the representable FP4 range [−6, 6].  We apply even rounding in calculating scales to obtain better accuracy. 

## SmoothQuant                                                              

We apply SmoothQuant as a pre-quantization transformation during MXFP4 
quantization using [AMD Quark](https://quark.docs.amd.com/latest/). SmoothQuant migrates the quantization 
difficulty from activations to weights by mathematically transforming the
weight and activation tensors with a per-channel scaling factor. For a
linear layer $Y = XW$, SmoothQuant applies:

$$\hat{Y} = (X \text{diag}(s)^{-1}) \cdot (\text{diag}(s) W) =
\hat{X}\hat{W}$$

where $s$ is a per-channel smoothing factor computed as:

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

The hyper-parameter $\alpha$ controls the trade-off between activation
and weight quantization difficulty. We set $\alpha = 0.5$, selected
empirically through trial and error to maximize accuracy for the
DeepSeek-R1 model under Quark OCP MXFP4 quantization. The smoothing scales are
computed using the provided MLPerf calibration dataset and applied as a
static offline transformation during quantization.

## Summarizing the quantization per model

#### GPT-OSS-120B

* No quantization is used

#### DLRM v3

* No quantization is needed

#### DS R1

* SmoothQuant

#### LLaMA-2-70B

* OCP MXFP4 quantization
