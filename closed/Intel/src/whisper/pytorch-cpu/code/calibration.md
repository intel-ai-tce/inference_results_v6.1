## Intel MLPerf Inference Calibration and Quantization Details

### Llama3.1-8B Quantization
Model Source: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

Model Quantization: BF16 -> INT4

Details: /closed/Intel/src/llama3.1-8b/pytorch-cpu/code/calibration/run_calibration.sh

### R-GAT Quantization
Model Source: https://github.com/IllinoisGraphBenchmark/IGB-Datasets/

Model Quantization: FP32 -> INT8

Implementation: /closed/Intel/code/rgat/pytorch-cpu/backend.py

### Whisper Quantization
Model Source: https://huggingface.co/openai/whisper-large-v3

Model Quantization: BF16 -> INT8 W8A8 compressed-tensors

Implementation: /closed/Intel/code/whisper/pytorch-cpu/code/calibration/run_calibration.sh

Details:
- Downloads `openai/whisper-large-v3` with `hf download` when `/model/whisper-large-v3` is missing
- Quantizes with `llmcompressor` `oneshot(...)` and `QuantizationModifier(targets="Linear", scheme="W8A8")`
- Saves the compressed checkpoint and Whisper processor to `/model/whisper-large-v3_calibrated-cpu`
- Removes `/model/whisper-large-v3` after the quantized checkpoint is written successfully
