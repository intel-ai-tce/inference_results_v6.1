## Intel MLPerf Inference Calibration and Quantization Details

### GPT-OSS-120B Quantization
Model Source: https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a

Details: Source model is not quantized or calibrated.

### Llama2-70B (BMG) Quantization (Intel Arc Pro)
Model Source: https://huggingface.co/meta-llama/Llama-2-70b-chat-hf

Model Quantization: BF16 -> INT8

Details: The calibrated model (`llama-2-70b-chat-hf_calibrated-xpu`) is downloaded pre-quantized from MLCommons storage via /closed/Intel/src/llama3_1-8b/pytorch-xpu/download_resources.sh

### Llama3.1-8B Quantization (Intel Xeon)
Model Source: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

Model Quantization: BF16 -> INT8

Details: /closed/Intel/src/llama3_1-8b/pytorch-cpu/code/calibration/quantize_model.py (recipe: /closed/Intel/src/llama3_1-8b/pytorch-cpu/code/calibration/recipe.yaml)

### Llama3.1-8B (BMG) Quantization (Intel Arc Pro)
Model Source: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

Model Quantization: BF16 -> INT8

Details: The calibrated model (`llama3-1-8b-instruct_calibrated-xpu`) is downloaded pre-quantized from MLCommons storage via /closed/Intel/src/llama3_1-8b/pytorch-xpu/download_resources.sh

### R-GAT Quantization (Intel Xeon)
Model Source: https://github.com/IllinoisGraphBenchmark/IGB-Datasets/

Model Quantization: FP32 -> INT8

Implementation: /closed/Intel/src/rgat/pytorch-cpu/backend.py

### Whisper Quantization (Intel Xeon)
Model Source: https://huggingface.co/openai/whisper-large-v3

Model Quantization: BF16 -> INT8 W8A8 compressed-tensors

Implementation: /closed/Intel/src/whisper/pytorch-cpu/code/calibration/run_calibration.sh

Details:
- Downloads `openai/whisper-large-v3` with `hf download` when `/model/whisper-large-v3` is missing
- Quantizes with `llmcompressor` `oneshot(...)` and `QuantizationModifier(targets="Linear", scheme="W8A8")`
- Saves the compressed checkpoint and Whisper processor to `/model/whisper-large-v3_calibrated-cpu`
- Removes `/model/whisper-large-v3` after the quantized checkpoint is written successfully

### Whisper Quantization (Intel Arc Pro)
Model Source: https://huggingface.co/openai/whisper-large-v3

Model Quantization: BF16 -> INT8

Details: /closed/Intel/src/whisper/pytorch-xpu/code/calibration/calibrate_whisper.py
