import shutil
from pathlib import Path

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_PATH = Path("/model/whisper-large-v3")
OUTPUT_DIR = Path("/model/whisper-large-v3_calibrated-cpu")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Expected full-precision Whisper model at {MODEL_PATH}. "
        "Run code/calibration/run_calibration.sh first."
    )

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
recipe = QuantizationModifier(targets="Linear", scheme="W8A8")

oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    output_dir=str(OUTPUT_DIR),
    save_compressed=True,
    pipeline="datafree",
    dataset=None,
    num_calibration_samples=None,
)
processor.save_pretrained(OUTPUT_DIR)

if not OUTPUT_DIR.exists():
    raise RuntimeError(f"Quantized model was not saved to {OUTPUT_DIR}")

required_files = ["config.json", "tokenizer_config.json"]
missing = [f for f in required_files if not (OUTPUT_DIR / f).exists()]
if missing:
    raise RuntimeError(
        f"Quantized model at {OUTPUT_DIR} is incomplete, missing: {missing}"
    )

shutil.rmtree(MODEL_PATH)
