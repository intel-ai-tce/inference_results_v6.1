# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

from pathlib import Path
from typing import Any


LEGACY_RELATIVE_PATH = (
    Path("deepseek-r1")
    / "fp4-quantized-modelopt"
    / "deepseek_r1-torch-fp4"
)
CENTML_OFFLINE_RELATIVE_PATH = (
    Path("deepseek-r1")
    / "official"
    / "centml-DeepSeek-R1-NVFP4-v2-mlpinf"
    / "93947a0d7bd04f73ff98636f6b18ff5839e7aaf9"
)


def _scenario_name(scenario: Any) -> str:
    return str(getattr(scenario, "valstr", scenario))


def select_harness_model_checkpoint(model_dir: Path, scenario: Any) -> Path:
    """Select the scenario-specific DeepSeek harness tokenizer/model path."""
    if _scenario_name(scenario) == "Offline":
        expected = (
            Path(model_dir)
            / "deepseek-r1"
            / "official"
            / "centml-DeepSeek-R1-NVFP4-v2-mlpinf"
            / "93947a0d7bd04f73ff98636f6b18ff5839e7aaf9"
        )
        selected = Path(model_dir) / CENTML_OFFLINE_RELATIVE_PATH
        if selected != expected:
            raise RuntimeError(
                "DeepSeek Offline harness checkpoint is not the exact CentML release"
            )
        return selected
    return Path(model_dir) / LEGACY_RELATIVE_PATH
