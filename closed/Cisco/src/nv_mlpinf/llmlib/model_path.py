# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

from pathlib import Path
from typing import Any


def resolve_harness_model_path(benchmark_module: Any, scenario: Any) -> str:
    """Resolve a model-specific harness path, retaining legacy fallback behavior."""
    resolver = getattr(benchmark_module, "get_harness_model_checkpoint_path", None)
    selected = (
        resolver(scenario)
        if callable(resolver)
        else getattr(benchmark_module, "MODEL_CHECKPOINT_PATH", None)
    )
    if selected is None:
        raise RuntimeError("Benchmark did not provide a harness model checkpoint path")
    path = Path(selected)
    if not path.is_absolute():
        raise RuntimeError(f"Harness model checkpoint path must be absolute: {path}")
    return str(path)
