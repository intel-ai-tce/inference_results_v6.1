"""ROCm/HIP env for NVIDIA harness on MI355 (gfx950)."""
from __future__ import annotations

import os
import sys
import types


def _stub_nvtx() -> None:
    if "nvtx" in sys.modules:
        return
    nvtx = types.ModuleType("nvtx")

    class _Annotate:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def __call__(self, fn):
            return fn

    def annotate(*_args, **_kwargs):
        return _Annotate()

    nvtx.annotate = annotate  # type: ignore[attr-defined]
    sys.modules["nvtx"] = nvtx


def _stub_cuda_nvtx() -> None:
    try:
        import torch

        if not hasattr(torch.cuda, "nvtx"):

            class _CudaNvtx:
                @staticmethod
                def range_push(_msg: str) -> None:
                    pass

                @staticmethod
                def range_pop() -> None:
                    pass

            torch.cuda.nvtx = _CudaNvtx()  # type: ignore[attr-defined]
    except ImportError:
        pass


def apply_rocm_env() -> None:
    """Defaults for HIP bring-up; call once per MPI rank before torch-heavy imports."""
    _stub_nvtx()
    _stub_cuda_nvtx()
    os.environ.setdefault("DLRM_SAFE_FBGEMM_CUMSUM", "1")
    os.environ.setdefault("DLRM_SKIP_DENSE_BATCH_CLONE", "1")
    os.environ.setdefault("DLRM_SKIP_AUTOTUNE", "1")
    if os.environ.get("DLRM_SKIP_FBGEMM_PATCH", "0") != "1":
        try:
            from generative_recommenders.ops.rocm_compat import (
                apply_rocm_fbgemm_cumsum_patch,
            )

            apply_rocm_fbgemm_cumsum_patch()
        except ImportError:
            pass
