"""Python (ctypes) front-end for the pinned-solution fp8 GEMM op.

scaled_mm_tuned(a, b, scale_a, scale_b, m, n, k, lda, ldb, trans_a, trans_b,
                bias=None, out_bf16=True, pin_index=-1) -> Tensor D[n, m]

Faithful to torch._scaled_mm: a=weight (op A), b=activation (op B), scale_a->A,
scale_b->B. pin_index<0 just uses the heuristic (behaviourally a no-op). The .so is
built (once) by build_lib.sh with hipcc; no torch headers (avoids the rocThrust/cub
include that breaks torch cpp_extension in this image).
"""
import ctypes
import os
import subprocess

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = None

_BIAS_DT = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}


def _arch() -> str:
    """Live GPU target (e.g. 'gfx950'), feature suffix stripped. GPU_ARCH overrides."""
    a = os.environ.get("GPU_ARCH")
    if not a:
        try:
            a = torch.cuda.get_device_properties(0).gcnArchName
        except Exception:  # noqa: BLE001
            a = ""
    return a.split(":")[0]


def _ensure():
    global _lib
    if _lib is not None:
        return _lib
    arch = _arch()
    # Arch-tagged binary so a gfx942 host never loads a gfx950 .so (and vice versa).
    so = os.path.join(_HERE, f"libfp8tuned_{arch}.so" if arch else "libfp8tuned.so")
    if not os.path.exists(so):
        env = dict(os.environ)
        if arch:
            env["GPU_ARCH"] = arch
        subprocess.check_call(["bash", os.path.join(_HERE, "build_lib.sh")], env=env)
    _lib = ctypes.CDLL(so)
    _lib.fp8_scaled_mm_tuned.restype = ctypes.c_int
    _lib.fp8_scaled_mm_tuned.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # a,b,sa,sb
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,                      # d,bias,bias_dt
        ctypes.c_void_p,                                                     # c (residual)
        ctypes.c_long, ctypes.c_long, ctypes.c_long,                         # m,n,k
        ctypes.c_long, ctypes.c_long,                                        # lda,ldb
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long,            # ta,tb,out_bf16,pin
        ctypes.c_void_p,                                                     # stream
    ]
    return _lib


def scaled_mm_tuned(a, b, scale_a, scale_b, m, n, k, lda, ldb,
                    trans_a, trans_b, bias=None, out_bf16=True, pin_index=-1,
                    c=None):
    """D = op(A)@op(B) (+ bias) (+ c).

    c: optional residual matrix folded via the hipBLASLt beta*C epilogue (beta=1).
    It must match D's layout/dtype: D is [n, m] row-major (== [m, n] col-major ld=m),
    so c must be a contiguous [n, m] tensor of the output dtype. Returns D[n, m].
    """
    lib = _ensure()
    out_dt = torch.bfloat16 if out_bf16 else torch.float8_e4m3fn
    d = torch.empty((n, m), device=a.device, dtype=out_dt)
    bias_ptr = 0
    bias_dt = 0
    if bias is not None:
        bias = bias.contiguous()
        bias_ptr = bias.data_ptr()
        bias_dt = _BIAS_DT.get(bias.dtype, 2)
    c_ptr = 0
    if c is not None:
        if c.dtype != out_dt:
            raise ValueError(
                f"residual c dtype {c.dtype} must equal output dtype {out_dt}"
            )
        c = c.contiguous()
        if tuple(c.shape) != (n, m):
            raise ValueError(f"residual c shape {tuple(c.shape)} must be ({n}, {m})")
        c_ptr = c.data_ptr()
    stream = torch.cuda.current_stream(a.device).cuda_stream
    rc = lib.fp8_scaled_mm_tuned(
        a.data_ptr(), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(),
        d.data_ptr(), bias_ptr, bias_dt, c_ptr,
        int(m), int(n), int(k), int(lda), int(ldb),
        1 if trans_a else 0, 1 if trans_b else 0, 1 if out_bf16 else 0, int(pin_index),
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"fp8_scaled_mm_tuned failed rc={rc} (m={m} n={n} k={k} pin={pin_index})")
    return d
