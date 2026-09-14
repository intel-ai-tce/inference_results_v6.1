"""Pinned-solution fp8 GEMM op (DLRM-v3 GEMM tuning).

Public API:
    scaled_mm_tuned(a, b, scale_a, scale_b, m, n, k, lda, ldb, trans_a, trans_b,
                    bias=None, out_bf16=True, pin_index=-1, c=None) -> Tensor

    c: optional residual matrix folded via the hipBLASLt beta*C epilogue (beta=1),
       i.e. D = op(A)@op(B) (+bias) + c. Must match D's layout/dtype ([n, m]).
"""
from .fp8tuned import scaled_mm_tuned  # noqa: F401
