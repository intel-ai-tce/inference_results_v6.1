"""Plan 51 — int8 embedding storage as packed-fp16 (no NVE C++ rebuild).

The dominant ``item_id`` table (1e9 x 512 fp16, ~1 TB) is stored as **per-row
absmax int8 + a per-row fp32 scale**, halving its footprint. The NVE binding
hardcodes element_size = (fp32 ? 4 : 2) bytes, so a true 1-byte dtype would need
C++ changes + a rebuild of the certified .so. Instead we exploit the fact that the
LinearUVM gather is a *dtype-agnostic byte copy*: store the int8 row as a plain
``float16`` row of half-ish width whose bytes encode the int8 data + the scale.

  row bytes = embedding_dim int8 (data)  ++  1 fp32 (scale)        = dim + 4 bytes
            = (dim + 4) / 2 float16 slots

For dim=512: 516 bytes = 258 fp16 slots (vs 1024 bytes fp16 today -> ~50% smaller).
NVE sees a plain fp16 [N, 258] table (existing supported path; the auto
ManagedMemBlock sizes itself to 516 B/row). The int8 semantics live entirely in
the offline pack (load path) and the at-use unpack+dequant (gather path), so the
model still sees a 512-wide bf16 embedding. The fp32 scale makes the dequant
exactly reproduce the per-row int8 numerics that PASSED the M1 GAUC cert.

GAUC was certified at 99.9505% of fp16 ref (bar 99.9%) via the fake-quant
emulation; this module is the real storage path it stands in for.
"""
import os

import torch

SCALE_BYTES = 4  # per-row scale stored as fp32 (exact -> matches the M1 numerics)
INT8_MAX = 127.0

# Physical NVE tables stored packed-int8 when DLRM_NVE_INT8_GATHER is on. item_id
# backs both the item_id and item_candidate_id features (one physical table).
INT8_TABLES = {"item_id"}


def int8_gather_enabled() -> bool:
    return os.environ.get("DLRM_NVE_INT8_GATHER", "0").strip().lower() in (
        "1", "item_id", "true", "yes",
    )


def packed_fp16_width(embedding_dim: int) -> int:
    """Number of fp16 slots that hold ``embedding_dim`` int8 bytes + a fp32 scale."""
    row_bytes = embedding_dim + SCALE_BYTES
    assert row_bytes % 2 == 0, f"packed row bytes {row_bytes} must be even (dim must be even)"
    return row_bytes // 2


def pack_int8_fp16(ref: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """[N, embedding_dim] (fp16/fp32) table -> [N, packed_fp16_width] fp16.

    Per-row absmax symmetric int8 quantization; bytes are laid out as
    ``[dim int8 data][fp32 scale]`` then reinterpreted as fp16 slots. Runs on the
    input tensor's device (GPU at load time).
    """
    x = ref.to(torch.float32)
    n = x.shape[0]
    amax = x.abs().amax(dim=1, keepdim=True)
    scale = (amax / INT8_MAX).clamp_min(1e-12)  # fp32 [N,1]
    q = torch.round(x / scale).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)  # [N,dim]
    row_bytes = embedding_dim + SCALE_BYTES
    packed = torch.empty(n, row_bytes, dtype=torch.uint8, device=ref.device)
    packed[:, :embedding_dim] = q.view(torch.uint8)
    packed[:, embedding_dim:row_bytes] = (
        scale.squeeze(1).contiguous().view(torch.uint8).reshape(n, SCALE_BYTES)
    )
    return packed.view(torch.float16).reshape(n, row_bytes // 2)


def unpack_dequant(res: torch.Tensor, embedding_dim: int,
                   out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """[M, packed_fp16_width] fp16 gather result -> [M, embedding_dim] out_dtype."""
    row_bytes = embedding_dim + SCALE_BYTES
    rb = res.contiguous().view(torch.uint8).reshape(-1, row_bytes)
    data = rb[:, :embedding_dim].view(torch.int8).to(torch.float32)            # [M,dim]
    scale = rb[:, embedding_dim:row_bytes].contiguous().view(torch.float32)    # [M,1]
    return (data * scale).to(out_dtype)
