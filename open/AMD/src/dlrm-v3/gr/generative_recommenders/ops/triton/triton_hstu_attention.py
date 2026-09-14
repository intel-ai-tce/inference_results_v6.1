# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3

import os
from typing import Dict, List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore

    HAS_TLX = True
except ImportError:
    # suppress type checking errors
    tlx = None

    HAS_TLX = False

from generative_recommenders.common import (
    autotune_max_seq_len,
    prev_power_of_2,
    switch_to_contiguous_if_needed,
    triton_autotune,
)

from triton.language.extra.libdevice import (  # @manual=//triton:triton
    fast_dividef,
    fast_expf,
)


@triton.jit
def _silu_exact(qk):
    return fast_dividef(qk, 1.0 + fast_expf(-qk))


@triton.jit
def _silu_poly(qk):
    # Plan 42 P1 — exp-free fp8-safe SiLU gate (the −4.6% GOLD-attn winner). Approximate the odd
    # sigmoid: sigmoid(x) ≈ clamp(0.5 + oddpoly(clamp(x,±4)), 0,1), silu = x·sigmoid. Evaluated in
    # u=x² (short dependency chain, ~10 VALU ops, 0 transcendentals ⇒ removes v_exp/v_rcp hazard
    # stalls). fp8-bit-safe for the production score range (σ≲1.5: 0.18% mismatch vs exact→fp8);
    # the wide-σ tail error is a MONOTONIC under-estimate (ranking-preserving) ⇒ GAUC-confirmed
    # in step 3, not bit-exact. The ±4 clamp keeps the odd poly from diverging.
    xc = tl.minimum(tl.maximum(qk, -4.0), 4.0)
    u = xc * xc
    p = xc * (0.24984827 + u * (-0.020384835 + u * (0.0017372594 + u * (-9.992902e-05 + u * 2.52397e-06))))
    return qk * tl.minimum(tl.maximum(0.5 + p, 0.0), 1.0)


@triton.jit
def _silu_poly_deg5(qk):
    # GOLD q12,200 gate: degree-5 sigmoid approximation promoted after
    # Offline GAUC, q12,200 Server PROD10min, and TEST08 verification.
    # Degree-9 remains available with DLRM_HSTU_GATE_POLY_DEG=9.
    xc = tl.minimum(tl.maximum(qk, -4.0), 4.0)
    u = xc * xc
    p = xc * (0.24984827 + u * (-0.020384835 + u * 0.0017372594))
    return qk * tl.minimum(tl.maximum(0.5 + p, 0.0), 1.0)


# Plan 42 P1 + q12,200 GOLD — select the SiLU gate at import. DLRM_HSTU_GATE_POLY=1 uses an
# exp-free polynomial gate; DLRM_HSTU_GATE_POLY_DEG selects the promoted degree-5 GOLD gate
# or the prior degree-9 gate. Default off ⇒ exact gate. Read once; the chosen @triton.jit fn
# is traced into the kernel.
_HSTU_GATE_POLY: bool = os.environ.get("DLRM_HSTU_GATE_POLY", "0") == "1"
_HSTU_GATE_POLY_DEG: str = os.environ.get("DLRM_HSTU_GATE_POLY_DEG", "9").strip()
_GATE = _silu_exact
if _HSTU_GATE_POLY:
    _GATE = _silu_poly_deg5 if _HSTU_GATE_POLY_DEG == "5" else _silu_poly

# Plan 22 A1 — fp8 attention matmuls (DLRM_HSTU_FP8_ATTN=1). When enabled, the
# QK^T and SiLU·V dots in ``_hstu_attn_fwd_one_block`` run in e4m3 with the fp32
# accumulator preserved (~2x matmul throughput on CDNA4 / gfx950). Off by default
# so the fp16 path stays bit-identical. Read once at import; the value flows to
# the kernel as a ``tl.constexpr`` so fp8 vs fp16 specialize to distinct kernels.
_HSTU_FP8_ATTN: bool = os.environ.get("DLRM_HSTU_FP8_ATTN", "0") == "1"

# Plan 36 / fp4 — QK^T in mixed precision (Q e4m3, K MXFP4 e2m1). Microbench on
# gfx950 showed native fp4 MFMA makes the QK^T dot ~1.9x faster than e4m3 (AV
# stays e4m3; fp4 is slower at the BN=32 reduction). The accuracy cost is the
# gate: in the fused-attn microkernel, fp4-K raised the attention-output rel-err
# from 0.053 (e4m3) to ~0.125 (mixed Q-e4m3/K-fp4). DLRM_HSTU_FP4_QK_EMU=1 is a
# numerics-only GAUC probe: it round-trips K through MXFP4 (quantize->dequantize)
# before the existing e4m3 attention, faithfully reproducing the mixed-QK score
# numerics (e2m1 values are a subset of e4m3, so the e4m3 cast is lossless) with
# zero kernel changes — so a harness accuracy run measures the true GAUC impact
# before we invest in the in-kernel dot_scaled path. Off by default.
_HSTU_FP4_QK_EMU: bool = os.environ.get("DLRM_HSTU_FP4_QK_EMU", "0") == "1"

# Plan 36 / fp4 — in-kernel mixed QK^T (Q e4m3 x K MXFP4 e2m1) via tl.dot_scaled
# (native fp4 MFMA on gfx950, ~1.9x the e4m3 QK^T dot). K is pre-quantized to
# packed e2m1 + e8m0 in the launcher; the inner blocks consume it with dot_scaled
# while AV stays e4m3 (fp4 is slower at the BN reduction). Requires the fp8-attn
# path (q is e4m3) and the non-TMA block path; falls back to e4m3 otherwise.
# GAUC-validated safe (0.7862402 vs certified 0.7862408). Off by default ⇒ the
# certified fp8 kernel is byte-identical.
_HSTU_FP4_QK: bool = os.environ.get("DLRM_HSTU_FP4_QK", "0") == "1"

# Plan 37 — fused MXFP4-pack K-projection. The K slice of the UVQK projection is
# emitted *already packed* (e2m1 ``kp`` + e8m0 ``ks``) by a single fused hipBLASLt
# GEMM (transposed K^T output + RELU_AUX pack epilogue), replacing the standalone
# Triton ``_quantize_k_mxfp4`` pass. The producer (preprocess) hands ``kp``/``ks``
# straight to the attention launcher, which forces the FP4_QK dot path. Bit-exact
# vs ``_quantize_k_mxfp4`` (both RNE). Off by default ⇒ certified path untouched.
_HSTU_FP4_QK_FUSED: bool = os.environ.get("DLRM_HSTU_FP4_QK_FUSED", "0") == "1"


@triton.jit
def _mxfp4_quant_khd_kernel(
    x_ptr, p_ptr, s_ptr, M, K, NBLK, sxm, sxk, BLOCK_M: tl.constexpr
):
    # Single-pass MXFP4 quantizer: per 32-element block along K, compute the e8m0
    # (ceil-log2) shared scale and round-to-nearest e2m1 codes, pack 2 codes/byte.
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    e16 = tl.arange(0, 16)
    off_lo = rows[:, None] * sxm + (pid_b * 32 + e16[None, :] * 2) * sxk
    off_hi = off_lo + sxk
    m2 = rmask[:, None]
    xlo = tl.load(x_ptr + off_lo, mask=m2, other=0.0).to(tl.float32)
    xhi = tl.load(x_ptr + off_hi, mask=m2, other=0.0).to(tl.float32)
    amax = tl.maximum(
        tl.maximum(tl.max(tl.abs(xlo), 1), tl.max(tl.abs(xhi), 1)), 1e-20
    )
    eexp = tl.ceil(tl.log2(amax / 6.0))
    inv = (1.0 / tl.exp2(eexp))[:, None]
    # Round-to-nearest-EVEN on the e2m1 ladder, matching the fused hipBLASLt MXFP4-pack
    # epilogue (HW v_cvt_scalef32_pk_fp4_f32 is RNE) so this producer is a bit-exact
    # drop-in. Even-mantissa levels {0,1.0,2.0,4.0} break midpoint ties downward ('>'),
    # odd-mantissa levels {0.5,1.5,3.0,6.0} break upward ('>='). (Was round-half-up:
    # all '>='; verified bit-exact vs the kernel across shapes via derisk_pack_axis_pure.)
    ql = xlo * inv
    al = tl.abs(ql)
    cl = (
        tl.where(ql < 0, 8, 0).to(tl.int32)
        + (al > 0.25).to(tl.int32) + (al >= 0.75).to(tl.int32)
        + (al > 1.25).to(tl.int32) + (al >= 1.75).to(tl.int32)
        + (al > 2.5).to(tl.int32) + (al >= 3.5).to(tl.int32)
        + (al > 5.0).to(tl.int32)
    )
    qh = xhi * inv
    ah = tl.abs(qh)
    ch = (
        tl.where(qh < 0, 8, 0).to(tl.int32)
        + (ah > 0.25).to(tl.int32) + (ah >= 0.75).to(tl.int32)
        + (ah > 1.25).to(tl.int32) + (ah >= 1.75).to(tl.int32)
        + (ah > 2.5).to(tl.int32) + (ah >= 3.5).to(tl.int32)
        + (ah > 5.0).to(tl.int32)
    )
    byte = (cl | (ch << 4)).to(tl.uint8)
    tl.store(p_ptr + rows[:, None] * (K // 2) + pid_b * 16 + e16[None, :], byte, mask=m2)
    sb = tl.maximum(tl.minimum(eexp.to(tl.int32) + 127, 254), 0).to(tl.uint8)
    tl.store(s_ptr + rows * NBLK + pid_b, sb, mask=rmask)


def _quantize_k_mxfp4(k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack K [L,H,D] -> (kp [L,H,D/2] uint8 e2m1, ks [L,H,D/32] uint8 e8m0) for the
    mixed QK^T dot_scaled (per-32-block scale along D)."""
    L, H, D = k.shape
    x = k.reshape(L * H, D).contiguous()
    M = L * H
    NBLK = D // 32
    p = torch.empty(M, D // 2, dtype=torch.uint8, device=k.device)
    s = torch.empty(M, NBLK, dtype=torch.uint8, device=k.device)
    _mxfp4_quant_khd_kernel[(triton.cdiv(M, 16), NBLK)](
        x, p, s, M, D, NBLK, x.stride(0), x.stride(1), BLOCK_M=16
    )
    return p.view(L, H, D // 2), s.view(L, H, NBLK)
_FP4_QK_LEVELS: Dict[torch.device, torch.Tensor] = {}


def _mxfp4_roundtrip_k(t: torch.Tensor) -> torch.Tensor:
    """Quantize the last dim of ``t`` to MXFP4 (shared per-32-block power-of-two
    e8m0 scale, e2m1 round-to-nearest) and dequantize back to ``t``'s dtype.
    Reproduces the numerics of storing/consuming K as MXFP4 in the QK^T dot."""
    lv = _FP4_QK_LEVELS.get(t.device)
    if lv is None:
        lv = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=t.device)
        _FP4_QK_LEVELS[t.device] = lv
    *lead, D = t.shape
    r = t.reshape(*lead, D // 32, 32).float()
    amax = r.abs().amax(-1, keepdim=True).clamp(min=1e-20)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    q = r / scale
    # Round-to-nearest-EVEN e2m1 selection (matches _mxfp4_quant_khd_kernel and the
    # fused HW pack). Alternating '>'/'>=' midpoint compares break ties to the even
    # mantissa. (Was torch.bucketize == round-half-down, inconsistent with the kernel.)
    a = q.abs()
    idx = (
        (a > 0.25).to(torch.int64) + (a >= 0.75).to(torch.int64)
        + (a > 1.25).to(torch.int64) + (a >= 1.75).to(torch.int64)
        + (a > 2.5).to(torch.int64) + (a >= 3.5).to(torch.int64)
        + (a > 5.0).to(torch.int64)
    )
    mag = lv[idx]
    deq = (torch.sign(q) * mag * scale).reshape(*lead, D)
    return deq.to(t.dtype)

# Plan 30 #3 — interior/boundary mask split (DLRM_HSTU_ATTN_FASTMASK=1). In the
# full-causal (no-window) regime, KV blocks strictly below the diagonal and within
# the history region have keep≡1, so the per-block [BLOCK_M,BLOCK_N] causal/target/
# contextual mask is a provable no-op there. When enabled, those interior blocks skip
# the mask construction (bit-exact). Off by default ⇒ the certified path is byte-
# identical. Flows to the kernel as a ``tl.constexpr`` so it specializes a distinct
# kernel; ignored when HAS_MAX_ATTN_LEN (windowed) since the upper window bound makes
# below-diagonal blocks maskable.
_HSTU_ATTN_FASTMASK: bool = os.environ.get("DLRM_HSTU_ATTN_FASTMASK", "0") == "1"

# Plan 30 #3 deploy guard. fp8 attention output bits are fixed only by the N-reduction
# grouping — BLOCK_N, matrix_instr_nonkdim, kpack (empirically confirmed); BLOCK_M /
# num_warps / num_stages re-partition independent work and are bit-exact-preserving. So
# when FASTMASK is on we pin the autotune grid to the certified bit-params (BLOCK_N=32 /
# matrix_instr=16 / kpack=2) by default → whatever config autotune picks (e.g. the faster
# 128/32/8w/num_stages=1) is byte-identical to the certified run ⇒ NO GAUC re-cert
# ("Win A"). Set DLRM_HSTU_ATTN_FULLGRID=1 to lift that pin and let autotune explore
# bit-breaking tiles (e.g. BLOCK_N=64) for the larger but re-cert-requiring win ("Win B").
_HSTU_ATTN_FULLGRID: bool = os.environ.get("DLRM_HSTU_ATTN_FULLGRID", "0") == "1"

# Plan 30 #4 — occupancy lever. The HIP fwd autotune grid hardcodes waves_per_eu=0, so
# occupancy is never explored. On gfx950 the C1-off BLOCK_N=64 tile is VGPR-bound (88
# VGPR ⇒ ~23% occupancy), which leaves the QKᵀ→A·V dependency chain exposed. A
# waves_per_eu hint forces the compiler to shrink the live VGPR set (88→64) and ~doubles
# occupancy (23%→40%), hiding the stall — bit-exact, since waves_per_eu only bounds
# register allocation and never touches arithmetic (num_warps/num_stages/waves_per_eu are
# all bit-preserving for a fixed BLOCK_N/matrix_instr/kpack; verified torch.equal). When
# set, autotune additionally explores waves_per_eu∈{3,4} so it can pick the higher-
# occupancy config per shape. Off by default ⇒ grid + selection are unchanged. Composes
# with both Win A (bit-exact, no re-cert) and Win B (BLOCK_N=64, already re-certified).
_HSTU_ATTN_OCCTUNE: bool = os.environ.get("DLRM_HSTU_ATTN_OCCTUNE", "0") == "1"
_HSTU_ATTN_MASK_SUBTILE_N: int = int(
    os.environ.get("DLRM_HSTU_ATTN_MASK_SUBTILE_N", "0")
)
_HSTU_ATTN_MASK_ZERO_QK: bool = (
    os.environ.get("DLRM_HSTU_ATTN_MASK_ZERO_QK", "0") == "1"
)


def _fw_waves_per_eu() -> List[int]:
    return [0, 3, 4] if _HSTU_ATTN_OCCTUNE else [0]

try:
    # @manual=//triton:triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    tensor_descriptor_tma = True
except ImportError:
    tensor_descriptor_tma = False

try:
    from generative_recommenders.ops.triton.fb.triton_attention_utils import acc_dq
except ImportError:
    from generative_recommenders.ops.triton.triton_attention_utils import acc_dq


def _host_descriptor_pre_hook(nargs):
    if not tensor_descriptor_tma:
        return

    if not isinstance(nargs["Q"], TensorDescriptor):
        return
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_D_Q = nargs["BLOCK_D_Q"]
    BLOCK_D_V = nargs["BLOCK_D_V"]
    if "USE_TLX" in nargs and nargs["USE_TLX"]:
        BLOCK_M = BLOCK_M // nargs["NUM_MMA_GROUPS"]
    nargs["Q"].block_shape = [BLOCK_M, BLOCK_D_Q]
    nargs["V"].block_shape = [BLOCK_N, BLOCK_D_V]
    nargs["K"].block_shape = [BLOCK_N, BLOCK_D_Q]


def _hip_use_safe_fw_configs() -> bool:
    """Prune HIP autotune grid on gfx950 until full 48-config grid is validated."""
    if not torch.version.hip:
        return False
    env = os.environ.get("DLRM_HSTU_TRITON_SAFE_CONFIG", "").lower()
    if env in ("0", "false", "no"):
        return False
    if env in ("1", "true", "yes"):
        return True
    if os.environ.get("DLRM_HSTU_TRITON_FULL_AUTOTUNE", "0") == "1":
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
        return arch.startswith("gfx95")
    except Exception:
        return False


def _get_fw_configs() -> List[triton.Config]:  # noqa: C901
    configs = []
    if torch.version.hip:
        block_m_values = [32, 64] if _hip_use_safe_fw_configs() else [32, 64, 128]
        matrix_dims = [16] if _hip_use_safe_fw_configs() else [16, 32]
        for BLOCK_M in block_m_values:
            for BLOCK_N in [32, 64]:
                for num_stages in [1, 2]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in matrix_dims:
                            for waves_per_eu in _fw_waves_per_eu():
                                configs.append(
                                    triton.Config(
                                        {
                                            "BLOCK_M": BLOCK_M,
                                            "BLOCK_N": BLOCK_N,
                                            "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                            "waves_per_eu": waves_per_eu,
                                            "kpack": 2,
                                        },
                                        num_stages=num_stages,
                                        num_warps=num_warps,
                                    )
                                )
        if _HSTU_ATTN_FASTMASK and not _HSTU_ATTN_FULLGRID:
            # Win A: keep only tiles bit-identical to the certified run (see the
            # _HSTU_ATTN_FULLGRID note). BLOCK_M / num_warps / num_stages stay free.
            bitexact = [
                c
                for c in configs
                if c.kwargs.get("BLOCK_N") == 32
                and c.kwargs.get("matrix_instr_nonkdim") == 16
                and c.kwargs.get("kpack") == 2
            ]
            if bitexact:
                configs = bitexact
    else:
        configs = [
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
        ]

        # Add TLX configs if TLX is available
        if HAS_TLX:
            try:
                device_capability = torch.cuda.get_device_capability()[0]
            except (RuntimeError, AssertionError):
                # No CUDA device available
                device_capability = None

            if device_capability == 9:
                # H100 configs
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": 128,
                            "BLOCK_N": 64,
                            "USE_TLX": True,
                            "NUM_BUFFERS": 2,
                            "NUM_MMA_WARPS_PER_GROUP": 4,
                            "NUM_MMA_GROUPS": 2,
                        },
                        num_stages=0,
                        num_warps=4,
                        pre_hook=_host_descriptor_pre_hook,
                    ),
                )


    # ROCm autotune configs omit these constexpr keys unless added here (CUDA path adds them inside else).
    for config in configs:
        if not config.kwargs.get("USE_TLX", False):
            config.kwargs.setdefault("USE_TLX", False)
            config.kwargs.setdefault("NUM_BUFFERS", 1)
            config.kwargs.setdefault("NUM_MMA_WARPS_PER_GROUP", 1)
            config.kwargs.setdefault("NUM_MMA_GROUPS", 1)

    return configs


@triton.jit
def _hstu_attn_fwd_one_block(  # noqa: C901
    start_n,
    seq_len,
    offs_m,
    offs_n,
    q,
    K,
    V,
    K_block_ptr,
    V_block_ptr,
    offset_kh,
    offset_vh,
    seq_start,
    n_targets,
    alpha,
    MAX_SEQ_LEN,
    contextual_seq_len,
    max_attn_len,
    Kp,
    Ks,
    stride_kpn,
    stride_ksn,
    offset_kph,
    offset_ksh,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    FP8: tl.constexpr,
    FP4_QK: tl.constexpr = False,
    MASK_ZERO_QK: tl.constexpr = False,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = None
    qk = None
    if FP4_QK:
        # Plan 36: mixed QK^T — q (e4m3) x K (MXFP4 e2m1) via native scaled MFMA.
        # K is pre-packed [L,H,D/2] e2m1 + [L,H,D/32] e8m0; load the [BLOCK_N,*] tile
        # like V (row-major). Mask rows past seq_len (boundary block) -> 0.
        offs_nk = (seq_start + start_n + tl.arange(0, BLOCK_N)).to(tl.int64)
        kmask = (start_n + tl.arange(0, BLOCK_N))[:, None] < seq_len
        d2 = tl.arange(0, BLOCK_D_Q // 2)
        sg = tl.arange(0, BLOCK_D_Q // 32)
        kp = tl.load(
            Kp + offset_kph + offs_nk[:, None] * stride_kpn + d2[None, :],
            mask=kmask, other=0,
        )
        ks = tl.load(
            Ks + offset_ksh + offs_nk[:, None] * stride_ksn + sg[None, :],
            mask=kmask, other=0,
        )
        qk = tl.dot_scaled(
            q, None, "e4m3", kp.T, ks, "e2m1", out_dtype=tl.float32
        ) * alpha
    elif ENABLE_TMA:
        k = K.load(
            [(seq_start + start_n).to(tl.int32), offset_kh.to(tl.int32)],
        )
        # tma can only be loaded in one order, use trans afterwards
        if FP8:
            # Plan 22 A1: e4m3 QK^T, fp32 accumulate. q/k are post-projection
            # activations ~O(1), well inside e4m3 range (max 448) — no input scale.
            qk = tl.dot(
                q.to(tl.float8e4nv),
                tl.trans(k).to(tl.float8e4nv),
                allow_tf32=ALLOW_TF32,
            ) * alpha
        else:
            qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * alpha
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        if FP8:
            qk = tl.dot(
                q.to(tl.float8e4nv),
                k.to(tl.float8e4nv),
                allow_tf32=ALLOW_TF32,
            ) * alpha
        else:
            qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        offs_m = offs_m - contextual_seq_len + 1
        offs_m = tl.where(
            offs_m > 0,
            offs_m,
            0,
        )
        offs_n = offs_n - contextual_seq_len + 1
        offs_n = tl.where(
            offs_n > 0,
            offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(
            offs_m < max_ids,
            offs_m,
            max_ids,
        )
        offs_n = tl.where(
            offs_n < max_ids,
            offs_n,
            max_ids,
        )
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    invalid_mask = invalid_mask | (offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        invalid_mask = invalid_mask & (offs_m_minus_n <= max_attn_len)
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask = invalid_mask | (
            (offs_m[:, None] == 0) & (offs_n[None, :] < max_ids)
        )
    if MASK_ZERO_QK and FP8:
        qk = tl.where(invalid_mask, qk, 0.0)
    gated = _GATE(qk)
    v = None
    if ENABLE_TMA:
        v = V.load(
            [(seq_start + start_n).to(tl.int32), offset_vh.to(tl.int32)],
        )
    else:
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    if FP8:
        # Plan 22 A1: the 1/MAX_SEQ_LEN factor (~1/16k) would underflow e4m3, so
        # keep the SiLU gate at native scale (×{0,1} mask), do the e4m3 ·V dot,
        # then apply 1/MAX_SEQ_LEN to the fp32 block result. Summing the scaled
        # per-block results is identical to scaling once at the end.
        if MASK_ZERO_QK:
            silu8 = gated.to(tl.float8e4nv)
        else:
            keep = tl.where(invalid_mask, 1.0, 0.0)
            silu8 = (gated * keep).to(tl.float8e4nv)
        return tl.dot(silu8, v.to(tl.float8e4nv), allow_tf32=ALLOW_TF32) * (
            1.0 / MAX_SEQ_LEN
        )
    scale = tl.where(invalid_mask, (1.0 / MAX_SEQ_LEN), 0.0)
    silu = gated * scale
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


@triton.jit
def _hstu_attn_fwd_one_block_mask_subtile(  # noqa: C901
    start_n,
    seq_len,
    offs_m,
    q,
    K,
    V,
    offset_kh,
    offset_vh,
    seq_start,
    n_targets,
    alpha,
    MAX_SEQ_LEN,
    contextual_seq_len,
    max_attn_len,
    stride_kn,
    stride_vn,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N_SUB: tl.constexpr,
    FP8: tl.constexpr,
):
    # Plan 65 4.B experiment: keep the outer Win-B BLOCK_N=64 schedule, but
    # consume masked boundary blocks as two raw-loaded N=32 subtiles. This
    # shortens the live score/gate tile in the pressure-heavy masked path.
    start_n = tl.multiple_of(start_n, BLOCK_N_SUB)
    offs_n = start_n + tl.arange(0, BLOCK_N_SUB)

    offs_kd = tl.arange(0, BLOCK_D_Q)
    offs_vd = tl.arange(0, BLOCK_D_V)
    abs_n = (seq_start + offs_n).to(tl.int64)
    nmask = offs_n < seq_len
    k = tl.load(
        K + offset_kh + offs_kd[:, None] + abs_n[None, :] * stride_kn,
        mask=nmask[None, :],
        other=0.0,
    )
    if FP8:
        qk = tl.dot(
            q.to(tl.float8e4nv),
            k.to(tl.float8e4nv),
            allow_tf32=ALLOW_TF32,
        ) * alpha
    else:
        qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha

    invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        offs_m = offs_m - contextual_seq_len + 1
        offs_m = tl.where(offs_m > 0, offs_m, 0)
        offs_n = offs_n - contextual_seq_len + 1
        offs_n = tl.where(offs_n > 0, offs_n, 0)
        max_ids = max_ids - contextual_seq_len + 1
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(offs_m < max_ids, offs_m, max_ids)
        offs_n = tl.where(offs_n < max_ids, offs_n, max_ids)
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    invalid_mask = invalid_mask | (offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        invalid_mask = invalid_mask & (offs_m_minus_n <= max_attn_len)
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask = invalid_mask | (
            (offs_m[:, None] == 0) & (offs_n[None, :] < max_ids)
        )

    gated = _GATE(qk)
    v = tl.load(
        V + offset_vh + abs_n[:, None] * stride_vn + offs_vd[None, :],
        mask=nmask[:, None],
        other=0.0,
    )
    if FP8:
        keep = tl.where(invalid_mask, 1.0, 0.0)
        silu8 = (gated * keep).to(tl.float8e4nv)
        return tl.dot(silu8, v.to(tl.float8e4nv), allow_tf32=ALLOW_TF32) * (
            1.0 / MAX_SEQ_LEN
        )
    scale = tl.where(invalid_mask, (1.0 / MAX_SEQ_LEN), 0.0)
    silu = (gated * scale).to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


@triton.jit
def _hstu_attn_fwd_one_block_nomask(
    start_n,
    q,
    K,
    V,
    K_block_ptr,
    V_block_ptr,
    offset_kh,
    offset_vh,
    seq_start,
    alpha,
    MAX_SEQ_LEN,
    Kp,
    Ks,
    stride_kpn,
    stride_ksn,
    offset_kph,
    offset_ksh,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    FP8: tl.constexpr,
    FP4_QK: tl.constexpr = False,
):
    # Plan 30 #3 — interior KV block: strictly below the causal diagonal and entirely
    # inside the history region ⇒ keep≡1 for every (m, n). This is _hstu_attn_fwd_one_block
    # with the per-block causal/target/contextual mask specialized to all-ones: gated*1.0
    # is exact in fp32 and 1.0→e4m3 is exact, so the result is bit-identical to the masked
    # path while skipping the entire mask VALU. Kept as a separate (branch-free) function
    # so the interior loop software-pipelines cleanly.
    start_n = tl.multiple_of(start_n, BLOCK_N)
    if FP4_QK:
        # Plan 36: mixed QK^T (q e4m3 x K e2m1). Interior blocks are fully in-history,
        # so the [BLOCK_N,*] packed-K tile is in-bounds (no row mask needed).
        offs_nk = (seq_start + start_n + tl.arange(0, BLOCK_N)).to(tl.int64)
        d2 = tl.arange(0, BLOCK_D_Q // 2)
        sg = tl.arange(0, BLOCK_D_Q // 32)
        kp = tl.load(Kp + offset_kph + offs_nk[:, None] * stride_kpn + d2[None, :])
        ks = tl.load(Ks + offset_ksh + offs_nk[:, None] * stride_ksn + sg[None, :])
        qk = tl.dot_scaled(
            q, None, "e4m3", kp.T, ks, "e2m1", out_dtype=tl.float32
        ) * alpha
    elif ENABLE_TMA:
        k = K.load(
            [(seq_start + start_n).to(tl.int32), offset_kh.to(tl.int32)],
        )
        if FP8:
            qk = tl.dot(
                q.to(tl.float8e4nv),
                tl.trans(k).to(tl.float8e4nv),
                allow_tf32=ALLOW_TF32,
            ) * alpha
        else:
            qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * alpha
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        if FP8:
            qk = tl.dot(
                q.to(tl.float8e4nv),
                k.to(tl.float8e4nv),
                allow_tf32=ALLOW_TF32,
            ) * alpha
        else:
            qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    gated = _GATE(qk)
    if ENABLE_TMA:
        v = V.load(
            [(seq_start + start_n).to(tl.int32), offset_vh.to(tl.int32)],
        )
    else:
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    if FP8:
        silu8 = gated.to(tl.float8e4nv)
        return tl.dot(silu8, v.to(tl.float8e4nv), allow_tf32=ALLOW_TF32) * (
            1.0 / MAX_SEQ_LEN
        )
    silu = (gated * (1.0 / MAX_SEQ_LEN)).to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


@triton.jit
def _hstu_attn_fwd_compute(  # noqa C901
    Q,
    K,
    V,
    H,
    DimQ,
    DimV,
    workspace_ptr,
    seq_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    MAX_SEQ_LEN,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    off_z,
    off_h,
    pid,
    Kp,
    Ks,
    stride_kpn,
    stride_kph,
    stride_ksn,
    stride_ksh,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
    FP8: tl.constexpr,
    FAST_INTERIOR: tl.constexpr = False,
    FP4_QK: tl.constexpr = False,
    MASK_SUBTILE_N: tl.constexpr = 0,
    MASK_ZERO_QK: tl.constexpr = False,
):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)

    if IS_DELTA_Q:
        start_m_delta = pid * BLOCK_M
        start_m = (start_m_delta + seq_len - DeltaSize).to(tl.int32)
    else:
        start_m_delta = 0
        start_m = pid * BLOCK_M
    if start_m < seq_len:
        if HAS_MULTIPLE_TARGETS:
            n_targets = tl.load(num_targets + off_z).to(tl.int32)
        else:
            n_targets = None

        # initialize offsets
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        Q_block_ptr = None
        K_block_ptr = None
        V_block_ptr = None
        if not ENABLE_TMA:
            if IS_DELTA_Q:
                Q_block_ptr = tl.make_block_ptr(
                    base=Q + off_h * stride_qh + off_z * DeltaSize * stride_qm,
                    shape=(DeltaSize, BLOCK_D_Q),
                    strides=(stride_qm, 1),
                    offsets=(start_m_delta, 0),
                    block_shape=(BLOCK_M, BLOCK_D_Q),
                    order=(1, 0),
                )
            else:
                Q_block_ptr = tl.make_block_ptr(
                    base=Q + off_h * stride_qh + seq_start * stride_qm,
                    shape=(seq_len, BLOCK_D_Q),
                    strides=(stride_qm, 1),
                    offsets=(start_m, 0),
                    block_shape=(BLOCK_M, BLOCK_D_Q),
                    order=(1, 0),
                )
            q = tl.load(
                Q_block_ptr, boundary_check=(
                    0,), padding_option="zero")

            K_block_ptr = tl.make_block_ptr(
                base=K + off_h * stride_kh + seq_start * stride_kn,
                shape=(BLOCK_D_Q, seq_len),
                strides=(1, stride_kn),
                offsets=(0, 0),
                block_shape=(BLOCK_D_Q, BLOCK_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=V + off_h * stride_vh + seq_start * stride_vn,
                shape=(seq_len, BLOCK_D_V),
                strides=(stride_vn, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D_V),
                order=(1, 0),
            )
        else:
            if IS_DELTA_Q:
                q = Q.load(
                    [
                        (off_z * DeltaSize + start_m_delta).to(tl.int32),
                        (off_h * stride_qh).to(tl.int32),
                    ]
                )
            else:
                q = Q.load(
                    [
                        (seq_start + start_m).to(tl.int32),
                        (off_h * stride_qh).to(tl.int32),
                    ]
                )

        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        if HAS_MULTIPLE_TARGETS:
            uih_end = seq_len - n_targets
        else:
            uih_end = seq_len
        # Plan 30 #3: unrounded history boundary for the interior-block predicate
        # (uih_end is later rounded up to BLOCK_N for the delta bound).
        uih_unrounded = uih_end
        if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
            # uih_end must be larger than start_m
            low = 0
            high = seq_len
        else:
            low = 0
            high = start_m + BLOCK_M
            if HAS_MAX_ATTN_LEN:
                if start_m > uih_end:
                    low = uih_end - max_attn_len
                else:
                    low = start_m - max_attn_len
                if HAS_CONTEXTUAL_SEQ_LEN:
                    low = low if low > contextual_seq_len else 0
                else:
                    low = low if low > 0 else 0
            if HAS_MULTIPLE_TARGETS:
                uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
                if uih_end < start_m:
                    high = seq_len - n_targets

        # Plan 30 #3: interior/boundary split. Blocks in [low, n_split) lie strictly
        # below the causal diagonal AND fully inside history (cols < uih_unrounded), so
        # keep≡1 there (no contextual/target/diagonal interaction) ⇒ the per-block mask
        # is a provable no-op and is skipped. Disabled under a window (HAS_MAX_ATTN_LEN
        # adds an upper bound that can mask below-diagonal blocks).
        n_split = low
        if FAST_INTERIOR and not HAS_MAX_ATTN_LEN:
            interior_bound = uih_unrounded
            if start_m < interior_bound:
                interior_bound = start_m
            n_split = (interior_bound // BLOCK_N) * BLOCK_N
            if n_split < low:
                n_split = low

        if low > 0:
            if not ENABLE_TMA:
                K_block_ptr = tl.advance(K_block_ptr, (0, low))
                V_block_ptr = tl.advance(V_block_ptr, (low, 0))
        end_n = low
        if FAST_INTERIOR and not HAS_MAX_ATTN_LEN:
            # Plan 30 #3: tight, branch-free interior loop [low, n_split) — keep≡1, so
            # no mask VALU. Split from the boundary loop (not a per-block branch) so this
            # hot loop still software-pipelines on the certified tiling.
            for start_n in range(low, n_split, BLOCK_N):
                acc += _hstu_attn_fwd_one_block_nomask(
                    start_n=start_n,
                    q=q,
                    K=K,
                    V=V,
                    K_block_ptr=K_block_ptr,
                    V_block_ptr=V_block_ptr,
                    offset_kh=off_h * stride_kh,
                    offset_vh=off_h * stride_vh,
                    seq_start=seq_start,
                    alpha=alpha,
                    MAX_SEQ_LEN=MAX_SEQ_LEN,
                    ALLOW_TF32=ALLOW_TF32,
                    BLOCK_D_Q=BLOCK_D_Q,
                    BLOCK_D_V=BLOCK_D_V,
                    BLOCK_N=BLOCK_N,
                    Kp=Kp,
                    Ks=Ks,
                    stride_kpn=stride_kpn,
                    stride_ksn=stride_ksn,
                    offset_kph=off_h * stride_kph,
                    offset_ksh=off_h * stride_ksh,
                    FP4_QK=FP4_QK,
                    ENABLE_TMA=ENABLE_TMA,
                    FP8=FP8,
                )
                if not ENABLE_TMA:
                    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
                end_n += BLOCK_N
            # boundary loop [n_split, high): diagonal / target / contextual edge — full mask
            for start_n in range(n_split, high, BLOCK_N):
                if (
                    MASK_SUBTILE_N == 32
                    and BLOCK_N == 64
                    and FP8
                    and not ENABLE_TMA
                    and not FP4_QK
                ):
                    acc += _hstu_attn_fwd_one_block_mask_subtile(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        q=q,
                        K=K,
                        V=V,
                        offset_kh=off_h * stride_kh,
                        offset_vh=off_h * stride_vh,
                        seq_start=seq_start,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                        BLOCK_N_SUB=32,
                        FP8=FP8,
                    )
                    acc += _hstu_attn_fwd_one_block_mask_subtile(
                        start_n=start_n + 32,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        q=q,
                        K=K,
                        V=V,
                        offset_kh=off_h * stride_kh,
                        offset_vh=off_h * stride_vh,
                        seq_start=seq_start,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                        BLOCK_N_SUB=32,
                        FP8=FP8,
                    )
                else:
                    acc += _hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=offs_n + start_n,
                        q=q,
                        K=K,
                        V=V,
                        K_block_ptr=K_block_ptr,
                        V_block_ptr=V_block_ptr,
                        offset_kh=off_h * stride_kh,
                        offset_vh=off_h * stride_vh,
                        seq_start=seq_start,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                        BLOCK_N=BLOCK_N,
                        Kp=Kp,
                        Ks=Ks,
                        stride_kpn=stride_kpn,
                        stride_ksn=stride_ksn,
                        offset_kph=off_h * stride_kph,
                        offset_ksh=off_h * stride_ksh,
                        FP4_QK=FP4_QK,
                        ENABLE_TMA=ENABLE_TMA,
                        FP8=FP8,
                        MASK_ZERO_QK=MASK_ZERO_QK,
                    )
                if not ENABLE_TMA:
                    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
                end_n += BLOCK_N
        else:
            for start_n in range(low, high, BLOCK_N):
                if (
                    MASK_SUBTILE_N == 32
                    and BLOCK_N == 64
                    and FP8
                    and not ENABLE_TMA
                    and not FP4_QK
                ):
                    acc += _hstu_attn_fwd_one_block_mask_subtile(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        q=q,
                        K=K,
                        V=V,
                        offset_kh=off_h * stride_kh,
                        offset_vh=off_h * stride_vh,
                        seq_start=seq_start,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                        BLOCK_N_SUB=32,
                        FP8=FP8,
                    )
                    acc += _hstu_attn_fwd_one_block_mask_subtile(
                        start_n=start_n + 32,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        q=q,
                        K=K,
                        V=V,
                        offset_kh=off_h * stride_kh,
                        offset_vh=off_h * stride_vh,
                        seq_start=seq_start,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        stride_kn=stride_kn,
                        stride_vn=stride_vn,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                        BLOCK_N_SUB=32,
                        FP8=FP8,
                    )
                else:
                    acc += _hstu_attn_fwd_one_block(
                        start_n=start_n,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=offs_n + start_n,
                        q=q,
                        K=K,
                        V=V,
                        K_block_ptr=K_block_ptr,
                        V_block_ptr=V_block_ptr,
                        offset_kh=off_h * stride_kh,
                        offset_vh=off_h * stride_vh,
                        seq_start=seq_start,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                        BLOCK_N=BLOCK_N,
                        Kp=Kp,
                        Ks=Ks,
                        stride_kpn=stride_kpn,
                        stride_ksn=stride_ksn,
                        offset_kph=off_h * stride_kph,
                        offset_ksh=off_h * stride_ksh,
                        FP4_QK=FP4_QK,
                        ENABLE_TMA=ENABLE_TMA,
                        FP8=FP8,
                        MASK_ZERO_QK=MASK_ZERO_QK,
                    )
                if not ENABLE_TMA:
                    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
                end_n += BLOCK_N

        if HAS_MULTIPLE_TARGETS:
            # pyre-ignore[61]
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                offset = (low_delta - end_n).to(tl.int32)
                if not ENABLE_TMA:
                    K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                    V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
                for start_delta in tl.range(
                    low_delta, high_delta, BLOCK_N, num_stages=0
                ):
                    if (
                        MASK_SUBTILE_N == 32
                        and BLOCK_N == 64
                        and FP8
                        and not ENABLE_TMA
                        and not FP4_QK
                    ):
                        acc += _hstu_attn_fwd_one_block_mask_subtile(
                            start_n=start_delta,
                            seq_len=seq_len,
                            offs_m=offs_m,
                            q=q,
                            K=K,
                            V=V,
                            offset_kh=off_h * stride_kh,
                            offset_vh=off_h * stride_vh,
                            seq_start=seq_start,
                            n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                            alpha=alpha,
                            MAX_SEQ_LEN=MAX_SEQ_LEN,
                            contextual_seq_len=contextual_seq_len,
                            max_attn_len=max_attn_len,
                            stride_kn=stride_kn,
                            stride_vn=stride_vn,
                            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                            ALLOW_TF32=ALLOW_TF32,
                            BLOCK_D_Q=BLOCK_D_Q,
                            BLOCK_D_V=BLOCK_D_V,
                            BLOCK_N_SUB=32,
                            FP8=FP8,
                        )
                        acc += _hstu_attn_fwd_one_block_mask_subtile(
                            start_n=start_delta + 32,
                            seq_len=seq_len,
                            offs_m=offs_m,
                            q=q,
                            K=K,
                            V=V,
                            offset_kh=off_h * stride_kh,
                            offset_vh=off_h * stride_vh,
                            seq_start=seq_start,
                            n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                            alpha=alpha,
                            MAX_SEQ_LEN=MAX_SEQ_LEN,
                            contextual_seq_len=contextual_seq_len,
                            max_attn_len=max_attn_len,
                            stride_kn=stride_kn,
                            stride_vn=stride_vn,
                            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                            ALLOW_TF32=ALLOW_TF32,
                            BLOCK_D_Q=BLOCK_D_Q,
                            BLOCK_D_V=BLOCK_D_V,
                            BLOCK_N_SUB=32,
                            FP8=FP8,
                        )
                    else:
                        acc += _hstu_attn_fwd_one_block(
                            start_n=start_delta,
                            seq_len=seq_len,
                            offs_m=offs_m,
                            offs_n=offs_n + start_delta,
                            q=q,
                            K=K,
                            V=V,
                            K_block_ptr=K_block_ptr,
                            V_block_ptr=V_block_ptr,
                            offset_kh=off_h * stride_kh,
                            offset_vh=off_h * stride_vh,
                            seq_start=seq_start,
                            n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                            alpha=alpha,
                            MAX_SEQ_LEN=MAX_SEQ_LEN,
                            contextual_seq_len=contextual_seq_len,
                            max_attn_len=max_attn_len,
                            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                            ALLOW_TF32=ALLOW_TF32,
                            BLOCK_D_Q=BLOCK_D_Q,
                            BLOCK_D_V=BLOCK_D_V,
                            BLOCK_N=BLOCK_N,
                            Kp=Kp,
                            Ks=Ks,
                            stride_kpn=stride_kpn,
                            stride_ksn=stride_ksn,
                            offset_kph=off_h * stride_kph,
                            offset_ksh=off_h * stride_ksh,
                            FP4_QK=FP4_QK,
                            ENABLE_TMA=ENABLE_TMA,
                            FP8=FP8,
                            MASK_ZERO_QK=MASK_ZERO_QK,
                        )
                    if not ENABLE_TMA:
                        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        # Don't use TMA in Jagged case since we don't want to overwrite
        # the output of another sequence
        if IS_DELTA_Q:
            start_m_delta = pid * BLOCK_M
            offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + off_z * DeltaSize * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m_delta[:,
                                            None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
        else:
            # rematerialize offsets to save registers
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + seq_start * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


@triton.jit
def _hstu_attn_fwd_compute_main_loop_tlx(  # noqa C901
    low,
    high,
    seq_len,
    offs_m,
    offs_n,
    acc,
    q_tiles,
    k_tiles,
    v_tiles,
    q_fulls,
    k_fulls,
    v_fulls,
    k_empties,
    v_empties,
    v_dtype,
    n_targets,
    alpha,
    end_n,
    loop_trip_cnt,
    max_attn_len,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    cid: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    WAIT_FOR_Q: tl.constexpr,
):
    if WAIT_FOR_Q:
        # wait for the Q buffer to be populated by the producer
        q_full = tlx.local_view(q_fulls, cid)
        tlx.barrier_wait(q_full, 0)

    q_tile = tlx.local_view(q_tiles, cid)

    for start in tl.range(low + BLOCK_N, high, BLOCK_N, num_stages=0):
        buf_id = loop_trip_cnt % NUM_BUFFERS
        # buffers in a row share the same phase
        kv_phase = (loop_trip_cnt // NUM_BUFFERS) % 2

        start_n = tl.multiple_of(start, BLOCK_N)
        offs_n_start = offs_n
        offs_n = offs_n_start + start_n

        # wait for the K buffer to be populated by the producer
        k_full = tlx.local_view(k_fulls, buf_id)
        tlx.barrier_wait(k_full, kv_phase)
        k_tile = tlx.local_view(k_tiles, buf_id)

        # tma can only be loaded in one order, use trans afterwards
        k_tile = tlx.local_trans(k_tile)
        # second
        qk = tlx.async_dot(q_tile, k_tile)
        # wait for the MMA using to complete
        qk = tlx.async_dot_wait(0, qk)
        # release the K buffer
        k_empty = tlx.local_view(k_empties, buf_id)
        tlx.barrier_arrive(k_empty, 1)

        qk = qk * alpha

        invalid_mask = offs_m[:, None] == offs_n[None, :]
        max_ids = seq_len
        if HAS_MULTIPLE_TARGETS:
            max_ids = max_ids - n_targets
            offs_m = tl.where(
                offs_m < max_ids,
                offs_m,
                max_ids,
            )
            offs_n = tl.where(
                offs_n < max_ids,
                offs_n,
                max_ids,
            )
        offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
        invalid_mask = invalid_mask | (offs_m_minus_n > 0)
        if HAS_MAX_ATTN_LEN:
            invalid_mask = invalid_mask & (offs_m_minus_n <= max_attn_len)
        if HAS_CONTEXTUAL_SEQ_LEN:
            invalid_mask = invalid_mask | (
                (offs_m[:, None] == 0) & (offs_n[None, :] < max_ids)
            )
        scale = tl.where(invalid_mask, (1.0 / MAX_SEQ_LEN), 0.0)
        silu = _GATE(qk) * scale
        silu = silu.to(v_dtype)

        # wait for the V buffer to be populated by the producer
        v_full = tlx.local_view(v_fulls, buf_id)
        tlx.barrier_wait(v_full, kv_phase)
        v_tile = tlx.local_view(v_tiles, buf_id)
        acc = tlx.async_dot(silu, v_tile, acc)
        # wait for the MMA using to complete
        acc = tlx.async_dot_wait(0, acc)
        # release the V buffer
        v_empty = tlx.local_view(v_empties, buf_id)
        tlx.barrier_arrive(v_empty, 1)

        end_n += BLOCK_N

        # increment loop trip counts
        loop_trip_cnt += 1

    return acc, end_n, loop_trip_cnt


@triton.jit
def _hstu_attn_fwd_compute_main_loop_tlx_pipelined(  # noqa C901
    low,
    high,
    seq_len,
    offs_m,
    offs_n,
    acc,
    q_tiles,
    k_tiles,
    v_tiles,
    q_fulls,
    k_fulls,
    v_fulls,
    k_empties,
    v_empties,
    v_dtype,
    n_targets,
    alpha,
    end_n,
    loop_trip_cnt,
    max_attn_len,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    cid: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    WAIT_FOR_Q: tl.constexpr,
):
    if WAIT_FOR_Q:
        # wait for the Q buffer to be populated by the producer
        q_full = tlx.local_view(q_fulls, cid)
        tlx.barrier_wait(q_full, 0)
    q_tile = tlx.local_view(q_tiles, cid)

    # wait for the K buffer to be populated by the producer
    k_buf_id = loop_trip_cnt % NUM_BUFFERS
    # buffers in a row share the same phase
    k_phase = (loop_trip_cnt // NUM_BUFFERS) % 2

    k_full = tlx.local_view(k_fulls, k_buf_id)
    tlx.barrier_wait(k_full, k_phase)
    k_tile = tlx.local_view(k_tiles, k_buf_id)

    # tma can only be loaded in one order, use trans afterwards
    k_tile = tlx.local_trans(k_tile)

    # Pingpong
    if cid == 0:
        # Consumer 0 waits for Consumer 1 to reach synchronization point at
        # barrier 9.
        tlx.named_barrier_wait(9, 256)
    else:
        # Consumer 1 signals its arrival at barrier 9.
        tlx.named_barrier_arrive(9, 256)
        # Then waits at barrier 10 until Consumer 0 finishes issuing its
        # async_dot.
        tlx.named_barrier_wait(10, 256)

    qk = tlx.async_dot(q_tile, k_tile)

    if cid == 0:
        # After issuing async_dot, Consumer 0 signals barrier 10 to unblock
        # Consumer 1.
        tlx.named_barrier_arrive(10, 256)

    # wait for the MMA using to complete
    qk = tlx.async_dot_wait(0, qk)
    # release the K buffer
    k_empty = tlx.local_view(k_empties, k_buf_id)
    tlx.barrier_arrive(k_empty, 1)

    qk = qk * alpha

    start_n = tl.multiple_of(low, BLOCK_N)
    offs_n_start = offs_n
    offs_n = offs_n_start + start_n

    invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(
            offs_m < max_ids,
            offs_m,
            max_ids,
        )
        offs_n = tl.where(
            offs_n < max_ids,
            offs_n,
            max_ids,
        )
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    invalid_mask = invalid_mask | (offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        invalid_mask = invalid_mask & (offs_m_minus_n <= max_attn_len)
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask = invalid_mask | (
            (offs_m[:, None] == 0) & (offs_n[None, :] < max_ids)
        )
    scale = tl.where(invalid_mask, (1.0 / MAX_SEQ_LEN), 0.0)
    silu = _GATE(qk) * scale
    silu = silu.to(v_dtype)

    loop_trip_cnt += 1

    for start in tl.range(low + BLOCK_N, high, BLOCK_N, num_stages=0):
        start_n = tl.multiple_of(start, BLOCK_N)
        offs_n = offs_n_start + start_n

        k_buf_id = loop_trip_cnt % NUM_BUFFERS
        # buffers in a row share the same phase
        k_phase = k_phase ^ (k_buf_id == 0)

        # wait for the K buffer to be populated by the producer
        k_full = tlx.local_view(k_fulls, k_buf_id)
        tlx.barrier_wait(k_full, k_phase)
        k_tile = tlx.local_view(k_tiles, k_buf_id)

        # tma can only be loaded in one order, use trans afterwards
        k_tile = tlx.local_trans(k_tile)

        qk = tlx.async_dot(q_tile, k_tile)
        # wait for the MMA using to complete
        prev_silu = silu

        v_buf_id = (loop_trip_cnt - 1) % NUM_BUFFERS
        # v_phase = v_phase ^ (v_buf_id == 0)
        v_phase = ((loop_trip_cnt - 1) // NUM_BUFFERS) % 2
        v_full = tlx.local_view(v_fulls, v_buf_id)
        tlx.barrier_wait(v_full, v_phase)
        v_tile = tlx.local_view(v_tiles, v_buf_id)
        acc = tlx.async_dot(prev_silu, v_tile, acc)
        qk = tlx.async_dot_wait(1, qk)

        # release the K buffer
        k_empty = tlx.local_view(k_empties, k_buf_id)
        tlx.barrier_arrive(k_empty, 1)

        qk = qk * alpha
        invalid_mask = offs_m[:, None] == offs_n[None, :]
        max_ids = seq_len
        if HAS_MULTIPLE_TARGETS:
            max_ids = max_ids - n_targets
            offs_m = tl.where(
                offs_m < max_ids,
                offs_m,
                max_ids,
            )
            offs_n = tl.where(
                offs_n < max_ids,
                offs_n,
                max_ids,
            )
        offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
        invalid_mask = invalid_mask | (offs_m_minus_n > 0)
        if HAS_MAX_ATTN_LEN:
            invalid_mask = invalid_mask & (offs_m_minus_n <= max_attn_len)
        if HAS_CONTEXTUAL_SEQ_LEN:
            invalid_mask = invalid_mask | (
                (offs_m[:, None] == 0) & (offs_n[None, :] < max_ids)
            )
        scale = tl.where(invalid_mask, (1.0 / MAX_SEQ_LEN), 0.0)
        silu = _GATE(qk) * scale
        silu = silu.to(v_dtype)

        acc = tlx.async_dot_wait(0, acc)
        # release the V buffer
        v_empty = tlx.local_view(v_empties, v_buf_id)
        tlx.barrier_arrive(v_empty, 1)

        end_n += BLOCK_N

        # increment loop trip counts
        loop_trip_cnt += 1
        # v_buf_id = loop_trip_cnt % NUM_BUFFERS
        # v_phase = (loop_trip_cnt // NUM_BUFFERS) % 2

    # wait for the V buffer to be populated by the producer
    v_buf_id = (loop_trip_cnt - 1) % NUM_BUFFERS
    v_phase = ((loop_trip_cnt - 1) // NUM_BUFFERS) % 2
    v_full = tlx.local_view(v_fulls, v_buf_id)
    # tlx.barrier_wait(v_full, v_buf_id)
    v_tile = tlx.local_view(v_tiles, v_buf_id)
    tlx.barrier_wait(v_full, v_phase)
    acc = tlx.async_dot(silu, v_tile, acc)
    acc = tlx.async_dot_wait(0, acc)
    # release the V buffer
    v_empty = tlx.local_view(v_empties, v_buf_id)
    tlx.barrier_arrive(v_empty, 1)

    return acc, end_n, loop_trip_cnt


@triton.jit
def _hstu_attn_fwd_load_K_or_V(
    K,
    k_tiles,
    k_empties,
    k_fulls,
    buf_id,
    k_phase,
    start_n,
    seq_start,
    offset_kh,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # wait for the K buffer to be released by the consumer
    k_empty = tlx.local_view(k_empties, buf_id)
    tlx.barrier_wait(k_empty, k_phase)
    # load K
    k_full = tlx.local_view(k_fulls, buf_id)
    k_tile = tlx.local_view(k_tiles, buf_id)
    tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * BLOCK_D_Q)  # float16
    tlx.async_descriptor_load(
        K,
        k_tile,
        [(seq_start + start_n).to(tl.int32), offset_kh.to(tl.int32)],
        k_full,
    )


@triton.jit
def _hstu_attn_fwd_load_Q(
    Q,
    q_tiles,
    q_fulls,
    cid,
    off_z,
    off_h,
    stride_qh,
    start_m,
    seq_start,
    DeltaSize,
    IS_DELTA_Q: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    q_full = tlx.local_view(q_fulls, cid)
    tlx.barrier_expect_bytes(q_full, 2 * BLOCK_M * BLOCK_D_Q)  # float16
    q_tile = tlx.local_view(q_tiles, cid)
    seq_offset = start_m + cid * BLOCK_M
    if IS_DELTA_Q:
        tlx.async_descriptor_load(
            Q,
            q_tile,
            [
                (off_z * DeltaSize + start_m).to(tl.int32),
                (off_h * stride_qh).to(tl.int32),
            ],
            q_full,
        )
    else:
        tlx.async_descriptor_load(
            Q,
            q_tile,
            [
                (seq_start + seq_offset).to(tl.int32),
                (off_h * stride_qh).to(tl.int32),
            ],
            q_full,
        )


@triton.jit
def _hstu_attn_fwd_caculate_range(
    seq_len,
    start_m,
    n_targets,
    contextual_seq_len,
    max_attn_len,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if HAS_MULTIPLE_TARGETS:
        uih_end = seq_len - n_targets
    else:
        uih_end = seq_len

    if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
        # uih_end must be larger than start_m
        low = 0
        high = seq_len
    else:
        low = 0
        high = start_m + BLOCK_M
        if HAS_MAX_ATTN_LEN:
            if start_m > uih_end:
                low = uih_end - max_attn_len
            else:
                low = start_m - max_attn_len
            if HAS_CONTEXTUAL_SEQ_LEN:
                low = low if low > contextual_seq_len else 0
            else:
                low = low if low > 0 else 0
        if HAS_MULTIPLE_TARGETS:
            uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
            if uih_end < start_m:
                high = seq_len - n_targets

    return low, high, uih_end


@triton.jit
def _hstu_attn_fwd_load_Q_K_V(
    Q,
    K,
    V,
    q_tiles,
    k_tiles,
    v_tiles,
    q_fulls,
    k_fulls,
    v_fulls,
    k_empties,
    v_empties,
    stride_qh,
    stride_kh,
    stride_vh,
    contextual_seq_len,
    max_attn_len,
    DeltaSize,
    off_z,
    off_h,
    start_m,
    seq_start,
    seq_len,
    n_targets,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    # load q: it will stay in SRAM throughout
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    _hstu_attn_fwd_load_Q(
        Q=Q,
        q_tiles=q_tiles,
        q_fulls=q_fulls,
        cid=0,
        off_z=off_z,
        off_h=off_h,
        stride_qh=stride_qh,
        start_m=start_m,
        seq_start=seq_start,
        DeltaSize=DeltaSize,
        IS_DELTA_Q=IS_DELTA_Q,
        BLOCK_D_Q=BLOCK_D_Q,
        BLOCK_M=BLOCK_M_SPLIT,
    )

    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    offset_kh = off_h * stride_kh
    offset_vh = off_h * stride_vh

    low, high, uih_end = _hstu_attn_fwd_caculate_range(
        seq_len,
        start_m,
        n_targets,
        contextual_seq_len,
        max_attn_len,
        HAS_MULTIPLE_TARGETS,
        HAS_CONTEXTUAL_SEQ_LEN,
        HAS_MAX_ATTN_LEN,
        BLOCK_M,
        BLOCK_N,
    )

    kv_phase = 0
    loop_trip_cnt = 0

    # pyre-ignore[58]
    buf_id = loop_trip_cnt % NUM_BUFFERS
    # buffers in a row share the same phase
    kv_phase = kv_phase ^ (buf_id == 0)

    start_n = tl.multiple_of(low, BLOCK_N)

    _hstu_attn_fwd_load_K_or_V(
        K,
        k_tiles,
        k_empties,
        k_fulls,
        buf_id,
        kv_phase,
        start_n,
        seq_start,
        offset_kh,
        BLOCK_D_Q,
        BLOCK_N,
    )

    for cid in tl.range(1, NUM_MMA_GROUPS,
                        loop_unroll_factor=NUM_MMA_GROUPS - 1):
        _hstu_attn_fwd_load_Q(
            Q,
            q_tiles,
            q_fulls,
            cid,
            off_z,
            off_h,
            stride_qh,
            start_m,
            seq_start,
            DeltaSize,
            IS_DELTA_Q,
            BLOCK_D_Q,
            BLOCK_M_SPLIT,
        )

    _hstu_attn_fwd_load_K_or_V(
        V,
        v_tiles,
        v_empties,
        v_fulls,
        buf_id,
        kv_phase,
        start_n,
        seq_start,
        offset_vh,
        BLOCK_D_V,
        BLOCK_N,
    )

    loop_trip_cnt += 1

    for start in range(low + BLOCK_N, high, BLOCK_N):
        # pyre-ignore[58]
        buf_id = loop_trip_cnt % NUM_BUFFERS
        # buffers in a row share the same phase
        kv_phase = kv_phase ^ (buf_id == 0)

        start_n = tl.multiple_of(start, BLOCK_N)

        _hstu_attn_fwd_load_K_or_V(
            K,
            k_tiles,
            k_empties,
            k_fulls,
            buf_id,
            kv_phase,
            start_n,
            seq_start,
            offset_kh,
            BLOCK_D_Q,
            BLOCK_N,
        )

        _hstu_attn_fwd_load_K_or_V(
            V,
            v_tiles,
            v_empties,
            v_fulls,
            buf_id,
            kv_phase,
            start_n,
            seq_start,
            offset_vh,
            BLOCK_D_V,
            BLOCK_N,
        )

        # increment loop trip counts
        loop_trip_cnt += 1

    # pyre-ignore[61]
    if uih_end < start_m:
        low_delta = start_m
        high_delta = start_m + BLOCK_M
        for start_delta in tl.range(
                low_delta, high_delta, BLOCK_N, num_stages=0):
            # pyre-ignore[58]
            buf_id = loop_trip_cnt % NUM_BUFFERS
            # buffers in a row share the same phase
            kv_phase = kv_phase ^ (buf_id == 0)

            start_n = tl.multiple_of(start_delta, BLOCK_N)

            _hstu_attn_fwd_load_K_or_V(
                K,
                k_tiles,
                k_empties,
                k_fulls,
                buf_id,
                kv_phase,
                start_n,
                seq_start,
                offset_kh,
                BLOCK_D_Q,
                BLOCK_N,
            )

            _hstu_attn_fwd_load_K_or_V(
                V,
                v_tiles,
                v_empties,
                v_fulls,
                buf_id,
                kv_phase,
                start_n,
                seq_start,
                offset_vh,
                BLOCK_D_V,
                BLOCK_N,
            )

            # increment loop trip counts
            loop_trip_cnt += 1


@triton.jit
def _hstu_attn_fwd_compute_tlx(  # noqa C901
    Q,
    K,
    V,
    H,
    DimQ,
    DimV,
    seq_offsets,
    num_targets,
    Out,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    MAX_SEQ_LEN,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    off_z,
    off_h,
    pid,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,  #
    NUM_MMA_WARPS_PER_GROUP: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)

    if IS_DELTA_Q:
        start_m = pid * BLOCK_M
        start_m = (start_m + seq_len - DeltaSize).to(tl.int32)
    else:
        start_m = pid * BLOCK_M

    if start_m >= seq_len:
        return

    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    # allocate buffers
    q_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_D_Q), tlx.dtype_of(Q), NUM_MMA_GROUPS
    )
    k_tiles = tlx.local_alloc(
        (BLOCK_N, BLOCK_D_Q), tlx.dtype_of(K), NUM_BUFFERS)
    v_tiles = tlx.local_alloc(
        (BLOCK_N, BLOCK_D_V), tlx.dtype_of(V), NUM_BUFFERS)

    # allocate barriers
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)
    k_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS
    )
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    v_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS
    )
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)

    with tlx.async_tasks():
        # producer group
        with tlx.async_task("default"):
            _hstu_attn_fwd_load_Q_K_V(
                Q=Q,
                K=K,
                V=V,
                q_tiles=q_tiles,
                k_tiles=k_tiles,
                v_tiles=v_tiles,
                q_fulls=q_fulls,
                k_fulls=k_fulls,
                v_fulls=v_fulls,
                k_empties=k_empties,
                v_empties=v_empties,
                stride_qh=stride_qh,
                stride_kh=stride_kh,
                stride_vh=stride_vh,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                DeltaSize=DeltaSize,
                off_z=off_z,
                off_h=off_h,
                start_m=start_m,
                seq_start=seq_start,
                seq_len=seq_len,
                n_targets=n_targets,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                IS_DELTA_Q=IS_DELTA_Q,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                NUM_BUFFERS=NUM_BUFFERS,
                NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            )

        # consumer groups
        with tlx.async_task(
            num_warps=NUM_MMA_WARPS_PER_GROUP, registers=232, replicate=NUM_MMA_GROUPS
        ):
            cid = tlx.async_task_replica_id()
            acc = tl.zeros([BLOCK_M_SPLIT, BLOCK_D_V], dtype=tl.float32)
            # initialize offsets
            offs_m = start_m + tl.arange(0,
                                         BLOCK_M_SPLIT) + cid * BLOCK_M_SPLIT
            offs_n = tl.arange(0, BLOCK_N)

            low, high, uih_end = _hstu_attn_fwd_caculate_range(
                seq_len,
                start_m,
                n_targets,
                contextual_seq_len,
                max_attn_len,
                HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN,
                BLOCK_M,
                BLOCK_N,
            )

            end_n = low
            loop_trip_cnt = 0

            acc, end_n, loop_trip_cnt = _hstu_attn_fwd_compute_main_loop_tlx_pipelined(
                low=low,
                high=high,
                seq_len=seq_len,
                offs_m=offs_m,
                offs_n=offs_n,
                acc=acc,
                q_tiles=q_tiles,
                k_tiles=k_tiles,
                v_tiles=v_tiles,
                q_fulls=q_fulls,
                k_fulls=k_fulls,
                v_fulls=v_fulls,
                k_empties=k_empties,
                v_empties=v_empties,
                v_dtype=tlx.dtype_of(V),
                n_targets=n_targets,
                alpha=alpha,
                end_n=end_n,
                loop_trip_cnt=loop_trip_cnt,
                max_attn_len=max_attn_len,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                cid=cid,
                BLOCK_N=BLOCK_N,
                NUM_BUFFERS=NUM_BUFFERS,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                WAIT_FOR_Q=1,
            )

            # pyre-ignore[61]
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                acc, end_n, loop_trip_cnt = _hstu_attn_fwd_compute_main_loop_tlx(
                    low=low_delta,
                    high=high_delta,
                    seq_len=seq_len,
                    offs_m=offs_m,
                    offs_n=offs_n,
                    acc=acc,
                    q_tiles=q_tiles,
                    k_tiles=k_tiles,
                    v_tiles=v_tiles,
                    q_fulls=q_fulls,
                    k_fulls=k_fulls,
                    v_fulls=v_fulls,
                    k_empties=k_empties,
                    v_empties=v_empties,
                    v_dtype=tlx.dtype_of(V),
                    n_targets=n_targets,
                    alpha=alpha,
                    end_n=end_n,
                    loop_trip_cnt=loop_trip_cnt,
                    max_attn_len=max_attn_len,
                    HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                    HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                    HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                    cid=cid,
                    BLOCK_N=BLOCK_N,
                    NUM_BUFFERS=NUM_BUFFERS,
                    MAX_SEQ_LEN=MAX_SEQ_LEN,
                    WAIT_FOR_Q=0,
                )

            # Don't use TMA in Jagged case since we don't want to overwrite
            # the output of another sequence
            if IS_DELTA_Q:
                start_m_delta = pid * BLOCK_M + cid * BLOCK_M_SPLIT
                offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M_SPLIT)
                offs_v_d = tl.arange(0, BLOCK_D_V)
                off_o = Out + off_z * DeltaSize * stride_om + off_h * stride_oh
                out_ptrs = off_o + \
                    offs_m_delta[:, None] * stride_om + offs_v_d[None, :]
                tl.store(
                    out_ptrs, acc, mask=(
                        offs_m_delta < DeltaSize)[
                        :, None])
            else:
                # rematerialize offsets to save registers
                start_m = pid * BLOCK_M + cid * BLOCK_M_SPLIT
                offs_m = start_m + tl.arange(0, BLOCK_M_SPLIT)
                offs_v_d = tl.arange(0, BLOCK_D_V)
                off_o = Out + seq_start * stride_om + off_h * stride_oh
                out_ptrs = off_o + offs_m[:, None] * \
                    stride_om + offs_v_d[None, :]
                tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


@triton_autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)
@triton.jit
def _hstu_attn_fwd(  # noqa C901
    Q,
    K,
    V,
    workspace_ptr,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    Kp,
    Ks,
    stride_kpn,
    stride_kph,
    stride_ksn,
    stride_ksh,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_TLX: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,  #
    NUM_MMA_WARPS_PER_GROUP: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
    FP8: tl.constexpr = False,
    FAST_INTERIOR: tl.constexpr = False,
    FP4_QK: tl.constexpr = False,
    MASK_SUBTILE_N: tl.constexpr = 0,
    MASK_ZERO_QK: tl.constexpr = False,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    pid = tl.program_id(0)
    if USE_TLX:
        _hstu_attn_fwd_compute_tlx(
            Q=Q,
            K=K,
            V=V,
            H=H,
            DimQ=DimQ,
            DimV=DimV,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            Out=Out,
            stride_qh=stride_qh,
            stride_kh=stride_kh,
            stride_vh=stride_vh,
            stride_om=stride_om,
            stride_oh=stride_oh,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            DeltaSize=DeltaSize,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            off_z=off_z,
            off_h=off_h,
            pid=pid,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            IS_DELTA_Q=IS_DELTA_Q,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            NUM_BUFFERS=NUM_BUFFERS,
            NUM_MMA_WARPS_PER_GROUP=NUM_MMA_WARPS_PER_GROUP,
            NUM_MMA_GROUPS=NUM_MMA_GROUPS,
        )
    else:
        _hstu_attn_fwd_compute(
            Q=Q,
            K=K,
            V=V,
            H=H,
            DimQ=DimQ,
            DimV=DimV,
            workspace_ptr=workspace_ptr,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            Out=Out,
            stride_qm=stride_qm,
            stride_qh=stride_qh,
            stride_kn=stride_kn,
            stride_kh=stride_kh,
            stride_vn=stride_vn,
            stride_vh=stride_vh,
            stride_om=stride_om,
            stride_oh=stride_oh,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            DeltaSize=DeltaSize,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            off_z=off_z,
            off_h=off_h,
            pid=pid,
            Kp=Kp,
            Ks=Ks,
            stride_kpn=stride_kpn,
            stride_kph=stride_kph,
            stride_ksn=stride_ksn,
            stride_ksh=stride_ksh,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            IS_DELTA_Q=IS_DELTA_Q,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            ENABLE_TMA=ENABLE_TMA,
            TMA_DESC_SIZE=TMA_DESC_SIZE,
            FP8=FP8,
            FAST_INTERIOR=FAST_INTERIOR,
            FP4_QK=FP4_QK,
            MASK_SUBTILE_N=MASK_SUBTILE_N,
            MASK_ZERO_QK=MASK_ZERO_QK,
        )


@triton_autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)
@triton.jit
def _hstu_attn_fwd_persistent(  # noqa C901
    Q,
    K,
    V,
    workspace_ptr,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    Kp,
    Ks,
    stride_kpn,
    stride_kph,
    stride_ksn,
    stride_ksh,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_TLX: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,  #
    NUM_MMA_WARPS_PER_GROUP: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
    FP4_QK: tl.constexpr = False,
):
    n_tile_num = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)

    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    for _ in range(0, tiles_per_sm):
        pid = (total_tiles - tile_idx - 1) // (Z * H)
        off_hz = (total_tiles - tile_idx - 1) % (Z * H)
        off_z = off_hz // H
        off_h = off_hz % H
        _hstu_attn_fwd_compute(
            Q=Q,
            K=K,
            V=V,
            H=H,
            DimQ=DimQ,
            DimV=DimV,
            workspace_ptr=workspace_ptr,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            Out=Out,
            stride_qm=stride_qm,
            stride_qh=stride_qh,
            stride_kn=stride_kn,
            stride_kh=stride_kh,
            stride_vn=stride_vn,
            stride_vh=stride_vh,
            stride_om=stride_om,
            stride_oh=stride_oh,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            DeltaSize=DeltaSize,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            off_z=off_z,
            off_h=off_h,
            pid=pid,
            Kp=Kp,
            Ks=Ks,
            stride_kpn=stride_kpn,
            stride_kph=stride_kph,
            stride_ksn=stride_ksn,
            stride_ksh=stride_ksh,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            IS_DELTA_Q=IS_DELTA_Q,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            ENABLE_TMA=ENABLE_TMA,
            TMA_DESC_SIZE=TMA_DESC_SIZE,
            FP4_QK=FP4_QK,
        )
        tile_idx += num_progs


@triton.jit
def _hstu_attn_bwd_one_block(  # noqa C901
    start_m,
    offs_n,
    offs_m,
    q_ptrs_trans,
    dq_ptrs_trans,
    do_ptrs,
    device_desc_q,
    device_desc_do,
    dk,
    dv,
    k,
    v,
    pos_offs_n,
    seq_len,
    max_ids,
    contextual_seq_len,
    max_attn_len,
    LOCK,
    off_h,
    stride_qh,
    stride_doh,
    stride_qm,
    stride_dom,
    stride_dqm,
    alpha,
    MAX_SEQ_LEN,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
):
    pos_offs_m = offs_m + start_m
    mask_m = pos_offs_m < seq_len
    invalid_mask_trans = pos_offs_m[None, :] == offs_n[:, None]
    # recompute qk and silu
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_m = pos_offs_m - contextual_seq_len + 1
        pos_offs_m = tl.where(
            pos_offs_m > 0,
            pos_offs_m,
            0,
        )
    if HAS_MULTIPLE_TARGETS:
        pos_offs_m = tl.where(
            pos_offs_m < max_ids,
            pos_offs_m,
            max_ids,
        )
    if ENABLE_TMA:
        q = device_desc_q.load(
            [start_m, (off_h * stride_qh).to(tl.int32)],
        )
        q_trans = tl.trans(q)
    else:
        q_trans = tl.load(
            q_ptrs_trans + start_m * stride_qm,
            mask=mask_m[None, :],
            other=0.0,
        )
    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32) * alpha
    sig_trans = fast_dividef(1.0, 1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
    pos_offs_m_minus_n = pos_offs_m[None, :] - pos_offs_n[:, None]
    invalid_mask_trans = invalid_mask_trans | (pos_offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        invalid_mask_trans = invalid_mask_trans & (pos_offs_m_minus_n <= max_attn_len)
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask_trans = invalid_mask_trans | (
            (pos_offs_m[None, :] == 0) & (pos_offs_n[:, None] < max_ids)
        )
    silu_trans = tl.where(invalid_mask_trans, silu_trans, 0)
    silu_trans = silu_trans.to(k.dtype)
    # compute dv
    if ENABLE_TMA:
        do = device_desc_do.load(
            [start_m, (off_h * stride_doh).to(tl.int32)],
        )
    else:
        do = tl.load(
            do_ptrs + start_m * stride_dom,
            mask=mask_m[:, None],
            other=0.0,
        )
    dv += tl.dot(silu_trans, do, allow_tf32=ALLOW_TF32)

    # compute dk and dq
    dqk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
    dqk_trans = (
        dqk_trans * sig_trans *
        (1 + qk_trans * (1 - sig_trans)) * (1.0 / MAX_SEQ_LEN)
    )
    dqk_trans = tl.where(invalid_mask_trans, dqk_trans, 0)
    dqk_trans = dqk_trans.to(k.dtype)

    # Note: the factor `alpha` is delayed until the end of the function to
    # reduce the cost
    dk += tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32)
    acc_dq(
        dq_ptrs_trans=dq_ptrs_trans,
        start_m=start_m,
        stride_dqm=stride_dqm,
        k=k,
        dqk_trans=dqk_trans,
        alpha=alpha,
        mask_m=mask_m,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        LOCK=LOCK,
        BLOCK_M=BLOCK_M,
        ATOMIC_ADD=ATOMIC_ADD,
        ALLOW_TF32=ALLOW_TF32,
    )
    return dk, dv


@triton.jit
def _hstu_attn_bwd_one_col_block(  # noqa C901
    start_n,
    seq_len,
    n_targets,
    contextual_seq_len,
    max_attn_len,
    Q,
    K,
    V,
    DOut,
    DQ,
    DK,
    DV,
    device_desc_q,
    device_desc_k,
    device_desc_v,
    device_desc_do,
    device_desc_dk,
    device_desc_dv,
    LOCK,
    off_h,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_doh,
    stride_dkh,
    stride_dvh,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    alpha,
    MAX_SEQ_LEN,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
):
    if HAS_MULTIPLE_TARGETS:
        low = start_n
        if HAS_MAX_ATTN_LEN:
            high = start_n + max_attn_len + BLOCK_N
            high = high if high + n_targets < seq_len else seq_len
        else:
            high = seq_len
    else:
        low = start_n
        if HAS_MAX_ATTN_LEN:
            high = start_n + max_attn_len + BLOCK_N
            high = high if high < seq_len else seq_len
        else:
            high = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        contextual_block_end = tl.cdiv(contextual_seq_len, BLOCK_M) * BLOCK_M
        if low < contextual_block_end:
            low = contextual_block_end

    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    dq_ptrs_trans = DQ + (offs_m[None, :] * stride_dqm + offs_qk_d[:, None])
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    if ENABLE_TMA:
        q_ptrs_trans = None
        do_ptrs = None
        k = device_desc_k.load(
            [start_n, (off_h * stride_kh).to(tl.int32)],
        )
        v = device_desc_v.load(
            [start_n, (off_h * stride_vh).to(tl.int32)],
        )
    else:
        mask_n = offs_n < seq_len
        q_ptrs_trans = Q + (offs_m[None, :] * stride_qm + offs_qk_d[:, None])
        do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_n = offs_n - contextual_seq_len + 1
        pos_offs_n = tl.where(
            pos_offs_n > 0,
            pos_offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    else:
        pos_offs_n = offs_n
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        pos_offs_n = tl.where(
            pos_offs_n < max_ids,
            pos_offs_n,
            max_ids,
        )
    # loop over rows
    if HAS_CONTEXTUAL_SEQ_LEN:
        for start_m in range(0, contextual_seq_len, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            dk, dv = _hstu_attn_bwd_one_block(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs_trans=q_ptrs_trans,
                dq_ptrs_trans=dq_ptrs_trans,
                do_ptrs=do_ptrs,
                device_desc_q=device_desc_q,
                device_desc_do=device_desc_do,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                pos_offs_n=pos_offs_n,
                seq_len=seq_len,
                max_ids=max_ids,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                LOCK=LOCK,
                off_h=off_h,
                stride_qh=stride_qh,
                stride_doh=stride_doh,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                ATOMIC_ADD=ATOMIC_ADD,
                ENABLE_TMA=ENABLE_TMA,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
            )
    for start_m in tl.range(low, high, BLOCK_M, loop_unroll_factor=UNROLL):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        dk, dv = _hstu_attn_bwd_one_block(
            start_m=start_m,
            offs_n=offs_n,
            offs_m=offs_m,
            q_ptrs_trans=q_ptrs_trans,
            dq_ptrs_trans=dq_ptrs_trans,
            do_ptrs=do_ptrs,
            device_desc_q=device_desc_q,
            device_desc_do=device_desc_do,
            dk=dk,
            dv=dv,
            k=k,
            v=v,
            pos_offs_n=pos_offs_n,
            seq_len=seq_len,
            max_ids=max_ids,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            LOCK=LOCK,
            off_h=off_h,
            stride_qh=stride_qh,
            stride_doh=stride_doh,
            stride_qm=stride_qm,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_M=BLOCK_M,
            ATOMIC_ADD=ATOMIC_ADD,
            ENABLE_TMA=ENABLE_TMA,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
        )
    # write-back
    dk = dk * alpha
    if ENABLE_TMA:
        device_desc_dv.store(
            [start_n, (off_h * stride_dvh).to(tl.int32)],
            dv.to(k.dtype),
        )
        device_desc_dk.store(
            [start_n, (off_h * stride_dkh).to(tl.int32)],
            dk.to(k.dtype),
        )
    else:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
        tl.store(dv_ptrs, dv.to(k.dtype),
                 mask=mask_n[:, None])  # pyre-ignore[61]
        tl.store(dk_ptrs, dk.to(k.dtype),
                 mask=mask_n[:, None])  # pyre-ignore[61]


def _bwd_pre_hook(nargs):
    nargs["DQ"].zero_()
    if nargs["SEQUENCE_PARALLEL"] is True:
        nargs["LOCK"].zero_()


def _get_bw_configs() -> List[triton.Config]:
    if torch.version.hip:
        configs = []
        for BLOCK_M in [32, 64]:
            for BLOCK_N in [32, 64, 128]:
                for num_stages in [1, 2]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            for waves_per_eu in [0, 2, 4]:
                                for sp in [True, False]:
                                    configs.append(
                                        triton.Config(
                                            {
                                                "BLOCK_M": BLOCK_M,
                                                "BLOCK_N": BLOCK_N,
                                                "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                                "waves_per_eu": waves_per_eu,
                                                "SEQUENCE_PARALLEL": sp,
                                                "UNROLL": 1,
                                            },
                                            num_stages=num_stages,
                                            num_warps=num_warps,
                                            pre_hook=_bwd_pre_hook,
                                        )
                                    )
        return configs

    configs = [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 16, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=3,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 4},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
    ]
    if torch.cuda.is_available() and torch.version.cuda < "12.8":
        configs += [
            triton.Config(
                {"BLOCK_M": 16,
                 "BLOCK_N": 64,
                 "SEQUENCE_PARALLEL": False,
                 "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32,
                 "BLOCK_N": 64,
                 "SEQUENCE_PARALLEL": False,
                 "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32,
                 "BLOCK_N": 64,
                 "SEQUENCE_PARALLEL": False,
                 "UNROLL": 1},
                num_stages=1,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64,
                    "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=1,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128,
                    "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=3,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64,
                    "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64,
                    "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=2,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {
                    "BLOCK_M": 32,
                    "BLOCK_N": 128,
                    "SEQUENCE_PARALLEL": False,
                    "UNROLL": 2,
                },
                num_stages=2,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
        ]
    else:
        print("WARNING: temporarily disabled some autotune configs for CUDA 12.8+")
    return configs


@triton_autotune(
    configs=_get_bw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
    ],
)
@triton.jit
def _hstu_attn_bwd(  # noqa C901
    Q,
    K,
    V,
    tma_workspace_ptr,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
    DOut,
    DQ,
    DK,
    DV,
    LOCK,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    contextual_seq_len,
    max_attn_len,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
    ENABLE_BUFFER_OPS_ASSUMES: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    off_h = off_h.to(tl.int64)
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None
    if ENABLE_BUFFER_OPS_ASSUMES:
        tl.assume(off_hz >= 0)
        tl.assume(off_z >= 0)
        tl.assume(off_h >= 0)
        tl.assume(seq_start >= 0)
        tl.assume(stride_qm >= 0)
        tl.assume(stride_qh >= 0)
        tl.assume(stride_kn >= 0)
        tl.assume(stride_kh >= 0)
        tl.assume(stride_vn >= 0)
        tl.assume(stride_vh >= 0)
        tl.assume(stride_dom >= 0)
        tl.assume(stride_doh >= 0)
        tl.assume(stride_dqm >= 0)
        tl.assume(stride_dqh >= 0)
        tl.assume(stride_dkn >= 0)
        tl.assume(stride_dkh >= 0)
        tl.assume(stride_dvn >= 0)
        tl.assume(stride_dvh >= 0)

    # offset pointers for batch/head
    Q = Q + seq_start * stride_qm
    K = K + seq_start * stride_kn
    V = V + seq_start * stride_vn
    DOut = DOut + seq_start * stride_dom
    DQ = DQ + seq_start * stride_dqm + off_h * stride_dqh
    DK = DK + seq_start * stride_dkn
    DV = DV + seq_start * stride_dvn
    device_desc_q = None
    device_desc_k = None
    device_desc_v = None
    device_desc_do = None
    device_desc_dk = None
    device_desc_dv = None
    if ENABLE_TMA:
        device_desc_q = tl.make_tensor_descriptor(
            Q,
            shape=[seq_len, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=[BLOCK_M, BLOCK_D_Q],
        )
        device_desc_do = tl.make_tensor_descriptor(
            DOut,
            shape=[seq_len, H * DimV],
            strides=[H * DimV, 1],
            block_shape=[BLOCK_M, BLOCK_D_V],
        )
        device_desc_k = tl.make_tensor_descriptor(
            K,
            shape=[seq_len, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=[BLOCK_N, BLOCK_D_Q],
        )
        device_desc_dk = tl.make_tensor_descriptor(
            DK,
            shape=[seq_len, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=[BLOCK_N, BLOCK_D_Q],
        )
        device_desc_v = tl.make_tensor_descriptor(
            V,
            shape=[seq_len, H * DimV],
            strides=[H * DimV, 1],
            block_shape=[BLOCK_N, BLOCK_D_V],
        )
        device_desc_dv = tl.make_tensor_descriptor(
            DV,
            shape=[seq_len, H * DimV],
            strides=[H * DimV, 1],
            block_shape=[BLOCK_N, BLOCK_D_V],
        )
    else:
        Q += off_h * stride_qh
        K += off_h * stride_kh
        V += off_h * stride_vh
        DOut += off_h * stride_doh
        DK += off_h * stride_dkh
        DV += off_h * stride_dvh
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1) * BLOCK_N
        if start_n >= seq_len:
            return
        _hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len=seq_len,
            n_targets=n_targets,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            Q=Q,
            K=K,
            V=V,
            DOut=DOut,
            DQ=DQ,
            DK=DK,
            DV=DV,
            device_desc_q=device_desc_q,
            device_desc_k=device_desc_k,
            device_desc_v=device_desc_v,
            device_desc_do=device_desc_do,
            device_desc_dk=device_desc_dk,
            device_desc_dv=device_desc_dv,
            LOCK=LOCK,
            off_h=off_h,
            stride_qh=stride_qh,
            stride_kh=stride_kh,
            stride_vh=stride_vh,
            stride_doh=stride_doh,
            stride_dkh=stride_dkh,
            stride_dvh=stride_dvh,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            stride_dkn=stride_dkn,
            stride_dvn=stride_dvn,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            UNROLL=UNROLL,
            ATOMIC_ADD=True,
            ENABLE_TMA=ENABLE_TMA,
        )
    else:
        for start_n in range(0, seq_len, BLOCK_N):
            _hstu_attn_bwd_one_col_block(
                start_n=start_n,
                seq_len=seq_len,
                n_targets=n_targets,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                Q=Q,
                K=K,
                V=V,
                DOut=DOut,
                DQ=DQ,
                DK=DK,
                DV=DV,
                device_desc_q=device_desc_q,
                device_desc_k=device_desc_k,
                device_desc_v=device_desc_v,
                device_desc_do=device_desc_do,
                device_desc_dk=device_desc_dk,
                device_desc_dv=device_desc_dv,
                LOCK=LOCK,
                off_h=off_h,
                stride_qh=stride_qh,
                stride_kh=stride_kh,
                stride_vh=stride_vh,
                stride_doh=stride_doh,
                stride_dkh=stride_dkh,
                stride_dvh=stride_dvh,
                stride_qm=stride_qm,
                stride_kn=stride_kn,
                stride_vn=stride_vn,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                stride_dkn=stride_dkn,
                stride_dvn=stride_dvn,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                UNROLL=UNROLL,
                ATOMIC_ADD=False,
                ENABLE_TMA=ENABLE_TMA,
            )


_ATTN_DBG_COUNT = 0


def _attn_determ_norm_seqlen(n: int) -> int:
    """Server-mode determinism guard for HSTU attention.

    HSTU attention normalizes its output by ``1.0 / MAX_SEQ_LEN`` (in-kernel), and
    ``MAX_SEQ_LEN`` (=N) is passed as the *dynamic per-batch* max sequence length.
    Under nondeterministic Server batching that value changes run-to-run, which
    rescales every query's attention output (and, via the downstream SiLU / fp8 /
    residual + LayerNorm, its final prediction) -> the TEST08 acc-vs-acc drift.

    Setting ``$DLRM_ATTN_DETERM_MAXLEN=<model_max_seq_len>`` pins the normalization
    (and, for consistency, the autotune key + grid) to a fixed constant. We take
    ``max(fixed, n)`` so the grid always covers the longest sequence even if the env
    value is set too low; with the true model max (>= any batch max) the effective
    value is constant across runs -> batch-invariant, deterministic attention.
    """
    _dbg = os.environ.get("DLRM_ATTN_DEBUG_MAXLEN")
    if _dbg:
        global _ATTN_DBG_COUNT
        if _ATTN_DBG_COUNT < 4000:
            _ATTN_DBG_COUNT += 1
            try:
                with open(_dbg, "a") as _fh:
                    _fh.write(f"{int(n)}\n")
            except Exception:
                pass
    v = os.environ.get("DLRM_ATTN_DETERM_MAXLEN")
    if v:
        try:
            return max(int(v), int(n))
        except (TypeError, ValueError):
            return n
    return n


def triton_hstu_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
    enable_tma: bool,
    kp_ext: Optional[torch.Tensor] = None,
    ks_ext: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Plan 2 determinism guard: pin ONLY the 1/MAX_SEQ_LEN normalization + the
    # autotune key to a fixed constant so Server dynamic batching cannot rescale
    # attention run-to-run. The launch grid deliberately stays at the *actual*
    # per-batch max N: query-tile programs with ``start_m >= seq_len`` are masked
    # no-ops (see _hstu_attn_fwd_compute), so padding the grid up to the pinned
    # MAX_SEQ_LEN (~1.7x extra tiles at N~9.5k, MAX=16384) only burns grid slots
    # with zero numeric effect. Decoupling the two recovers that lost throughput
    # while keeping attention bit-identical + batch-invariant.
    N_norm = _attn_determ_norm_seqlen(N)
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
    # Plan 22 A-FUSE Φ2: q/k/v may arrive already-e4m3 (emitted by the UVQK GEMM).
    # The attention output must stay fp16, so don't inherit v's fp8 dtype.
    _out_dtype = torch.float16 if v.dtype == torch.float8_e4m3fn else v.dtype
    out = torch.empty_like(v, dtype=_out_dtype)
    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    has_sort_by_length_indices = sort_by_length_indices is not None
    if L == 0:
        return out

    # Plan 22 A1: pre-cast Q/K/V to e4m3 *once* here rather than inside the
    # block loop. The attention inner loop re-loads K/V per block and would
    # otherwise re-cast q every N-iteration; pre-casting halves the repeated
    # K/V load bandwidth and removes redundant casts. ``out`` keeps v's original
    # (fp16) dtype — it is allocated above before the cast.
    # Φ2: skip the pre-cast when the UVQK GEMM already emitted e4m3 q/k/v.
    # fp4-QK GAUC probe: round-trip K through MXFP4 before the e4m3 cast so the
    # QK^T dot sees the e2m1-quantized K values (q stays e4m3). Numerics-only.
    # Plan 37: the producer already packed K (fused hipBLASLt epilogue) and passes
    # kp/ks in directly — force the FP4_QK dot path and skip the standalone Triton
    # pack and the (dead under FP4_QK) e4m3 K materialization/cast. ``k`` arrives as
    # a cheap uninitialized placeholder (never dereferenced when FP4_QK is on).
    _fused_kpks = kp_ext is not None and ks_ext is not None
    if _fused_kpks:
        assert not enable_tma, "fused MXFP4 K-pack requires the non-TMA block path"
        if k is None:
            k = torch.empty((L, H, DimQ), dtype=torch.float8_e4m3fn, device=q.device)
        kp, ks = kp_ext, ks_ext
        fp4_qk = True
    else:
        if _HSTU_FP4_QK_EMU and k.dtype != torch.float8_e4m3fn:
            k = _mxfp4_roundtrip_k(k)
        # Plan 36: in-kernel mixed QK^T. Pack K to MXFP4 from its pre-e4m3 (bf16) value;
        # q stays e4m3 (requires fp8 attn) and the non-TMA block path consumes the pack.
        fp4_qk = _HSTU_FP4_QK and _HSTU_FP8_ATTN and not enable_tma and k.dtype != torch.float8_e4m3fn
        if fp4_qk:
            kp, ks = _quantize_k_mxfp4(k)
        else:
            kp, ks = k, k  # placeholder pointers; FP4_QK=False ⇒ dead branch in-kernel
    if _HSTU_FP8_ATTN:
        if q.dtype != torch.float8_e4m3fn:
            q = q.to(torch.float8_e4m3fn)
        if not _fused_kpks and k.dtype != torch.float8_e4m3fn:
            k = k.to(torch.float8_e4m3fn)
        if v.dtype != torch.float8_e4m3fn:
            v = v.to(torch.float8_e4m3fn)

    TMA_DESC_SIZE = 128
    workspace = None
    desc_q = q
    desc_k = k
    desc_v = v

    if enable_tma and tensor_descriptor_tma:
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(
            q,
            shape=[L, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[L, H * DimV],
            strides=[H * DimV, 1],
            block_shape=dummy_block,
        )
        desc_k = TensorDescriptor(
            k,
            shape=[L, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=dummy_block,
        )

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == TMA_DESC_SIZE
        return torch.empty(size, dtype=torch.int8, device="cuda")

    # pyre-ignore [6]
    triton.set_allocator(alloc_fn)

    def grid(meta): return (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )

    _hstu_attn_fwd[grid](
        Q=desc_q,
        K=desc_k,
        V=desc_v,
        workspace_ptr=workspace,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N_norm,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N_norm),
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=0,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        Kp=kp,
        Ks=ks,
        stride_kpn=kp.stride(0),
        stride_kph=kp.stride(1),
        stride_ksn=ks.stride(0),
        stride_ksh=ks.stride(1),
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        IS_DELTA_Q=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
        HAS_SORT_BY_LENGTH_INDICES=has_sort_by_length_indices,
        ENABLE_TMA=enable_tma,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
        FP8=_HSTU_FP8_ATTN,
        FAST_INTERIOR=_HSTU_ATTN_FASTMASK,
        FP4_QK=fp4_qk,
        MASK_SUBTILE_N=_HSTU_ATTN_MASK_SUBTILE_N,
        MASK_ZERO_QK=_HSTU_ATTN_MASK_ZERO_QK,
    )
    return out


def triton_hstu_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    N: int,
    alpha: float,
    max_attn_len: int,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
    enable_tma: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout = switch_to_contiguous_if_needed(dout)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    dv = switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    _, _, DimV = v.shape

    def grid(meta): return (  # noqa E731
        Z * H,
        (triton.cdiv(N, meta["BLOCK_N"]) if meta["SEQUENCE_PARALLEL"] else 1),
    )
    # The minimum size of BLOCK_M used in `_get_bw_configs`.
    # TODO (linjianma): avoid hardcoding the value.
    MIN_BLOCK_M = 16
    lock = torch.empty(
        (Z * H, triton.cdiv(N, MIN_BLOCK_M)),
        dtype=torch.int32,
        device=q.device,
    )
    AUTOTUNE_Z = prev_power_of_2(Z)
    TMA_DESC_SIZE = 128
    tma_workspace = None

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == TMA_DESC_SIZE
        return torch.empty(size, dtype=torch.int8, device="cuda")

    # pyre-ignore [6]
    triton.set_allocator(alloc_fn)

    # Enable BufferOps on AMD
    ENABLE_BUFFER_OPS_ASSUMES = torch.version.hip is not None
    _hstu_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        tma_workspace_ptr=tma_workspace,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        DOut=dout,
        DQ=dq,
        DK=dk,
        DV=dv,
        LOCK=lock,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        alpha=alpha,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        HAS_CONTEXTUAL_SEQ_LEN=contextual_seq_len > 0,
        HAS_MAX_ATTN_LEN=max_attn_len > 0,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
        ENABLE_TMA=enable_tma,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
        ENABLE_BUFFER_OPS_ASSUMES=ENABLE_BUFFER_OPS_ASSUMES,
    )

    return dq, dk, dv


class _AttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
        sort_by_length: bool,
        enable_tma: bool,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved_tensors = [q, k, v, seq_offsets]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.enable_tma = enable_tma
        return triton_hstu_attention_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=enable_tma,
        )

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor
    ) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            idx = 4
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            else:
                num_targets = None
            if ctx.sort_by_length:
                sort_by_length_indices = ctx.saved_tensors[idx]
            else:
                sort_by_length_indices = None

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dq, dk, dv = triton_hstu_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=ctx.N,
                alpha=ctx.alpha,
                max_attn_len=ctx.max_attn_len,
                contextual_seq_len=ctx.contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
                enable_tma=ctx.enable_tma,
            )
            return (
                None,
                None,
                dq,
                dk,
                dv,
                None,
                None,
                None,
                None,
                None,
                None,
            )


@torch.fx.wrap
def triton_hstu_mha(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    sort_by_length: bool = False,
    enable_tma: bool = False,
) -> torch.Tensor:
    return _AttentionFunction.apply(
        N,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        sort_by_length,
        enable_tma,
    )


@torch.fx.wrap
def triton_cached_hstu_mha(
    N: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    enable_tma: bool = False,
) -> torch.Tensor:
    Z = seq_offsets.size(0) - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    DELTA_L, H, DimQ = delta_q.shape
    DeltaSize = DELTA_L // Z
    L, _, DimV = v.shape
    out = torch.empty(
        (DELTA_L,
         H,
         DimV),
        dtype=delta_q.dtype,
        device=delta_q.device)

    # Plan 22 A1: pre-cast to e4m3 once (see triton_hstu_attention_fwd). ``out``
    # keeps delta_q's original dtype (allocated above before the cast).
    if _HSTU_FP4_QK_EMU and k.dtype != torch.float8_e4m3fn:
        k = _mxfp4_roundtrip_k(k)
    fp4_qk = _HSTU_FP4_QK and _HSTU_FP8_ATTN and not enable_tma and k.dtype != torch.float8_e4m3fn
    if fp4_qk:
        kp, ks = _quantize_k_mxfp4(k)
    else:
        kp, ks = k, k
    if _HSTU_FP8_ATTN:
        delta_q = delta_q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)

    TMA_DESC_SIZE = 128
    desc_q = delta_q
    desc_k = k
    desc_v = v

    if enable_tma and tensor_descriptor_tma:
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(
            delta_q,
            shape=[DELTA_L, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[L, H * DimV],
            strides=[H * DimV, 1],
            block_shape=dummy_block,
        )
        desc_k = TensorDescriptor(
            k,
            shape=[L, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=dummy_block,
        )

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == TMA_DESC_SIZE
        return torch.empty(size, dtype=torch.int8, device="cuda")

    # pyre-ignore [6]
    triton.set_allocator(alloc_fn)

    def grid(meta): return (  # noqa E731
        triton.cdiv(DeltaSize, meta["BLOCK_M"]),
        Z * H,
    )

    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    _hstu_attn_fwd[grid](
        Q=desc_q,
        K=desc_k,
        V=desc_v,
        workspace_ptr=None,
        sort_by_length_indices=None,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=out,
        stride_qm=delta_q.stride(0),
        stride_qh=delta_q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        Kp=kp,
        Ks=ks,
        stride_kpn=kp.stride(0),
        stride_kph=kp.stride(1),
        stride_ksn=ks.stride(0),
        stride_ksh=ks.stride(1),
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=DeltaSize,
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        IS_DELTA_Q=True,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
        HAS_SORT_BY_LENGTH_INDICES=False,
        ENABLE_TMA=enable_tma,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
        FP8=_HSTU_FP8_ATTN,
        FAST_INTERIOR=_HSTU_ATTN_FASTMASK,
        FP4_QK=fp4_qk,
        MASK_SUBTILE_N=_HSTU_ATTN_MASK_SUBTILE_N,
        MASK_ZERO_QK=_HSTU_ATTN_MASK_ZERO_QK,
    )
    return out
