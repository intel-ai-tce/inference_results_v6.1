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

# pyre-strict

import os
from typing import Dict, Optional, Tuple

import torch
from generative_recommenders.ops.triton.triton_addmm import (
    _FP8_ASCALE_MARGIN,
    _FP8_MAX,
    _HSTU_FP8_GEMM,
    _dump_fp8_scale,
    _fp8_static_scale,
    _scaled_addmm_fp8_preq,
    _scaled_mm_fp8_out_preq,
    _scaled_mm_mxfp4_aux_kpks,
    maybe_triton_addmm_fwd,
    triton_addmm_bwd,
    triton_addmm_fwd,
)
from generative_recommenders.ops.triton.triton_hstu_attention import (
    _HSTU_FP4_QK_FUSED,
    _HSTU_FP8_ATTN,
    triton_hstu_attention_bwd,
    triton_hstu_attention_fwd,
)
from generative_recommenders.ops.triton.triton_layer_norm import (
    triton_weighted_layer_norm_bwd,
    triton_weighted_layer_norm_fwd,
)
from torch.nn import functional as F

# Plan 22 A-FUSE Φ1: fuse the fp8 activation quantization into the input LN epilogue
# so the UVQK GEMM consumes fp8 normed_x with no standalone cast. Gated under the A2
# fp8-GEMM flag (the GEMM must be fp8 to accept fp8 in). DLRM_HSTU_FP8_FUSE_LN=0 reverts
# to the A2 "cast at the GEMM boundary" path for A/B.
_FP8_FUSE_LN: bool = (
    _HSTU_FP8_GEMM and os.environ.get("DLRM_HSTU_FP8_FUSE_LN", "1") == "1"
)
# Plan 22 A-FUSE Φ2 (A1↔A2 coupling): split the UVQK projection into a fp16 `u`
# GEMM (needs SiLU) and a fp8-out `v/q/k` GEMM, so the fused fp8 attention consumes
# e4m3 q/k/v with no standalone cast. Needs both the fused-LN fp8 activation and A1
# fp8 attention. DLRM_HSTU_FP8_FUSE_QKV=0 reverts to the single fp16-out UVQK GEMM.
_FP8_FUSE_QKV: bool = (
    _FP8_FUSE_LN
    and _HSTU_FP8_ATTN
    and os.environ.get("DLRM_HSTU_FP8_FUSE_QKV", "1") == "1"
)
# Plan 24 D2 (epilogue fusion): the standalone ``F.silu(u)`` below is the only
# unfused *functional* op left on the jagged [N, d] activation (5x/forward). The
# output LN kernel (`_ln_mul_dropout_fwd`) already has an (unused) ``SILU_U``
# epilogue path, so when this gate is on we skip the standalone SiLU launch here
# and let the output kernel apply SiLU instead — numerically identical (same gated
# ``y`` and same concat slot-0 ``u``), one fewer [N, d] kernel per layer.
# DLRM_HSTU_FUSE_EPILOGUE=0 (default) keeps the standalone F.silu(u). Assumes the
# fused HSTU path (not DLRM_HSTU_FORCE_UNFUSED, which applies SiLU elsewhere).
_FUSE_EPILOGUE: bool = os.environ.get("DLRM_HSTU_FUSE_EPILOGUE", "0") == "1"
# u-only SiLU epilogue fold for the MAIN (non-last) layers ONLY. This defers the
# standalone F.silu(u) into the output LN/concat epilogue (silu_u=True) exactly like
# D2/_FUSE_EPILOGUE, but ONLY for the fused preprocess+attention main path; the delta
# (last-layer targets-only) path keeps its standalone SiLU and is untouched. Unlike the
# global D2 flag this composes with DLRM_HSTU_LASTLAYER_TARGETS_ONLY (which trims only
# the last layer). DLRM_HSTU_FUSE_SILU_MAINONLY=0 (default) is byte-for-byte GOLD.
_FUSE_SILU_MAINONLY: bool = (
    os.environ.get("DLRM_HSTU_FUSE_SILU_MAINONLY", "0") == "1"
)
# Delta (last-layer targets-only) V/K fp8-out fold: emit e4m3 v/k directly from the
# delta projection GEMM (mirrors the certified main-layer Phi2 fp8-out path) so the
# delta attention skips the standalone bf16->e4m3 cast of k/v over ALL rows
# (`float8_copy`, ~2% of worker GPU). Default off; not bit-exact (one fp32->e4m3 round
# vs the current two-round fp32->bf16->e4m3), but mirrors what the main layers already do.
_FP8_DELTA_VK_FP8OUT: bool = (
    os.environ.get("DLRM_HSTU_FP8_DELTA_VK_FP8OUT", "0") == "1"
)
# Per UVQK-GEMM site: (scale_a tensor [1,1] fp32, inv_scale float), calibrated once.
_ln_fp8_scale_cache: Dict[int, Tuple[torch.Tensor, float]] = {}
# Per UVQK-GEMM site: split weights/biases (u | vqk), contiguous, built once.
_uvqk_split_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
# Plan 37: per UVQK-GEMM site, split (v+q | k) for the fused MXFP4-pack K path.
_vqk_ksplit_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
# Per UVQK-GEMM site for the final-layer split projection experiment:
# (W_uq, b_uq, W_vk, b_vk), where U/Q are needed only for target rows and V/K are
# needed for all rows.
_uvqk_delta_split_cache: Dict[
    int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _get_uvqk_split(
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    u_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split [K, u+vqk] weight/bias into contiguous (W_u, b_u, W_vqk, b_vqk),
    cached by uvqk_weight pointer (weights are static at inference)."""
    key = uvqk_weight.data_ptr()
    cached = _uvqk_split_cache.get(key)
    if cached is None:
        w_u = uvqk_weight[:, :u_width].contiguous()
        w_vqk = uvqk_weight[:, u_width:].contiguous()
        b_u = uvqk_bias[:u_width].contiguous()
        b_vqk = uvqk_bias[u_width:].contiguous()
        cached = (w_u, b_u, w_vqk, b_vqk)
        _uvqk_split_cache[key] = cached
    return cached


def _get_vqk_ksplit(
    w_vqk: torch.Tensor,
    b_vqk: torch.Tensor,
    vq_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plan 37: split the [K, v+q+k] weight/bias into contiguous (W_vq, b_vq, W_k, b_k),
    cached by w_vqk pointer. Lets the fused MXFP4-pack path run K as its own transposed
    GEMM (so K is emitted already-packed) while v/q stay in the e4m3 projection."""
    key = w_vqk.data_ptr()
    cached = _vqk_ksplit_cache.get(key)
    if cached is None:
        w_vq = w_vqk[:, :vq_width].contiguous()
        w_k = w_vqk[:, vq_width:].contiguous()
        b_vq = b_vqk[:vq_width].contiguous()
        b_k = b_vqk[vq_width:].contiguous()
        cached = (w_vq, b_vq, w_k, b_k)
        _vqk_ksplit_cache[key] = cached
    return cached


def _get_delta_uqkv_split(
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    u_width: int,
    v_width: int,
    q_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build contiguous split matrices for the final-layer delta projection.

    The native UVQK layout is [U | V | Q | K]. For target-only final-layer compute,
    target rows need U/Q, while all rows need V/K. Those column groups are not
    contiguous in the native layout, so cache packed [U | Q] and [V | K] matrices.
    """
    key = uvqk_weight.data_ptr()
    cached = _uvqk_delta_split_cache.get(key)
    if cached is None:
        u_start = 0
        v_start = u_width
        q_start = v_start + v_width
        k_start = q_start + q_width
        w_uq = torch.cat(
            (
                uvqk_weight[:, u_start:v_start],
                uvqk_weight[:, q_start:k_start],
            ),
            dim=1,
        ).contiguous()
        b_uq = torch.cat(
            (
                uvqk_bias[u_start:v_start],
                uvqk_bias[q_start:k_start],
            ),
            dim=0,
        ).contiguous()
        w_vk = torch.cat(
            (
                uvqk_weight[:, v_start:q_start],
                uvqk_weight[:, k_start:],
            ),
            dim=1,
        ).contiguous()
        b_vk = torch.cat(
            (
                uvqk_bias[v_start:q_start],
                uvqk_bias[k_start:],
            ),
            dim=0,
        ).contiguous()
        cached = (w_uq, b_uq, w_vk, b_vk)
        _uvqk_delta_split_cache[key] = cached
    return cached


def compute_uqvk_for_delta(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plan 25-B: fp8-preserving UVQK projection for the last-layer targets-only lever.

    Mirrors the input-LN + UVQK-GEMM of the certified fused forward (A-FUSE Φ1/A2) so
    the delta last layer's projection runs as a fp8 GEMM (fp16-out) instead of the
    fp16 ``torch.addmm`` in ``hstu_compute_uqvk``. q/k/v stay fp16-out and are cast to
    e4m3 inside ``triton_cached_hstu_mha`` (so the delta attention is fp8 either way);
    only the dense projection is upgraded. ``u`` is returned SiLU'd unless the epilogue
    fusion is on (matching the fused path), so it composes with ``hstu_compute_output``.
    Falls back to the fp16 LN+addmm when fp8 is not applicable (same as the fused path).
    """
    _use_fp8 = (
        _FP8_FUSE_LN
        and torch.version.hip is not None
        and x.dtype in (torch.float16, torch.bfloat16)
        and uvqk_weight.shape[0] % 16 == 0
        and uvqk_weight.shape[1] % 16 == 0
    )
    if _use_fp8:
        fp8_key = uvqk_weight.data_ptr()
        cal = _ln_fp8_scale_cache.get(fp8_key)
        if cal is None:
            # Calibration batch: fp16 LN, freeze scale from normed_x amax, fp16 GEMM.
            normed_x, _, _, _ = triton_weighted_layer_norm_fwd(
                x=x, weight=norm_weight, bias=norm_bias, eps=norm_eps
            )
            amax = normed_x.detach().abs().amax().clamp(min=1e-12)
            scale_val = _fp8_static_scale(
                "uvqk_ln", float(amax) * _FP8_ASCALE_MARGIN / _FP8_MAX
            )
            scale_a = torch.tensor(
                [[scale_val]], dtype=torch.float32, device=x.device
            )
            _ln_fp8_scale_cache[fp8_key] = (scale_a, 1.0 / scale_val)
            _dump_fp8_scale("uvqk_ln", float(amax), scale_val, tuple(normed_x.shape))
            uvqk = maybe_triton_addmm_fwd(
                x=normed_x, w=uvqk_weight, y=uvqk_bias
            ).contiguous()
        else:
            scale_a, inv_scale = cal
            normed_x, _, _, _ = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                output_fp8=True,
                inv_scale=inv_scale,
            )
            uvqk = _scaled_addmm_fp8_preq(
                normed_x, scale_a, uvqk_weight, uvqk_bias
            ).contiguous()
    else:
        normed_x, _, _, _ = triton_weighted_layer_norm_fwd(
            x=x, weight=norm_weight, bias=norm_bias, eps=norm_eps
        )
        uvqk = maybe_triton_addmm_fwd(
            x=normed_x, w=uvqk_weight, y=uvqk_bias
        ).contiguous()
    u, v, q, k = uvqk.split(
        [
            hidden_dim * num_heads,
            hidden_dim * num_heads,
            attn_dim * num_heads,
            attn_dim * num_heads,
        ],
        dim=1,
    )
    q = q.view(-1, num_heads, attn_dim)
    k = k.view(-1, num_heads, attn_dim)
    v = v.view(-1, num_heads, hidden_dim)
    u = u if _FUSE_EPILOGUE else F.silu(u)
    return u, q, k, v


def compute_split_uqkv_for_delta(
    x: torch.Tensor,
    tgt_idx: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Final-layer split projection for target-only delta attention.

    Target-aware attention clamps all candidate positions to the UIH boundary, so a
    candidate query attends to history plus its own diagonal candidate row, not other
    candidates. Therefore the final layer needs:
      - U/Q only for target rows.
      - V/K for all rows.

    This keeps the same fp8 input-LN/GEMM calibration path as the fused forward, but
    replaces one full [U|V|Q|K] projection with two narrower projections:
    [U|Q] on target rows and [V|K] on all rows.
    """
    u_width = hidden_dim * num_heads
    v_width = hidden_dim * num_heads
    q_width = attn_dim * num_heads
    w_uq, b_uq, w_vk, b_vk = _get_delta_uqkv_split(
        uvqk_weight=uvqk_weight,
        uvqk_bias=uvqk_bias,
        u_width=u_width,
        v_width=v_width,
        q_width=q_width,
    )
    _use_fp8 = (
        _FP8_FUSE_LN
        and torch.version.hip is not None
        and x.dtype in (torch.float16, torch.bfloat16)
        and uvqk_weight.shape[0] % 16 == 0
        and uvqk_weight.shape[1] % 16 == 0
    )
    if _use_fp8:
        fp8_key = uvqk_weight.data_ptr()
        cal = _ln_fp8_scale_cache.get(fp8_key)
        if cal is None:
            # Calibration batch: compute fp16 LN, freeze scale, use fp16 GEMMs.
            normed_x, _, _, _ = triton_weighted_layer_norm_fwd(
                x=x, weight=norm_weight, bias=norm_bias, eps=norm_eps
            )
            amax = normed_x.detach().abs().amax().clamp(min=1e-12)
            scale_val = _fp8_static_scale(
                "uvqk_delta", float(amax) * _FP8_ASCALE_MARGIN / _FP8_MAX
            )
            scale_a = torch.tensor(
                [[scale_val]], dtype=torch.float32, device=x.device
            )
            _ln_fp8_scale_cache[fp8_key] = (scale_a, 1.0 / scale_val)
            _dump_fp8_scale("uvqk_delta", float(amax), scale_val, tuple(normed_x.shape))
            normed_tgt = normed_x.index_select(0, tgt_idx)
            uq = maybe_triton_addmm_fwd(x=normed_tgt, w=w_uq, y=b_uq).contiguous()
            vk = maybe_triton_addmm_fwd(x=normed_x, w=w_vk, y=b_vk).contiguous()
        else:
            scale_a, inv_scale = cal
            normed_x, _, _, _ = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                output_fp8=True,
                inv_scale=inv_scale,
            )
            normed_tgt = normed_x.index_select(0, tgt_idx)
            uq = _scaled_addmm_fp8_preq(normed_tgt, scale_a, w_uq, b_uq).contiguous()
            if _FP8_DELTA_VK_FP8OUT:
                # e4m3-out vk GEMM: delta_hstu_mha then sees k/v already e4m3 and its
                # .to(e4m3) is a no-op (no `float8_copy` over all rows).
                vk = _scaled_mm_fp8_out_preq(
                    normed_x, scale_a, w_vk, b_vk
                ).contiguous()
            else:
                vk = _scaled_addmm_fp8_preq(normed_x, scale_a, w_vk, b_vk).contiguous()
    else:
        normed_x, _, _, _ = triton_weighted_layer_norm_fwd(
            x=x, weight=norm_weight, bias=norm_bias, eps=norm_eps
        )
        normed_tgt = normed_x.index_select(0, tgt_idx)
        uq = maybe_triton_addmm_fwd(x=normed_tgt, w=w_uq, y=b_uq).contiguous()
        vk = maybe_triton_addmm_fwd(x=normed_x, w=w_vk, y=b_vk).contiguous()
    u, q = uq.split([u_width, q_width], dim=1)
    v, k = vk.split([v_width, q_width], dim=1)
    q = q.view(-1, num_heads, attn_dim)
    k = k.view(-1, num_heads, attn_dim)
    v = v.view(-1, num_heads, hidden_dim)
    u = u if _FUSE_EPILOGUE else F.silu(u)
    return u, q, k, v


class _HSTUPreprocessAndAttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore [14]
    def forward(
        ctx,  # pyre-ignore [2]
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        norm_eps: float,
        num_heads: int,
        attn_dim: int,
        hidden_dim: int,
        uvqk_weight: torch.Tensor,
        uvqk_bias: torch.Tensor,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        attn_alpha: float,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
        recompute_uvqk_in_backward: bool,
        recompute_normed_x_in_backward: bool,
        sort_by_length: bool,
        enable_tma: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Plan 37: when the fused MXFP4-pack K path runs, K is emitted already-packed
        # and handed to attention via these planes (k itself is not materialized).
        kp_ext: Optional[torch.Tensor] = None
        ks_ext: Optional[torch.Tensor] = None
        _use_fp8_fuse = (
            _FP8_FUSE_LN
            and torch.version.hip is not None
            and x.dtype in (torch.float16, torch.bfloat16)
            and uvqk_weight.shape[0] % 16 == 0
            and uvqk_weight.shape[1] % 16 == 0
        )
        if _use_fp8_fuse:
            fp8_key = uvqk_weight.data_ptr()
            cal = _ln_fp8_scale_cache.get(fp8_key)
            if cal is None:
                # Calibration batch: fp16 LN, freeze scale from normed_x amax,
                # then fp16-path GEMM so this first batch is exact.
                normed_x, x_mean, x_rstd, BLOCK_D = triton_weighted_layer_norm_fwd(
                    x=x, weight=norm_weight, bias=norm_bias, eps=norm_eps
                )
                amax = normed_x.detach().abs().amax().clamp(min=1e-12)
                scale_val = _fp8_static_scale(
                    "uvqk_fuse", float(amax) * _FP8_ASCALE_MARGIN / _FP8_MAX
                )
                scale_a = torch.tensor(
                    [[scale_val]], dtype=torch.float32, device=x.device
                )
                _ln_fp8_scale_cache[fp8_key] = (scale_a, 1.0 / scale_val)
                _dump_fp8_scale("uvqk_fuse", float(amax), scale_val, tuple(normed_x.shape))
                uvqk = maybe_triton_addmm_fwd(
                    x=normed_x, w=uvqk_weight, y=uvqk_bias
                ).contiguous()
            else:
                scale_a, inv_scale = cal
                # LN emits e4m3 normed_x directly (cast folded into its epilogue).
                normed_x, x_mean, x_rstd, BLOCK_D = triton_weighted_layer_norm_fwd(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=norm_eps,
                    output_fp8=True,
                    inv_scale=inv_scale,
                )
                if _FP8_FUSE_QKV and _HSTU_FP4_QK_FUSED:
                    # Plan 37: `u` + `v/q` stay in the e4m3 projection; K is split into
                    # its own fused transposed GEMM that emits the MXFP4 pack (kp/ks)
                    # in-epilogue, so the standalone Triton K-pack and the dense K cast
                    # both disappear. K is never materialized as [tokens, D].
                    w_u, b_u, w_vqk, b_vqk = _get_uvqk_split(
                        uvqk_weight, uvqk_bias, hidden_dim * num_heads
                    )
                    u = _scaled_addmm_fp8_preq(normed_x, scale_a, w_u, b_u)
                    w_vq, b_vq, w_k, b_k = _get_vqk_ksplit(
                        w_vqk, b_vqk, (hidden_dim + attn_dim) * num_heads
                    )
                    vq = _scaled_mm_fp8_out_preq(normed_x, scale_a, w_vq, b_vq)
                    v, q = vq.split(
                        [hidden_dim * num_heads, attn_dim * num_heads], dim=1
                    )
                    kp_flat, ks_flat = _scaled_mm_mxfp4_aux_kpks(
                        normed_x, scale_a, w_k, b_k
                    )
                    kp_ext = kp_flat.view(-1, num_heads, attn_dim // 2)
                    ks_ext = ks_flat.view(-1, num_heads, attn_dim // 32)
                    k = None
                    uvqk = None
                elif _FP8_FUSE_QKV:
                    # Φ2: split UVQK — `u` stays fp16 (SiLU), v/q/k emitted in e4m3
                    # straight from the GEMM epilogue so the fused fp8 attention
                    # consumes them with no standalone cast.
                    w_u, b_u, w_vqk, b_vqk = _get_uvqk_split(
                        uvqk_weight, uvqk_bias, hidden_dim * num_heads
                    )
                    u = _scaled_addmm_fp8_preq(normed_x, scale_a, w_u, b_u)
                    vqk = _scaled_mm_fp8_out_preq(normed_x, scale_a, w_vqk, b_vqk)
                    v, q, k = vqk.split(
                        [
                            hidden_dim * num_heads,
                            attn_dim * num_heads,
                            attn_dim * num_heads,
                        ],
                        dim=1,
                    )
                    uvqk = None
                else:
                    uvqk = _scaled_addmm_fp8_preq(
                        normed_x, scale_a, uvqk_weight, uvqk_bias
                    ).contiguous()
        else:
            normed_x, x_mean, x_rstd, BLOCK_D = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
            )
            uvqk = maybe_triton_addmm_fwd(
                x=normed_x, w=uvqk_weight, y=uvqk_bias
            ).contiguous()
        if uvqk is not None:
            u, v, q, k = uvqk.split(
                [
                    hidden_dim * num_heads,
                    hidden_dim * num_heads,
                    attn_dim * num_heads,
                    attn_dim * num_heads,
                ],
                dim=1,
            )
        q = q.view(-1, num_heads, attn_dim)
        # Plan 37: under the fused MXFP4-pack K path, K is not materialized — kp/ks
        # carry the packed K and `k` stays None (a placeholder is made in attention).
        if k is not None:
            k = k.view(-1, num_heads, attn_dim)
        v = v.view(-1, num_heads, hidden_dim)
        # Plan 24 D2 (+ main-only fold): when fusing the epilogue, return RAW u; the
        # output LN kernel applies SiLU in its epilogue (SILU_U=True), removing this
        # standalone launch. _FUSE_SILU_MAINONLY folds it for the main path only so it
        # composes with last-layer targets-only.
        silu_u = u if (_FUSE_EPILOGUE or _FUSE_SILU_MAINONLY) else F.silu(u)
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        out = triton_hstu_attention_fwd(
            N=max_seq_len,
            alpha=attn_alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=enable_tma,
            kp_ext=kp_ext,
            ks_ext=ks_ext,
        )
        # update ctx
        saved_tensors = [
            x,
            norm_weight,
            norm_bias,
            x_mean,
            x_rstd,
            uvqk_weight,
            seq_offsets,
        ]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if not recompute_normed_x_in_backward:
            saved_tensors.append(normed_x)
        if recompute_uvqk_in_backward:
            saved_tensors.append(uvqk_bias)
        else:
            saved_tensors.append(uvqk)
        if sort_by_length:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.attn_alpha = attn_alpha
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_seq_len = max_seq_len
        ctx.max_attn_len = max_attn_len
        ctx.recompute_normed_x_in_backward = recompute_normed_x_in_backward
        ctx.recompute_uvqk_in_backward = recompute_uvqk_in_backward
        ctx.hidden_dim = hidden_dim
        ctx.attn_dim = attn_dim
        ctx.num_heads = num_heads
        ctx.uvqk_bias_1d = uvqk_bias.dim() == 1
        ctx.norm_eps = norm_eps
        ctx.norm_BLOCK_D = BLOCK_D
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.enable_tma = enable_tma
        return silu_u, out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx,  # pyre-ignore[2]
        dsilu_u: torch.Tensor,
        dout: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,  # d_x
        torch.Tensor,  # d_norm_weight
        torch.Tensor,  # d_norm_bias
        None,
        None,
        None,
        None,
        torch.Tensor,  # d_uvqk_weight
        torch.Tensor,  # d_uvqk_bias
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        x, norm_weight, norm_bias, x_mean, x_rstd, uvqk_weight, seq_offsets = (
            ctx.saved_tensors[:7]
        )
        idx = 7
        if ctx.has_multiple_targets:
            num_targets = ctx.saved_tensors[idx]
            idx += 1
        else:
            num_targets = None
        if ctx.recompute_normed_x_in_backward:
            normed_x, _, _, _ = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=ctx.norm_eps,
                mean=x_mean,
                rstd=x_rstd,
            )
        else:
            normed_x = ctx.saved_tensors[idx]
            idx += 1
        if ctx.recompute_uvqk_in_backward:
            uvqk_bias = ctx.saved_tensors[idx]
            uvqk = maybe_triton_addmm_fwd(
                x=normed_x, w=uvqk_weight, y=uvqk_bias)
            idx += 1
        else:
            uvqk = ctx.saved_tensors[idx]
            idx += 1
        if ctx.sort_by_length:
            sort_by_length_indices = ctx.saved_tensors[idx]
        else:
            sort_by_length_indices = None

        duvqk = torch.empty_like(uvqk)
        du, dv, dq, dk = duvqk.split(
            [
                ctx.hidden_dim * ctx.num_heads,
                ctx.hidden_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
            ],
            dim=1,
        )
        u, v, q, k = uvqk.split(
            [
                ctx.hidden_dim * ctx.num_heads,
                ctx.hidden_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
            ],
            dim=1,
        )
        q = q.view(-1, ctx.num_heads, ctx.attn_dim)
        k = k.view(-1, ctx.num_heads, ctx.attn_dim)
        v = v.view(-1, ctx.num_heads, ctx.hidden_dim)
        dq = dq.view(-1, ctx.num_heads, ctx.attn_dim)
        dk = dk.view(-1, ctx.num_heads, ctx.attn_dim)
        dv = dv.view(-1, ctx.num_heads, ctx.hidden_dim)
        # Note: the two operations below update duvqk in place
        (
            _dq,
            _dk,
            _dv,
        ) = triton_hstu_attention_bwd(
            dout=dout,
            q=q,
            k=k,
            v=v,
            dq=dq,
            dk=dk,
            dv=dv,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            N=ctx.max_seq_len,
            max_attn_len=ctx.max_attn_len,
            alpha=ctx.attn_alpha,
            contextual_seq_len=ctx.contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=ctx.enable_tma,
        )
        if dq.data_ptr() != _dq.data_ptr():
            dq.copy_(_dq)
        if dk.data_ptr() != _dk.data_ptr():
            dk.copy_(_dk)
        if dv.data_ptr() != _dv.data_ptr():
            dv.copy_(_dv)
        torch.ops.aten.silu_backward(dsilu_u, u, grad_input=du)
        d_normed_x, d_uvqk_weight, d_uvqk_bias = triton_addmm_bwd(
            x=normed_x,
            w=uvqk_weight,
            dz=duvqk,
            is_y_1d=ctx.uvqk_bias_1d,
        )
        d_x, d_norm_weight, d_norm_bias = triton_weighted_layer_norm_bwd(
            dy=d_normed_x,
            x=x,
            weight=norm_weight,
            bias=norm_bias,
            mean=x_mean,
            rstd=x_rstd,
            learnable=True,
            eps=ctx.norm_eps,
            BLOCK_D=ctx.norm_BLOCK_D,
        )
        # pyre-ignore[7]
        return (
            d_x,
            d_norm_weight,
            d_norm_bias,
            None,
            None,
            None,
            None,
            d_uvqk_weight,
            d_uvqk_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def triton_hstu_preprocess_and_attention(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    attn_alpha: float,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    recompute_uvqk_in_backward: bool = False,
    recompute_normed_x_in_backward: bool = False,
    sort_by_length: bool = False,
    enable_tma: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _HSTUPreprocessAndAttentionFunction.apply(
        x,
        norm_weight,
        norm_bias,
        norm_eps,
        num_heads,
        attn_dim,
        hidden_dim,
        uvqk_weight,
        uvqk_bias,
        max_seq_len,
        seq_offsets,
        attn_alpha,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        recompute_uvqk_in_backward,
        recompute_normed_x_in_backward,
        sort_by_length,
        enable_tma,
    )
