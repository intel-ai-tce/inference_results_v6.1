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
import abc
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from generative_recommenders.common import fx_unwrap_optional_tensor, HammerModule
from generative_recommenders.ops.hstu_attention import delta_hstu_mha
from generative_recommenders.ops.hstu_compute import (
    hstu_compute_output,
    hstu_compute_uqvk,
    hstu_preprocess_and_attention,
)
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from torch.autograd.profiler import record_function

try:
    import fbgemm_gpu  # noqa: F401
except ImportError:
    pass

# Last-layer target-only lever (exact, Closed-safe predict lever). At inference the
# model runs all STU layers over [history || candidates] and then DISCARDS the entire
# history half (HSTUTransducer._postprocess keeps only candidate embeddings). History-
# token outputs are therefore unused at the FINAL layer (earlier layers still need them
# to feed forward). So the last layer can compute attention + output for ONLY the target
# (candidate) queries via the existing delta-query path — bit-exact for the returned
# candidate scores, just not wasting work on outputs that are thrown away.
#
# Formalized (2026-06-06): `DLRM_HSTU_LASTLAYER_TARGETS_ONLY=1` is the single switch for
# the certified best path — it now also turns on the three validated follow-ups below
# (fp8 UVQK / split UQ-KV / target-only return) by default. It is most valuable with the
# 1024 window OFF (DLRM_HSTU_MAX_ATTN_LEN=0), where the last layer's full-causal
# attention is the dominant cost: it certified at b40 Server 7,400 q/s VALID
# (+11.3% over the un-levered full-causal baseline). 0/unset (default) is the certified
# C1-on path, byte-for-byte unchanged. Full write-up: docs/last_layer_targets_only.md.
#
# NOTE: the delta path is the unfused route (hstu_compute_uqvk applies F.silu(u) and
# hstu_compute_output must NOT re-apply it), so this lever is only correct with
# DLRM_HSTU_FUSE_EPILOGUE=0 (D2 off). The stack guards on that to stay single-SiLU.
_LASTLAYER_TARGETS_ONLY: bool = (
    os.environ.get("DLRM_HSTU_LASTLAYER_TARGETS_ONLY", "0") == "1"
)
_FUSE_EPILOGUE_ON: bool = os.environ.get("DLRM_HSTU_FUSE_EPILOGUE", "0") == "1"
# u-only SiLU epilogue fold for the MAIN (non-last) layers only. Unlike global D2 above,
# this does NOT touch the delta/targets-only path, so it composes with the last-layer
# targets-only lever: main layers fold SiLU(u) into the output LN epilogue while the
# last layer keeps its standalone SiLU. Default off => byte-for-byte GOLD.
_FUSE_SILU_MAINONLY: bool = (
    os.environ.get("DLRM_HSTU_FUSE_SILU_MAINONLY", "0") == "1"
)
# Validated follow-ups (certified 2026-06-06). These three were each A/B-validated as
# accuracy-neutral and together form the certified full-causal best path (b40 Server
# 7,400 q/s VALID, p99 59.9 ms — see docs/last_layer_targets_only.md). They therefore
# DEFAULT ON whenever the lever is on, so `DLRM_HSTU_LASTLAYER_TARGETS_ONLY=1` alone
# selects the certified path. Each remains individually overridable for A/B by setting
# its own env var to 0. With the lever OFF they default OFF, so the certified
# C1-on / D2-on path is byte-for-byte unchanged.
_lastlayer_default: str = "1" if _LASTLAYER_TARGETS_ONLY else "0"
# Keep the final-layer UVQK projection on the same fp8 GEMM as the certified fused path
# (instead of the fp16 torch.addmm in hstu_compute_uqvk), so going off the fused path
# for the delta attention does not regress the dense projection to fp16.
_LASTLAYER_FP8_UVQK: bool = (
    os.environ.get("DLRM_HSTU_LASTLAYER_FP8_UVQK", _lastlayer_default) == "1"
)
# Target rows attend to history plus their own candidate row only, so the final layer
# needs K/V for every row but U/Q only for target rows. This splits the final-layer
# projection accordingly (all-row V/K, target-row U/Q).
_LASTLAYER_SPLIT_UQKV: bool = (
    os.environ.get("DLRM_HSTU_LASTLAYER_SPLIT_UQKV", _lastlayer_default) == "1"
)
# Return only final-layer target rows and let HSTUTransducer._postprocess skip the
# embedding split. This removes the full-length zero scatter and immediate re-split of
# discarded history rows.
_LASTLAYER_RETURN_TARGETS_ONLY: bool = (
    os.environ.get("DLRM_HSTU_LASTLAYER_RETURN_TARGETS_ONLY", _lastlayer_default) == "1"
)


class STU(HammerModule, abc.ABC):
    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
        targets_are_uniform: Optional[bool] = None,
        tgt_per_seq: Optional[int] = None,
    ) -> torch.Tensor:
        pass


@dataclass
class STULayerConfig:
    embedding_dim: int
    num_heads: int
    hidden_dim: int
    attention_dim: int
    output_dropout_ratio: float = 0.3
    causal: bool = True
    target_aware: bool = True
    max_attn_len: Optional[int] = None
    attn_alpha: Optional[float] = None
    use_group_norm: bool = False
    recompute_normed_x: bool = True
    recompute_uvqk: bool = True
    recompute_y: bool = True
    sort_by_length: bool = True
    contextual_seq_len: int = 0


@torch.fx.wrap
def _update_kv_cache(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    max_kv_caching_len: int,
    kv_caching_lengths: Optional[torch.Tensor],
    orig_k_cache: Optional[torch.Tensor],
    orig_v_cache: Optional[torch.Tensor],
    orig_max_kv_caching_len: int,
    orig_kv_caching_offsets: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, Optional[torch.Tensor]]:
    if kv_caching_lengths is not None:
        kv_caching_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            kv_caching_lengths
        )
        delta_offsets = seq_offsets - kv_caching_offsets
        k_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=fx_unwrap_optional_tensor(k).flatten(1, 2),
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        v_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=fx_unwrap_optional_tensor(v).flatten(1, 2),
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        if max_kv_caching_len == 0:
            max_kv_caching_len = int(kv_caching_lengths.max().item())
        return (
            k_cache,
            v_cache,
            max_kv_caching_len,
            kv_caching_offsets,
        )
    else:
        return (
            orig_k_cache,
            orig_v_cache,
            orig_max_kv_caching_len,
            orig_kv_caching_offsets,
        )


@torch.fx.wrap
def _construct_full_kv(
    delta_k: torch.Tensor,
    delta_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    max_kv_caching_len: int,
    kv_caching_offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    L, _ = delta_k.shape
    B = kv_caching_offsets.shape[0] - 1
    delta_size = L // B
    full_k = concat_2D_jagged(
        max_seq_len=max_kv_caching_len + delta_size,
        values_left=k_cache,
        values_right=delta_k,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    full_v = concat_2D_jagged(
        max_seq_len=max_kv_caching_len + delta_size,
        values_left=v_cache,
        values_right=delta_v,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    full_kv_caching_offsets = kv_caching_offsets + delta_size * torch.arange(
        B + 1, device=delta_k.device
    )
    return (
        full_k,
        full_v,
        max_kv_caching_len + delta_size,
        full_kv_caching_offsets,
    )


class STULayer(STU):
    max_kv_caching_len: int
    k_cache: Optional[torch.Tensor]
    v_cache: Optional[torch.Tensor]
    kv_caching_offsets: Optional[torch.Tensor]

    def __init__(
        self,
        config: STULayerConfig,
        is_inference: bool = False,
    ) -> None:
        super().__init__(
            is_inference=is_inference,
        )
        self.reset_kv_cache()
        self._num_heads: int = config.num_heads
        self._embedding_dim: int = config.embedding_dim
        self._hidden_dim: int = config.hidden_dim
        self._attention_dim: int = config.attention_dim
        self._output_dropout_ratio: float = config.output_dropout_ratio
        self._target_aware: bool = config.target_aware
        self._causal: bool = config.causal
        self._max_attn_len: int = config.max_attn_len or 0
        # Plan 22 C1: optional sliding-window attention. DLRM_HSTU_MAX_ATTN_LEN>0
        # bounds each query to the most recent N causal positions, letting the
        # Triton HSTU kernel skip KV blocks outside the window (cuts the dominant
        # attention L^2 for long-UIH users). 0 / unset = unbounded (trained default).
        _env_max_attn_len = os.environ.get("DLRM_HSTU_MAX_ATTN_LEN", "").strip()
        if _env_max_attn_len:
            self._max_attn_len = int(_env_max_attn_len)
        self._attn_alpha: float = config.attn_alpha or 1.0 / \
            (self._attention_dim**0.5)
        self._use_group_norm: bool = config.use_group_norm
        self._recompute_normed_x: bool = config.recompute_normed_x
        self._recompute_uvqk: bool = config.recompute_uvqk
        self._recompute_y: bool = config.recompute_y
        self._sort_by_length: bool = config.sort_by_length
        self._contextual_seq_len: int = config.contextual_seq_len

        self._uvqk_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    self._embedding_dim,
                    (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
                )
            ),
        )
        torch.nn.init.xavier_uniform_(self._uvqk_weight)
        self._uvqk_beta: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(
                (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
            ),
        )
        self._input_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((self._embedding_dim,)),
        )
        self._input_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((self._embedding_dim,)),
        )
        self._output_weight = torch.nn.Parameter(
            torch.empty(
                (
                    self._hidden_dim * self._num_heads * 3,
                    self._embedding_dim,
                )
            ),
        )
        torch.nn.init.xavier_uniform_(self._output_weight)
        output_norm_shape: int = (
            self._hidden_dim * self._num_heads
            if not self._use_group_norm
            else self._num_heads
        )
        self._output_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((output_norm_shape,)),
        )
        self._output_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((output_norm_shape,)),
        )

    def reset_kv_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.kv_caching_offsets = None
        self.max_kv_caching_len = 0

    def update_kv_cache(
        self,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        max_kv_caching_len: int,
        kv_caching_lengths: Optional[torch.Tensor],
    ) -> None:
        self.k_cache, self.v_cache, self.max_kv_caching_len, self.kv_caching_offsets = (
            _update_kv_cache(
                max_seq_len=max_seq_len,
                seq_offsets=seq_offsets,
                k=k,
                v=v,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
                orig_k_cache=self.k_cache,
                orig_v_cache=self.v_cache,
                orig_max_kv_caching_len=self.max_kv_caching_len,
                orig_kv_caching_offsets=self.kv_caching_offsets,
            )
        )

    def construct_full_kv(
        self,
        delta_k: torch.Tensor,
        delta_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        return _construct_full_kv(
            delta_k=delta_k,
            delta_v=delta_v,
            k_cache=fx_unwrap_optional_tensor(self.k_cache),
            v_cache=fx_unwrap_optional_tensor(self.v_cache),
            max_kv_caching_len=self.max_kv_caching_len,
            kv_caching_offsets=self.kv_caching_offsets,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
        targets_are_uniform: Optional[bool] = None,
        tgt_per_seq: Optional[int] = None,
    ) -> torch.Tensor:
        with record_function("## stu_preprocess_and_attention ##"):
            u, attn_output, k, v = hstu_preprocess_and_attention(
                x=x,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(x.dtype),
                uvqk_bias=self._uvqk_beta.to(x.dtype),
                max_seq_len=max_seq_len,
                seq_offsets=x_offsets,
                attn_alpha=self._attn_alpha,
                causal=self._causal,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                recompute_uvqk_in_backward=self._recompute_uvqk,
                recompute_normed_x_in_backward=self._recompute_normed_x,
                sort_by_length=self._sort_by_length,
                prefill=kv_caching_lengths is not None,
                kernel=self.hammer_kernel(),
            )

        self.update_kv_cache(
            max_seq_len=max_seq_len,
            seq_offsets=x_offsets,
            k=k,
            v=v,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )

        with record_function("## stu_compute_output ##"):
            return hstu_compute_output(
                attn=attn_output,
                u=u,
                x=x,
                norm_weight=self._output_norm_weight.to(x.dtype),
                norm_bias=self._output_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_ux=True,
                training=self.training,
                kernel=self.hammer_kernel(),
                recompute_y_in_backward=self._recompute_y,
                # Main-path SiLU(u) fold: preprocess returns raw u (deferred), so the
                # output LN epilogue applies SiLU here. Delta path is unaffected.
                silu_u=_FUSE_SILU_MAINONLY,
            )

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with record_function("## stu_compute_uqvk ##"):
            delta_u, delta_q, delta_k, delta_v = hstu_compute_uqvk(
                x=delta_x,
                norm_weight=self._input_norm_weight.to(delta_x.dtype),
                norm_bias=self._input_norm_bias.to(delta_x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(delta_x.dtype),
                uvqk_bias=self._uvqk_beta.to(delta_x.dtype),
                kernel=self.hammer_kernel(),
            )
        k, v, max_seq_len, seq_offsets = self.construct_full_kv(
            delta_k=delta_k.flatten(1, 2),
            delta_v=delta_v.flatten(1, 2),
        )
        self.update_kv_cache(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            k=k,
            v=v,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )
        k = k.view(-1, self._num_heads, self._attention_dim)
        v = v.view(-1, self._num_heads, self._hidden_dim)
        with record_function("## delta_hstu_mha ##"):
            delta_attn_output = delta_hstu_mha(
                max_seq_len=max_seq_len,
                alpha=self._attn_alpha,
                delta_q=delta_q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                kernel=self.hammer_kernel(),
            ).view(-1, self._hidden_dim * self._num_heads)
        with record_function("## stu_compute_output ##"):
            return hstu_compute_output(
                attn=delta_attn_output,
                u=delta_u,
                x=delta_x,
                norm_weight=self._output_norm_weight.to(delta_x.dtype),
                norm_bias=self._output_norm_bias.to(delta_x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(delta_x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_ux=True,
                training=self.training,
                kernel=self.hammer_kernel(),
                recompute_y_in_backward=self._recompute_y,
            )

    def forward_targets_only(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        tgt_per_seq: Optional[int] = None,
    ) -> torch.Tensor:
        """Last-layer-only exact lever: compute attention + output for ONLY the
        target (candidate) queries, over the FULL freshly-computed K/V. Returns a
        full-length tensor with history rows zeroed (those are discarded downstream),
        so the caller / _postprocess split is unchanged. Bit-exact for candidate rows
        (same math as the full forward, just skipping the discarded history outputs).

        Requires uniform num_targets per sequence (the delta kernel assumes a uniform
        DeltaSize); the caller guards on this and on DLRM_HSTU_FUSE_EPILOGUE=0.
        """
        kernel = self.hammer_kernel()
        # Gather only the target (candidate) rows: the last `tgt_per_seq` rows of each
        # sequence. num_targets is uniform here (caller guards), so this is a regular
        # index pattern. tgt_idx[i*T + j] = x_offsets[i+1] - T + j.
        n_seq = x_offsets.numel() - 1
        if tgt_per_seq is None:
            tgt_per_seq = int(num_targets[0].item())
        else:
            tgt_per_seq = int(tgt_per_seq)
        tgt_idx = (
            x_offsets[1:].view(n_seq, 1)
            - tgt_per_seq
            + torch.arange(tgt_per_seq, device=x.device)
        ).reshape(-1)
        x_tgt = x.index_select(0, tgt_idx)
        # Full UVQK for ALL positions (needed for history K/V that candidates attend
        # to). Both paths apply F.silu(u) once (FUSE_EPILOGUE is off here, so the
        # output kernel does not re-apply it). The fp8 path mirrors the certified
        # fused projection so the dense GEMM stays fp8 off the fused path.
        if _LASTLAYER_FP8_UVQK and _LASTLAYER_SPLIT_UQKV:
            from generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention import (
                compute_split_uqkv_for_delta,
            )

            u_tgt, q_tgt, k, v = compute_split_uqkv_for_delta(
                x=x,
                tgt_idx=tgt_idx,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(x.dtype),
                uvqk_bias=self._uvqk_beta.to(x.dtype),
            )
        elif _LASTLAYER_FP8_UVQK:
            from generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention import (
                compute_uqvk_for_delta,
            )

            u, q, k, v = compute_uqvk_for_delta(
                x=x,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(x.dtype),
                uvqk_bias=self._uvqk_beta.to(x.dtype),
            )
            q_tgt = q.reshape(q.shape[0], -1).index_select(0, tgt_idx)
            u_tgt = u.index_select(0, tgt_idx)
        else:
            u, q, k, v = hstu_compute_uqvk(
                x=x,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(x.dtype),
                uvqk_bias=self._uvqk_beta.to(x.dtype),
                kernel=kernel,
            )
            q_tgt = q.reshape(q.shape[0], -1).index_select(0, tgt_idx)
            u_tgt = u.index_select(0, tgt_idx)
        delta_q = q_tgt.reshape(-1, self._num_heads, self._attention_dim)
        k = k.reshape(-1, self._num_heads, self._attention_dim)
        v = v.reshape(-1, self._num_heads, self._hidden_dim)
        with record_function("## stu_lastlayer_delta_mha ##"):
            delta_attn = delta_hstu_mha(
                max_seq_len=max_seq_len,
                alpha=self._attn_alpha,
                delta_q=delta_q,
                k=k,
                v=v,
                seq_offsets=x_offsets,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                kernel=kernel,
            ).reshape(-1, self._hidden_dim * self._num_heads)
        out_tgt = hstu_compute_output(
            attn=delta_attn,
            u=u_tgt,
            x=x_tgt,
            norm_weight=self._output_norm_weight.to(x.dtype),
            norm_bias=self._output_norm_bias.to(x.dtype),
            norm_eps=1e-6,
            dropout_ratio=self._output_dropout_ratio,
            output_weight=self._output_weight.to(x.dtype),
            group_norm=self._use_group_norm,
            num_heads=self._num_heads,
            linear_dim=self._hidden_dim,
            concat_ux=True,
            training=self.training,
            kernel=kernel,
            recompute_y_in_backward=self._recompute_y,
        )
        if _LASTLAYER_RETURN_TARGETS_ONLY:
            return out_tgt

        # Scatter target outputs back into a full-length tensor (history rows stay zero;
        # they are discarded by _postprocess). Layout matches [history || candidates].
        full = out_tgt.new_zeros((x.shape[0], out_tgt.shape[1]))
        full.index_copy_(0, tgt_idx, out_tgt)
        return full


class STUStack(STU):
    def __init__(
        self,
        stu_list: List[STU],
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=stu_list)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
        targets_are_uniform: Optional[bool] = None,
        tgt_per_seq: Optional[int] = None,
    ) -> torch.Tensor:
        # Plan 25-B: on the final layer, compute only the target (candidate) queries
        # since history outputs are discarded downstream. Gated + guarded so the
        # certified path (flag off) is byte-for-byte unchanged.
        if targets_are_uniform is None and num_targets is not None:
            targets_are_uniform = bool((num_targets.min() == num_targets.max()).item())
        last_targets_only = (
            _LASTLAYER_TARGETS_ONLY
            and not _FUSE_EPILOGUE_ON
            and num_targets is not None
            and not self.training
            and kv_caching_lengths is None
            and bool(targets_are_uniform)
        )
        n_layers = len(self._stu_layers)
        for i, layer in enumerate(self._stu_layers):
            if last_targets_only and i == n_layers - 1:
                x = layer.forward_targets_only(  # pyre-ignore [29]
                    x=x,
                    x_lengths=x_lengths,
                    x_offsets=x_offsets,
                    max_seq_len=max_seq_len,
                    num_targets=num_targets,
                    tgt_per_seq=tgt_per_seq,
                )
            else:
                x = layer(
                    x=x,
                    x_lengths=x_lengths,
                    x_offsets=x_offsets,
                    max_seq_len=max_seq_len,
                    num_targets=num_targets,
                    max_kv_caching_len=max_kv_caching_len,
                    kv_caching_lengths=kv_caching_lengths,
                )
        return x

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self._stu_layers:
            delta_x = layer.cached_forward(  # pyre-ignore [29]
                delta_x=delta_x,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
        return delta_x
