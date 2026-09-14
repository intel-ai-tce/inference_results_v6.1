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

"""ROCm bring-up shims (gfx950 / fbgemm-gpu 1.5.0+rocm7.0)."""

from __future__ import annotations

import os

import torch

_PATCHED = False


def asynchronous_complete_cumsum_torch(lengths: torch.Tensor) -> torch.Tensor:
    z = torch.zeros(1, device=lengths.device, dtype=lengths.dtype)
    return torch.cat([z, lengths.cumsum(0)], dim=0)


def apply_rocm_fbgemm_cumsum_patch() -> None:
    """
    fbgemm GPU jagged ops SIGSEGV on HIP (gfx950, fbgemm-gpu 1.5.0+rocm7.0).
    - cumsum: torch on CUDA (preserves input dtype — Plan 10 §10.1 surfaced
      that torch.cumsum upcasts int32 to int64 unless the dtype= arg is
      passed, breaking ``kjt_batched_func_cuda_upgrade``'s
      ``reorder_batched_ad_lengths`` call which expects int32).
    - jagged_to_padded_dense / dense_to_jagged: CPU fbgemm + H2D/D2H
    - Plan 10 §10.1.b: reorder_batched_ad_lengths / reorder_batched_ad_indices
      both SIGSEGV on HIP (same fbgemm-gpu 1.5.0+rocm7.0 op-family bug as
      cumsum / jagged_to_padded_dense). Required by NV's
      ``kjt_batched_func_cuda_upgrade`` (gpu-batching). Falling back to CPU
      keeps the path runnable — at smoke cap (b<=24) the D2H/H2D round-trip
      costs ~20 µs, negligible vs the per-sample CPU concat the CPU
      ``kjt_batch_func`` would do.  Once upstream fbgemm-gpu lands a ROCm
      reorder fix these fallbacks become no-ops on GPU input.
    Disable via DLRM_SAFE_FBGEMM_CUMSUM=0.
    """
    global _PATCHED
    if _PATCHED:
        return
    if not getattr(torch.version, "hip", None):
        return
    flag = os.environ.get("DLRM_SAFE_FBGEMM_CUMSUM", "1").upper()
    if flag in ("0", "OFF", "FALSE", "NO"):
        return

    try:
        import fbgemm_gpu  # noqa: F401
    except ImportError:
        return

    orig_cumsum = torch.ops.fbgemm.asynchronous_complete_cumsum
    orig_j2p = torch.ops.fbgemm.jagged_to_padded_dense
    orig_d2j = torch.ops.fbgemm.dense_to_jagged
    orig_rbal = torch.ops.fbgemm.reorder_batched_ad_lengths
    orig_rbai = torch.ops.fbgemm.reorder_batched_ad_indices

    def safe_asynchronous_complete_cumsum(lengths):
        if lengths.is_cuda:
            return asynchronous_complete_cumsum_torch(lengths)
        return orig_cumsum(lengths)

    def safe_jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
        if values.is_cuda:
            out = orig_j2p(
                values=values.cpu(),
                offsets=[o.cpu() for o in offsets],
                max_lengths=max_lengths,
                padding_value=padding_value,
            )
            return out.to(device=values.device)
        return orig_j2p(
            values=values,
            offsets=offsets,
            max_lengths=max_lengths,
            padding_value=padding_value,
        )

    def safe_dense_to_jagged(dense, offsets, total_len):
        if dense.is_cuda:
            vals = orig_d2j(
                dense.cpu(),
                [o.cpu() for o in offsets],
                int(total_len),
            )
            return (vals[0].to(device=dense.device),)
        return orig_d2j(dense, offsets, total_len)

    def safe_reorder_batched_ad_lengths(batched_length, bs_offset, bs):
        """Plan 10 §10.1.b — HIP fbgemm SIGSEGV; CPU fallback."""
        if batched_length.is_cuda or bs_offset.is_cuda:
            tgt = batched_length.device
            out = orig_rbal(batched_length.cpu(), bs_offset.cpu(), int(bs))
            return out.to(device=tgt, non_blocking=True)
        return orig_rbal(batched_length, bs_offset, int(bs))

    def safe_reorder_batched_ad_indices(
        cat_ad_offsets,
        cat_ad_indices,
        reordered_cat_ad_offsets,
        batch_offsets,
        num_ads_in_batch,
        *args,
        **kwargs,
    ):
        """Plan 10 §10.1.b — HIP fbgemm SIGSEGV; CPU fallback."""
        any_cuda = any(
            t.is_cuda for t in (
                cat_ad_offsets, cat_ad_indices, reordered_cat_ad_offsets, batch_offsets,
            )
        )
        if any_cuda:
            tgt = cat_ad_indices.device
            out = orig_rbai(
                cat_ad_offsets.cpu(),
                cat_ad_indices.cpu(),
                reordered_cat_ad_offsets.cpu(),
                batch_offsets.cpu(),
                int(num_ads_in_batch),
                *args,
                **kwargs,
            )
            return out.to(device=tgt, non_blocking=True)
        return orig_rbai(
            cat_ad_offsets,
            cat_ad_indices,
            reordered_cat_ad_offsets,
            batch_offsets,
            int(num_ads_in_batch),
            *args,
            **kwargs,
        )

    torch.ops.fbgemm.asynchronous_complete_cumsum = safe_asynchronous_complete_cumsum
    torch.ops.fbgemm.jagged_to_padded_dense = safe_jagged_to_padded_dense
    torch.ops.fbgemm.dense_to_jagged = safe_dense_to_jagged
    # Plan 10 §10.1.b — reorder fallbacks gated separately because they are
    # only required when --batching-on-gpu (= NV ``kjt_batched_func_cuda_upgrade``)
    # is active. The existing CPU collate path (``kjt_batch_func``) calls
    # ``reorder_batched_ad_*`` on CPU tensors, where the patched wrapper
    # would be a no-op in principle — but in the multi-rank lockstep
    # harness installing the wrappers regresses cold-start collate by
    # ~100 ms per batch for the first ~500 batches (Plan 10 Phase 10.1.b
    # bisection ISOLATION vs CONTROL runs, 2026-05-28).  Default OFF;
    # gate via DLRM_SAFE_FBGEMM_REORDER=1 only when DLRM_BATCHING_ON_GPU=1.
    if os.environ.get("DLRM_SAFE_FBGEMM_REORDER", "0") == "1":
        torch.ops.fbgemm.reorder_batched_ad_lengths = safe_reorder_batched_ad_lengths
        torch.ops.fbgemm.reorder_batched_ad_indices = safe_reorder_batched_ad_indices
    _PATCHED = True
