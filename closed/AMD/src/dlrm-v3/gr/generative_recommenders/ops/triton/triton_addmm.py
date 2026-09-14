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
    tlx = None
    HAS_TLX = False

from generative_recommenders.common import triton_autotune, triton_cc
from generative_recommenders.ops.utils import is_sm100

try:
    # @manual=//triton:triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    TMA_AVAILABLE = True
except ImportError:
    TMA_AVAILABLE = False
    pass


ENABLE_FULL_TURNING_SPACE = False


def _check_tma_alignment(
    x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, min_alignment: int = 16
) -> bool:
    """Check if tensors meet TMA alignment requirements.

    TMA (Tensor Memory Accelerator) on H100 requires:
    1. Base addresses to be 64-byte aligned
    2. Dimensions to be multiples of 64 for optimal performance
    3. Contiguous inner dimensions (stride=1)

    Args:
        x: Input tensor [M, K]
        w: Weight tensor [K, N]
        y: Bias tensor [N] or [M, N]
        min_alignment: Minimum alignment requirement (default: 64)

    Returns:
        True if all tensors meet TMA alignment requirements
    """
    _, K = x.shape
    KB, N = w.shape
    assert K == KB, f"incompatible dimensions {K}, {KB}"

    is_y_1d = y.dim() == 1
    NY = y.shape[0] if is_y_1d else y.shape[1]
    assert N == NY, f"incompatible dimensions {N}, {NY}"

    return (K % min_alignment == 0) and (N % min_alignment == 0)


def get_mm_configs(pre_hook=None) -> List[triton.Config]:
    if torch.version.hip:
        if ENABLE_FULL_TURNING_SPACE:
            block_m_range = [32, 64, 128, 256]
            block_n_range = [32, 64, 128, 256]
            block_k_range = [32, 64]
            group_m_range = [4, 8]
            matrix_instr_nonkdim_range = [16]
            waves_per_eu_range = [0]
            kpack_range = [1, 2]
            num_warps_range = [4, 8]
            num_stage_range = [2] if triton.__version__ >= "3.2.0" else [0]
        else:
            block_m_range = [256]
            block_n_range = [256]
            block_k_range = [32]
            group_m_range = [8]
            matrix_instr_nonkdim_range = [16]
            waves_per_eu_range = [0]
            kpack_range = [2]
            num_warps_range = [8]
            num_stage_range = [2] if triton.__version__ >= "3.2.0" else [0]

        return [
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": block_k,
                    "GROUP_M": group_m,
                    "matrix_instr_nonkdim": matrix_instr_nonkdim,
                    "waves_per_eu": waves_per_eu,
                    "kpack": kpack,
                },
                num_stages=num_stages,
                num_warps=num_warps,
                pre_hook=pre_hook,
            )
            for block_m in block_m_range
            for block_n in block_n_range
            for block_k in block_k_range
            for group_m in group_m_range
            for matrix_instr_nonkdim in matrix_instr_nonkdim_range
            for waves_per_eu in waves_per_eu_range
            for kpack in kpack_range
            for num_stages in num_stage_range
            for num_warps in num_warps_range
        ]
    else:
        block_m_range = [32, 64, 128, 256]
        block_n_range = [32, 64, 128, 256]
        block_k_range = [32, 64]
        group_m_range = [4, 8]
        # WARP_SPECIALIZE only works with num_warps >=4
        num_warps_range = [4, 8] if is_sm100() else [2, 4, 8]
        num_stage_range = [2, 3, 4, 5]
        if ENABLE_FULL_TURNING_SPACE:
            return [
                triton.Config(
                    {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                        "GROUP_M": group_m,
                    },
                    num_stages=num_stages,
                    num_warps=num_warps,
                    pre_hook=pre_hook,
                )
                for block_m in block_m_range
                for block_n in block_n_range
                for block_k in block_k_range
                for group_m in group_m_range
                for num_stages in num_stage_range
                for num_warps in num_warps_range
            ]
        else:
            configs = [
                triton.Config(
                    {
                        "BLOCK_M": 32,
                        "BLOCK_N": 64,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=5,
                    num_warps=2,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 256,
                        "BLOCK_K": 64,
                        "GROUP_M": 8,
                    },
                    num_stages=3,
                    num_warps=8,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 256,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=4,
                    num_warps=4,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 128,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=4,
                    num_warps=4,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 64,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=4,
                    num_warps=4,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 128,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=4,
                    num_warps=4,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 32,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=4,
                    num_warps=4,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 32,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=5,
                    num_warps=2,
                    pre_hook=pre_hook,
                ),
            ]
            if is_sm100():
                configs += [
                    triton.Config(
                        {
                            "BLOCK_M": 128,
                            "BLOCK_N": 256,
                            "BLOCK_K": 64,
                            "GROUP_M": 8,
                        },
                        num_stages=3,
                        num_warps=4,
                        pre_hook=pre_hook,
                    ),
                ]
                return [c for c in configs if c.num_warps >= 4]

            return configs


@triton_cc(
    annotations={
        "M": "i32",
        "N": ("i32", 16),
        "K": ("i32", 16),
        "stride_xm": ("i32", 16),
        "stride_xk": ("i32", 1),
        "stride_wk": ("i32", 16),
        "stride_wn": ("i32", 1),
        "stride_ym": ("i32", 16),
        "stride_yn": ("i32", 1),
        "stride_zm": ("i32", 16),
        "stride_zn": ("i32", 1),
    },
)
@triton_autotune(
    configs=get_mm_configs(),
    key=["N", "K"],
)
@triton.jit
def _addmm_fwd(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    stride_zm,
    stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
):
    pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
    pid = pid_0 * tl.num_programs(axis=1) + pid_1
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    x_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_xm
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm +
                      offs_k[None, :] * stride_xk)
    w_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_wn
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk +
                      offs_n[None, :] * stride_wn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=mask_k & mask_n, other=0.0)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    z_mask = mask_m & mask_n
    if BROADCAST_Y:
        # y is a vector, broadcast to add to z
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_n)
    else:
        y_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_ym
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_ym * \
            offs_m[:, None] + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=z_mask)
    z = (accumulator + y.to(tl.float32)).to(z_ptr.dtype.element_ty)
    z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
    z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=z_mask)


def _addmm_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]
    nargs["x_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["w_desc"].block_shape = [BLOCK_K, BLOCK_N]
    nargs["z_desc"].block_shape = [BLOCK_M, BLOCK_N]
    if nargs["BROADCAST_Y"]:
        nargs["y_desc"].block_shape = [1, BLOCK_N]
    else:
        nargs["y_desc"].block_shape = [BLOCK_M, BLOCK_N]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton_autotune(
    configs=get_mm_configs(pre_hook=_addmm_tma_set_block_size_hook),
    key=["N", "K", "WARP_SPECIALIZE"],
)
@triton.jit
def _addmm_fwd_tma_persistent(
    x_desc,
    w_desc,
    y_desc,
    z_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_M * num_pid_n

    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS
        )
        offs_xm = pid_m * BLOCK_M
        offs_wn = pid_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
            offs_k = k * BLOCK_K
            x = x_desc.load([offs_xm, offs_k])
            w = w_desc.load([offs_k, offs_wn])
            accumulator = tl.dot(x, w, accumulator, allow_tf32=ALLOW_TF32)
        if BROADCAST_Y:
            y = y_desc.load([0, offs_wn])
        else:
            y = y_desc.load([offs_xm, offs_wn])
        z = (accumulator + y.to(tl.float32)).to(z_desc.dtype)
        z_desc.store([offs_xm, offs_wn], z)


@triton_autotune(
    configs=get_mm_configs(pre_hook=_addmm_tma_set_block_size_hook),
    key=["N", "K"],
)
@triton.jit
def _addmm_fwd_tma_ws(
    x_desc,
    w_desc,
    y_desc,
    z_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
):
    x_buffers = tlx.local_alloc(
        (BLOCK_M, BLOCK_K), x_desc.dtype, NUM_SMEM_BUFFERS)
    w_buffers = tlx.local_alloc(
        (BLOCK_K, BLOCK_N), w_desc.dtype, NUM_SMEM_BUFFERS)
    acc_tmem_buffer = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem
    )

    if BROADCAST_Y:
        y_buffer = tlx.local_alloc((1, BLOCK_N), y_desc.dtype, tl.constexpr(1))
    else:
        y_buffer = tlx.local_alloc(
            (BLOCK_M, BLOCK_N), y_desc.dtype, tl.constexpr(1))
    z_buffer = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), z_desc.dtype, tl.constexpr(1))

    smem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    y_load_barrier = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    with tlx.async_tasks():
        # Producer task: TMA loads
        with tlx.async_task("default"):
            pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
            pid = pid_0 * tl.num_programs(axis=1) + pid_1
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_xm = pid_m * BLOCK_M
            offs_wn = pid_n * BLOCK_N
            k_tiles = tl.cdiv(K, BLOCK_K)

            load_phase = 0
            for k in range(0, k_tiles):
                buf = k % int(NUM_SMEM_BUFFERS)

                # Wait for buffer to be free
                if k >= NUM_SMEM_BUFFERS:
                    tlx.barrier_wait(smem_empty_bars[buf], load_phase ^ 1)

                offs_k = k * BLOCK_K
                tlx.barrier_expect_bytes(
                    smem_full_bars[buf],
                    2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N),
                )
                tlx.async_descriptor_load(
                    x_desc, x_buffers[buf], [
                        offs_xm, offs_k], smem_full_bars[buf]
                )
                tlx.async_descriptor_load(
                    w_desc, w_buffers[buf], [
                        offs_k, offs_wn], smem_full_bars[buf]
                )

                load_phase = load_phase ^ (buf == NUM_SMEM_BUFFERS - 1)

        # Consumer task: async_dot MMA
        with tlx.async_task(num_warps=4, num_regs=232):
            pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
            pid = pid_0 * tl.num_programs(axis=1) + pid_1
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_xm = pid_m * BLOCK_M
            offs_wn = pid_n * BLOCK_N
            k_tiles = tl.cdiv(K, BLOCK_K)

            # Start async load of y early
            y_buf_view = tlx.local_view(y_buffer, 0)
            y_load_bar = tlx.local_view(y_load_barrier, 0)
            if BROADCAST_Y:
                tlx.barrier_expect_bytes(y_load_bar, 1 * BLOCK_N * 2)
                tlx.async_descriptor_load(
                    y_desc, y_buf_view, [
                        0, offs_wn], y_load_bar)
            else:
                tlx.barrier_expect_bytes(y_load_bar, BLOCK_M * BLOCK_N * 2)
                tlx.async_descriptor_load(
                    y_desc, y_buf_view, [offs_xm, offs_wn], y_load_bar
                )

            dot_phase = 0
            for k in range(0, k_tiles):
                buf = k % int(NUM_SMEM_BUFFERS)
                tlx.barrier_wait(smem_full_bars[buf], dot_phase)

                tlx.async_dot(
                    x_buffers[buf],
                    w_buffers[buf],
                    acc_tmem_buffer[0],
                    use_acc=k > 0,
                    mBarriers=[smem_empty_bars[buf]],
                    out_dtype=tl.float32,
                )

                dot_phase = dot_phase ^ (buf == NUM_SMEM_BUFFERS - 1)

            last_buf = (k_tiles - 1) % NUM_SMEM_BUFFERS
            last_dot_phase = dot_phase ^ (last_buf == NUM_SMEM_BUFFERS - 1)
            tlx.barrier_wait(smem_empty_bars[last_buf], last_dot_phase)

            tmem_result = tlx.local_load(acc_tmem_buffer[0])

            tlx.barrier_wait(y_load_bar, 0)
            y = tlx.local_load(y_buf_view)

            z = (tmem_result + y.to(tl.float32)).to(z_desc.dtype)
            z_buf_view = tlx.local_view(z_buffer, 0)
            tlx.local_store(z_buf_view, z)
            tlx.async_descriptor_store(z_desc, z_buf_view, [offs_xm, offs_wn])
            tlx.async_descriptor_store_wait(0)


@triton_autotune(
    configs=get_mm_configs(pre_hook=_addmm_tma_set_block_size_hook),
    key=["N", "K"],
)
@triton.jit
def _addmm_fwd_tma_ws_persistent(
    x_desc,
    w_desc,
    y_desc,
    z_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    # Allocate buffers once for all tiles
    x_buffers = tlx.local_alloc(
        (BLOCK_M, BLOCK_K), x_desc.dtype, NUM_SMEM_BUFFERS)
    w_buffers = tlx.local_alloc(
        (BLOCK_K, BLOCK_N), w_desc.dtype, NUM_SMEM_BUFFERS)
    tmem_buffers = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem
    )

    # Barriers for producer <-> MMA
    smem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    # Barriers for MMA <-> Epilogue
    tmem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    with tlx.async_tasks():
        # Epilogue consumer: loads Y, adds bias, stores Z
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            tmem_read_phase = 0
            cur_tmem_buf = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS
                )
                offs_xm = pid_m * BLOCK_M
                offs_wn = pid_n * BLOCK_N

                # Wait for MMA to finish computing this tile
                tlx.barrier_wait(tmem_full_bars[cur_tmem_buf], tmem_read_phase)
                tmem_read_phase = tmem_read_phase ^ (
                    cur_tmem_buf == int(NUM_TMEM_BUFFERS) - 1
                )

                # Load Y synchronously
                if BROADCAST_Y:
                    y = y_desc.load([0, offs_wn])
                else:
                    y = y_desc.load([offs_xm, offs_wn])

                # Load result from TMEM and add bias
                acc_tmem = tmem_buffers[cur_tmem_buf]
                result = tlx.local_load(acc_tmem)
                z = (result + y.to(tl.float32)).to(z_desc.dtype)

                # Store result directly via TMA
                z_desc.store([offs_xm, offs_wn], z)

                # Signal MMA that this TMEM buffer is now free
                tlx.barrier_arrive(tmem_empty_bars[cur_tmem_buf], 1)

                cur_tmem_buf = (cur_tmem_buf + 1) % int(NUM_TMEM_BUFFERS)

        # MMA consumer: performs matrix multiplication
        with tlx.async_task(num_warps=4, num_regs=232):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_K)

            dot_phase = 0
            tmem_write_phase = 1
            cur_tmem_buf = 0
            processed_k_iters = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS
                )

                # Wait for epilogue to finish with this TMEM buffer
                tlx.barrier_wait(
                    tmem_empty_bars[cur_tmem_buf],
                    tmem_write_phase)
                tmem_write_phase = tmem_write_phase ^ (
                    cur_tmem_buf == int(NUM_TMEM_BUFFERS) - 1
                )

                # Perform K-dimension reduction
                for k in range(0, k_tiles):
                    buf = (processed_k_iters + k) % int(NUM_SMEM_BUFFERS)
                    tlx.barrier_wait(smem_full_bars[buf], dot_phase)

                    tlx.async_dot(
                        x_buffers[buf],
                        w_buffers[buf],
                        tmem_buffers[cur_tmem_buf],
                        use_acc=(k > 0),
                        mBarriers=[smem_empty_bars[buf]],
                        out_dtype=tl.float32,
                    )

                    dot_phase = dot_phase ^ (buf == int(NUM_SMEM_BUFFERS) - 1)

                # Wait for last MMA to complete
                last_buf = (processed_k_iters + k_tiles -
                            1) % int(NUM_SMEM_BUFFERS)
                last_dot_phase = dot_phase ^ (
                    last_buf == int(NUM_SMEM_BUFFERS) - 1)
                tlx.barrier_wait(smem_empty_bars[last_buf], last_dot_phase)

                # Signal epilogue that result is ready
                tlx.barrier_arrive(tmem_full_bars[cur_tmem_buf], 1)

                cur_tmem_buf = (cur_tmem_buf + 1) % int(NUM_TMEM_BUFFERS)
                processed_k_iters += k_tiles

        # Producer: TMA loads for X and W
        with tlx.async_task(num_warps=1, num_regs=24):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_K)

            load_phase = 0
            processed_k_iters = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS
                )
                offs_xm = pid_m * BLOCK_M
                offs_wn = pid_n * BLOCK_N

                for k in range(0, k_tiles):
                    buf = (processed_k_iters + k) % int(NUM_SMEM_BUFFERS)

                    # Wait for buffer to be free
                    tlx.barrier_wait(smem_empty_bars[buf], load_phase ^ 1)

                    offs_k = k * BLOCK_K
                    tlx.barrier_expect_bytes(
                        smem_full_bars[buf],
                        2 * (BLOCK_M + BLOCK_N) * BLOCK_K,
                    )
                    tlx.async_descriptor_load(
                        x_desc, x_buffers[buf], [
                            offs_xm, offs_k], smem_full_bars[buf]
                    )
                    tlx.async_descriptor_load(
                        w_desc, w_buffers[buf], [
                            offs_k, offs_wn], smem_full_bars[buf]
                    )

                    load_phase = load_phase ^ (
                        buf == int(NUM_SMEM_BUFFERS) - 1)

                processed_k_iters += k_tiles


@torch.fx.wrap
def triton_addmm_fwd_tma_persistent(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    warp_specialize: bool = False,
) -> torch.Tensor:
    M, K = x.shape
    _, N = w.shape

    is_y_1d = y.dim() == 1

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    # A dummy block value that will be overwritten when we have the real block
    # size
    dummy_block = [1, 1]
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    w_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    y = y.reshape(1, -1) if is_y_1d else y
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    y_desc = TensorDescriptor(y, y.shape, y.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    z_desc = TensorDescriptor(z, z.shape, z.stride(), dummy_block)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(meta):
        nonlocal x_desc, w_desc, z_desc
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            ),
        )

    _addmm_fwd_tma_persistent[grid](
        x_desc,
        w_desc,
        y_desc,
        z_desc,
        M,
        N,
        K,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=is_y_1d,
        WARP_SPECIALIZE=warp_specialize,
        NUM_SMS=NUM_SMS,
    )
    return z


@torch.fx.wrap
def triton_addmm_fwd_tma_ws_tlx(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    _, N = w.shape

    is_y_1d = y.dim() == 1

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    # A dummy block value that will be overwritten when we have the real block
    # size
    dummy_block = [1, 1]
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    w_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    y = y.reshape(1, -1) if is_y_1d else y
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    y_desc = TensorDescriptor(y, y.shape, y.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    z_desc = TensorDescriptor(z, z.shape, z.stride(), dummy_block)

    def grid(meta):
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        return (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

    _addmm_fwd_tma_ws[grid](
        x_desc,
        w_desc,
        y_desc,
        z_desc,
        M,
        N,
        K,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=is_y_1d,
        NUM_SMEM_BUFFERS=2,  # Double buffering
    )
    return z


@torch.fx.wrap
def triton_addmm_fwd_tma_ws_persistent_tlx(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    _, N = w.shape

    is_y_1d = y.dim() == 1

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    NUM_SMEM_BUFFERS = 2
    NUM_TMEM_BUFFERS = 2
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten by the hook
    dummy_block = [1, 1]
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    w_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    y = y.reshape(1, -1) if is_y_1d else y
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    y_desc = TensorDescriptor(y, y.shape, y.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    z_desc = TensorDescriptor(z, z.shape, z.stride(), dummy_block)

    def grid(meta):
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            ),
        )

    _addmm_fwd_tma_ws_persistent[grid](
        x_desc,
        w_desc,
        y_desc,
        z_desc,
        M,
        N,
        K,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=is_y_1d,
        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
        NUM_SMS=NUM_SMS,
    )
    return z


@torch.fx.wrap
def triton_addmm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    KB, N = w.shape
    assert K == KB, f"incompatible dimensions {K}, {KB}"

    is_y_1d = y.dim() == 1
    NY = y.shape[0] if is_y_1d else y.shape[1]
    assert N == NY, f"incompatible dimensions {N}, {NY}"

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    def grid(meta): return (  # noqa E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _addmm_fwd[grid](
        x,
        w,
        y,
        z,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0) if not is_y_1d else 0,
        y.stride(1) if not is_y_1d else y.stride(0),
        z.stride(0),
        z.stride(1),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=is_y_1d,
    )
    return z


def triton_addmm_bwd(
    x: torch.Tensor,
    w: torch.Tensor,
    dz: torch.Tensor,
    is_y_1d: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_y_1d:
        dy = torch.sum(dz, dim=0)
    else:
        dy = dz
    dw = torch.mm(x.t(), dz)
    dx = torch.mm(dz, w.t())

    return dx, dw, dy


# Plan 22 A2 — fp8 (e4m3) GEMMs for the two HSTU dense projections (UVQK and
# output) via torch._scaled_mm / hipBLASLt. Gated by DLRM_HSTU_FP8_GEMM=1; off by
# default so the fp16 rocBLAS path is unchanged. The UVQK (K=512→2048) and output
# (K=1536→512) GEMMs are ~19% of GPU at the production shape (Plan 22 §8 E1).
_HSTU_FP8_GEMM: bool = os.environ.get("DLRM_HSTU_FP8_GEMM", "0") == "1"
# use_fast_accum: fp8 accumulation in lower precision (faster); we have large
# accuracy headroom (≥99.9%-relative bar). Set DLRM_HSTU_FP8_GEMM_FASTACC=0 to disable.
_HSTU_FP8_GEMM_FASTACC: bool = (
    os.environ.get("DLRM_HSTU_FP8_GEMM_FASTACC", "1") == "1"
)
# Plan 22 §8: dynamic per-call activation amax (reduce+abs) costs ~3.6 ms/batch —
# more than the fp8 GEMM saving. With STATIC_ASCALE the activation scale is
# calibrated once per GEMM site (first call, ×safety margin) and frozen, so steady
# state pays only the cast, not the reduction. LayerNorm-normalized activations are
# range-stable, so this is safe; values above the calibrated max just saturate at
# e4m3's 448. Default ON when fp8 GEMM is enabled.
_HSTU_FP8_GEMM_STATIC_ASCALE: bool = (
    os.environ.get("DLRM_HSTU_FP8_GEMM_STATIC_ASCALE", "1") == "1"
)
_FP8_ASCALE_MARGIN: float = float(
    os.environ.get("DLRM_HSTU_FP8_GEMM_ASCALE_MARGIN", "1.5")
)
# Residual fold: fold the OUTPROJ residual skip (`out + x`, a full [M,N] matrix) into the
# fp8 GEMM via the hipBLASLt beta*C epilogue (route_a_probe/fp8tuned_ext), removing the
# standalone elementwise add kernel (CUDAFunctor_add<bf16> in ## stu_compute_output ##).
# torch._scaled_mm only fuses a 1-D bias, so the matrix residual needs the custom op.
# NOT bit-exact: the residual accumulates in fp32 inside the epilogue and rounds once
# (measured slightly MORE accurate than the bf16 add) -> needs an accuracy re-cert.
# Single switch DLRM_HSTU_FP8_RESID, bf16-output only, gated under DLRM_HSTU_FP8_GEMM:
#   off  : disabled — separate bf16 add (default)
#   pin  : fold + pinned OUTPROJ Tensile solution. The stock top-1 heuristic mis-picks at
#          the production token count (~131072): a slow solution that *cancels* the saved
#          add (-4.8% vs +30% expected). A pinned solution is robustly fast across the whole
#          inference token range (+23..+40% on the OUTPROJ GEMM block). RECOMMENDED.
#   heur : fold + stock hipBLASLt heuristic (no pin) — probing/debug only.
# e2e (b64, MI350, saturation probe): fold halves OUTPROJ-driven tail latency (p99 636->293ms);
# pin vs heur is an e2e wash (the GEMM time is hidden behind attention), so the win is the
# deleted add kernel, not the GEMM solution.
_FP8_RESID_MODE: str = os.environ.get("DLRM_HSTU_FP8_RESID", "off").strip().lower()
if _FP8_RESID_MODE not in ("off", "pin", "heur"):
    _FP8_RESID_MODE = "off"
_HSTU_FP8_FUSE_RESID: bool = _HSTU_FP8_GEMM and _FP8_RESID_MODE in ("pin", "heur")
# Tensile solution index used in `pin` mode for the OUTPROJ (N=512,K=1536) shape. Advanced
# override only (the index is tied to the hipBLASLt build); `pin`/`heur` is the real switch.
_FP8_RESID_PIN: int = int(os.environ.get("DLRM_HSTU_FP8_RESID_PIN", "454444"))
_FP8_E4M3 = torch.float8_e4m3fn
_FP8_MAX = 448.0
# scale_result=1 ⇒ _scaled_mm emits a raw saturating cast of the true projection,
# matching A1's `q.to(e4m3)` (no output rescale). Lazily created on first use.
_FP8_ONE: Optional[torch.Tensor] = None
# Static-weight fp8 cache keyed by weight storage ptr: weights don't change at
# inference, so quantize each weight to e4m3 (col-major for _scaled_mm mat2) +
# its per-tensor dequant scale exactly once. Value: (w_fp8_colmajor, scale_b, shape).
_fp8_weight_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Size]] = {}
# Calibrated static activation scale per GEMM site (keyed by weight ptr).
_fp8_ascale_cache: Dict[int, torch.Tensor] = {}
_fp8_gemm_fallback_warned: bool = False
# Replayable activation-scale store (DLRM_FP8_SCALE_STORE). Loaded lazily per rank.
_fp8_scale_store: Optional[Dict[str, float]] = None
_fp8_scale_counter: Dict[Tuple[int, str], int] = {}


def _fp8_current_rank() -> int:
    """Best-effort process rank (dist -> MPI/env fallbacks -> -1)."""
    try:
        import torch.distributed as _dist

        if _dist.is_available() and _dist.is_initialized():
            return _dist.get_rank()
    except Exception:  # noqa: BLE001
        pass
    for _k in ("RANK", "OMPI_COMM_WORLD_RANK", "LOCAL_RANK"):
        _v = os.environ.get(_k)
        if _v is not None:
            try:
                return int(_v)
            except Exception:  # noqa: BLE001
                pass
    return -1


def _fp8_static_scale(tag: str, computed: float) -> float:
    """Run-to-run-stable fp8 activation scale.

    The per-tensor activation scale is otherwise frozen from the ``amax`` of the
    first (dynamically-composed, nondeterministic) batch to hit each site, so it
    varies run-to-run and shifts every downstream fp8 quantization -> the TEST08
    drift.     With ``$DLRM_FP8_SCALE_STORE=<path>`` set, replay a persisted scale for
    this site (key = ``rank:tag:call-index``, a deterministic layer-traversal
    order) or persist the freshly computed one on first sight, so two runs use
    byte-identical scales. No-op (returns ``computed``) when the env var is unset.

    ``$DLRM_FP8_SCALE_STORE_SHARED=1`` unifies the scale ACROSS ranks: every rank
    reads rank 0's store (``.rank0``, key ``0:tag:idx``) and only rank 0 writes.
    The per-rank scale is otherwise calibrated from that rank's own first live
    batch, so the 8 ranks hold *different* scales; under batch=1 async dispatch a
    query lands on a different rank run-to-run and is quantized with a different
    scale -> pervasive run-to-run drift. Sharing makes every rank use one scale so
    which-rank-served-the-query no longer perturbs the output. Pre-populate
    ``.rank0`` with a generation run before the compare runs (fresh shared file =>
    non-rank-0 ranks miss and fall back to their own computed scale for that run).
    """
    global _fp8_scale_store
    path = os.environ.get("DLRM_FP8_SCALE_STORE")
    if not path:
        return computed
    import json

    rank = _fp8_current_rank()
    shared = os.environ.get("DLRM_FP8_SCALE_STORE_SHARED", "0") == "1"
    eff_rank = 0 if shared else rank
    rank_path = f"{path}.rank{eff_rank}"
    if _fp8_scale_store is None:
        try:
            with open(rank_path) as _fh:
                _fp8_scale_store = json.load(_fh)
        except Exception:  # noqa: BLE001
            _fp8_scale_store = {}
    idx = _fp8_scale_counter.get((rank, tag), 0)
    _fp8_scale_counter[(rank, tag)] = idx + 1
    key = f"{eff_rank}:{tag}:{idx}"
    if key in _fp8_scale_store:
        return float(_fp8_scale_store[key])
    _fp8_scale_store[key] = float(computed)
    # In shared mode only rank 0 owns the file, so concurrent ranks never clobber
    # it with their own (divergent) calibration values.
    if not shared or rank == 0:
        try:
            with open(rank_path, "w") as _fh:
                json.dump(_fp8_scale_store, _fh)
        except Exception:  # noqa: BLE001
            pass
    return computed


def _dump_fp8_scale(tag: str, amax_val: float, scale_val: float, shape) -> None:
    """Env-gated ($DLRM_FP8_DUMP_SCALE=path) one-line dump of a frozen fp8
    calibration scale, for cross-run determinism diffing. No-op unless set."""
    path = os.environ.get("DLRM_FP8_DUMP_SCALE")
    if not path:
        return
    rank = _fp8_current_rank()
    try:
        shp = "x".join(str(int(s)) for s in shape)
        with open(path, "a") as _fh:
            _fh.write(
                f"pid={os.getpid()} rank={rank} tag={tag} shape={shp} "
                f"amax={amax_val:.10e} scale={scale_val:.10e}\n"
            )
    except Exception:  # noqa: BLE001
        pass


def _fp8_quantize_per_tensor(
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """e4m3 quantize with a per-tensor dequant scale (amax / 448)."""
    amax = t.detach().abs().amax().clamp(min=1e-12)
    scale = (amax / _FP8_MAX).to(torch.float32).reshape(1, 1)
    q = (t / scale).to(_FP8_E4M3)
    return q, scale


def _scaled_addmm_fp8(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """y + x @ w with x,w in e4m3 (fp32-dequant), via torch._scaled_mm.

    x: [M,K] activation (dynamic per-tensor scale). w: [K,N] static weight
    (cached fp8, col-major). y: bias ([N] or [M,N]). Returns x.dtype.
    """
    out_dtype = x.dtype
    key = w.data_ptr()

    if _HSTU_FP8_GEMM_STATIC_ASCALE:
        # Calibrate once per GEMM site, then reuse (no per-call amax reduction).
        scale_a = _fp8_ascale_cache.get(key)
        if scale_a is None:
            amax = x.detach().abs().amax().clamp(min=1e-12)
            scale_val = _fp8_static_scale(
                "addmm", float(amax * _FP8_ASCALE_MARGIN / _FP8_MAX)
            )
            scale_a = torch.tensor(
                [[scale_val]], dtype=torch.float32, device=x.device
            )
            _fp8_ascale_cache[key] = scale_a
            _dump_fp8_scale("addmm", float(amax), scale_val, x.shape)
        xq = (x.contiguous() / scale_a).to(_FP8_E4M3)
    else:
        xq, scale_a = _fp8_quantize_per_tensor(x.contiguous())

    cached = _fp8_weight_cache.get(key)
    if cached is None or cached[2] != w.shape:
        wq, scale_b = _fp8_quantize_per_tensor(w)
        # _scaled_mm needs mat2 column-major ([K,N] with stride (1,K)).
        wq_cm = wq.t().contiguous().t()
        cached = (wq_cm, scale_b, w.shape)
        _fp8_weight_cache[key] = cached
    wq_cm, scale_b, _ = cached

    if y.dim() == 1:
        return torch._scaled_mm(
            xq,
            wq_cm,
            scale_a=scale_a,
            scale_b=scale_b,
            bias=y.to(out_dtype),
            out_dtype=out_dtype,
            use_fast_accum=_HSTU_FP8_GEMM_FASTACC,
        )
    if (
        _HSTU_FP8_FUSE_RESID
        and out_dtype == torch.bfloat16
        and y.dtype == torch.bfloat16
    ):
        # Fold the [M,N] residual into the GEMM epilogue (D = xq@wq + 1.0*y) instead
        # of a standalone `out + y` kernel. Falls back on any unsupported shape/op error.
        try:
            return _scaled_addmm_fp8_resid(xq, scale_a, wq_cm, scale_b, y)
        except Exception as exc:  # noqa: BLE001
            global _fp8_resid_fallback_warned
            if not _fp8_resid_fallback_warned:
                _fp8_resid_fallback_warned = True
                print(
                    f"[resid-fold] fp8 beta*C residual fold fell back to "
                    f"separate add: {exc!r}"
                )
    out = torch._scaled_mm(
        xq,
        wq_cm,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
        use_fast_accum=_HSTU_FP8_GEMM_FASTACC,
    )
    return out + y


def _scaled_addmm_fp8_preq(
    xq: torch.Tensor,
    scale_a: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Plan 22 A-FUSE Φ1: y + xq @ w where `xq` is ALREADY e4m3-quantized by the
    producer (the fused LN epilogue) with dequant scale `scale_a`. Skips the
    activation amax+cast entirely — that was A2's dominant overhead. Weight is
    quantized+cached exactly as in `_scaled_addmm_fp8`.
    """
    out_dtype = y.dtype if y.is_floating_point() else torch.float16
    key = w.data_ptr()
    cached = _fp8_weight_cache.get(key)
    if cached is None or cached[2] != w.shape:
        wq, scale_b = _fp8_quantize_per_tensor(w)
        wq_cm = wq.t().contiguous().t()
        cached = (wq_cm, scale_b, w.shape)
        _fp8_weight_cache[key] = cached
    wq_cm, scale_b, _ = cached

    if y.dim() == 1:
        return torch._scaled_mm(
            xq,
            wq_cm,
            scale_a=scale_a,
            scale_b=scale_b,
            bias=y.to(out_dtype),
            out_dtype=out_dtype,
            use_fast_accum=_HSTU_FP8_GEMM_FASTACC,
        )
    if (
        _HSTU_FP8_FUSE_RESID
        and out_dtype == torch.bfloat16
        and y.dtype == torch.bfloat16
    ):
        # Fold the [M,N] residual into the GEMM epilogue (D = xq@wq + 1.0*y) instead
        # of a standalone `out + y` kernel. Falls back on any unsupported shape/op error.
        try:
            return _scaled_addmm_fp8_resid(xq, scale_a, wq_cm, scale_b, y)
        except Exception as exc:  # noqa: BLE001
            global _fp8_resid_fallback_warned
            if not _fp8_resid_fallback_warned:
                _fp8_resid_fallback_warned = True
                print(
                    f"[resid-fold] fp8 beta*C residual fold fell back to "
                    f"separate add: {exc!r}"
                )
    out = torch._scaled_mm(
        xq,
        wq_cm,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
        use_fast_accum=_HSTU_FP8_GEMM_FASTACC,
    )
    return out + y


def _scaled_mm_fp8_out_preq(
    xq: torch.Tensor,
    scale_a: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Plan 22 A-FUSE Φ2: (xq @ w + bias) emitted directly in e4m3 (scale_result=1,
    i.e. a raw saturating cast of the real projection — matches A1's `q.to(e4m3)`).
    `xq` is the already-fp8 activation (from the fused LN); `w` is the static weight
    (quantized+cached). Used for the v/q/k slice of the UVQK projection so the fused
    attention consumes fp8 q/k/v with no standalone cast.
    """
    global _FP8_ONE
    if _FP8_ONE is None or _FP8_ONE.device != xq.device:
        _FP8_ONE = torch.tensor([[1.0]], dtype=torch.float32, device=xq.device)
    key = w.data_ptr()
    cached = _fp8_weight_cache.get(key)
    if cached is None or cached[2] != w.shape:
        wq, scale_b = _fp8_quantize_per_tensor(w)
        wq_cm = wq.t().contiguous().t()
        cached = (wq_cm, scale_b, w.shape)
        _fp8_weight_cache[key] = cached
    wq_cm, scale_b, _ = cached
    return torch._scaled_mm(
        xq,
        wq_cm,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=None if bias is None else bias.to(torch.bfloat16),
        out_dtype=_FP8_E4M3,
        scale_result=_FP8_ONE,
        use_fast_accum=_HSTU_FP8_GEMM_FASTACC,
    )


# Residual fold: lazily-loaded hipBLASLt op (route_a_probe/fp8tuned_ext) that supports a
# fused beta*C epilogue (D = A@B + 1.0*C). Only imported when DLRM_HSTU_FP8_RESID is
# pin/heur, so the default (off) path never touches it. Override its location with
# DLRM_FP8TUNED_EXT_PATH (falls back to DLRM_MXFP4_EXT_PATH, then route_a_probe).
_fp8tuned_mod = None
_fp8_resid_fallback_warned: bool = False


def _get_fp8tuned():
    global _fp8tuned_mod
    if _fp8tuned_mod is None:
        try:
            from generative_recommenders.ops.triton import fp8tuned_ext as _m  # vendored
        except Exception:  # noqa: BLE001
            import sys
            ext_root = os.environ.get(
                "DLRM_FP8TUNED_EXT_PATH",
                os.environ.get(
                    "DLRM_MXFP4_EXT_PATH", "/work/route_a_probe"
                ),
            )
            if ext_root not in sys.path:
                sys.path.insert(0, ext_root)
            import fp8tuned_ext as _m
        _fp8tuned_mod = _m
    return _fp8tuned_mod


def _scaled_addmm_fp8_resid(
    xq: torch.Tensor,
    scale_a: torch.Tensor,
    wq_cm: torch.Tensor,
    scale_b: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Residual fold: out = (xq @ wq) + y in one hipBLASLt call (beta*C, beta=1).

    `xq` is the already-e4m3 activation [M,K] with dequant scale `scale_a`; `wq_cm` is the
    fp8 weight ([K,N] col-major) with scale `scale_b`; `y` is the [M,N] bf16 residual skip.
    Mirrors the OUTPROJ mapping in fp8tuned_ext.test (a=weight, b=activation, trans_a=True,
    scales swapped): the op returns D[n=M, m=N] == [M,N] bf16 with the residual already
    added. bf16 output only; the caller must guard dtype and fall back on any exception.
    """
    M, K = xq.shape
    N = wq_cm.shape[1]
    # `pin` mode: known-good OUTPROJ solution (heuristic mis-picks here). `heur`/other: -1.
    pin = (
        _FP8_RESID_PIN
        if (_FP8_RESID_MODE == "pin" and N == 512 and K == 1536)
        else -1
    )
    mod = _get_fp8tuned()
    return mod.scaled_mm_tuned(
        wq_cm,
        xq,
        scale_b,
        scale_a,
        N,
        M,
        K,
        K,
        K,
        True,
        False,
        bias=None,
        out_bf16=True,
        pin_index=pin,
        c=y.contiguous(),
    )


# Plan 37 (H/C): fused MXFP4-pack epilogue op (route_a_probe/mxfp4_ext). Loaded lazily
# and only when DLRM_HSTU_FP4_QK_FUSED is on, so the import/compile never runs on the
# default path. Override its location with DLRM_MXFP4_EXT_PATH.
_mxfp4_aux_mod = None


def _get_mxfp4_aux():
    global _mxfp4_aux_mod
    if _mxfp4_aux_mod is None:
        try:
            from generative_recommenders.ops.triton import mxfp4_ext as _m  # vendored
        except Exception:  # noqa: BLE001
            import sys
            ext_root = os.environ.get(
                "DLRM_MXFP4_EXT_PATH", "/work/route_a_probe"
            )
            if ext_root not in sys.path:
                sys.path.insert(0, ext_root)
            import mxfp4_ext as _m
        _mxfp4_aux_mod = _m
    return _mxfp4_aux_mod


_fused_pack_checked = False


def _assert_fused_pack_selected(aux_mod) -> None:
    """Plan 37 (L): guard against a silent heuristic miss. If HIPBLASLT_TENSILE_LIBPATH
    is not the merged/lib_mxpack library, stock hipBLASLt may return a generic RELU_AUX
    solution (got>0) that writes fp16 into AUX instead of the MXFP4 pack — yielding
    garbage kp/ks. Exercise the SAME call path the K-projection uses (the TN
    scaled_mm_mxfp4_aux wrapper, trans_a=True -> Alik_Bljk contraction) so the canary
    actually probes the solution the consumer selects, then verify the AUX scale-plane
    equals the expected e8m0 (deterministic — no tie ambiguity); raise loudly otherwise."""
    global _fused_pack_checked
    if _fused_pack_checked:
        return
    dev = "cuda"
    tokens, hidden, d_head = 256, 512, 512
    g = torch.Generator(device=dev).manual_seed(0)
    xq = (torch.rand(tokens, hidden, generator=g, device=dev) * 3.0).to(_FP8_E4M3)
    w = (torch.rand(hidden, d_head, generator=g, device=dev) * 3.0).to(_FP8_E4M3)
    wq_cm = w.t().contiguous().t()
    one = torch.ones(1, dtype=torch.float32, device=dev)
    D, aux = aux_mod.scaled_mm_mxfp4_aux(xq, one, wq_cm, one, None)
    _kp, ks = aux_mod.split_aux(aux, d_head, tokens)
    Df = D.float().reshape(tokens, d_head // 32, 32)
    amax = Df.abs().amax(-1).clamp(min=1e-20)
    exp = torch.ceil(torch.log2(amax / 6.0))
    sb = (exp.to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)
    if not torch.equal(ks, sb):
        nbad = int((ks != sb).sum().item())
        raise RuntimeError(
            f"[Plan37] fused MXFP4-pack (MXScaleE) solution NOT selected "
            f"({nbad}/{ks.numel()} scale-plane mismatches). Point "
            f"HIPBLASLT_TENSILE_LIBPATH at the merged lib_mxpack (TN) library so the "
            f"fused epilogue is used for the K-projection."
        )
    _fused_pack_checked = True


def _scaled_mm_mxfp4_aux_kpks(
    xq: torch.Tensor,
    scale_a: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plan 37: fused transposed K-projection + MXFP4 pack in one hipBLASLt call.

    Computes K^T = (w^T) @ (xq^T) (out = D head-dim x tokens) and emits the packed
    K (e2m1) + e8m0 scales in the epilogue, returning the two dense planes:
      kp [tokens, D_out//2] uint8, ks [tokens, D_out//32] uint8   (D_out = w.shape[1])
    laid out row-major so a caller `.view(L, H, D//2)` / `.view(L, H, D//32)` matches
    `_quantize_k_mxfp4` bit-exactly (both RNE). `xq` is the already-e4m3 activation
    (fused LN) with dequant scale `scale_a`; `w` is the static weight (fp8-cached).
    """
    key = w.data_ptr()
    cached = _fp8_weight_cache.get(key)
    if cached is None or cached[2] != w.shape:
        wq, scale_b = _fp8_quantize_per_tensor(w)
        wq_cm = wq.t().contiguous().t()
        cached = (wq_cm, scale_b, w.shape)
        _fp8_weight_cache[key] = cached
    wq_cm, scale_b, _ = cached
    d_out = w.shape[1]
    tokens = xq.shape[0]
    assert d_out % 32 == 0, f"MXFP4 K-pack needs out width % 32 == 0, got {d_out}"
    aux_mod = _get_mxfp4_aux()
    _assert_fused_pack_selected(aux_mod)
    bias_t = None if bias is None else bias.to(torch.float16)
    _D, aux = aux_mod.scaled_mm_mxfp4_aux(xq, scale_a, wq_cm, scale_b, bias_t)
    kp, ks = aux_mod.split_aux(aux, d_out, tokens)
    return kp, ks


@torch.fx.wrap
def maybe_triton_addmm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    # Plan 22 A2: fp8 path for the HSTU dense GEMMs on HIP (env-gated, off by
    # default). Falls back to fp16 addmm on any shape/dtype it can't handle.
    if _HSTU_FP8_GEMM and torch.version.hip is not None and x.dim() == 2:
        M, K = x.shape
        N = w.shape[1]
        if (
            M > 0
            and N > 0
            and K % 16 == 0
            and N % 16 == 0
            and x.dtype in (torch.float16, torch.bfloat16)
        ):
            try:
                return _scaled_addmm_fp8(x, w, y)
            except Exception as exc:  # noqa: BLE001
                global _fp8_gemm_fallback_warned
                if not _fp8_gemm_fallback_warned:
                    _fp8_gemm_fallback_warned = True
                    print(
                        f"[Plan22 A2] fp8 GEMM fell back to fp16 addmm "
                        f"(M={M},K={K},N={N}): {exc!r}"
                    )
    # triton addmm is slower than torch (cublas) on AMD/Blackwell.
    # Default to pytorch addmm on AMD/Blackwell for now.
    if is_sm100() or torch.version.hip is not None:
        return torch.addmm(y, x, w)
    else:
        return triton_addmm_fwd(x=x, w=w, y=y)


class _AddMmFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        ctx.is_y_1d = y.dim() == 1
        if is_sm100() and TMA_AVAILABLE and _check_tma_alignment(x, w, y):
            if x.dtype == torch.float32 or HAS_TLX == False:
                # use TMA persistent kernel on sm100
                return triton_addmm_fwd_tma_persistent(
                    x, w, y, warp_specialize=True)
            else:
                return triton_addmm_fwd_tma_ws_persistent_tlx(
                    x, w, y
                )  # tlx.async_dot doesn't support fp32 inputs because of WGMMA requirements
        else:
            return triton_addmm_fwd(x, w, y)

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (x, w) = ctx.saved_tensors
        return triton_addmm_bwd(x, w, dz, ctx.is_y_1d)


def triton_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> torch.Tensor:
    return _AddMmFunction.apply(mat1, mat2, input)
