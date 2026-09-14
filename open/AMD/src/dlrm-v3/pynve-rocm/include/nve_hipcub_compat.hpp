/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/*
 * Plan 14.2 (ROCm/HIP kernels) — CUB -> hipCUB shim.
 *
 * The NVE device kernels use NVIDIA CUB (`#include <cub/cub.cuh>`, `cub::Device*`).
 * hipCUB is the ROCm-portable wrapper over rocPRIM with a CUB-identical API, so a
 * namespace alias lets the call sites keep their `cub::` spelling unchanged.
 * Kept separate from nve_hip_compat.hpp so non-kernel TUs don't pull in hipCUB.
 */
#pragma once

#ifdef NVE_ROCM
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

// Portable warp/wavefront shuffle masks for the cub-using device kernels. gfx950
// wavefronts are 64 lanes and HIP statically requires a 64-bit mask; NVIDIA warps
// are 32 lanes. __activemask() returns a 32-bit lane mask (rejected by HIP where a
// 64-bit mask is needed), so on ROCm we use the all-lanes mask (advisory on AMD).
#ifndef NVE_SHFL_MASK
#ifdef NVE_ROCM
#define NVE_SHFL_MASK 0xffffffffffffffffULL
#else
#define NVE_SHFL_MASK 0xffffffffu
#endif
#endif
#ifndef NVE_ACTIVEMASK
#ifdef NVE_ROCM
#define NVE_ACTIVEMASK() NVE_SHFL_MASK
#else
#define NVE_ACTIVEMASK() __activemask()
#endif
#endif

// Host-side warp/wavefront width for kernel-launch geometry. Device code uses the
// builtin `warpSize` (64 on gfx950, 32 on NVIDIA), but `warpSize` is device-only, so
// host launchers must hardcode it — NVIDIA's upstream used the literal 32. Any launch
// whose block dim / grid stride must equal the device `warpSize` (e.g. ComputeSetKernel,
// which strides `sets[]` by `warpSize`) MUST use NVE_WARP_SIZE here, otherwise a 32-wide
// block under a 64-wide device stride leaves half of sets[] uninitialized and indexes
// out of bounds.
#ifndef NVE_WARP_SIZE
#ifdef NVE_ROCM
#define NVE_WARP_SIZE 64u
#else
#define NVE_WARP_SIZE 32u
#endif
#endif
