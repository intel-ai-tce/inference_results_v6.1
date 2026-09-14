// Plan 14 (ROCm/HIP port) — device-side CUDA intrinsic compatibility shim.
//
// Pulled in transitively by the <cuda_fp16.h> redirect (rocm_cuda_shim/cuda_fp16.h)
// so every device translation unit that spells a CUDA fp16 include gets:
//   * NVE_SHFL_MASK         — 64-bit wavefront shuffle mask (gfx950 has 64 lanes)
//   * __stcs/__stcg/__stwb  — CUDA streaming/cache-hint *stores*  -> plain stores
//   * __ldcs/__ldcg         — CUDA streaming/cache-hint *loads*   -> plain loads
//   * atomicAdd(__half*)    — scalar half atomic (HIP only ships the packed form)
//   * atomicAdd(__hip_bfloat16*)
// __ldg already exists in HIP, so it is intentionally NOT redefined here.
#pragma once
#ifdef NVE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// Portable warp/wavefront shuffle mask. Mirrors cuda_ops/kernels_common.cuh so the
// two definitions never disagree (guarded -> first one wins).
#ifndef NVE_SHFL_MASK
#define NVE_SHFL_MASK 0xffffffffffffffffULL
#endif

// CUDA cache-streaming load/store hints have no HIP equivalent; the hint is purely
// advisory, so a plain dereference is functionally identical.
template <typename T> __device__ __forceinline__ void __stcs(T* ptr, T val) { *ptr = val; }
template <typename T> __device__ __forceinline__ void __stcg(T* ptr, T val) { *ptr = val; }
template <typename T> __device__ __forceinline__ void __stwb(T* ptr, T val) { *ptr = val; }
template <typename T> __device__ __forceinline__ T    __ldcs(const T* ptr)  { return *ptr; }
template <typename T> __device__ __forceinline__ T    __ldcg(const T* ptr)  { return *ptr; }
template <typename T> __device__ __forceinline__ T    __ldcv(const T* ptr)  { return *ptr; }

// HIP provides packed (half2) atomicAdd but not the scalar __half / bf16 overloads
// the NVE accumulate kernels use. gfx950's unsafeAtomicAdd covers both fp16 & bf16.
__device__ __forceinline__ __half atomicAdd(__half* addr, __half val) {
  return unsafeAtomicAdd(addr, val);
}
__device__ __forceinline__ __hip_bfloat16 atomicAdd(__hip_bfloat16* addr, __hip_bfloat16 val) {
  return unsafeAtomicAdd(addr, val);
}
#endif  // NVE_ROCM
