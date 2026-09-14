// Plan 14.3 cuEmbed hipify shim — CUDA async-copy pipeline -> HIP no-op fallback.
//
// cuEmbed uses __pipeline_memcpy_async/commit/wait_prior only as a global->shared
// prefetch optimization (embedding_lookup_ops.cuh LoadIndexToShmemAndSync), always
// immediately followed by __pipeline_wait_prior(0) + __syncthreads(). A synchronous
// copy is functionally identical (just no prefetch overlap). gfx950 has async LDS
// copy but correctness only needs the sync fallback for the spike.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstring>

// cudaStream_t / cudaError_t arrive transitively via <cuda_pipeline_primitives.h>
// on the NVIDIA side; provide them here for the shim.
using cudaStream_t = hipStream_t;

// Runtime/device-query API the cuEmbed BACKWARD launch heuristic
// (GetGradKernelLaunchParams) references. amdclang does early (two-phase) lookup
// of these non-dependent names even though forward never instantiates that
// template, so they must resolve for any TU that includes the header.
#define cudaGetDevice                          hipGetDevice
#define cudaDeviceGetAttribute                 hipDeviceGetAttribute
#define cudaDevAttrMaxThreadsPerMultiProcessor hipDeviceAttributeMaxThreadsPerMultiProcessor
#define cudaDevAttrMultiProcessorCount         hipDeviceAttributeMultiprocessorCount

// NOTE: the scalar __half / bf16 atomicAdd shims (used by cuEmbed's eagerly-compiled
// VecAtomicAdd backward specializations) now live in nve_hip_device_compat.cuh, which
// is force-included into every HIP TU -> do not redefine them here.

__device__ __forceinline__ void __pipeline_memcpy_async(void* dst, const void* src, size_t size) {
  // Single-thread synchronous copy; the caller already guards per-thread element ranges.
  for (size_t i = 0; i < size; ++i) {
    reinterpret_cast<char*>(dst)[i] = reinterpret_cast<const char*>(src)[i];
  }
}
__device__ __forceinline__ void __pipeline_commit() {}
__device__ __forceinline__ void __pipeline_wait_prior(int) {}
