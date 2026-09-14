/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/*
 * Plan 14 (ROCm/HIP port) — central CUDA->HIP compatibility shim.
 *
 * Included (under -DNVE_ROCM) in place of <cuda.h>/<cuda_runtime.h>/<cuda_fp16.h>
 * so the NVE sources keep their CUDA spelling (cuMem*, CUmem*, cuda*) while
 * compiling against the HIP runtime + driver API. The mapping is the one the
 * Tier-1 de-risk spike validated 1:1 on 8x gfx950 (HIP VMM + pidfd over xGMI).
 *
 * Single-source-of-truth for the type/enum/function aliases. Error-type wrappers
 * (is_success/RuntimeError specializations) live in cuda_support.hpp because they
 * need common.hpp; HIP merges the CUDA runtime (cudaError_t) and driver (CUresult)
 * error enums into a single hipError_t, so there is exactly ONE specialization.
 */
#pragma once

#ifndef NVE_ROCM
#error "nve_hip_compat.hpp is only for the ROCm/HIP build (-DNVE_ROCM)"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// ------------------------------------------------------------- reduced-precision
// nve_types.hpp spells the NVIDIA bf16/fp8 type names; alias them to the HIP types
// (gfx950 supports the OCP e4m3/e5m2 fp8 encodings).
using nv_bfloat16   = __hip_bfloat16;
using __nv_bfloat16 = __hip_bfloat16;
using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8_e5m2 = __hip_fp8_e5m2;

// ------------------------------------------------------------------ runtime API
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
#define cudaSuccess            hipSuccess
#define cudaGetDevice          hipGetDevice
#define cudaSetDevice          hipSetDevice
#define cudaGetDeviceCount     hipGetDeviceCount
#define cudaMalloc             hipMalloc
#define cudaFree               hipFree
#define cudaMemset             hipMemset
#define cudaMemsetAsync        hipMemsetAsync
#define cudaMemcpy             hipMemcpy
#define cudaMemcpyAsync        hipMemcpyAsync
#define cudaMemcpyDefault      hipMemcpyDefault
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaDeviceSynchronize  hipDeviceSynchronize
#define cudaStreamSynchronize  hipStreamSynchronize
#define cudaGetLastError       hipGetLastError
#define cudaPeekAtLastError    hipPeekAtLastError
#define cudaGetErrorName       hipGetErrorName
#define cudaGetErrorString     hipGetErrorString
// stream/event lifecycle (GPUEmbeddingLayer / execution contexts).
using cudaEvent_t = hipEvent_t;
#define cudaStreamCreate         hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy        hipStreamDestroy
#define cudaStreamWaitEvent      hipStreamWaitEvent
#define cudaStreamNonBlocking    hipStreamNonBlocking
#define cudaEventCreate          hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy         hipEventDestroy
#define cudaEventRecord          hipEventRecord
#define cudaEventSynchronize     hipEventSynchronize
#define cudaEventQuery           hipEventQuery
#define cudaEventElapsedTime     hipEventElapsedTime
#define cudaEventDisableTiming   hipEventDisableTiming
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemGetInfo           hipMemGetInfo
#define cudaDeviceGetAttribute   hipDeviceGetAttribute
#define cudaFuncSetAttribute     hipFuncSetAttribute
// memory pools + pinned host memory (DefaultAllocator).
using cudaMemPool_t = hipMemPool_t;
#define cudaMallocAsync                 hipMallocAsync
#define cudaMallocFromPoolAsync         hipMallocFromPoolAsync
#define cudaFreeAsync                   hipFreeAsync
#define cudaMallocHost                  hipHostMalloc   // hipMallocHost is deprecated
#define cudaFreeHost                    hipHostFree     // hipFreeHost is deprecated
#define cudaHostRegister                hipHostRegister
#define cudaHostUnregister              hipHostUnregister
#define cudaHostRegisterDefault         hipHostRegisterDefault
#define cudaHostRegisterPortable        hipHostRegisterPortable
#define cudaHostRegisterMapped          hipHostRegisterMapped
#define cudaDeviceGetDefaultMemPool     hipDeviceGetDefaultMemPool
#define cudaMemPoolSetAttribute         hipMemPoolSetAttribute
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported
// pointer / memory-type queries (BufferWrapper).
using cudaPointerAttributes = hipPointerAttribute_t;
using cudaMemoryType        = hipMemoryType;
#define cudaPointerGetAttributes   hipPointerGetAttributes
#define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered
#define cudaMemoryTypeHost         hipMemoryTypeHost
#define cudaMemoryTypeDevice       hipMemoryTypeDevice
#define cudaMemoryTypeManaged      hipMemoryTypeManaged
// managed / unified memory (ManagedMemBlock).
#define cudaMallocManaged                  hipMallocManaged
#define cudaMemAdvise                      hipMemAdvise
#define cudaMemPrefetchAsync               hipMemPrefetchAsync
#define cudaMemAdviseSetPreferredLocation  hipMemAdviseSetPreferredLocation
#define cudaMemAdviseSetAccessedBy         hipMemAdviseSetAccessedBy
#define cudaCpuDeviceId                    hipCpuDeviceId
// error codes.
#define cudaErrorMemoryAllocation       hipErrorMemoryAllocation
#define cudaErrorInvalidValue           hipErrorInvalidValue
#define cudaErrorNotReady               hipErrorNotReady
#define cudaErrorNotSupported           hipErrorNotSupported
#define cudaErrorInvalidDevicePointer   hipErrorInvalidDevicePointer
#define cudaErrorInvalidDevice          hipErrorInvalidDevice

// ------------------------------------------------------------------ driver API
// HIP unifies runtime+driver error codes: CUresult -> hipError_t (== cudaError_t).
using CUresult = hipError_t;
using CUdevice = hipDevice_t;
using CUdeviceptr = hipDeviceptr_t;  // void* on HIP (CUdeviceptr is ull on CUDA)
using CUmemGenericAllocationHandle = hipMemGenericAllocationHandle_t;
using CUmemAllocationProp = hipMemAllocationProp;
using CUmemAccessDesc = hipMemAccessDesc;

#define CUDA_SUCCESS hipSuccess

// CUmemAllocationProp field rename: CUDA `requestedHandleTypes` -> HIP `requestedHandleType`.
#define requestedHandleTypes requestedHandleType

// Virtual Memory Management (the MPIMemBlock / CUDADistributedBuffer mechanism).
#define cuMemCreate                    hipMemCreate
#define cuMemRelease                   hipMemRelease
#define cuMemMap                       hipMemMap
#define cuMemUnmap                     hipMemUnmap
#define cuMemSetAccess                 hipMemSetAccess
#define cuMemAddressReserve            hipMemAddressReserve
#define cuMemAddressFree               hipMemAddressFree
#define cuMemGetAllocationGranularity  hipMemGetAllocationGranularity
#define cuMemExportToShareableHandle   hipMemExportToShareableHandle
#define cuMemImportFromShareableHandle hipMemImportFromShareableHandle

// Device init/query (MPIEnv).
#define cuInit               hipInit
#define cuDeviceGet          hipDeviceGet
#define cuDeviceGetCount     hipGetDeviceCount
#define cuDeviceGetName      hipDeviceGetName
#define cuDeviceGetAttribute hipDeviceGetAttribute

// Allocation property / access enums.
#define CU_MEM_ALLOCATION_TYPE_PINNED            hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE              hipMemLocationTypeDevice
#define CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR hipMemHandleTypePosixFileDescriptor
#define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED     hipMemAllocationGranularityRecommended
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE       hipMemAccessFlagsProtReadWrite

// Device attribute enums.
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID hipDeviceAttributePciDomainID
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID    hipDeviceAttributePciBusId
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID hipDeviceAttributePciDeviceId
