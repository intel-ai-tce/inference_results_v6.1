// Plan 14 ROCm: CUDA fp16 -> HIP fp16, plus the device-intrinsic compat shims
// (shuffle mask, streaming load/store, scalar half/bf16 atomicAdd) that the NVE
// device kernels need. Every .cu/.cuh that includes <cuda_fp16.h> picks these up.
#pragma once
#include <hip/hip_fp16.h>
#include "nve_hip_device_compat.cuh"
