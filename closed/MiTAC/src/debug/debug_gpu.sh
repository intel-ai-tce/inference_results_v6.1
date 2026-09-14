#!/bin/bash

# source this before running your workload

export AMD_LOG_LEVEL="3"
export AMD_SERIALIZE_KERNEL="3"
export AMD_SERIALIZE_COPY="3"
export HSA_ENABLE_DEBUG="1"
export HSA_TOOLS_LIB="/opt/rocm/lib/librocm-debug-agent.so.2"
export NCCL_NVLS_ENABLE="0"
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
