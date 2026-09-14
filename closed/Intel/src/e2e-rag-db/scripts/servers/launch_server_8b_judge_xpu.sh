#!/bin/bash

# Load config (repo root is two levels up); safe if absent.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "${_ROOT}/config.sh" ] && source "${_ROOT}/config.sh"
[ -f "${_ROOT}/config.default.sh" ] && source "${_ROOT}/config.default.sh"
source "$(dirname "${BASH_SOURCE[0]}")/health_check.sh"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MODEL_PATH="${MODEL_JUDGE_PATH}"
PORT=${PORT:-${SERVER_JUDGE_XPU_PORT:-8125}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-${SERVER_JUDGE_XPU_MAX_MODEL_LEN:-16384}}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-${SERVER_JUDGE_XPU_GPU_MEM_UTIL:-0.70}}

# No NUMA/core pinning on XPU; pin to a single device instead.
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-${SERVER_JUDGE_XPU_DEVICE:-0}}
TP=1

# B70 arch-parser fix (triton-xpu 3.6.0) + Triton-XPU attention kernels from vllm-xpu-kernels.
export TRITON_INTEL_DEVICE_ARCH=bmg
export VLLM_USE_TRITON_XPU_ATTN=1

# Unquantized Llama auto-selects vLLM's buggy XPUModelRunnerV2 (Triton Gumbel sampler crash); force V1.
export VLLM_USE_V2_MODEL_RUNNER=0

echo "[8b-judge-xpu] ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK TP=$TP"

# Launch unattended; log to repo root.
LOG="${_ROOT}/log-server-8b-judge-xpu.log"
echo "[8b-judge-xpu] port=${PORT} -> ${LOG}"
nohup vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port ${PORT} \
        --served-model-name meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len ${MAX_MODEL_LEN} \
        --data-parallel-size 1 \
        --tensor-parallel-size ${TP} \
        --max-num-seqs 128 \
        --kv-cache-dtype fp8 \
        --enforce-eager \
        --gpu-memory-utilization ${GPU_MEM_UTIL} > "${LOG}" 2>&1 &
echo "[8b-judge-xpu] started PID $! (unattended)"

health_check_v1
