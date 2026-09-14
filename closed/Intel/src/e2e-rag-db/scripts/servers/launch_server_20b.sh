#!/bin/bash

# Load config (repo root is two levels up); safe if absent.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "${_ROOT}/config.sh" ] && source "${_ROOT}/config.sh"
[ -f "${_ROOT}/config.default.sh" ] && source "${_ROOT}/config.default.sh"
source "$(dirname "${BASH_SOURCE[0]}")/health_check.sh"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export KV_BUFFER_DEVICE="cpu"
export DECODER_KV_LAYOUT="HND"
export UCX_TLS=sm,cma,self,tcp
export UCX_LOG_LEVEL=error
export UCX_NET_DEVICES="all"
export MODEL_PATH="${MODEL_20B_PATH}"
PORT="${PORT:-${SERVER_20B_PORT:-8192}}"
TP="${TP:-${SERVER_20B_TP:-4}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${SERVER_20B_MAX_MODEL_LEN:-8192}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${SERVER_20B_MAX_NUM_SEQS:-1024}}"
# export VLLM_TORCH_PROFILER_DIR=/workspace/traces

GPU_MEM_UTIL=${GPU_MEM_UTIL:-${SERVER_20B_GPU_MEM_UTIL:-0.3}}
CORES_PER_NODE=${CORES_PER_NODE:-${SERVER_20B_CORES_PER_NODE:-43}}
NODE_STARTS=(0 43 86 129)
OMP_BIND="" ; TASKSET_LIST=""
for s in "${NODE_STARTS[@]}"; do
    e=$((s + CORES_PER_NODE - 1))
    OMP_BIND="${OMP_BIND:+$OMP_BIND|}${s}-${e}"
    TASKSET_LIST="${TASKSET_LIST:+$TASKSET_LIST,}${s}-${e}"
done
export OMP_NUM_THREADS=${CORES_PER_NODE}
export VLLM_CPU_OMP_THREADS_BIND="$OMP_BIND"
echo "[20b] CORES_PER_NODE=$CORES_PER_NODE  bind=$VLLM_CPU_OMP_THREADS_BIND  taskset=$TASKSET_LIST"

# Launch vLLM unattended; log to repo root.
LOG="${_ROOT}/log-server-20b.log"
echo "[20b] port=${PORT} tp=${TP} -> ${LOG}"
nohup taskset -c "$TASKSET_LIST" vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port ${PORT} \
        --trust-request-chat-template \
        --data-parallel-size 1 \
        --enable-prefix-caching \
        --tensor-parallel-size ${TP} \
        --max-model-len ${MAX_MODEL_LEN} \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --kv-cache-dtype fp8 \
        --gpu-memory-utilization ${GPU_MEM_UTIL} > "${LOG}" 2>&1 &
echo "[20b] started PID $! (unattended)"

health_check_v1
