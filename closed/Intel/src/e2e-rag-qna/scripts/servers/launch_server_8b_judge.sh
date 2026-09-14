#!/bin/bash

# Load config (repo root is two levels up); safe if absent.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "${_ROOT}/config.sh" ] && source "${_ROOT}/config.sh"
[ -f "${_ROOT}/config.default.sh" ] && source "${_ROOT}/config.default.sh"
source "$(dirname "${BASH_SOURCE[0]}")/health_check.sh"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MODEL_PATH="${MODEL_JUDGE_PATH}"
PORT=${PORT:-${SERVER_JUDGE_PORT:-8125}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-${SERVER_JUDGE_MAX_MODEL_LEN:-16384}}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-${SERVER_JUDGE_GPU_MEM_UTIL:-0.12}}

NODES=${NODES:-${SERVER_JUDGE_NODES:-"0 1"}}
NODE_START=(0 43 86 129)
read -ra NODE_ARR <<< "$NODES"
TP=${#NODE_ARR[@]}

OMP_BIND="" ; TASKSET_LIST=""
for n in "${NODE_ARR[@]}"; do
    s=${NODE_START[$n]}
    e=$((s + 42))
    OMP_BIND="${OMP_BIND:+$OMP_BIND|}${s}-${e}"
    TASKSET_LIST="${TASKSET_LIST:+$TASKSET_LIST,}${s}-${e}"
done
export OMP_NUM_THREADS=43
export VLLM_CPU_OMP_THREADS_BIND="$OMP_BIND"
echo "[8b-judge] NODES=$NODES TP=$TP bind=$VLLM_CPU_OMP_THREADS_BIND taskset=$TASKSET_LIST"

# Launch unattended; log to repo root.
LOG="${_ROOT}/log-server-8b-judge.log"
echo "[8b-judge] port=${PORT} -> ${LOG}"
nohup taskset -c "$TASKSET_LIST" vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port ${PORT} \
        --served-model-name meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len ${MAX_MODEL_LEN} \
        --data-parallel-size 1 \
        --tensor-parallel-size ${TP} \
        --max-num-seqs 128 \
        --kv-cache-dtype fp8 \
        --gpu-memory-utilization ${GPU_MEM_UTIL} > "${LOG}" 2>&1 &
echo "[8b-judge] started PID $! (unattended)"

health_check_v1
