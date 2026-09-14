#!/bin/bash
# Launch the ColBERT reranker as a vLLM OpenAI-compatible server exposing
# /v1/score. Consumed by engine/ragdb.py::rerank via requests.post.
#
# Pooling flags mirror multiturn/benchmark_vllm.py::build_offline_llm
# (--runner pooling --enforce-eager --dtype bfloat16) and its MAX_MODEL_LEN=512.

# Load config (repo root is two levels up); safe if absent.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "${_ROOT}/config.sh" ] && source "${_ROOT}/config.sh"
[ -f "${_ROOT}/config.default.sh" ] && source "${_ROOT}/config.default.sh"
cd "${_ROOT}" || exit 1

PORT=${PORT:-${RERANK_PORT:-8193}}
RERANKER_MODEL="${RERANKER_MODEL_PATH}"

LOG="${_ROOT}/log-server-rerank-vllm.log"
echo "[rerank-vllm] port=${PORT} model=${RERANKER_MODEL} -> ${LOG}"

# Health check helper.
function health_check_v1() {
    echo "[rerank-vllm] waiting for server on port ${PORT}..."
    RETRY_COUNT=0
    MAX_RETRIES=100
    while ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -gt $MAX_RETRIES ]; then
            echo "[rerank-vllm] server failed to start (see ${LOG})"
            exit 1
        fi
        sleep 5
    done
    echo "[rerank-vllm] server ready at http://localhost:${PORT}"
}

# Core affinity: reuse the SAME budget as the ColBERT pool
# (SERVER_RERANK_WORKER_CORES) so the two backends are comparable. That var is
# ';'-separated, one group per worker; vLLM wants '|'-separated, one group per
# data-parallel rank -- so translate and set dp = group count.
#
# The earlier hardcoded "37-42|123-128" gave vLLM only 12 cores vs ColBERT's 21
# and overlapped 37-38 with the 120B API host, so any perf comparison against
# the ColBERT path was measuring the core budget, not the backend.
RERANK_CORES="${SERVER_RERANK_VLLM_CORES:-${SERVER_RERANK_WORKER_CORES}}"
OMP_BIND="${RERANK_CORES//;/|}"
# dp ranks must match the number of bind groups or vLLM mis-pins / errors.
DP_SIZE="${RERANK_VLLM_DP_SIZE:-$(awk -F'|' '{print NF}' <<< "${OMP_BIND}")}"
# taskset bounds the parent (and the ranks' non-OMP threads) to the same set.
TASKSET_LIST="${OMP_BIND//|/,}"

export VLLM_CPU_OMP_THREADS_BIND="${OMP_BIND}"
echo "[rerank-vllm] dp=${DP_SIZE} bind=${VLLM_CPU_OMP_THREADS_BIND}"
taskset -c "${TASKSET_LIST}" vllm serve "${RERANKER_MODEL}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --runner pooling \
        --enforce-eager \
        --dtype bfloat16 \
	--data-parallel-size ${DP_SIZE} \
        --max-num-seqs 128 \
        --max-num-batched-tokens 8192 \
	--gpu-memory-utilization 0.01 \
        --max-model-len 512 2>&1 | tee ${LOG} &
        # If default init fails on the pooler task, flip to token_embed:
        # --override-pooler-config '{"task": "token_embed"}'
echo "[rerank-vllm] started PID $!"

health_check_v1
