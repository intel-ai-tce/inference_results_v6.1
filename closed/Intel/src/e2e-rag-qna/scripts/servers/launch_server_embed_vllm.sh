#!/bin/bash
# Launch the retriever (e5-base-v2 by default) as a vLLM OpenAI-compatible
# embedding server exposing /v1/embeddings. Consumed by
# sut/pipeline_stage_embed.py::stage2_embed_worker via aiohttp.

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "${_ROOT}/config.sh" ] && source "${_ROOT}/config.sh"
[ -f "${_ROOT}/config.default.sh" ] && source "${_ROOT}/config.default.sh"
cd "${_ROOT}" || exit 1

PORT=${PORT:-${SERVER_EMBED_VLLM_PORT:-8194}}
EMBED_MODEL="${RETRIEVER_MODEL_PATH}"

LOG="${_ROOT}/log-server-embed-vllm.log"
echo "[embed-vllm] port=${PORT} model=${EMBED_MODEL} -> ${LOG}"

function health_check_v1() {
    echo "[embed-vllm] waiting for server on port ${PORT}..."
    RETRY_COUNT=0
    MAX_RETRIES=100
    while ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -gt $MAX_RETRIES ]; then
            echo "[embed-vllm] server failed to start (see ${LOG})"
            exit 1
        fi
        sleep 5
    done
    echo "[embed-vllm] server ready at http://localhost:${PORT}"
}

export VLLM_CPU_OMP_THREADS_BIND="0-36|43-79|86-112|129-165"
vllm serve "${EMBED_MODEL}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --runner pooling \
        --dtype bfloat16 \
        --max-num-seqs 1024 \
	--data-parallel-size 4 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.1 \
        --max-model-len 512 > ${LOG} 2>&1 &
        # If default pooler init picks the wrong task for this model,
        # flip explicitly:
        # --override-pooler-config '{"task": "embed"}'
echo "[embed-vllm] started PID $!"

health_check_v1
