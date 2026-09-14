#!/bin/bash
# Launch the ColBERT rerank service: ONE http port, an internal pull-based load
# balancer across N pinned worker processes (rerank does not batch, so throughput
# scales by worker count). See servers/rerank_server.py.

# Load config (repo root is two levels up); safe if absent.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "${_ROOT}/config.sh" ] && source "${_ROOT}/config.sh"
[ -f "${_ROOT}/config.default.sh" ] && source "${_ROOT}/config.default.sh"
cd "${_ROOT}" || exit 1

PORT=${PORT:-${RERANK_PORT:-8101}}
RERANKER_MODEL=${RERANKER_MODEL:-${SERVER_RERANK_MODEL:-/data/models/colbert-ir_colbertv2.0/colbertv2.0}}
# N worker processes behind the one port, and their per-worker core pins.
RERANK_NUM_WORKERS=${RERANK_NUM_WORKERS:-${SERVER_RERANK_NUM_WORKERS:-2}}
RERANK_WORKER_CORES=${RERANK_WORKER_CORES:-${SERVER_RERANK_WORKER_CORES:-"126-128;169-171"}}

LOG="${_ROOT}/log-server-rerank.log"
echo "[rerank] port=${PORT} workers=${RERANK_NUM_WORKERS} cores=${RERANK_WORKER_CORES} -> ${LOG}"
export RERANK_NUM_WORKERS RERANK_WORKER_CORES
nohup python3 -u -m servers.rerank_server \
    --reranker-model "${RERANKER_MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" > "${LOG}" 2>&1 &
echo "[rerank] started PID $! (unattended)"
