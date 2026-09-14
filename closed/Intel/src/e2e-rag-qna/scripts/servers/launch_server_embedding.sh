#!/bin/bash
# Launch the e5 embedding + FAISS search service on ONE http port. A single
# process with a gather-window batcher (MAX_BATCH) coalesces concurrent requests
# into one fused forward pass

# Load config (repo root is two levels up); safe if absent.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "${_ROOT}/config.sh" ] && source "${_ROOT}/config.sh"
[ -f "${_ROOT}/config.default.sh" ] && source "${_ROOT}/config.default.sh"
cd "${_ROOT}" || exit 1

PORT=${PORT:-${EMBED_PORT:-8100}}
# Same DB knob as the run scripts: RUN_DATABASE (full path). One source of truth.
DB=${RUN_DATABASE:-data/vector_html_hnsw_len768_ov32_word.db}
EMB=${EMBEDDING_MODEL:-${SERVER_EMBED_MODEL:-intfloat_e5-base-v2/e5-base-v2}}
EMBED_CORES=${EMBED_CORES:-${SERVER_EMBED_CORES:-84-85}}

if [ ! -e "${DB}" ]; then
    echo "ERROR: embed DB not found: ${DB} (set RUN_DATABASE)"
    exit 1
fi

LOG="${_ROOT}/log-server-embedding.log"
echo "[embed] port=${PORT} cores=${EMBED_CORES} db=${DB} -> ${LOG}"
nohup taskset -c "${EMBED_CORES}" python3 -u -m servers.embed_search_server \
    --db "${DB}" \
    --embedding-model "${EMB}" \
    --host 0.0.0.0 \
    --port "${PORT}" > "${LOG}" 2>&1 &
echo "[embed] started PID $! (unattended)"
