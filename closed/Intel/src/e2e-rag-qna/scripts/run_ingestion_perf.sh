#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

echo "Time Start: $(date +%s)"

[ -f config.sh ] && source config.sh
source config.default.sh

if [ ! -d "${INGESTION_DOC_DIR}" ]; then
    echo "ERROR: Documents directory not found: ${INGESTION_DOC_DIR}"
    exit 1
fi

HTML_COUNT=$(find "${INGESTION_DOC_DIR}" -maxdepth 1 -name "*.html" | wc -l)
echo "  HTML files found: ${HTML_COUNT}"
if [ ${HTML_COUNT} -eq 0 ]; then
    echo "ERROR: No HTML files found in ${INGESTION_DOC_DIR}"
    exit 1
fi

# Loadgen dispatches all HTML files at once (one query per document).
if [ -f "user.conf" ]; then
    for key in max_async_queries min_query_count; do
        if grep -q "e2e-rag-db.Offline.${key}" user.conf; then
            sed -i "s/^e2e-rag-db.Offline.${key} = .*/e2e-rag-db.Offline.${key} = ${HTML_COUNT}/" user.conf
        else
            echo "e2e-rag-db.Offline.${key} = ${HTML_COUNT}" >> user.conf
        fi
    done
fi

BENCHMARK_ARG=""
[ "${INGESTION_BENCHMARK}" = "true" ] && BENCHMARK_ARG="--benchmark"

PIPELINE_ARGS=""
if [ "${INGESTION_PIPELINED}" = "true" ]; then
    PIPELINE_ARGS="--pipelined --embed_url ${INGESTION_EMBED_URL}"
    echo "  Pipelined: embed_url=${INGESTION_EMBED_URL}"
    if ! curl -sf --max-time 3 "${INGESTION_EMBED_URL}/v1/models" > /dev/null; then
        echo "[FATAL] embed server not reachable at ${INGESTION_EMBED_URL}"
        echo "        Start it with: bash scripts/servers/launch_server_embed_vllm.sh"
        exit 2
    fi
fi

# main_ingestion.py prints the full resolved config.
python3 -u main_ingestion.py \
    --documents_dir ${INGESTION_DOC_DIR} \
    --database ${INGESTION_DB} \
    --scenario ${RUN_SCENARIO} \
    --log_dir ${INGESTION_OUTPUT_DIR} \
    --output_dir ${INGESTION_OUTPUT_DIR} \
    --chunk_size ${INGESTION_CHUNK_LEN} \
    --chunk_overlap ${INGESTION_CHUNK_OVERLAP} \
    --text_boundary ${INGESTION_TEXT_BOUNDARY} \
    --embedding_model ${EMBEDDING_MODEL} \
    --reranker_model ${INGESTION_RERANKER_MODEL} \
    --device ${INGESTION_DEVICE} \
    --num_embedding_devices ${INGESTION_NUM_EMBEDDING_DEVICES} \
    --vector_index_method ${INGESTION_VECTOR_INDEX_METHOD} \
    --max_workers ${INGESTION_MAX_WORKERS} \
    ${PIPELINE_ARGS} \
    ${BENCHMARK_ARG} 2<&1 | tee ingestion_output.log &

EXIT_CODE=$?
echo "Time Stop: $(date +%s)"
exit ${EXIT_CODE}
