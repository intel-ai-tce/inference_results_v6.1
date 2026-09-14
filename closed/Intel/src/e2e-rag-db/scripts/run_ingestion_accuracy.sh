#!/bin/bash

# Run from repo root (script lives in scripts/); resolves user.conf, main_*.py, outputs.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Accuracy test script for E2E-RAG-Datasetup workload with MLPerf Loadgen

echo "Time Start: $(date +%s)"

# Load config. Order matters: config.sh first, then the template — both use
# ${VAR:-...} guards, so precedence is env var > config.sh > config.default.sh.
# To override, copy a line from config.default.sh into config.sh, or export it.
[ -f config.sh ] && source config.sh
source config.default.sh

OUTPUT_DIR="${INGESTION_ACCURACY_OUTPUT_DIR}"

# Reference DB manifest for cross-system behavioral-equivalence verification
# (passage count, index params, probe-query top-K retrieval overlap).
# Set to "" or "none" to skip the manifest check. Note: ${VAR:-default} treats
# an empty value the same as unset, so an explicit "none" sentinel is the
# reliable way to skip from a parent script that exports MANIFEST="".
export MANIFEST=${MANIFEST-assets/db_manifest_intel_xpu.json.gz}
export RETRIEVAL_THRESHOLD=${RETRIEVAL_THRESHOLD:-0.95}

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
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --chunk_size ${INGESTION_CHUNK_LEN} \
    --chunk_overlap ${INGESTION_CHUNK_OVERLAP} \
    --text_boundary ${INGESTION_TEXT_BOUNDARY} \
    --embedding_model ${EMBEDDING_MODEL} \
    --reranker_model ${INGESTION_RERANKER_MODEL} \
    --device ${INGESTION_DEVICE} \
    --num_embedding_devices ${INGESTION_NUM_EMBEDDING_DEVICES} \
    --vector_index_method ${INGESTION_VECTOR_INDEX_METHOD} \
    --max_workers ${INGESTION_MAX_WORKERS} \
    --accuracy \
    ${PIPELINE_ARGS} \
    ${BENCHMARK_ARG}

EXIT_CODE=$?
echo "Time Stop: $(date +%s)"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "=== Running Accuracy Evaluation ==="
    # Optional manifest argument (skip the check if MANIFEST is empty or "none").
    # The manifest gate produces the headline retrieval-accuracy metric that
    # accuracy.txt reports, which the submission checker validates.
    MANIFEST_ARG=""
    if [ -n "${MANIFEST}" ] && [ "${MANIFEST,,}" != "none" ]; then
        MANIFEST_ARG="--manifest ${MANIFEST} --retrieval_threshold ${RETRIEVAL_THRESHOLD}"
    fi

    python3 -u -m evaluation.accuracy_eval_ingestion \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --database ${INGESTION_DB}.db \
        --embedding_model ${EMBEDDING_MODEL} \
        ${MANIFEST_ARG}
    EVAL_EXIT_CODE=$?
    [ ${EVAL_EXIT_CODE} -ne 0 ] && { echo "ERROR: Accuracy evaluation failed"; exit ${EVAL_EXIT_CODE}; }
fi

exit ${EXIT_CODE}
