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

# Accuracy test script for E2E DocGrader workload with MLPerf Loadgen

echo "Time Start: $(date +%s)"

# Raise the open-file limit: concurrent requests to all servers each hold a
# socket -> OSError: Too many open files
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# Load config. Order matters: config.sh first, then the template — both use
# ${VAR:-...} guards, so precedence is env var > config.sh > config.default.sh.
# To override, copy a line from config.default.sh into config.sh, or export it.
[ -f config.sh ] && source config.sh
source config.default.sh

mkdir -p "${RUN_OUTPUT_DIR}/results"
exec > >(tee -a "${RUN_OUTPUT_DIR}/results/run.log") 2>&1

# (main_qna.py prints the full resolved config + endpoints.)
# Section is e2e-rag-qna (the checker's workload name), not the older rag-qna.
# A bare sed on the old key silently no-ops, leaving RUN_MAX_ASYNC_QUERIES
# unapplied; append the key if the section doesn't already carry it.
if grep -q "^e2e-rag-qna.Offline.max_async_queries" user.conf; then
    sed -i "s/^e2e-rag-qna.Offline.max_async_queries = .*/e2e-rag-qna.Offline.max_async_queries = ${RUN_MAX_ASYNC_QUERIES}/" user.conf
else
    echo "e2e-rag-qna.Offline.max_async_queries = ${RUN_MAX_ASYNC_QUERIES}" >> user.conf
fi

# Async pipelined SUT is the default (needs the embed/rerank services from
# launch_servers.sh). Set ASYNC_PIPELINE=0 to use the sequential SUT.
ASYNC_ARGS=""
if [ "${ASYNC_PIPELINE:-1}" = "1" ]; then
    export EMBED_URL=${EMBED_URL:-http://127.0.0.1:${SERVER_EMBED_PORT:-8100}}
    # RERANK_API=score -> vLLM /v1/score server (different port), else the
    # ColBERT /rerank server. Model name is required by the /v1/score payload.
    export RERANK_API=${RERANK_API:-colbert}
    if [ "${RERANK_API}" = "score" ]; then
        export RERANK_URL=${RERANK_URL:-http://127.0.0.1:${SERVER_RERANK_VLLM_PORT:-8193}}
        export RERANKER_MODEL_PATH=${RERANKER_MODEL_PATH:-${INFERENCE_RERANKER_MODEL}}
    else
        export RERANK_URL=${RERANK_URL:-http://127.0.0.1:${SERVER_RERANK_PORT:-8101}}
    fi
    export TRACE=${TRACE:-batch-curve}
    export TRACE_DIR=${TRACE_DIR:-${RUN_OUTPUT_DIR}/results}
    export SERVER_LIMITS=${SERVER_LIMITS:-}
    [ -n "${SERVER_LIMITS}" ] && ASYNC_ARGS="--server_limits ${SERVER_LIMITS}"
    echo "ASYNC PIPELINE: embed=${EMBED_URL} rerank=${RERANK_URL} (api=${RERANK_API}) server_limits='${SERVER_LIMITS}' trace='${TRACE}' trace_dir=${TRACE_DIR}"
else
    # Sequential SUT loads the DB + reranker in-process (async uses the servers).
    ASYNC_ARGS="--sequential --database ${RUN_DATABASE} --reranker_model ${INFERENCE_RERANKER_MODEL}"
    echo "SEQUENTIAL SUT (ASYNC_PIPELINE=0) db=${RUN_DATABASE}"
fi

python3 -u main_qna.py \
    ${ASYNC_ARGS} \
    --dataset_path ${RUN_DATASET} \
    --scenario ${RUN_SCENARIO} \
    --output_dir ${RUN_OUTPUT_DIR} \
    --perf_count ${RUN_PERF_COUNT} \
    --max-iterations ${INFERENCE_MAX_ITERATIONS} \
    --max-sub-queries ${INFERENCE_MAX_SUB_QUERIES} \
    --top_k_retriever ${INFERENCE_TOP_K_RETRIEVER} \
    --top_k_reranking ${INFERENCE_TOP_K_RERANKING} \
    --max_workers ${RUN_MAX_WORKERS} \
    --embedding_model ${EMBEDDING_MODEL} \
    --llm_service_url ${INFERENCE_LLM_URL} \
    --llm_model ${INFERENCE_MODEL} \
    --query_model ${INFERENCE_QUERY_MODEL} \
    --query-service-url ${INFERENCE_QUERY_URL} \
    --sufficiency-service-url ${INFERENCE_SUFFICIENCY_URL} \
    --sufficiency-model ${INFERENCE_SUFFICIENCY_MODEL} \
    --judge_service_url ${INFERENCE_JUDGE_URL} \
    --judge_model ${INFERENCE_JUDGE_MODEL} \
    --accuracy

echo "Time Stop: $(date +%s)"
