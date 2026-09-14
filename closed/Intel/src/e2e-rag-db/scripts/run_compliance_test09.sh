#!/bin/bash
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

# TEST09 Compliance Test Runner for E2E DocGrader Workload
# Automates: setup -> run -> verify -> cleanup workflow

set -e  # Exit on error

echo "=============================================================================="
echo "TEST09 Compliance Test for E2E-RAG Workload"
echo "=============================================================================="
echo "Start time: $(date)"
echo ""

# Configuration
# Script lives in scripts/; SCRIPT_DIR is repo root so all paths below resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}" || exit 1
# The TEST09 config comes from the main inference repo's compliance tree, vendored
# under third_party/. Upstream renamed e2e-rag/ -> e2e-rag-qna/, so accept either;
# override COMPLIANCE_DIR to point somewhere else.
TEST09_TREE="${SCRIPT_DIR}/third_party/mlperf-inference/compliance/TEST09"
if [ -z "${COMPLIANCE_DIR}" ]; then
    if [ -d "${TEST09_TREE}/e2e-rag-qna" ]; then
        COMPLIANCE_DIR="${TEST09_TREE}/e2e-rag-qna"
    else
        COMPLIANCE_DIR="${TEST09_TREE}/e2e-rag"
    fi
fi
AUDIT_CONFIG="${COMPLIANCE_DIR}/audit.config"
WORKING_AUDIT_CONFIG="${SCRIPT_DIR}/audit.config"
TEST09_VERIFICATION="${SCRIPT_DIR}/third_party/mlperf-inference/compliance/TEST09/run_verification.py"

# Host config, same precedence as the other run scripts: env > config.sh >
# config.default.sh. Without this the defaults below (relative dataset/DB paths,
# bare HF model ids) don't resolve on a real host, so every value had to be
# passed on the command line.
[ -f config.sh ] && source config.sh
source config.default.sh

# Directories
export WORKSPACE_DIR=${WORKSPACE_DIR:-"${SCRIPT_DIR}"}
export DATA_DIR=${DATA_DIR:-"frames-benchmark-dataset"}
# Prefer the config's resolved dataset/DB (RUN_*) over the relative defaults.
export DATASET_PATH="${DATASET_PATH:-${RUN_DATASET:-${DATA_DIR}/frames_dataset.tsv}}"
export DATABASE="${DATABASE:-${RUN_DATABASE:-vector_html_hnsw_len768_ov32_word.db}}"
# main_qna.py has no --log_dir: it writes the loadgen logs (mlperf_log_*.json)
# straight into --output_dir and puts results.json etc. in <output_dir>/results.
# So RUN_LOGS -- where Part III looks for mlperf_log_accuracy.json -- IS
# OUTPUT_DIR. (Previously these were two directories and --log_dir was passed,
# which argparse rejected outright.)
export OUTPUT_DIR=${WORKSPACE_DIR}/output_test09
export RUN_LOGS=${OUTPUT_DIR}
export SUBMISSION_DIR=${WORKSPACE_DIR}/submission/compliance/e2e-rag-qna/Offline
export SCENARIO="${SCENARIO:-Offline}"

# Performance testing - full dataset for compliance.
# Overridable (e.g. for a smoke run); a valid TEST09 submission needs 824.
export PERF_COUNT=${PERF_COUNT:-824}

# Threading configuration
export MAX_ASYNC_QUERIES=${MAX_ASYNC_QUERIES:-10}
export MAX_WORKERS=${MAX_WORKERS:-10}

# Multi-shot retrieval parameters
export MAX_ITERATIONS=${MAX_ITERATIONS:-5}
export MAX_SUB_QUERIES=${MAX_SUB_QUERIES:-3}
# Rules fix both at 10; defer to the config so compliance and the qna runs
# can't silently diverge.
export TOP_K_RETRIEVER=${TOP_K_RETRIEVER:-${INFERENCE_TOP_K_RETRIEVER:-10}}
export TOP_K_RERANKING=${TOP_K_RERANKING:-${INFERENCE_TOP_K_RERANKING:-10}}

# Model paths. Config values (EMBEDDING_MODEL / INFERENCE_RERANKER_MODEL) win:
# the fallbacks are bare relative ids that only resolve in the reference tree.
export EMBEDDING_MODEL=${EMBEDDING_MODEL:-intfloat_e5-base-v2/e5-base-v2}
export RERANKER_MODEL=${RERANKER_MODEL:-${INFERENCE_RERANKER_MODEL:-colbert-ir_colbertv2.0/colbertv2.0}}

# LLM service configuration. The servers register served-model-name as the FULL
# on-disk path, so the bare HF ids below give HTTP 404 -- prefer the config's
# INFERENCE_* values, which carry this host's paths.
export LLM_SERVICE_URL=${LLM_SERVICE_URL:-${INFERENCE_LLM_URL:-http://127.0.0.1:8192/v1/chat/completions}}
export LLM_MODEL=${LLM_MODEL:-${INFERENCE_MODEL:-gpt-oss-20b-mxfp4}}
export QUERY_SERVICE_URL=${QUERY_SERVICE_URL:-${INFERENCE_QUERY_URL:-http://127.0.0.1:8123/v1/chat/completions}}
export QUERY_MODEL=${QUERY_MODEL:-${INFERENCE_QUERY_MODEL:-gpt-oss-120b-mxfp4}}
export SUFFICIENCY_SERVICE_URL=${SUFFICIENCY_SERVICE_URL:-${INFERENCE_SUFFICIENCY_URL:-http://127.0.0.1:8123/v1/chat/completions}}
export SUFFICIENCY_MODEL=${SUFFICIENCY_MODEL:-${INFERENCE_SUFFICIENCY_MODEL:-gpt-oss-120b-mxfp4}}
# Reference judge is Llama-3.1-8B on its own vLLM (see config.default.sh), not the 20B.
export JUDGE_SERVICE_URL=${JUDGE_SERVICE_URL:-${INFERENCE_JUDGE_URL:-http://127.0.0.1:8125/v1/chat/completions}}
export JUDGE_MODEL=${JUDGE_MODEL:-${INFERENCE_JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}}

# Async pipeline endpoints + reranker backend, mirroring run_qna_perf.sh so the
# compliance run uses the same rerank path as the perf/accuracy runs.
export RERANK_API=${RERANK_API:-colbert}
if [ "${RERANK_API}" = "score" ]; then
    export RERANK_URL=${RERANK_URL:-http://127.0.0.1:${SERVER_RERANK_VLLM_PORT:-8193}}
    export RERANKER_MODEL_PATH=${RERANKER_MODEL_PATH:-${RERANKER_MODEL}}
else
    export RERANK_URL=${RERANK_URL:-http://127.0.0.1:${SERVER_RERANK_PORT:-8101}}
fi
export EMBED_URL=${EMBED_URL:-http://127.0.0.1:${SERVER_EMBED_PORT:-8100}}
echo "  RERANK: ${RERANK_URL} (api=${RERANK_API})"
echo "  EMBED:  ${EMBED_URL}"

# Performance cache file (optional - for faster testing)
export PERF_CACHE_FILE=${PERF_CACHE_FILE:-""}

echo "Configuration:"
echo "  DATASET_PATH: ${DATASET_PATH}"
echo "  DATABASE: ${DATABASE}"
echo "  SCENARIO: ${SCENARIO}"
echo "  PERF_COUNT: ${PERF_COUNT}"
echo "  RUN_LOGS: ${RUN_LOGS}"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  SUBMISSION_DIR: ${SUBMISSION_DIR}"
echo "  MAX_ASYNC_QUERIES: ${MAX_ASYNC_QUERIES}"
echo "  MAX_WORKERS: ${MAX_WORKERS}"
echo ""

# ============================================================================
# Part I: Setup
# ============================================================================
echo "=============================================================================="
echo "PART I: Setup"
echo "=============================================================================="

# Verify audit.config exists
if [ ! -f "${AUDIT_CONFIG}" ]; then
    echo "ERROR: audit.config not found at ${AUDIT_CONFIG}"
    echo "Please ensure compliance configuration is set up."
    exit 1
fi

# Verify verification script exists
if [ ! -f "${TEST09_VERIFICATION}" ]; then
    echo "ERROR: run_verification.py not found at ${TEST09_VERIFICATION}"
    exit 1
fi

# Create directories
mkdir -p "${RUN_LOGS}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SUBMISSION_DIR}"

# Copy audit.config to working directory
echo "Copying audit.config to working directory..."
cp "${AUDIT_CONFIG}" "${WORKING_AUDIT_CONFIG}"
# Keep the audit.config's min_query_count in sync with PERF_COUNT. Otherwise
# loadgen honors the config's min_query_count (824) and loops back up to it even
# when PERF_COUNT is smaller (e.g. a smoke run). A real submission uses 824.
sed -i "s/^\*\.\*\.min_query_count = .*/*.*.min_query_count = ${PERF_COUNT}/" "${WORKING_AUDIT_CONFIG}"
echo "✓ audit.config copied to ${WORKING_AUDIT_CONFIG} (min_query_count=${PERF_COUNT})"
echo ""

# ============================================================================
# Part II: Run Performance Test with Compliance Logging
# ============================================================================
echo "=============================================================================="
echo "PART II: Run Performance Test"
echo "=============================================================================="
echo "Running MLPerf LoadGen with TEST09 compliance logging..."
echo ""

# Build perf cache argument if file exists
PERF_CACHE_ARG=""
if [ -n "${PERF_CACHE_FILE}" ] && [ -f "${PERF_CACHE_FILE}" ]; then
    PERF_CACHE_ARG="--perf-test-mode ${PERF_CACHE_FILE}"
    echo "Using cached LLM responses from: ${PERF_CACHE_FILE}"
fi

# Run loadgen performance test
# main_qna.py passes --audit_conf explicitly to StartTestWithLogSettings, so we
# must point it at the copied audit.config (named audit.config, whereas the
# default arg is audit.conf). Without this, loadgen never applies the TEST09
# accuracy_log_sampling_target and mlperf_log_accuracy.json comes out empty.
python3 -u main_qna.py \
    --dataset_path ${DATASET_PATH} \
    --database ${DATABASE} \
    --scenario ${SCENARIO} \
    --audit_conf ${WORKING_AUDIT_CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --perf_count ${PERF_COUNT} \
    --max-iterations ${MAX_ITERATIONS} \
    --max-sub-queries ${MAX_SUB_QUERIES} \
    --top_k_retriever ${TOP_K_RETRIEVER} \
    --top_k_reranking ${TOP_K_RERANKING} \
    --max_workers ${MAX_WORKERS} \
    --embedding_model ${EMBEDDING_MODEL} \
    --reranker_model ${RERANKER_MODEL} \
    --llm_service_url ${LLM_SERVICE_URL} \
    --llm_model ${LLM_MODEL} \
    --query_service_url ${QUERY_SERVICE_URL} \
    --query_model ${QUERY_MODEL} \
    --sufficiency-service-url ${SUFFICIENCY_SERVICE_URL} \
    --sufficiency-model ${SUFFICIENCY_MODEL} \
    --judge_service_url ${JUDGE_SERVICE_URL} \
    --judge_model ${JUDGE_MODEL} \
    ${PERF_CACHE_ARG}

TEST_EXIT_CODE=$?

if [ ${TEST_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Performance test failed with exit code ${TEST_EXIT_CODE}"
    echo "Cleaning up audit.config..."
    rm -f "${WORKING_AUDIT_CONFIG}"
    exit ${TEST_EXIT_CODE}
fi

echo ""
echo "✓ Performance test completed successfully"
echo ""

# ============================================================================
# Part III: Verify Compliance
# ============================================================================
echo "=============================================================================="
echo "PART III: Verify Compliance"
echo "=============================================================================="

# Check if accuracy log exists
if [ ! -f "${RUN_LOGS}/mlperf_log_accuracy.json" ]; then
    echo "ERROR: mlperf_log_accuracy.json not found in ${RUN_LOGS}"
    echo "Compliance verification requires accuracy log."
    rm -f "${WORKING_AUDIT_CONFIG}"
    exit 1
fi

echo "Running TEST09 verification..."
echo ""

# Verify against the SAME config the run used (WORKING_AUDIT_CONFIG), not the
# pristine source: it is the one whose min_query_count was synced to PERF_COUNT.
# run_verification.py only reads test09_{min,max}_output_tokens, which the sed
# above does not touch, so the two agree today — but keeping one source of truth
# means a future edit to the working copy can't silently diverge from the check.
python3 "${TEST09_VERIFICATION}" \
    -c "${RUN_LOGS}" \
    -o "${SUBMISSION_DIR}/.." \
    --audit-config "${WORKING_AUDIT_CONFIG}"

VERIFY_EXIT_CODE=$?

echo ""
if [ ${VERIFY_EXIT_CODE} -eq 0 ]; then
    echo "✓ TEST09 verification PASSED"
else
    echo "✗ TEST09 verification FAILED"
fi
echo ""

# ============================================================================
# Part IV: Cleanup
# ============================================================================
echo "=============================================================================="
echo "PART IV: Cleanup"
echo "=============================================================================="

echo "Removing audit.config from working directory..."
rm -f "${WORKING_AUDIT_CONFIG}"
echo "✓ audit.config removed"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=============================================================================="
echo "TEST09 Compliance Test Summary"
echo "=============================================================================="
echo "End time: $(date)"
echo ""
echo "Logs saved to:"
echo "  Run logs:        ${RUN_LOGS}"
echo "  Output:          ${OUTPUT_DIR}"
echo "  Submission:      ${SUBMISSION_DIR}"
echo ""
echo "Submission artifacts (to be uploaded):"
echo "  ${SUBMISSION_DIR}/verify_output_len.txt"
echo "  ${SUBMISSION_DIR}/accuracy/mlperf_log_accuracy.json"
echo "  ${SUBMISSION_DIR}/performance/run_1/mlperf_log_summary.txt"
echo "  ${SUBMISSION_DIR}/performance/run_1/mlperf_log_detail.txt"
echo ""

if [ ${VERIFY_EXIT_CODE} -eq 0 ]; then
    echo "Status: ✓ COMPLIANCE TEST PASSED"
    echo "=============================================================================="
    exit 0
else
    echo "Status: ✗ COMPLIANCE TEST FAILED"
    echo "=============================================================================="
    exit 1
fi
