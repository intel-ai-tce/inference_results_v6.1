# =============================================================================
# DO NOT DELETE THIS FILE. Every script sources config.default.sh for its
# baseline defaults; without it, runs fall back to hard-coded script defaults or
# fail. To customize, edit config.sh (gitignored) or export env vars — never this.
# =============================================================================
# config.default.sh — GENERIC defaults, committed to the repo. This is the
# starting reference; you normally do NOT edit it.
#
# To override a value, either:
#   (a) copy the single line you want to change into config.sh (gitignored) and
#       edit it there, or
#   (b) export the env var in your shell before running (e.g.
#       `RUN_PERF_COUNT=100 bash scripts/run_qna_accuracy.sh`).
#
# Every value below is guarded with ${VAR:-default}, and scripts source config.sh
# BEFORE this file, so the resolution order per variable is (first wins):
#   1. Exported shell env var        (highest priority)
#   2. Value in config.sh            (device-specific overrides; gitignored)
#   3. Default here in config.default.sh
#
# Keep this file generic (localhost endpoints, relative paths). Put machine
# paths like /data/datasets and /data/models in config.sh, NOT here.
# =============================================================================

# ── Ingestion pipeline (scripts/run_ingestion.sh, run_ingestion_{perf,accuracy}.sh)
INGESTION_DEVICE="${INGESTION_DEVICE:-cpu}"
INGESTION_EMBEDDING_DEVICE="${INGESTION_EMBEDDING_DEVICE:-${INGESTION_DEVICE}}"
INGESTION_NUM_EMBEDDING_DEVICES="${INGESTION_NUM_EMBEDDING_DEVICES:-4}"
INGESTION_CHUNK_LEN="${INGESTION_CHUNK_LEN:-768}"
INGESTION_CHUNK_OVERLAP="${INGESTION_CHUNK_OVERLAP:-32}"
INGESTION_TEXT_BOUNDARY="${INGESTION_TEXT_BOUNDARY:-word}"
INGESTION_VECTOR_INDEX_METHOD="${INGESTION_VECTOR_INDEX_METHOD:-hnsw}"
# Embedding model — the SAME for building the DB and querying it (a mismatch
# makes stored vectors geometrically incompatible with query vectors).
EMBEDDING_MODEL="${EMBEDDING_MODEL:-intfloat_e5-base-v2/e5-base-v2}"
INGESTION_RERANKER_MODEL="${INGESTION_RERANKER_MODEL:-colbert-ir_colbertv2.0/colbertv2.0}"
INGESTION_DOC_DIR="${INGESTION_DOC_DIR:-doc_html}"
# Aliases used by the vLLM-embed / pipelined-ingestion scripts (from the
# vllm-reranker merge). Default to EMBEDDING_MODEL/INGESTION_RERANKER_MODEL.
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:-${EMBEDDING_MODEL}}"
RERANKER_MODEL_PATH="${RERANKER_MODEL_PATH:-${INGESTION_RERANKER_MODEL}}"
INGESTION_PASSAGES_JSON="${INGESTION_PASSAGES_JSON:-passages/doc_html_len768_ov32_word.json}"
INGESTION_DB="${INGESTION_DB:-vector_html_hnsw_len768_ov32_word}"
INGESTION_MAX_WORKERS="${INGESTION_MAX_WORKERS:-4}"
INGESTION_BENCHMARK="${INGESTION_BENCHMARK:-false}"
INGESTION_OUTPUT_DIR="${INGESTION_OUTPUT_DIR:-output_datasetup}"
INGESTION_ACCURACY_OUTPUT_DIR="${INGESTION_ACCURACY_OUTPUT_DIR:-output_datasetup_accuracy}"

# ── Pipelined ingestion (multiprocess parse/embed/index SUT) ──────────────────
# Embedding goes to the vLLM server at INGESTION_EMBED_URL (see
# scripts/servers/launch_server_embed_vllm.sh).
INGESTION_PIPELINED="${INGESTION_PIPELINED:-false}"
INGESTION_EMBED_URL="${INGESTION_EMBED_URL:-http://127.0.0.1:8194}"

# ── Inference pipeline (run_multi_shot.sh, run_single_shot.sh) ────────────────
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cpu}"
INFERENCE_EMBEDDING_DEVICE="${INFERENCE_EMBEDDING_DEVICE:-${INFERENCE_DEVICE}}"
INFERENCE_RERANKER_DEVICE="${INFERENCE_RERANKER_DEVICE:-${INFERENCE_DEVICE}}"
INFERENCE_DB="${INFERENCE_DB:-vector_html_hnsw_len768_ov32_word}"
# top_k_retriever/top_k_reranking are FIXED at 10 by the e2e-rag rules ("All
# submitters must use the following parameters"), matching the reference
# implementation's params.py defaults. Was 15 here, which is non-compliant.
INFERENCE_TOP_K_RETRIEVER="${INFERENCE_TOP_K_RETRIEVER:-10}"
INFERENCE_TOP_K_RERANKING="${INFERENCE_TOP_K_RERANKING:-10}"
# Rules cap max retrieval iterations at 5 and require submitters to set it to 5.
INFERENCE_MAX_ITERATIONS="${INFERENCE_MAX_ITERATIONS:-5}"
INFERENCE_MAX_SUB_QUERIES="${INFERENCE_MAX_SUB_QUERIES:-3}"
INFERENCE_TEMPERATURE="${INFERENCE_TEMPERATURE:-1.0}"
INFERENCE_MAX_RETRIES="${INFERENCE_MAX_RETRIES:-5}"
INFERENCE_N_QUERIES="${INFERENCE_N_QUERIES:-5}"
INFERENCE_NUM_WORKERS="${INFERENCE_NUM_WORKERS:-1}"

# LLM endpoints (vLLM, OpenRouter, etc.)
INFERENCE_LLM_URL="${INFERENCE_LLM_URL:-http://127.0.0.1:8192/v1/chat/completions}"
INFERENCE_MODEL="${INFERENCE_MODEL:-gpt-oss-20b-mxfp4}"

# Query + sufficiency components (larger model, separate server).
INFERENCE_QUERY_URL="${INFERENCE_QUERY_URL:-http://127.0.0.1:8123/v1/chat/completions}"
INFERENCE_QUERY_MODEL="${INFERENCE_QUERY_MODEL:-gpt-oss-120b-mxfp4}"
INFERENCE_SUFFICIENCY_URL="${INFERENCE_SUFFICIENCY_URL:-http://127.0.0.1:8123/v1/chat/completions}"
INFERENCE_SUFFICIENCY_MODEL="${INFERENCE_SUFFICIENCY_MODEL:-gpt-oss-120b-mxfp4}"

# Judge (reference judge = Llama-3.1-8B on a local vLLM, port 8125).
INFERENCE_JUDGE_URL="${INFERENCE_JUDGE_URL:-http://127.0.0.1:8125/v1/chat/completions}"
INFERENCE_JUDGE_MODEL="${INFERENCE_JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# ── MLPerf run wrappers (scripts/run_qna_*.sh, scripts/run_ingestion_*.sh) ────
RUN_DATABASE="${RUN_DATABASE:-data/vector_html_hnsw_len768_ov32_word.db}"
RUN_DATASET="${RUN_DATASET:-data/frames_dataset.tsv}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-output}"
RUN_SCENARIO="${RUN_SCENARIO:-Offline}"
RUN_PERF_COUNT="${RUN_PERF_COUNT:-824}"
RUN_MAX_WORKERS="${RUN_MAX_WORKERS:-10}"
RUN_MAX_ASYNC_QUERIES="${RUN_MAX_ASYNC_QUERIES:-10}"
RUN_PERF_CACHE_FILE="${RUN_PERF_CACHE_FILE:-assets/logs_result.json.gz}"

# Pipeline tracing (async pipeline only). TRACE selects which trace VIEWS to emit
# as a comma-separated list; empty disables tracing entirely (no hooks, no overhead).
# Available views:
#   query-timeline     per-query spans across stages + flow arrows (Perfetto lane view)
#   batch-curve        concurrency/cumulative step curve per resource (span-edge sweep)
#   batch-curve-detail batch-curve + live client-side inflight and sampled vLLM
#                      running/waiting tracks (extra per-request + /metrics-poll overhead)
#   all                every registered view
# Examples: TRACE="batch-curve"  |  TRACE="query-timeline,batch-curve"  |  TRACE=""
TRACE="${TRACE:-batch-curve}"
TRACE_DIR="${TRACE_DIR:-${RUN_OUTPUT_DIR}/results}"

# SERVER_LIMITS (async pipeline): per-server host-side concurrency caps.
# Servers: LLM-120B, LLM-20B, embedder, reranker. Empty = unlimited.
SERVER_LIMITS="${SERVER_LIMITS:-LLM-120B=1024,LLM-20B=256,reranker=128}"

# ── vLLM server launchers (scripts/servers/launch_server_*.sh) ───────────────
# Model paths are device-specific (set absolute /data paths in config.sh).
# Core-binding / mem defaults are the measured sweet spots for this host.
MODEL_20B_PATH="${MODEL_20B_PATH:-/data/gpt-oss-20b-mxfp4}"
SERVER_20B_PORT="${SERVER_20B_PORT:-8192}"
SERVER_20B_TP="${SERVER_20B_TP:-4}"
SERVER_20B_GPU_MEM_UTIL="${SERVER_20B_GPU_MEM_UTIL:-0.3}"
SERVER_20B_CORES_PER_NODE="${SERVER_20B_CORES_PER_NODE:-43}"
SERVER_20B_MAX_MODEL_LEN="${SERVER_20B_MAX_MODEL_LEN:-8192}"
SERVER_20B_MAX_NUM_SEQS="${SERVER_20B_MAX_NUM_SEQS:-1024}"

MODEL_120B_PATH="${MODEL_120B_PATH:-/data/gpt-oss-120b-mxfp4}"
SERVER_120B_PORT="${SERVER_120B_PORT:-8123}"
SERVER_120B_GPU_MEM_UTIL="${SERVER_120B_GPU_MEM_UTIL:-0.90}"
SERVER_120B_MAX_NUM_SEQS="${SERVER_120B_MAX_NUM_SEQS:-1024}"
SERVER_120B_MAX_CUDAGRAPH_CAPTURE_SIZE="${SERVER_120B_MAX_CUDAGRAPH_CAPTURE_SIZE:-784}"
SERVER_120B_HOST_CORES="${SERVER_120B_HOST_CORES:-40-42,83}"
SERVER_120B_MAX_MODEL_LEN="${SERVER_120B_MAX_MODEL_LEN:-131072}"

MODEL_JUDGE_PATH="${MODEL_JUDGE_PATH:-/data/models/Llama-3.1-8B-Instruct}"
SERVER_JUDGE_PORT="${SERVER_JUDGE_PORT:-8125}"
SERVER_JUDGE_GPU_MEM_UTIL="${SERVER_JUDGE_GPU_MEM_UTIL:-0.12}"
SERVER_JUDGE_MAX_MODEL_LEN="${SERVER_JUDGE_MAX_MODEL_LEN:-16384}"
SERVER_JUDGE_NODES="${SERVER_JUDGE_NODES:-0 1}"

# XPU variant (launch_server_8b_judge_xpu.sh). TP fixed at 1, pinned via ZE_AFFINITY_MASK.
SERVER_JUDGE_XPU_PORT="${SERVER_JUDGE_XPU_PORT:-8125}"
SERVER_JUDGE_XPU_GPU_MEM_UTIL="${SERVER_JUDGE_XPU_GPU_MEM_UTIL:-0.70}"
SERVER_JUDGE_XPU_MAX_MODEL_LEN="${SERVER_JUDGE_XPU_MAX_MODEL_LEN:-16384}"
SERVER_JUDGE_XPU_DEVICE="${SERVER_JUDGE_XPU_DEVICE:-0}"

# Retrieval services for the async pipeline (each: one http port).
# Embed = single batching process; rerank = N-worker pool w/ internal LB.
SERVER_EMBED_PORT="${SERVER_EMBED_PORT:-8100}"
SERVER_EMBED_CORES="${SERVER_EMBED_CORES:-84-85}"
SERVER_RERANK_PORT="${SERVER_RERANK_PORT:-8101}"
SERVER_RERANK_NUM_WORKERS="${SERVER_RERANK_NUM_WORKERS:-2}"
SERVER_RERANK_WORKER_CORES="${SERVER_RERANK_WORKER_CORES:-126-128;169-171}"

# Embedding vLLM server (launch_server_embed_vllm.sh); reads RETRIEVER_MODEL_PATH.
SERVER_EMBED_VLLM_PORT="${SERVER_EMBED_VLLM_PORT:-8194}"

# ── Oracle evaluation (run_oracle.sh) ─────────────────────────────────────────
INFERENCE_ORACLE_BATCH_SIZE="${INFERENCE_ORACLE_BATCH_SIZE:-4}"
INFERENCE_ORACLE_TIMEOUT="${INFERENCE_ORACLE_TIMEOUT:-2400}"
# INFERENCE_ORACLE_ENABLE_THINKING=1   # uncomment to pass --enable-thinking
INFERENCE_ORACLE_DATASET="${INFERENCE_ORACLE_DATASET:-/data/dataset/frames-benchmark-dataset/frames_dataset.tsv}"
INFERENCE_ORACLE_WIKI_DIR="${INFERENCE_ORACLE_WIKI_DIR:-wiki_articles}"

# ── Secrets (export from your shell instead of committing) ───────────────────
#source secrets
#   export OPENROUTER_API_KEY=sk-or-...


# Override GPU index allocator (e.g. when auto-detect picks the wrong devices).
# Comma-separated 0-based indices; subset of available CUDA/XPU devices.
# Per-component:
# INFERENCE_EMBEDDING_GPU_DEVICES="0,1"   # one entry per --num_embedding_devices worker
# INFERENCE_RERANKER_GPU_DEVICES="2"

# Per-component NUMA / OMP placement (applied inside the per-process worker).
# Pins the worker's CPU set + memory to the given node; OMP threads default
# to the node's physical core count if not set.
# INFERENCE_RERANKER_NUMA_NODE=0
# INFERENCE_RERANKER_OMP_NUM_THREADS=21
# INFERENCE_EMBEDDING_NUMA_NODES="0,0,1,1"     # one node per embedding worker
# INFERENCE_EMBEDDING_OMP_NUM_THREADS=21       # cap per worker; default = even split

# ── Python-side env vars (CPU_*) ──────────────────────────────────────────────
# These only fire when a model lands on CPU (gated in apply_cpu_threading_env).
# Uncomment to override Python defaults:
# CPU_DISABLE_NUMA=1
# CPU_NUMA_NODE=0
# CPU_NUMA_CORES="43-85"
# CPU_OMP_NUM_THREADS=43
