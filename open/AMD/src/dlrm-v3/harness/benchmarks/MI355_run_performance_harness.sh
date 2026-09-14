#!/usr/bin/env bash
# ROCm MI355x8 — NVIDIA closed harness (MPI + ZMQ + LoadGen rank)
# Phase 2a: GenerativeRecommenderBackend (open HSTU path) per worker, not NVE/TRT.
#
# Usage:
#   bash MI355_run_performance_harness.sh SCENARIO=Server MODE=performance OUTPUT_DIR=/tmp/dlrm_nvidia_out
#
# Requires preprocessed dataset (Phase 1):
#   python3 tools/preprocess_data.py --output-dir /data/dlrmv3_preprocessed ...

set -euo pipefail

SCENARIO=""
MODE=""
OUTPUT_DIR=""

for arg in "$@"; do
    case $arg in
        SCENARIO=*) SCENARIO="${arg#*=}" ;;
        MODE=*) MODE="${arg#*=}" ;;
        OUTPUT_DIR=*) OUTPUT_DIR="${arg#*=}" ;;
        *) echo "Unknown: $arg"; exit 1 ;;
    esac
done

: "${SCENARIO:?SCENARIO=Server|Offline}"
: "${MODE:?MODE=performance|accuracy}"
: "${OUTPUT_DIR:?OUTPUT_DIR=path}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_PATH="${DATASET_PATH:-/data/dlrmv3_preprocessed}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/model/dlrm-v3-checkpoint/}"
BATCH_SIZE="${BATCH_SIZE:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-15}"
DATASET_PERCENTAGE="${DATASET_PERCENTAGE:-1}"
# Phase 2a: each worker loads a full HSTUModelFamily — default 1 worker to avoid OOM on smoke.
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_TOTAL_PROCESSES=$((NUM_WORKERS + 1))
# Phase 3 multi-worker scale-out: one ZMQ shard per worker so the
# LoadGen-side round-robin counter (mpi_utils.send_to_worker) gives a
# deterministic 1/W split. With NUM_SHARDS=1 both workers share a
# single PUSH<->PULL pair and ZMQ delivers the entire pre-warmup
# backlog to whichever worker happened to finish warmup a few ms
# earlier (Phase 3 multi-worker scale-out repro 2026-05-26: 2W run
# saw all 1961 LoadGen batches arrive at rank 1, rank 0 stayed idle,
# completed QPS capped at 396 instead of ~650).
NUM_SHARDS="${NUM_SHARDS:-${NUM_WORKERS}}"
# gpus_per_node only affects shard_id = local_rank // gpus_per_node;
# we want each worker to be its own shard, so set gpus_per_node=1.
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
LOADGEN_HOSTNAME="${LOADGEN_HOSTNAME:-localhost}"
COMMUNICATOR_TYPE="${COMMUNICATOR_TYPE:-zmq}"
USER_CONF="${USER_CONF:-user_mi355x8.conf}"

export DLRM_ROCM_GR_BACKEND=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export DLRM_SAFE_FBGEMM_CUMSUM=1
export DLRM_SKIP_AUTOTUNE=1
export DLRM_SKIP_DENSE_BATCH_CLONE=1
export PYTHONPATH="/work/mlcommons-inference/recommendation/dlrm_v3:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_nvidia}"
export DLRM_ZMQ_LATENCY_REQUESTS="${DLRM_ZMQ_LATENCY_REQUESTS:-0}"

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: preprocessed dataset missing: $DATASET_PATH" >&2
    echo "Run tools/preprocess_data.py first (see NVIDIA_PORT_INVENTORY.md)." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHON_ARGS="$SCRIPT_DIR/run_benchmark.py \
    --dataset-path $DATASET_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    --scenario $SCENARIO \
    --mode $MODE \
    --batch-size $BATCH_SIZE \
    --warmup-steps $WARMUP_STEPS \
    --dataset-percentage $DATASET_PERCENTAGE \
    --user-conf $SCRIPT_DIR/$USER_CONF \
    --output-dir $OUTPUT_DIR \
    --communicator-type $COMMUNICATOR_TYPE \
    --loadgen-hostname $LOADGEN_HOSTNAME \
    --num-shards $NUM_SHARDS \
    --gpus-per-node $GPUS_PER_NODE"

# Plan 12 §3.3 — propagate ``--use-mpi-lookup`` to run_benchmark.py when
# the launcher sets DLRM_USE_MPI_LOOKUP=1. On the ROCm path run_benchmark.py
# re-exports DLRM_USE_MPI_LOOKUP=1 so sparse_routing._use_mpi_lookup_env()
# picks it up; the env-only path also works without the CLI flag, but the
# flag makes the harness logs explicit about which transport is requested.
if [[ "${DLRM_USE_MPI_LOOKUP:-0}" == "1" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --use-mpi-lookup"
fi

echo "=== MI355 NVIDIA harness ROCm (GR backend) ==="
echo "  workers=$NUM_WORKERS processes=$NUM_TOTAL_PROCESSES batch=$BATCH_SIZE"
echo "  dataset=$DATASET_PATH"
echo "  checkpoint=$CHECKPOINT_PATH"
echo "  out=$OUTPUT_DIR"

export PYTHON_ARGS
# Profiling hook (env-gated; no behavior change when ROCPROF!=1): wrap ONE worker rank
# with rocprofv3 kernel-trace, others run plain python. Rank 0 is the LoadGen rank, so
# ROCPROF_RANK defaults to 1 (first GPU worker). Output (CSV) lands under ROCPROF_OUT.
ROCPROF="${ROCPROF:-0}"
ROCPROF_RANK="${ROCPROF_RANK:-1}"
ROCPROF_OUT="${ROCPROF_OUT:-$OUTPUT_DIR/rocprof}"
ROCPROF_MODE="${ROCPROF_MODE:---kernel-trace}"
export ROCPROF ROCPROF_RANK ROCPROF_OUT ROCPROF_MODE
if [ "${BIND_NUMA:-0}" = "1" ]; then
  WRAPPER="$(cd "$(dirname "$0")" && pwd)/numa_wrapper.sh"
  echo "  [bind] NUMA pinning enabled via $WRAPPER"
  mpirun -n "$NUM_TOTAL_PROCESSES" --bind-to none bash -lc "exec bash $WRAPPER"
elif [ "$ROCPROF" = "1" ]; then
  mkdir -p "$ROCPROF_OUT"
  echo "  [rocprof] $ROCPROF_MODE on rank $ROCPROF_RANK -> $ROCPROF_OUT"
  mpirun -n "$NUM_TOTAL_PROCESSES" --bind-to none bash -lc '
    if [ "${OMPI_COMM_WORLD_RANK:-x}" = "$ROCPROF_RANK" ]; then
      exec rocprofv3 $ROCPROF_MODE --output-format csv -d "$ROCPROF_OUT" -o "rank${OMPI_COMM_WORLD_RANK}" -- python3 -u $PYTHON_ARGS
    else
      exec python3 -u $PYTHON_ARGS
    fi'
else
  mpirun -n "$NUM_TOTAL_PROCESSES" --bind-to none bash -lc 'exec python3 -u $PYTHON_ARGS'
fi
