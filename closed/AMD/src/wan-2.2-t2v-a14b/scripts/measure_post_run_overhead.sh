#!/usr/bin/env bash
# Profile post-run_unit overhead with the mock backend and real-sized payloads.
#
# Runs Offline performance under torchrun (no GPU model) with
# --measure-post-run-overhead. All stdout/stderr from every rank is tee'd
# to ${OUTPUT_DIR}/harness.log while still appearing on the terminal.
#
# Usage:
#   ./scripts/measure_post_run_overhead.sh [--dispatch async|wave] \
#       [--result-transport shm|gloo] [--output-dir DIR] [--nproc N] \
#       [-- extra wan-harness args]
#
# Examples:
#   ./scripts/measure_post_run_overhead.sh
#   ./scripts/measure_post_run_overhead.sh --dispatch wave
#   ./scripts/measure_post_run_overhead.sh --output-dir runs/overhead-probe-async
#   ./scripts/measure_post_run_overhead.sh -- --min-query-count 32
#
# Single-process sanity check (no torchrun, tiny query count still uses
# real frame dimensions):
#   ./scripts/measure_post_run_overhead.sh --single-process

set -euo pipefail

DISPATCH="async"
RESULT_TRANSPORT=""
OUTPUT_DIR=""
NPROC="${NPROC_PER_NODE:-8}"
SINGLE_PROCESS=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dispatch)       DISPATCH="$2"; shift 2 ;;
        --result-transport) RESULT_TRANSPORT="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --nproc)          NPROC="$2"; shift 2 ;;
        --single-process) SINGLE_PROCESS=1; shift ;;
        --)               shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '1,/^set -euo/p' "$0" | head -24
            exit 0 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ "${DISPATCH}" != "async" && "${DISPATCH}" != "wave" ]]; then
    echo "error: --dispatch must be 'async' or 'wave', got '${DISPATCH}'" >&2
    exit 2
fi

if [[ -n "${RESULT_TRANSPORT}" && "${RESULT_TRANSPORT}" != "shm" && "${RESULT_TRANSPORT}" != "gloo" ]]; then
    echo "error: --result-transport must be 'shm' or 'gloo', got '${RESULT_TRANSPORT}'" >&2
    exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    if [[ "${SINGLE_PROCESS}" -eq 1 ]]; then
        OUTPUT_DIR="runs/overhead-probe/single-process"
    else
        if [[ -n "${RESULT_TRANSPORT}" ]]; then
            OUTPUT_DIR="runs/overhead-probe/${DISPATCH}-${RESULT_TRANSPORT}"
        else
            OUTPUT_DIR="runs/overhead-probe/${DISPATCH}"
        fi
    fi
fi

LOG_FILE="${OUTPUT_DIR}/harness.log"
mkdir -p "${OUTPUT_DIR}"

echo "[measure_post_run_overhead] dispatch=${DISPATCH} output_dir=${OUTPUT_DIR}"
if [[ -n "${RESULT_TRANSPORT}" ]]; then
    echo "[measure_post_run_overhead] result_transport=${RESULT_TRANSPORT}"
fi
echo "[measure_post_run_overhead] logging stdout/stderr to ${LOG_FILE}"

_common_args=(
    run
    --backend mock
    --scenario Offline
    --mode performance
    --measure-post-run-overhead
    --set height=720
    --set width=1280
    --set num_frames=81
    --min-query-count 16
    --min-duration-ms 100
    --performance-sample-count 32
    --output-dir "${OUTPUT_DIR}"
    --log-level INFO
)

if [[ -n "${RESULT_TRANSPORT}" ]]; then
    _common_args+=(--result-transport "${RESULT_TRANSPORT}")
fi

_run_with_log() {
    # Preserve the child exit code when piping through tee.
    set -o pipefail
    "$@" 2>&1 | tee -a "${LOG_FILE}"
}

if [[ "${SINGLE_PROCESS}" -eq 1 ]]; then
    _run_with_log wan-harness \
        "${_common_args[@]}" \
        --min-query-count 4 \
        --min-duration-ms 10 \
        "${EXTRA_ARGS[@]}"
    exit 0
fi

_run_with_log torchrun --standalone --nproc-per-node="${NPROC}" -m wan_harness.cli \
    "${_common_args[@]}" \
    --mock-dispatch "${DISPATCH}" \
    "${EXTRA_ARGS[@]}"
