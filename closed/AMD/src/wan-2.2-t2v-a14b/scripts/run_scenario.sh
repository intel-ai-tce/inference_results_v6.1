#!/usr/bin/env bash
# Run a single (scenario, mode) test against a given backend.
#
# Usage:
#   ./scripts/run_scenario.sh [--backend mock|wan22] [--scenario Offline|SingleStream] \
#                              [--mode performance|accuracy] [--output-dir DIR] \
#                              [-- extra CLI args forwarded to wan-harness]
#
# Examples:
#   ./scripts/run_scenario.sh                                      # Mock + Offline + performance
#   ./scripts/run_scenario.sh --scenario SingleStream --mode accuracy
#   ./scripts/run_scenario.sh --backend wan22 --scenario Offline --mode performance

set -euo pipefail

BACKEND="mock"
SCENARIO="Offline"
MODE="performance"
OUTPUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)        BACKEND="$2"; shift 2 ;;
        --scenario)       SCENARIO="$2"; shift 2 ;;
        --mode)           MODE="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --)               shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '1,/^set -euo/p' "$0" | head -20
            exit 0 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
    if [[ "${MODE}" == "performance" ]]; then
        OUTPUT_DIR="runs/${BACKEND}/${SCENARIO}/performance/run_1"
    else
        OUTPUT_DIR="runs/${BACKEND}/${SCENARIO}/${MODE}"
    fi
fi

mkdir -p "${OUTPUT_DIR}"

echo "[run_scenario] backend=${BACKEND} scenario=${SCENARIO} mode=${MODE}"
echo "[run_scenario] output_dir=${OUTPUT_DIR}"

# Multi-rank backends are launched under torchrun. The world size is
# pulled from the per-scenario YAML so the launcher and the config
# stay in lock-step.
if [[ "${BACKEND}" == "wan22" ]]; then
    BACKEND_CONFIG="${BACKEND_CONFIG:-configs/wan22/${SCENARIO}.yaml}"
    NPROC="${NPROC_PER_NODE:-8}"

    case "${SCENARIO}" in
        Offline) KVARIANT=fast ;;
        *)       KVARIANT=safe ;;
    esac
    KERNEL_SRC="/opt/aiter-kernels/${KVARIANT}/fmha_v3_fwd"
    KERNEL_DST="/app/external/aiter/hsa/gfx950/fmha_v3_fwd"
    if [[ -d "${KERNEL_SRC}" && -d "${KERNEL_DST}" ]]; then
        echo "[run_scenario] installing '${KVARIANT}' fmha_v3 kernels"
        cp -f "${KERNEL_SRC}"/*.co "${KERNEL_DST}/"
    fi

    echo "[run_scenario] launching torchrun --nproc-per-node=${NPROC} (config=${BACKEND_CONFIG})"
    exec torchrun --standalone --nproc-per-node="${NPROC}" -m wan_harness.cli run \
        --backend "${BACKEND}" \
        --scenario "${SCENARIO}" \
        --mode "${MODE}" \
        --output-dir "${OUTPUT_DIR}" \
        --backend-config "${BACKEND_CONFIG}" \
        "${EXTRA_ARGS[@]}"
fi

# Mock / single-process path.
exec wan-harness run \
    --backend "${BACKEND}" \
    --scenario "${SCENARIO}" \
    --mode "${MODE}" \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"
