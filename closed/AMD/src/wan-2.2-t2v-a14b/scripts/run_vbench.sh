#!/usr/bin/env bash
# Score an accuracy-mode harness run with VBench and emit accuracy.txt.
#
# Reads ``artefacts/*.mp4`` and ``artefacts/prompts.json`` under the accuracy
# run directory, stages videos for VBench, and writes ``accuracy.txt`` plus
# a ``vbench/`` sidecar under the same directory. See README.md § VBench
# evaluation and ``tools/run_vbench.py``.
#
# Usage:
#   ./scripts/run_vbench.sh \
#       --backend wan22 \
#       --scenario Offline \
#       --accuracy-dir runs/wan22/<exp>/Offline/accuracy
#
# Optional env:
#   VBENCH_NPROC_PER_NODE   passed to --nproc-per-node (default: 1)

set -euo pipefail

SCENARIO=""
ACCURACY_DIR=""
BACKEND="${WAN_HARNESS_BACKEND:-mock}"
NPROC="${VBENCH_NPROC_PER_NODE:-1}"
NO_WITH_VBENCH=0
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)          BACKEND="$2"; shift 2 ;;
        --scenario)         SCENARIO="$2"; shift 2 ;;
        --accuracy-dir)     ACCURACY_DIR="$2"; shift 2 ;;
        --nproc-per-node)   NPROC="$2"; shift 2 ;;
        --no-with-vbench)   NO_WITH_VBENCH=1; shift ;;
        --dry-run)          DRY_RUN=1; shift ;;
        --)                 shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '1,/^set -euo/p' "$0" | head -28
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${SCENARIO}" || -z "${ACCURACY_DIR}" ]]; then
    echo "Usage: $0 --backend {mock|wan22} --scenario {Offline,SingleStream} --accuracy-dir <dir>" >&2
    exit 2
fi

case "${SCENARIO}" in
    Offline|SingleStream) ;;
    *) echo "[run_vbench] unknown scenario: ${SCENARIO}" >&2; exit 1 ;;
esac

ACCURACY_DIR="$(cd "${ACCURACY_DIR}" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARTEFACTS_DIR="${ACCURACY_DIR}/artefacts"
if [[ ! -d "${ARTEFACTS_DIR}" ]]; then
    echo "[run_vbench] no artefacts/ under ${ACCURACY_DIR}" >&2
    echo "[run_vbench] run accuracy mode first" >&2
    exit 1
fi
if [[ ! -f "${ARTEFACTS_DIR}/prompts.json" ]]; then
    echo "[run_vbench] no ${ARTEFACTS_DIR}/prompts.json" >&2
    exit 1
fi
if ! compgen -G "${ARTEFACTS_DIR}/*.mp4" > /dev/null; then
    echo "[run_vbench] no .mp4 files under ${ARTEFACTS_DIR}" >&2
    echo "[run_vbench] the mock backend writes .bin frames only; use --backend wan22" >&2
    exit 1
fi

echo "[run_vbench] backend=${BACKEND} scenario=${SCENARIO}"
echo "[run_vbench] accuracy_dir=${ACCURACY_DIR}"
echo "[run_vbench] nproc_per_node=${NPROC}"

cd "${REPO_ROOT}"

VBENCH_ARGS=(
    python3 -m tools.run_vbench
    "${ACCURACY_DIR}"
    --nproc-per-node "${NPROC}"
)
if [[ ${NO_WITH_VBENCH} -eq 1 ]]; then
    VBENCH_ARGS+=(--no-with-vbench)
fi
if [[ ${DRY_RUN} -eq 1 ]]; then
    VBENCH_ARGS+=(--dry-run)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    VBENCH_ARGS+=("${EXTRA_ARGS[@]}")
fi

exec "${VBENCH_ARGS[@]}"
