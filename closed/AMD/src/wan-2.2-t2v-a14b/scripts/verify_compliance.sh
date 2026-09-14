#!/usr/bin/env bash
# Run TEST04 compliance for wan-2.2-t2v-a14b.
#
# Per mlcommons/inference/compliance/README.md, Wan2.2 requires TEST04 only
# (verify the SUT is not caching repeated sample IDs). TEST01 is not required
# — accuracy is validated separately via VBench.
#
# TEST04 audit settings: harness copy (Wan min_query_count=64) by default;
# falls back to ${MLPERF_INFERENCE_DIR}/compliance/TEST04/audit.config.
#
# Usage:
#   ./scripts/verify_compliance.sh \
#       --backend wan22 \
#       --scenario Offline \
#       --scenario-dir runs/wan22/<exp>/Offline
#
# Legacy alias (--base-run points at the performance/ subdir):
#   ./scripts/verify_compliance.sh \
#       --scenario Offline \
#       --base-run runs/wan22/<exp>/Offline/performance

set -euo pipefail

SCENARIO=""
SCENARIO_DIR=""
BACKEND="${WAN_HARNESS_BACKEND:-mock}"
COMPLIANCE_REPO="${MLPERF_INFERENCE_DIR:-/opt/mlperf-inference}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)          BACKEND="$2"; shift 2 ;;
        --scenario)         SCENARIO="$2"; shift 2 ;;
        --scenario-dir)     SCENARIO_DIR="$2"; shift 2 ;;
        --base-run)         SCENARIO_DIR="$(dirname "$2")"; shift 2 ;;
        --compliance-repo)  COMPLIANCE_REPO="$2"; shift 2 ;;
        --)                 shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '1,/^set -euo/p' "$0" | head -30
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${SCENARIO}" || -z "${SCENARIO_DIR}" ]]; then
    echo "Usage: $0 --backend {mock|wan22} --scenario {Offline,SingleStream} --scenario-dir <dir>" >&2
    echo "       $0 --scenario {Offline,SingleStream} --base-run <performance-dir>" >&2
    exit 2
fi

case "${SCENARIO}" in
    Offline|SingleStream) ;;
    *) echo "[verify_compliance] unknown scenario: ${SCENARIO}" >&2; exit 1 ;;
esac

# Resolve early so verification can run from a temp cwd without breaking paths.
SCENARIO_DIR="$(cd "${SCENARIO_DIR}" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HARNESS_AUDIT="${REPO_ROOT}/configs/compliance/TEST04-audit.config"
UPSTREAM_AUDIT="${COMPLIANCE_REPO}/compliance/TEST04/audit.config"

if [[ -f "${HARNESS_AUDIT}" ]]; then
    AUDIT_CFG_SRC="${HARNESS_AUDIT}"
elif [[ -f "${UPSTREAM_AUDIT}" ]]; then
    AUDIT_CFG_SRC="${UPSTREAM_AUDIT}"
    echo "[verify_compliance] warning: using upstream audit.config (no harness copy at ${HARNESS_AUDIT})" >&2
else
    echo "[verify_compliance] no TEST04 audit.config at ${HARNESS_AUDIT} or ${UPSTREAM_AUDIT}" >&2
    exit 1
fi

PERF_RUN_DIR="${SCENARIO_DIR}/performance/run_1"
PERF_SUMMARY="${PERF_RUN_DIR}/mlperf_log_summary.txt"
# Legacy flat layout (pre-run_1): performance/mlperf_log_summary.txt directly.
if [[ ! -f "${PERF_SUMMARY}" && -f "${SCENARIO_DIR}/performance/mlperf_log_summary.txt" ]]; then
    PERF_SUMMARY="${SCENARIO_DIR}/performance/mlperf_log_summary.txt"
fi
if [[ ! -f "${PERF_SUMMARY}" ]]; then
    echo "[verify_compliance] no performance summary at ${PERF_RUN_DIR}/mlperf_log_summary.txt" >&2
    echo "[verify_compliance] run the performance scenario first" >&2
    exit 1
fi

TEST04_DIR="${SCENARIO_DIR}/TEST04"
COMPLIANCE_OUT="${SCENARIO_DIR}/compliance"
mkdir -p "${TEST04_DIR}" "${COMPLIANCE_OUT}"

echo "[verify_compliance] backend=${BACKEND} scenario=${SCENARIO}"
echo "[verify_compliance] audit.config=${AUDIT_CFG_SRC}"
echo "[verify_compliance] performance summary=${PERF_SUMMARY}"
echo "[verify_compliance] TEST04 output=${TEST04_DIR}"

./scripts/run_scenario.sh \
    --backend "${BACKEND}" \
    --scenario "${SCENARIO}" \
    --mode performance \
    --output-dir "${TEST04_DIR}" \
    -- --audit-conf "${AUDIT_CFG_SRC}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

VERIFY="${COMPLIANCE_REPO}/compliance/TEST04/run_verification.py"
if [[ -f "${VERIFY}" ]]; then
    # run_verification.py writes verify_performance.txt to its cwd before
    # copying it into --output_dir; isolate that in a temp dir so we do not
    # leave a stray file in the repo root.
    VERIFY_TMP="$(mktemp -d)"
    trap 'rm -rf "${VERIFY_TMP}"' EXIT
    (
        cd "${VERIFY_TMP}"
        python3 "${VERIFY}" \
            --results_dir "${SCENARIO_DIR}" \
            --compliance_dir "${TEST04_DIR}" \
            --output_dir "${COMPLIANCE_OUT}"
    )
else
    echo "[verify_compliance] no verifier at ${VERIFY}; running verify_performance.py directly" >&2
    VERIFY_PERF="${COMPLIANCE_REPO}/compliance/TEST04/verify_performance.py"
    if [[ -f "${VERIFY_PERF}" ]]; then
        mkdir -p "${COMPLIANCE_OUT}/TEST04"
        python3 "${VERIFY_PERF}" \
            -r "${PERF_SUMMARY}" \
            -t "${TEST04_DIR}/mlperf_log_summary.txt" \
            | tee "${COMPLIANCE_OUT}/TEST04/verify_performance.txt"
    else
        echo "[verify_compliance] no verify_performance.py either; skipping verification" >&2
        exit 1
    fi
fi

echo "[verify_compliance] TEST04 artefacts under ${COMPLIANCE_OUT}/TEST04"
