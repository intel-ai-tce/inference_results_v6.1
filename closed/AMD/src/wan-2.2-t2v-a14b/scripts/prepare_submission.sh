#!/usr/bin/env bash
# Build an MLPerf Inference submission tree from a run_all experiment.
#
# Intended to be invoked inside the wan-harness container (``./launch.sh``
# from the host). Checkout the experiment MANIFEST.json git SHA first.
#
# Copies LoadGen logs into the checker-expected layout, truncates
# mlperf_log_accuracy.json, and re-emits accuracy.txt with a hash that
# matches the truncated log. The source experiment directory is not modified.
#
# Usage:
#   ./scripts/prepare_submission.sh \
#       --experiment-root runs/wan22/latest \
#       --output submissions/my-org \
#       --submitter AMD \
#       [--system 8xMI355X_2xEPYC_9575F] \
#       [--system-desc systems/8xMI355X_2xEPYC_9575F.json]
#
# Optional:
#   --division closed|open|network   (default: closed)
#   --benchmark wan-2.2-t2v-a14b     (default)
#   --user-conf configs/user.conf
#   --measurements configs/measurements.json
#   --readme-template templates/submission_scenario_README.md
#   --skip-compliance                (omit TEST04)
#   --skip-vbench-refresh            (copy accuracy.txt as-is)
#   --skip-code                      (omit src/ snapshot)
#   --skip-measurements              (omit measurements.json)
#   --skip-readme                    (omit per-scenario README.md)
#   --dry-run
#
# Verify the system description against the current host (inside container):
#   python3 -m tools.verify_system_desc systems/8xMI355X_2xEPYC_9575F.json

set -euo pipefail

EXPERIMENT_ROOT=""
OUTPUT=""
SUBMITTER=""
SYSTEM=""
SYSTEM_DESC=""
DIVISION="closed"
BENCHMARK="wan-2.2-t2v-a14b"
USER_CONF=""
MEASUREMENTS=""
README_TEMPLATE=""
SKIP_COMPLIANCE=0
SKIP_VBENCH_REFRESH=0
SKIP_CODE=0
SKIP_MEASUREMENTS=0
SKIP_README=0
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-root)      EXPERIMENT_ROOT="$2"; shift 2 ;;
        --output)               OUTPUT="$2"; shift 2 ;;
        --submitter)            SUBMITTER="$2"; shift 2 ;;
        --system)               SYSTEM="$2"; shift 2 ;;
        --system-desc)          SYSTEM_DESC="$2"; shift 2 ;;
        --division)             DIVISION="$2"; shift 2 ;;
        --benchmark)            BENCHMARK="$2"; shift 2 ;;
        --user-conf)            USER_CONF="$2"; shift 2 ;;
        --measurements)         MEASUREMENTS="$2"; shift 2 ;;
        --readme-template)      README_TEMPLATE="$2"; shift 2 ;;
        --skip-compliance)      SKIP_COMPLIANCE=1; shift ;;
        --skip-vbench-refresh)  SKIP_VBENCH_REFRESH=1; shift ;;
        --skip-code)            SKIP_CODE=1; shift ;;
        --skip-measurements)    SKIP_MEASUREMENTS=1; shift ;;
        --skip-readme)          SKIP_README=1; shift ;;
        --dry-run)              DRY_RUN=1; shift ;;
        --)                     shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '1,/^set -euo/p' "$0" | head -28
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${EXPERIMENT_ROOT}" || -z "${OUTPUT}" || -z "${SUBMITTER}" ]]; then
    echo "Usage: $0 --experiment-root <dir> --output <dir> --submitter <org> [--system <id>]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ARGS=(
    python3 -m tools.prepare_submission
    "${EXPERIMENT_ROOT}"
    --output "${OUTPUT}"
    --submitter "${SUBMITTER}"
    --division "${DIVISION}"
    --benchmark "${BENCHMARK}"
)
[[ -n "${SYSTEM}" ]] && ARGS+=(--system "${SYSTEM}")
[[ -n "${SYSTEM_DESC}" ]] && ARGS+=(--system-desc "${SYSTEM_DESC}")
[[ -n "${USER_CONF}" ]] && ARGS+=(--user-conf "${USER_CONF}")
[[ -n "${MEASUREMENTS}" ]] && ARGS+=(--measurements "${MEASUREMENTS}")
[[ -n "${README_TEMPLATE}" ]] && ARGS+=(--readme-template "${README_TEMPLATE}")
[[ ${SKIP_COMPLIANCE} -eq 1 ]] && ARGS+=(--skip-compliance)
[[ ${SKIP_VBENCH_REFRESH} -eq 1 ]] && ARGS+=(--skip-vbench-refresh)
[[ ${SKIP_CODE} -eq 1 ]] && ARGS+=(--skip-code)
[[ ${SKIP_MEASUREMENTS} -eq 1 ]] && ARGS+=(--skip-measurements)
[[ ${SKIP_README} -eq 1 ]] && ARGS+=(--skip-readme)
[[ ${DRY_RUN} -eq 1 ]] && ARGS+=(--dry-run)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    ARGS+=("${EXTRA_ARGS[@]}")
fi

exec "${ARGS[@]}"
