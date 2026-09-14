#!/usr/bin/env bash
# Run the upstream MLPerf submission checker on a packaged tree.
#
# Uses tools/submission/submission_checker/main.py from mlcommons/inference.
# Stage 2 writes PREPARE_MANIFEST.json at the submission root; the checker
# flags that as an extra file, so we pass --skip-extra-files-in-root-check
# by default.
#
# Intended to be invoked inside the wan-harness container (MLPERF_INFERENCE_DIR
# defaults to /opt/mlperf-inference). On the host, start a shell with
# ``./launch.sh``, then run:
#
#   ./scripts/verify_submission.sh \
#       --input submissions/amd \
#       --submitter AMD
#
# For automation, ``./launch.sh ./scripts/verify_submission.sh ...`` from
# the host is equivalent.

set -euo pipefail

INPUT=""
SUBMITTER=""
VERSION="v6.1"
MLPERF_INFERENCE="${MLPERF_INFERENCE_DIR:-/opt/mlperf-inference}"
CHECKER="${MLPERF_INFERENCE}/tools/submission/submission_checker/main.py"
EXTRA_ARGS=(--skip-extra-files-in-root-check)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)            INPUT="$2"; shift 2 ;;
        --submitter)        SUBMITTER="$2"; shift 2 ;;
        --version)          VERSION="$2"; shift 2 ;;
        --mlperf-inference) MLPERF_INFERENCE="$2"; CHECKER="${MLPERF_INFERENCE}/tools/submission/submission_checker/main.py"; shift 2 ;;
        --strict-root)      EXTRA_ARGS=(); shift ;;
        --)                 shift; EXTRA_ARGS+=("$@"); break ;;
        -h|--help)
            sed -n '1,/^set -euo/p' "$0" | head -25
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${INPUT}" || -z "${SUBMITTER}" ]]; then
    echo "Usage: $0 --input <submission-dir> --submitter <ORG> [--version v6.1]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUT="$(cd "${INPUT}" && pwd)"

if [[ ! -f "${CHECKER}" ]]; then
    echo "[verify_submission] checker not found: ${CHECKER}" >&2
    echo "[verify_submission] set MLPERF_INFERENCE_DIR or use --mlperf-inference" >&2
    exit 1
fi

if [[ ! -d "${INPUT}/closed/${SUBMITTER}" ]]; then
    echo "[verify_submission] expected closed/${SUBMITTER} under ${INPUT}" >&2
    exit 1
fi

echo "[verify_submission] checker=${CHECKER}"
echo "[verify_submission] input=${INPUT}"
echo "[verify_submission] submitter=${SUBMITTER} version=${VERSION}"

cd "${MLPERF_INFERENCE}/tools/submission"
python3 "${CHECKER}" \
    --input "${INPUT}" \
    --version "${VERSION}" \
    --submitter "${SUBMITTER}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
