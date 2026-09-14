#!/bin/bash
# Build an MLPerf submission tree for the e2e-rag DB (data-setup) workload and
# run the official submission checker against it.
#
# The submission checker calls this workload "e2e-rag-db", matching the LoadGen
# user.conf section name (see
# third_party/mlperf-inference/tools/submission/submission_checker/constants.py).
# It was named "e2e_vectorDB" in checkers predating mlcommons/inference c22d843b;
# with that older name get_required() returns None and the checker crashes in
# lower_list() rather than reporting a clean error.
#
# Inputs are the two run output directories produced by:
#   bash scripts/run_ingestion_perf.sh      -> ${PERF_SRC}
#   bash scripts/run_ingestion_accuracy.sh  -> ${ACC_SRC}
#
# Usage: bash tools/make_submission_db.sh [SUBMISSION_ROOT]

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
REPO_ROOT="$(pwd)"

source config.sh 2>/dev/null || true
source config.default.sh

SUBMISSION_ROOT="${1:-${REPO_ROOT}/submission}"

DIVISION="${DIVISION:-closed}"
SUBMITTER="${SUBMITTER:-Intel}"
SYSTEM_ID="${SYSTEM_ID:-1-node-2S-Xeon6787P}"
BENCHMARK="${BENCHMARK:-e2e-rag-db}"
SCENARIO="${SCENARIO:-Offline}"
VERSION="${VERSION:-v6.1}"

PERF_SRC="${PERF_SRC:-${INGESTION_OUTPUT_DIR}}"
ACC_SRC="${ACC_SRC:-${INGESTION_ACCURACY_OUTPUT_DIR}}"

CHECKER_DIR="${REPO_ROOT}/third_party/mlperf-inference/tools/submission"

echo "============================================================"
echo "Building submission tree"
echo "============================================================"
echo "  root:      ${SUBMISSION_ROOT}"
echo "  division:  ${DIVISION}"
echo "  submitter: ${SUBMITTER}"
echo "  system:    ${SYSTEM_ID}"
echo "  benchmark: ${BENCHMARK}  scenario: ${SCENARIO}  version: ${VERSION}"
echo "  perf src:  ${PERF_SRC}"
echo "  acc src:   ${ACC_SRC}"
echo ""

for d in "${PERF_SRC}" "${ACC_SRC}"; do
    [ -d "${d}" ] || { echo "ERROR: source dir not found: ${d}"; exit 1; }
done

BASE="${SUBMISSION_ROOT}/${DIVISION}/${SUBMITTER}"
SCEN_DIR="${BASE}/results/${SYSTEM_ID}/${BENCHMARK}/${SCENARIO}"

# SUBMISSION_ROOT is user-supplied (argv[1] or env), and this rm is recursive.
# Refuse the values that would be catastrophic rather than merely wrong: empty
# (rm -rf "" with a trailing slash elsewhere), the filesystem root, and any
# path with no parent directory component (e.g. "/tmp" is fine, "/" is not).
case "${SUBMISSION_ROOT}" in
    ""|"/") echo "ERROR: refusing to delete unsafe SUBMISSION_ROOT='${SUBMISSION_ROOT}'" >&2; exit 2 ;;
    */*) ;;
    *) echo "ERROR: SUBMISSION_ROOT must be a path with a parent dir, got '${SUBMISSION_ROOT}'" >&2; exit 2 ;;
esac
if [ "${SUBMISSION_ROOT}" = "${HOME}" ] || [ "${SUBMISSION_ROOT}" = "${REPO_ROOT}" ]; then
    echo "ERROR: refusing to delete SUBMISSION_ROOT='${SUBMISSION_ROOT}'" >&2
    exit 2
fi
rm -rf "${SUBMISSION_ROOT}"
# NOTE: for v6.0/v6.1 the checker's SRC_PATH is "{division}/{submitter}/src"
# (it was "code" up to v5.1) -- MeasurementsCheck.directory_exist_check looks
# for exactly that name, so the implementation dir must be src/, not code/.
mkdir -p "${SCEN_DIR}/performance/run_1" "${SCEN_DIR}/accuracy" \
         "${BASE}/systems" "${BASE}/src/${BENCHMARK}" "${BASE}/documentation"

# ---- performance run (REQUIRED_PERF_FILES) ----
cp "${PERF_SRC}/mlperf_log_summary.txt" "${PERF_SRC}/mlperf_log_detail.txt" \
   "${SCEN_DIR}/performance/run_1/"

# ---- accuracy run (REQUIRED_ACC_FILES) ----
cp "${ACC_SRC}/mlperf_log_summary.txt" "${ACC_SRC}/mlperf_log_detail.txt" \
   "${ACC_SRC}/mlperf_log_accuracy.json" "${ACC_SRC}/accuracy.txt" \
   "${SCEN_DIR}/accuracy/"

# ---- measurements.json (SYSTEM_IMP_REQUIRED_FILES) ----
cat > "${SCEN_DIR}/measurements.json" <<EOF
{
    "input_data_types": "fp32",
    "retraining": "No",
    "starting_weights_filename": "intfloat/e5-base-v2 (MLCommons-hosted copy)",
    "weight_data_types": "fp32",
    "weight_transformations": "none"
}
EOF

# ---- user.conf + README.md (REQUIRED_MEASURE_FILES) ----
# Copied verbatim -- it must match the settings the copied mlperf_log_detail.txt
# was actually produced with. Two of the checker's performance checks
# (min_query_count_check, min_duration_check) currently reject what this
# workload+LoadGen can produce in Offline; see SUBMISSION_NOTES.md. Editing
# this file to satisfy them without re-running would make the submission
# inconsistent with its own logs, so it is left as-is.
cp "${REPO_ROOT}/user.conf" "${SCEN_DIR}/user.conf"

cat > "${SCEN_DIR}/README.md" <<EOF
# ${BENCHMARK} (${SCENARIO}) -- ${SUBMITTER} ${SYSTEM_ID}

E2E-RAG data-setup (vector database build) workload: parse the frozen Wikipedia
HTML corpus, chunk it, embed the passages with e5-base-v2, and build a FAISS
HNSW index.

## Reproduction

\`\`\`bash
bash scripts/run_ingestion_perf.sh       # performance run
bash scripts/run_ingestion_accuracy.sh   # accuracy run + DB manifest check
\`\`\`

## Accuracy

The reported accuracy is the DB manifest probe-query retrieval accuracy: the
mean top-K document-URL set overlap against the reference database over 50
fixed probe queries (see \`ingestion/db_manifest_v2.py\`). Gate: >= 0.95.

The accuracy run additionally verifies file-processing success rate, the
database MD5 reported by the SUT, vector/docstore consistency, index
dimension, and the FAISS index parameters.
EOF

# ---- system description ----
# CPU and memory are visible through to the container, so autodetect is right.
HOST_CPU="$(lscpu | sed -n 's/^Model name: *//p' | head -1)"
CORES="$(lscpu | sed -n 's/^Core(s) per socket: *//p' | head -1)"
SOCKETS="$(lscpu | sed -n 's/^Socket(s): *//p' | head -1)"
MEM_GB="$(free -g | awk '/^Mem:/{print $2}')"

# The OS and root-filesystem size, however, describe the *container* when this
# runs inside one -- the submission must describe the host. Override with
# HOST_OS / HOST_STORAGE / HOST_STORAGE_TYPE when running containerized.
CONTAINER_OS="$(. /etc/os-release && echo "${PRETTY_NAME}")"
if [ -f /.dockerenv ] && [ -z "${HOST_OS:-}" ]; then
    echo "WARNING: running inside a container and HOST_OS is unset."
    echo "         operating_system will report the container OS"
    echo "         (${CONTAINER_OS}), not the host's. Re-run with"
    echo "         HOST_OS='<host os>' HOST_STORAGE='<size>' to fix."
    echo ""
fi
OS_NAME="${HOST_OS:-${CONTAINER_OS}}"
DISK="${HOST_STORAGE:-$(df -BG --output=size / | tail -1 | tr -d ' ')}"
STORAGE_TYPE="${HOST_STORAGE_TYPE:-NVMe SSD}"

# Framework versions come from the interpreter actually running the workload,
# which is this one (the script is meant to run in the benchmark container).
TORCH_VER="$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo unknown)"
FAISS_VER="$(python3 -c 'import faiss;print(faiss.__version__)' 2>/dev/null || echo unknown)"
TRANSFORMERS_VER="$(python3 -c 'import transformers;print(transformers.__version__)' 2>/dev/null || echo unknown)"

cat > "${BASE}/systems/${SYSTEM_ID}.json" <<EOF
{
    "division": "${DIVISION}",
    "submitter": "${SUBMITTER}",
    "status": "available",
    "system_name": "${SYSTEM_ID}",
    "system_type": "datacenter",
    "system_type_detail": "N/A",
    "number_of_nodes": 1,

    "host_processor_model_name": "${HOST_CPU}",
    "host_processors_per_node": ${SOCKETS},
    "host_processor_core_count": "${CORES}",
    "host_processor_frequency": "3.8 GHz max turbo",
    "host_processor_caches": "N/A",
    "host_processor_interconnect": "Intel UPI",
    "host_memory_capacity": "${MEM_GB}GB",
    "host_memory_configuration": "DDR5",
    "host_storage_capacity": "${DISK}",
    "host_storage_type": "${STORAGE_TYPE}",
    "host_networking": "N/A",
    "host_network_card_count": "N/A",
    "host_networking_topology": "N/A",

    "accelerators_per_node": "0",
    "accelerator_model_name": "N/A",
    "accelerator_memory_capacity": "N/A",
    "accelerator_frequency": "N/A",
    "accelerator_host_interconnect": "N/A",
    "accelerator_interconnect": "N/A",
    "accelerator_interconnect_topology": "N/A",
    "accelerator_memory_configuration": "N/A",
    "accelerator_on-chip_memories": "N/A",

    "cooling": "Air",
    "framework": "PyTorch ${TORCH_VER}, FAISS ${FAISS_VER}",
    "operating_system": "${OS_NAME}",
    "other_software_stack": "transformers ${TRANSFORMERS_VER}, langchain-community, bm25s",
    "hw_notes": "CPU-only data-setup run; no accelerator used.",
    "sw_notes": "Vector DB built from the frozen docs.tar.gz Wikipedia corpus (2515 HTML pages)."
}
EOF

# ---- code + documentation (structure checks expect these to be non-empty) ----
cat > "${BASE}/src/${BENCHMARK}/README.md" <<EOF
# ${BENCHMARK} implementation

Source: this repository (E2E-RAG benchmark, Intel-optimized fork of the
MLCommons \`inference/e2e-rag\` reference).

Entry point: \`main_ingestion.py\` (LoadGen SUT in \`sut/SUT_ingestion.py\`,
pipelined variant in \`sut/SUT_ingestion_pipelined.py\`).
Driver scripts: \`scripts/run_ingestion_perf.sh\`, \`scripts/run_ingestion_accuracy.sh\`.
EOF

cat > "${BASE}/documentation/README.md" <<EOF
# ${SUBMITTER} ${BENCHMARK} submission

CPU-only vector-database build for the E2E-RAG benchmark. See
\`src/${BENCHMARK}/README.md\` for the implementation and
\`results/${SYSTEM_ID}/${BENCHMARK}/${SCENARIO}/README.md\` for reproduction steps.
EOF

# ---- truncate the accuracy log ----
# mlperf_log_accuracy.json must be <= MAX_ACCURACY_LOG_SIZE (10 KiB) and
# accuracy.txt must carry a "hash=<sha256>" line -- both are produced by the
# official truncate_accuracy_log.py, which backs up the full log first.
echo "============================================================"
echo "Truncating accuracy log"
echo "============================================================"
BACKUP_DIR="${SUBMISSION_ROOT}_accuracy_backup"
rm -rf "${BACKUP_DIR}"
mkdir -p "${BACKUP_DIR}"
PYTHONDONTWRITEBYTECODE=1 python3 "${CHECKER_DIR}/truncate_accuracy_log.py" \
    --input "${SUBMISSION_ROOT}" \
    --submitter "${SUBMITTER}" \
    --backup "${BACKUP_DIR}"
echo ""

echo "Submission tree:"
find "${SUBMISSION_ROOT}" -type f | sed "s|${SUBMISSION_ROOT}/|  |" | sort
echo ""

echo "============================================================"
echo "Running submission checker (${VERSION})"
echo "============================================================"

# LENIENT=1 used to add --skip-dataset-size-check, needed only because the old
# checker had dataset-size 824 (the QnA query count) for this workload. Upstream
# c22d843b sets it to 2515, so no skip flag should be necessary any more; kept as
# an escape hatch for older checkers.
LENIENT_ARGS=()
if [ "${LENIENT:-0}" = "1" ]; then
    LENIENT_ARGS=(--skip-dataset-size-check)
    echo "LENIENT=1: adding ${LENIENT_ARGS[*]}"
    echo ""
fi

# --csv defaults to the bare relative name "summary.csv", so it would land in
# CHECKER_DIR (inside third_party/, typically root-owned) and fail with
# PermissionError. Point it at the repo root instead.
cd "${CHECKER_DIR}"
PYTHONDONTWRITEBYTECODE=1 python3 -m submission_checker.main \
    --input "${SUBMISSION_ROOT}" \
    --version "${VERSION}" \
    --submitter "${SUBMITTER}" \
    --csv "${REPO_ROOT}/submission_summary.csv" \
    --skip_compliance \
    "${LENIENT_ARGS[@]}" \
    "${@:2}"
