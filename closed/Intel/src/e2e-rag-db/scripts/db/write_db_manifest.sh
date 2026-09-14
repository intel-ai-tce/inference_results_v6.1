#!/bin/bash
# =============================================================================
# Generate a reference DB manifest from this system's vector DB.
#
# Usage (from repo root):
#   bash scripts/db/write_db_manifest.sh [OUTPUT]
#
#   OUTPUT: optional path for the manifest JSON. Use .json.gz to compress.
#           default: db_manifest_$(hostname -s).json.gz
#
# Configuration: see config.default.sh (uses RUN_DATABASE and
# EMBEDDING_MODEL).
# =============================================================================

set -e

# Run from repo root so python3 -m pkg.module and relative paths resolve.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

DB="${RUN_DATABASE}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-intfloat_e5-base-v2/e5-base-v2}"
DATASET="${RUN_DATASET:-data/frames_dataset.tsv}"

OUTPUT="${1:-db_manifest_$(hostname -s).json.gz}"

echo "=== Writing DB manifest ==="
echo "  DB:        ${DB}"
echo "  Embedding model: ${EMBEDDING_MODEL}"
echo "  Dataset:   ${DATASET}"
echo "  Output:    ${OUTPUT}"
echo ""

python3 -u -m ingestion.db_manifest write \
    --db "${DB}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --dataset "${DATASET}" \
    --output "${OUTPUT}"
