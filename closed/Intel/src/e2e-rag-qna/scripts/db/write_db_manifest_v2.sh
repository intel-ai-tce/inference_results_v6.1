#!/bin/bash
# =============================================================================
# Generate a v2 BEHAVIORAL-EQUIVALENCE manifest from this system's vector DB.
#
# Unlike the v1 manifest (write_db_manifest.sh), the v2 manifest does NOT pin
# stored vectors or an order-dependent corpus hash. It records what a vendor's
# independently-built DB must match to be "the same DB": passage count, index
# params/algorithm, an order-INDEPENDENT corpus-set hash (same HTML + chunking +
# parsing), and reference-query top-K URLs (for a retrieval-overlap gate).
#
# Usage (from repo root):
#   bash scripts/db/write_db_manifest_v2.sh [OUTPUT]
#
#   OUTPUT: optional path for the manifest JSON. Use .json.gz to compress.
#           default: DB_MANIFEST_V2 from config, else
#           db_manifest_v2_$(hostname -s).json.gz
#
# Configuration: see config.default.sh (uses RUN_DATABASE,
# EMBEDDING_MODEL, RUN_DATASET, DB_MANIFEST_V2).
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

OUTPUT="${1:-${DB_MANIFEST_V2:-db_manifest_v2_$(hostname -s).json.gz}}"

echo "=== Writing v2 behavioral-equivalence manifest ==="
echo "  DB:        ${DB}"
echo "  Embedding model: ${EMBEDDING_MODEL}"
echo "  Dataset:   ${DATASET}"
echo "  Output:    ${OUTPUT}"
echo ""

python3 -u -m ingestion.db_manifest_v2 write \
    --db "${DB}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --dataset "${DATASET}" \
    --output "${OUTPUT}"
