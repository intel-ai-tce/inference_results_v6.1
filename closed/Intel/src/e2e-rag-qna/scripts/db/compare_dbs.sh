#!/bin/bash
# =============================================================================
# Directly compare two vector DBs on disk for BEHAVIORAL EQUIVALENCE (no manifest).
#
# Checks that CAND is "the same DB" as REF in the ways that matter: same passage
# count + FAISS index params, same corpus SET (order-independent — same HTML +
# chunking + parsing), and top-K retrieval overlap vs reference queries within a
# tolerance. Numerically different embeddings / different orderings are allowed;
# only the same embedding MODEL, corpus, chunking and index config are required.
#
# Usage (from repo root):
#   bash scripts/db/compare_dbs.sh CAND_DB [REF_DB] [OVERLAP_THRESHOLD]
#
#   CAND_DB: candidate DB to check (required).
#   REF_DB:  reference DB (default: REFERENCE_DATABASE from config).
#   OVERLAP_THRESHOLD: min mean top-K retrieval overlap (default: 0.95).
#
# Configuration: sources config.sh for REFERENCE_DATABASE,
# EMBEDDING_MODEL and RUN_DATASET.
# =============================================================================

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

if [[ -z "$1" ]]; then
    echo "ERROR: candidate DB path required" >&2
    echo "Usage: $0 CAND_DB [REF_DB] [OVERLAP_THRESHOLD]" >&2
    exit 1
fi

CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

EMBEDDING_MODEL="${EMBEDDING_MODEL:-intfloat_e5-base-v2/e5-base-v2}"
DATASET="${RUN_DATASET:-data/frames_dataset.tsv}"

CAND="$1"
REF="${2:-${REFERENCE_DATABASE}}"
OVERLAP_THRESHOLD="${3:-0.95}"

if [[ -z "${REF}" ]]; then
    echo "ERROR: no reference DB (pass REF_DB or set REFERENCE_DATABASE in config)" >&2
    exit 1
fi

echo "=== Comparing DBs (behavioral equivalence) ==="
echo "  REF:               ${REF}"
echo "  CAND:              ${CAND}"
echo "  Embedding model:   ${EMBEDDING_MODEL}"
echo "  Dataset:           ${DATASET}"
echo "  Overlap threshold: ${OVERLAP_THRESHOLD}"
echo ""

python3 -u -m ingestion.db_manifest_v2 compare \
    --ref "${REF}" \
    --db "${CAND}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --dataset "${DATASET}" \
    --overlap-threshold "${OVERLAP_THRESHOLD}"
