#!/bin/bash
# =============================================================================
# Verify this system's vector DB against a v2 BEHAVIORAL-EQUIVALENCE manifest.
#
# Checks the DB is "the same DB" as the reference without requiring byte
# identity: passage count + embedding dim, FAISS index params/algorithm, an
# order-INDEPENDENT corpus-set hash (same HTML + chunking + parsing), and top-K
# retrieval overlap vs reference queries within a tolerance. Numerically
# different embeddings / different orderings are allowed.
#
# Usage (from repo root):
#   bash scripts/db/verify_db_manifest_v2.sh [MANIFEST] [OVERLAP_THRESHOLD]
#
#   MANIFEST:          v2 manifest JSON (default: DB_MANIFEST_V2 from config).
#   OVERLAP_THRESHOLD: min mean top-K retrieval overlap (default: 0.95).
#
# Configuration: see config.default.sh (uses RUN_DATABASE,
# EMBEDDING_MODEL, DB_MANIFEST_V2).
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
EMBEDDING_MODEL="${EMBEDDING_MODEL:-}"

MANIFEST="${1:-${DB_MANIFEST_V2}}"
OVERLAP_THRESHOLD="${2:-0.95}"

if [[ -z "${MANIFEST}" ]]; then
    echo "ERROR: no manifest (pass MANIFEST or set DB_MANIFEST_V2 in config)" >&2
    echo "Usage: $0 [MANIFEST] [OVERLAP_THRESHOLD]" >&2
    exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: manifest not found: ${MANIFEST}" >&2
    exit 1
fi

# Print the manifest's sha256 so it's provable which manifest was verified against.
MANIFEST_SHA="$(sha256sum "${MANIFEST}" | cut -d' ' -f1)"

echo "=== Verifying DB against v2 behavioral manifest ==="
echo "  DB:                ${DB}"
echo "  Manifest:          ${MANIFEST}"
echo "  Manifest sha256:   ${MANIFEST_SHA}"
echo "  Embedding model:   ${EMBEDDING_MODEL:-(from manifest)}"
echo "  Overlap threshold: ${OVERLAP_THRESHOLD}"
echo ""

RM_ARG=()
[ -n "${EMBEDDING_MODEL}" ] && RM_ARG=(--embedding_model "${EMBEDDING_MODEL}")

python3 -u -m ingestion.db_manifest_v2 verify \
    --db "${DB}" \
    --manifest "${MANIFEST}" \
    "${RM_ARG[@]}" \
    --overlap-threshold "${OVERLAP_THRESHOLD}"
