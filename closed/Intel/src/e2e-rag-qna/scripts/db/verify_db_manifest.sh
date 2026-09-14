#!/bin/bash
# =============================================================================
# Verify this system's vector DB against a reference manifest.
#
# Usage (from repo root):
#   bash scripts/db/verify_db_manifest.sh [MANIFEST] [COSINE_THRESHOLD] [TOP_K_DEPTH]
#
#   MANIFEST:         path to the reference manifest JSON (default: DB_MANIFEST
#                     from config).
#   COSINE_THRESHOLD: minimum sample-embedding cosine similarity (default: 0.9999).
#   TOP_K_DEPTH:      probe-query top-K rank match depth (default: 3).
#
# Configuration: see config.default.sh (uses RUN_DATABASE, DB_MANIFEST and
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
EMBEDDING_MODEL="${EMBEDDING_MODEL:-}"

MANIFEST="${1:-${DB_MANIFEST}}"
COSINE_THRESHOLD="${2:-0.9999}"
TOP_K_DEPTH="${3:-3}"

if [[ -z "${MANIFEST}" ]]; then
    echo "ERROR: no manifest (pass MANIFEST or set DB_MANIFEST in config)" >&2
    echo "Usage: $0 [MANIFEST] [COSINE_THRESHOLD] [TOP_K_DEPTH]" >&2
    exit 1
fi

# Print the manifest's sha256 so it's provable which manifest was verified against.
if [[ -f "${MANIFEST}" ]]; then
    MANIFEST_SHA="$(sha256sum "${MANIFEST}" | cut -d' ' -f1)"
else
    echo "ERROR: manifest not found: ${MANIFEST}" >&2
    exit 1
fi

echo "=== Verifying DB against manifest ==="
echo "  DB:                ${DB}"
echo "  Manifest:          ${MANIFEST}"
echo "  Manifest sha256:   ${MANIFEST_SHA}"
echo "  Embedding model:   ${EMBEDDING_MODEL:-(from manifest)}"
echo "  Cosine threshold:  ${COSINE_THRESHOLD}"
echo "  Top-K depth:       ${TOP_K_DEPTH}"
echo ""

RM_ARG=()
[ -n "${EMBEDDING_MODEL}" ] && RM_ARG=(--embedding_model "${EMBEDDING_MODEL}")

python3 -u -m ingestion.db_manifest verify \
    --db "${DB}" \
    --manifest "${MANIFEST}" \
    "${RM_ARG[@]}" \
    --cosine-threshold "${COSINE_THRESHOLD}" \
    --top-k-depth "${TOP_K_DEPTH}"
