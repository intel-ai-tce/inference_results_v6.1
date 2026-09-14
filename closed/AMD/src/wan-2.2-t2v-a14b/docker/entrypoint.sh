#!/usr/bin/env bash
# Minimal entrypoint for the wan-harness container.
#
# - Ensures the workspace caches exist (and are writable by the container user).
# - Honours `WAN_HARNESS_*` env vars passed in by the launcher.
# - Exec's whatever command the caller provided (defaults to `bash`).
set -euo pipefail

for d in \
    "${HF_HOME:-}" \
    "${HF_HUB_CACHE:-}" \
    "${TRANSFORMERS_CACHE:-}" \
    "${TORCH_HOME:-}" \
    "${TORCH_EXTENSIONS_DIR:-}" ; do
    if [[ -n "${d}" ]]; then
        mkdir -p "${d}" || true
    fi
done

if [[ $# -eq 0 ]]; then
    exec bash
fi

exec "$@"
