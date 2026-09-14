#!/usr/bin/env bash
# setup_data.sh — REFERENCE prerequisite step: stage the MLPerf DLRM-v3 PREPROCESSED
# DATASET and the ~1 TB TRAINED CHECKPOINT into the canonical host paths that the
# launchers (run_gold.sh / run_nve_parallel_ckpt.sh, SETUP.md §6) point at.
#
# This is the OTHER "phase 0" that setup_submission.sh deliberately does NOT do.
# setup_submission.sh only readies the *software* (builds + patches the source trees);
# it never fetches or even checks the dataset/checkpoint. setup_workspace.sh fetches
# the 3rd-party *source trees*. THIS script handles the *data*.
#
# Like setup_workspace.sh it is intentionally NOT airtight: the dataset and the ~1 TB
# checkpoint are **fleet-local, obtained out-of-band** (PORTING.md §6 — "out of scope
# for this repo"), so there is no public URL baked in. You point the script at wherever
# your copy lives (local dir, another host, or an rclone remote) and it places it at the
# canonical path + verifies the layout. With no source set it just preflights + prints
# exactly where the two artifacts must go.
#
# ── Run on the HOST (the data lives on the host; the container sees it via /work). ────
#   # already-staged node: just verify the canonical paths + print the launcher mapping
#   bash scripts/build/setup_data.sh
#
#   # stage from a local staging dir / another mount:
#   DATASET_SRC=/staging/dlrmv3_preprocessed_full \
#   CHECKPOINT_SRC=/staging/dlrm-v3-checkpoint \
#     bash scripts/build/setup_data.sh
#
#   # stage from an rclone remote (form "remote:path", needs rclone configured):
#   DATASET_SRC=myremote:mlperf/dlrmv3_preprocessed_full \
#   CHECKPOINT_SRC=myremote:mlperf/dlrm-v3-checkpoint \
#     bash scripts/build/setup_data.sh
#
# ── Configuration (override via env) ──────────────────────────────────────────
#   WORKSPACE_HOST   where data lives (== /work in container)  [parent of this repo checkout]
#   DATASET_DIR      dataset dir name under WORKSPACE_HOST      [dlrmv3_preprocessed_full]
#   CHECKPOINT_DIR   checkpoint dir (RELATIVE, may be nested)   [dlrmv3_trained_checkpoint/dlrm-v3-checkpoint]
#   DATASET_SRC      where to copy the dataset FROM (optional; local path, host:path, or remote:path)
#   CHECKPOINT_SRC   where to copy the checkpoint FROM (optional; same forms)
#   FETCH            override the copy tool: "rsync" | "cp" | "rclone" (auto-detected by default)
set -uo pipefail

# WORKSPACE_HOST defaults to the folder CONTAINING this repo checkout (the execution
# workspace) — not a node-specific path. Override by exporting WORKSPACE_HOST.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${WORKSPACE_HOST:-}" ]]; then
  _RR="$(git -C "${SELF}" rev-parse --show-toplevel 2>/dev/null || true)"
  WORKSPACE_HOST="$(cd "${_RR:-${SELF}/../..}/.." && pwd)"
fi
DATASET_DIR="${DATASET_DIR:-dlrmv3_preprocessed_full}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-dlrmv3_trained_checkpoint/dlrm-v3-checkpoint}"
DATASET_SRC="${DATASET_SRC:-}"
CHECKPOINT_SRC="${CHECKPOINT_SRC:-}"
FETCH="${FETCH:-}"

DATASET_DST="${WORKSPACE_HOST}/${DATASET_DIR}"
CHECKPOINT_DST="${WORKSPACE_HOST}/${CHECKPOINT_DIR}"

log()  { echo "[setup-data $(date -u +%H:%M:%S)] $*"; }
note() { echo "    $*"; }

# Pick a copy tool for a given SRC form. "remote:path" (rclone) is detected by a ':'
# that is NOT a local path and NOT a "host:/abs" rsync target.
fetch_into() {  # $1 = SRC, $2 = DST
  local src="$1" dst="$2" tool="${FETCH}"
  if [[ -z "${tool}" ]]; then
    if [[ "${src}" == *:* && ! -e "${src%%:*}" && "${src}" != *:/* ]] && command -v rclone >/dev/null; then
      tool="rclone"
    elif command -v rsync >/dev/null; then
      tool="rsync"
    else
      tool="cp"
    fi
  fi
  note "fetch (${tool}): ${src} -> ${dst}"
  case "${tool}" in
    rclone) mkdir -p "${dst}"; rclone copy --progress "${src}" "${dst}" ;;
    rsync)  mkdir -p "${dst}"; rsync -a --info=progress2 "${src%/}/" "${dst}/" ;;
    cp)     mkdir -p "${dst}"; cp -a "${src%/}/." "${dst}/" ;;
    *) echo "ERROR: unknown FETCH=${tool}" >&2; return 1 ;;
  esac
}

# Stage one artifact: skip if present & no SRC; fetch if SRC set; warn if missing & no SRC.
stage() {  # $1 = label, $2 = SRC, $3 = DST
  local label="$1" src="$2" dst="$3"
  log "=== ${label} -> ${dst} ==="
  if [[ -n "${src}" ]]; then
    fetch_into "${src}" "${dst}" || { note "WARN: fetch failed — stage ${label} manually into ${dst}"; return; }
  elif [[ -d "${dst}" ]] && [[ -n "$(ls -A "${dst}" 2>/dev/null)" ]]; then
    note "present & non-empty — reusing (set ${label^^}_SRC to re-stage)"
  else
    note "MISSING and no source given."
    note "  -> place your ${label} at: ${dst}"
    note "  -> or re-run with ${label^^}_SRC=<local-dir | host:path | rclone-remote:path>"
    note "  -> these are fleet-local / out-of-band (PORTING.md §6); this repo ships no URL."
  fi
}

command -v git >/dev/null 2>&1 || true   # not required; here only for parity w/ setup_workspace
mkdir -p "${WORKSPACE_HOST}"
log "workspace: ${WORKSPACE_HOST}  (== /work inside the container)"

stage "dataset"    "${DATASET_SRC}"    "${DATASET_DST}"
stage "checkpoint" "${CHECKPOINT_SRC}" "${CHECKPOINT_DST}"

# ── Verify the layout the launchers / Plan 20 loader expect ───────────────────
log "verify layout"
DATA_OK=0; CKPT_OK=0
if [[ -d "${DATASET_DST}" ]] && [[ -n "$(ls -A "${DATASET_DST}" 2>/dev/null)" ]]; then
  note "dataset OK: $(find "${DATASET_DST}" -maxdepth 1 -type f | wc -l) top-level files in ${DATASET_DST}"
  DATA_OK=1
else
  note "dataset NOT staged (empty/missing): ${DATASET_DST}"
fi
if [[ -d "${CHECKPOINT_DST}" ]]; then
  n_distcp=$(find "${CHECKPOINT_DST}" -maxdepth 1 -name '*.distcp' 2>/dev/null | wc -l)
  if [[ "${n_distcp}" -gt 0 ]]; then
    note "checkpoint OK: ${n_distcp} *.distcp shard(s) (Plan 20 reads __{rank}_0.distcp)"
    [[ -f "${CHECKPOINT_DST}/.metadata" ]] && note "  .metadata present (DCP)" || note "  WARN: no .metadata — verify this is a DCP-format checkpoint"
    CKPT_OK=1
  else
    note "WARN: no *.distcp shards under ${CHECKPOINT_DST} — verify this is the sharded sparse checkpoint dir"
  fi
else
  note "checkpoint NOT staged (missing): ${CHECKPOINT_DST}"
fi

cat <<EOF

=== setup_data: DONE ===
  dataset    : ${DATASET_DST}   $([[ ${DATA_OK} -eq 1 ]] && echo OK || echo MISSING)
  checkpoint : ${CHECKPOINT_DST}   $([[ ${CKPT_OK} -eq 1 ]] && echo OK || echo MISSING)

Inside the container these map (via the /work symlink) to the launcher env:
  DATASET_PATH    = /work/${DATASET_DIR}
  CHECKPOINT_PATH = /work/${CHECKPOINT_DIR}
(run_gold.sh / run_nve_parallel_ckpt.sh already export exactly these — see SETUP.md §6.)

Next:  bash scripts/build/setup_submission.sh   # readies the software (separate concern)
EOF
