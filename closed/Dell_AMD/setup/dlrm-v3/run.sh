#!/usr/bin/env bash
# run.sh — one-shot orchestrator for the certified DLRM-v3 ROCm/gfx950 Server cert.
#
# Chains the four stages, each of which is also runnable on its own:
#   [data]        scripts/build/setup_data.sh        stage/verify dataset + checkpoint
#   [workspace]   scripts/build/setup_workspace.sh   clone the cert port repos + loadgen baseline
#   [submission]  scripts/build/setup_submission.sh  (re)create the container, build fbgemm+pynve, verify
#   [run]         scripts/run/run_gold.sh            launch the GOLD 10-min Server cert (b64 full-causal q12,200)
#
# The dataset (~140 GB) + checkpoint (~964 GB) are fleet-local / out-of-band (no public
# URL). The [data] stage PULLS them only if you point it at a source; otherwise it just
# verifies what is already on the host.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash run.sh                                            # all four stages (perf cert)
#   STAGES=workspace,submission,run bash run.sh            # skip data (already present)
#   DATASET_SRC=<src> CHECKPOINT_SRC=<src> bash run.sh     # pull data if missing, then build+run
#   STAGES=run bash run.sh                                # just re-run the GOLD perf cert
#   STAGES=run WINDOW=1 bash run.sh                        # legacy windowed C1-on path (NOT submission-legal)
#   STAGES=accuracy bash run.sh                            # Offline AccuracyOnly cert (+ GAUC score)
#   STAGES=submission,run,accuracy bash run.sh             # build, then perf + accuracy certs
#
# A 5th stage, [accuracy], runs scripts/run/run_accuracy.sh (Offline AccuracyOnly on the same
# certified stack + GAUC scoring). It is NOT in the default set; add it via STAGES.
#
# ── Env ───────────────────────────────────────────────────────────────────────
#   STAGES          comma list to run            [data,workspace,submission,run] (+ optional 'accuracy')
#   DATASET_SRC     dataset source (local dir | host:path | rclone remote:path)   [pull if set]
#   CHECKPOINT_SRC  checkpoint source (same forms)                                [pull if set]
#   WORKSPACE_HOST  where the trees + data live  [parent of this repo checkout]
#   …plus any env the underlying scripts accept (CONTAINER, MOUNT_ROOT, CONF, FUSE_EPILOGUE,
#     HARNESS_REPO_REF, GR_REPO_REF, PYNVE_REPO_REF, BUILD_JOBS, WAIT, …) — passed through.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES="${STAGES:-data,workspace,submission,run}"

log()  { echo "[run $(date -u +%H:%M:%S)] $*"; }
has()  { [[ ",${STAGES}," == *",$1,"* ]]; }

# Guard a run/accuracy stage: the run needs the data present (the build does not). Fail with
# a clear message rather than launching a job that dies ~10 min in.
require_data() {
  local ds="${WORKSPACE_HOST}/${DATASET_DIR}" ck="${WORKSPACE_HOST}/${CHECKPOINT_DIR}" missing=0
  [[ -d "${ds}" && -n "$(ls -A "${ds}" 2>/dev/null)" ]] || { echo "[run] ERROR: dataset missing/empty: ${ds}"; missing=1; }
  [[ -d "${ck}" && -n "$(ls -A "${ck}" 2>/dev/null)" ]] || { echo "[run] ERROR: checkpoint missing/empty: ${ck}"; missing=1; }
  if [[ "${missing}" -eq 1 ]]; then
    echo "[run] stage the data first:  DATASET_SRC=<src> CHECKPOINT_SRC=<src> STAGES=data bash run.sh"
    exit 1
  fi
}

# Resolve the workspace the same way the stage scripts do, so the pre-run data guard
# checks the right paths.
if [[ -z "${WORKSPACE_HOST:-}" ]]; then
  _RR="$(git -C "${SELF}" rev-parse --show-toplevel 2>/dev/null || true)"
  WORKSPACE_HOST="$(cd "${_RR:-${SELF}}/.." && pwd)"
fi
export WORKSPACE_HOST
DATASET_DIR="${DATASET_DIR:-dlrmv3_preprocessed_full}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-dlrmv3_trained_checkpoint/dlrm-v3-checkpoint}"

log "workspace : ${WORKSPACE_HOST}"
log "stages    : ${STAGES}"

if has data; then
  log "═══ [data] setup_data.sh ═══"
  bash "${SELF}/scripts/build/setup_data.sh"
fi

if has workspace; then
  log "═══ [workspace] setup_workspace.sh ═══"
  bash "${SELF}/scripts/build/setup_workspace.sh"
fi

if has submission; then
  log "═══ [submission] setup_submission.sh (container + fbgemm + pynve build; ~15 min) ═══"
  bash "${SELF}/scripts/build/setup_submission.sh"
fi

if has run; then
  require_data
  log "═══ [run] run_gold.sh (GOLD b64 full-causal Server cert; ~22 min wall) ═══"
  bash "${SELF}/scripts/run/run_gold.sh"
fi

if has accuracy; then
  require_data
  log "═══ [accuracy] run_accuracy.sh (Offline AccuracyOnly + GAUC scoring) ═══"
  bash "${SELF}/scripts/run/run_accuracy.sh"
fi

log "DONE (stages: ${STAGES})"
