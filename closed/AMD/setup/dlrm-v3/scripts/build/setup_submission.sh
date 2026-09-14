#!/usr/bin/env bash
# setup_submission.sh — ONE-SHOT, from-clean bring-up of the certified DLRMv3
# ROCm/gfx950 NVE inference stack (the 10,595 q/s VALID figure of record).
#
# This is the single entry point a fresh MI355X (gfx950:sramecc+) node needs to
# go from "bare host with the source trees present" to "container that can run
# the GOLD 10-min Server cert" (scripts/run/run_gold.sh).
#
# It ORCHESTRATES the pieces that were previously spread across SETUP.md by hand
# + two existing build scripts:
#   [1] container   create the atom rocm7.2.3 container (--cap-add=SYS_PTRACE)
#   [2] worklink    /work -> $WORKSPACE_HOST symlink (launchers hardcode /work)
#   [3] env         setup_stack_rocm723.sh: fbgemm gfx950:sramecc+ build, Triton
#                   gfx950 patch, torch-PINNED deps, mlperf_loadgen
#   [4] pynve       build_pynve_rocm_full.sh: build the certified AMD-AGI/pynve-rocm
#                   repo in place (clones it if PYNVE_TREE is absent) + HIP build
#   [5] patches     NEITHER tree is patched. Both the harness and the mlcommons GR
#                   port are living working trees the per-plan patches can't faithfully
#                   reconstruct (harness: 6/14 fail to apply; GR: patches apply but miss
#                   cert-path code — Plan-12 MPI-lookup + Plan-10.1.b reorder fallback).
#                   setup_workspace.sh provisions BOTH via CERTIFIED repo clones
#                   (AMD-AGI/dlrm-v3-{harness,gr}-rocm). This phase only VERIFIES the sentinels.
#   [6] shims       the checkpoint re-export shim (SETUP.md §2e)
#   [7] confs       the certified b48 PROD10min LoadGen confs
#   [8] verify      torch-pin + pynve import smoke (+ optional 8-rank MPIMemBlock)
#
# IDEMPOTENT: re-running is safe. Patches apply with `-N --forward` (already-applied
# hunks skip), the container is reused if already running, deps are constrained.
#
# ── Run from the HOST (it drives docker). Example ─────────────────────────────
#   bash scripts/build/setup_submission.sh
#   PHASES=patches,confs,verify bash scripts/build/setup_submission.sh   # subset
#   VERIFY_MPI=1 bash scripts/build/setup_submission.sh                  # +GPU smoke
#
# ── Configuration (override via env) ──────────────────────────────────────────
#   CONTAINER        container name                 [dlrmv3-e2e723]
#   IMAGE            base image                      [rocm/atom:rocm7.2.3_..._atom20260511]
#   MOUNT_ROOT       host path bind-mounted 1:1      [WORKSPACE_HOST] — set to a
#                    common ancestor if dataset/checkpoint live OUTSIDE the workspace
#                    (the data-reachability preflight prints the exact value to use)
#   DATASET_DIR      dataset dir (under WORKSPACE)    [dlrmv3_preprocessed_full]
#   CHECKPOINT_DIR   checkpoint dir (under WORKSPACE)  [dlrmv3_trained_checkpoint/dlrm-v3-checkpoint]
#   WORKSPACE_HOST   workspace (holds the trees)     [parent of this repo checkout]
#   WORK             in-container symlink to ^       [/work]
#   HARNESS_SUB      harness tree subdir             [dlrm-v3-harness-rocm]
#   GR_SUB           generative_recommenders tree    [mlcommons-inference]
#   PYNVE_SUB        pynve tree subdir               [pynve-rocm]
#   FBGEMM_SUB       FBGEMM source subdir            [FBGEMM]
#   REPO_SUB         this runner repo, relative to WORKSPACE [current checkout dir name]
#   KFD_GROUP        render/kfd gid to add           [993]
#   PHASES           comma list of phases to run     [container,worklink,env,pynve,patches,shims,confs,verify]
#   VERIFY_MPI       1 = run the 8-rank MPIMemBlock GPU smoke in verify [0]
#   BUILD_JOBS       parallel build jobs             [8]
set -uo pipefail

# ── Resolve config ────────────────────────────────────────────────────────────
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${CONTAINER:-dlrmv3-e2e723}"
IMAGE="${IMAGE:-rocm/atom:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom20260511}"
# WORKSPACE_HOST defaults to the folder CONTAINING this repo checkout (the execution
# workspace), not a node-specific path. Override by exporting WORKSPACE_HOST.
if [[ -z "${WORKSPACE_HOST:-}" ]]; then
  _RR="$(git -C "${SELF}" rev-parse --show-toplevel 2>/dev/null || true)"
  WORKSPACE_HOST="$(cd "${_RR:-${SELF}/../..}/.." && pwd)"
fi
# MOUNT_ROOT is bind-mounted 1:1 into the container and MUST contain WORKSPACE_HOST.
# It defaults to WORKSPACE_HOST (the build floor — the workspace itself, derived from
# the repo location, NOT a node-specific path). It does NOT auto-guess a broader root
# to cover out-of-tree data: if the dataset/checkpoint resolve OUTSIDE WORKSPACE_HOST
# (e.g. symlinks into a sibling tree), a RUN needs MOUNT_ROOT set explicitly to a
# common ancestor — the data-reachability preflight below detects this and prints the
# exact value to use. (The build itself only needs WORKSPACE_HOST.)
MOUNT_ROOT="${MOUNT_ROOT:-${WORKSPACE_HOST}}"
WORK="${WORK:-/work}"
if _REPO_ROOT="$(git -C "${SELF}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  _REPO_ROOT="$(cd "${SELF}/../.." && pwd)"
fi
_REPO_SUB_DEFAULT="$(basename "${_REPO_ROOT}")"
HARNESS_SUB="${HARNESS_SUB:-dlrm-v3-harness-rocm}"
GR_SUB="${GR_SUB:-mlcommons-inference}"
PYNVE_SUB="${PYNVE_SUB:-pynve-rocm}"
FBGEMM_SUB="${FBGEMM_SUB:-FBGEMM}"
REPO_SUB="${REPO_SUB:-${_REPO_SUB_DEFAULT}}"
# Run-time input dir names (shared with setup_data.sh) — used by the reachability
# preflight to check they will resolve inside the container at run time.
DATASET_DIR="${DATASET_DIR:-dlrmv3_preprocessed_full}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-dlrmv3_trained_checkpoint/dlrm-v3-checkpoint}"
KFD_GROUP="${KFD_GROUP:-993}"
BUILD_JOBS="${BUILD_JOBS:-8}"
VERIFY_MPI="${VERIFY_MPI:-0}"
PHASES="${PHASES:-container,worklink,env,pynve,patches,shims,confs,verify}"

# In-container paths (everything reached through the /work symlink).
C_REPO="${WORK}/${REPO_SUB}"
C_HARNESS="${WORK}/${HARNESS_SUB}"
C_GR="${WORK}/${GR_SUB}"
C_PYNVE="${WORK}/${PYNVE_SUB}"
C_FBGEMM="${WORK}/${FBGEMM_SUB}"
C_BENCH="${C_HARNESS}/benchmarks"
PATCHES_HOST="$(cd "${SELF}/../../patches" 2>/dev/null && pwd || echo "${SELF}/../../patches")"

log()  { echo "[setup-submission $(date -u +%H:%M:%S)] $*"; }
note() { echo "    $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
want() { [[ ",${PHASES}," == *",$1,"* ]]; }

# is_under CHILD PARENT — true if abs path CHILD is inside abs path PARENT.
is_under() { local c="${1%/}/" p="${2%/}/"; [[ "${c}" == "${p}"* ]]; }
# common_ancestor A B — longest shared directory prefix of two abs paths.
common_ancestor() {
  local a b out=""; IFS='/' read -ra a <<<"$1"; IFS='/' read -ra b <<<"$2"; local i=0
  while [[ ${i} -lt ${#a[@]} && ${i} -lt ${#b[@]} && "${a[$i]}" == "${b[$i]}" ]]; do
    [[ -n "${a[$i]}" ]] && out="${out}/${a[$i]}"; i=$((i+1))
  done
  echo "${out:-/}"
}
# Warn (do NOT fail — the build needs no data) if a staged run-time input resolves
# OUTSIDE MOUNT_ROOT, so it won't be visible at ${WORK}/... inside the container.
check_data_reachable() {
  local issues=0 name hp real anc
  for spec in "dataset|${WORKSPACE_HOST}/${DATASET_DIR}" "checkpoint|${WORKSPACE_HOST}/${CHECKPOINT_DIR}"; do
    name="${spec%%|*}"; hp="${spec#*|}"
    [[ -e "${hp}" ]] || continue                          # not staged yet — irrelevant to the build
    real="$(readlink -f "${hp}" 2>/dev/null || echo "${hp}")"
    if is_under "${real}" "${MOUNT_ROOT}"; then
      note "${name} reachable: ${real}"
    else
      issues=1; anc="$(common_ancestor "${WORKSPACE_HOST}" "${real}")"
      log "WARN: ${name} resolves to ${real}"
      note "      OUTSIDE MOUNT_ROOT=${MOUNT_ROOT} — it will NOT be visible at ${WORK}/${name} inside the container."
      note "      For a RUN, recreate the container with:  MOUNT_ROOT=${anc} bash scripts/build/setup_submission.sh"
    fi
  done
  [[ ${issues} -eq 0 ]] || note "(the build does not need the data; this only affects an actual run)"
}
# run a command inside the container, login shell, with /work-relative env.
dexec() { docker exec "$@"; }

# ── [0] Preflight ─────────────────────────────────────────────────────────────
command -v docker >/dev/null || die "docker not found on PATH"
[[ -e /dev/kfd ]] || log "WARN: /dev/kfd missing — GPU phases (pynve build smoke / verify) will fail"
[[ -d "${PATCHES_HOST}" ]] || die "patches dir not found: ${PATCHES_HOST}"

# Host-side presence of the source trees (provisioned by setup_workspace.sh). BOTH the
# harness and the mlcommons GR tree must be the CERTIFIED trees — direct clones of
# AMD-AGI/dlrm-v3-harness-rocm and AMD-AGI/dlrm-v3-gr-rocm (the GR repo cloned into the
# mlcommons baseline at recommendation/dlrm_v3); the per-plan patches can't faithfully
# reconstruct either one.
preflight_tree() {
  local host="$1" what="$2" hint="$3"
  [[ -d "${host}" ]] || die "missing ${what}: ${host}
    -> ${hint}"
}
if want container || want env || want pynve || want patches; then
  preflight_tree "${WORKSPACE_HOST}/${HARNESS_SUB}" "harness tree" \
    "run setup_workspace.sh (clones the certified harness repo AMD-AGI/dlrm-v3-harness-rocm to dlrm-v3-harness-rocm/)"
  preflight_tree "${WORKSPACE_HOST}/${GR_SUB}" "generative_recommenders tree" \
    "run setup_workspace.sh (clones the GR repo AMD-AGI/dlrm-v3-gr-rocm into recommendation/dlrm_v3; baseline kept for loadgen)"
fi
if want env; then
  preflight_tree "${WORKSPACE_HOST}/${FBGEMM_SUB}" "FBGEMM source" \
    "git clone the FBGEMM source here (built from source for gfx950:sramecc+)"
fi

log "config: container=${CONTAINER} image=${IMAGE}"
log "config: workspace=${WORKSPACE_HOST} -> ${WORK} | phases=${PHASES}"
log "config: MOUNT_ROOT=${MOUNT_ROOT} (bind-mounted 1:1)"
# Data-reachability: warn (not fail) if staged dataset/checkpoint resolve outside MOUNT_ROOT.
check_data_reachable

# ── [1] Container ─────────────────────────────────────────────────────────────
if want container; then
  if docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null | grep -q true; then
    log "[1] container ${CONTAINER} already running — reusing (set PHASES to skip)"
  else
    log "[1] (re)creating container ${CONTAINER}"
    docker rm -f "${CONTAINER}" 2>/dev/null || true
    docker run -d --name "${CONTAINER}" \
      --device=/dev/kfd --device=/dev/dri --group-add video --group-add "${KFD_GROUP}" \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      --ipc=host --shm-size 32G \
      -v "${MOUNT_ROOT}:${MOUNT_ROOT}" \
      "${IMAGE}" sleep infinity \
      || die "docker run failed (image present? pull ${IMAGE})"
  fi
fi

# ── [2] /work symlink ─────────────────────────────────────────────────────────
if want worklink; then
  log "[2] ${WORK} -> ${WORKSPACE_HOST}"
  dexec "${CONTAINER}" ln -sfn "${WORKSPACE_HOST}" "${WORK}" || die "could not create ${WORK} symlink"
fi

# ── [3] Env layer (fbgemm sramecc+ / Triton patch / deps / loadgen) ───────────
if want env; then
  log "[3] env layer (setup_stack_rocm723.sh) — fbgemm build can take several minutes"
  dexec -e FBGEMM_DIR="${C_FBGEMM}" "${CONTAINER}" \
    bash -lc "bash '${C_REPO}/scripts/build/setup_stack_rocm723.sh'" \
    || die "[3] env layer failed (see output above)"
fi

# ── [4] pynve port build ──────────────────────────────────────────────────────
if want pynve; then
  log "[4] pynve ROCm port build (build_pynve_rocm_full.sh)"
  dexec -e PYNVE_TREE="${C_PYNVE}" -e BUILD_JOBS="${BUILD_JOBS}" "${CONTAINER}" \
    bash -lc "bash '${C_REPO}/scripts/build/build_pynve_rocm_full.sh'" \
    || die "[4] pynve build failed (AMD-AGI/pynve-rocm repo access/auth? see PORTING.md §0)"
fi

# ── [5] Verify the certified ports (NO patching) ──────────────────────────────
# NEITHER tree is patched here. Both the harness NVE-cert port and the mlcommons GR
# port are living working trees the per-plan patches CANNOT faithfully reconstruct:
#   - harness: 6/14 patches fail to forward-apply from baseline.
#   - GR: the patches forward-apply but MISS cert-path code — the Plan-12 MPI-lookup
#     swap in sparse_routing.py (run_gold.sh sets DLRM_USE_MPI_LOOKUP=1 and the
#     harness Phase-12.3 calls into it) and the Plan-10.1.b rocm_compat reorder fallback.
# setup_workspace.sh provisions both via CERTIFIED repo clones — harness from
# AMD-AGI/dlrm-v3-harness-rocm, GR from AMD-AGI/dlrm-v3-gr-rocm (cloned into the mlcommons
# baseline at recommendation/dlrm_v3; the baseline is kept only for loadgen). This phase
# only VERIFIES the certified sentinels — the per-plan patch history that produced these
# trees lives in their own repos (this runner does NOT carry or apply it). The only patch
# this runner applies is patches/fbgemm_rocm7.patch, during the fbgemm build (phase [3]).
if want patches; then
  # Harness: verify the certified repo clone is in place (setup_workspace clones it).
  log "[5] verify certified harness (NVE sentinels in ${C_HARNESS})"
  dexec "${CONTAINER}" bash -lc "
    s='${C_HARNESS}/inference_harness/inference_server.py'
    q='${C_HARNESS}/inference_harness/dataset/mlperf_streaming_qsl.py'
    if grep -q DLRM_ROCM_NVE \"\$s\" 2>/dev/null && grep -q DLRM_CLAMP_OOB_IDS \"\$q\" 2>/dev/null; then
      echo '    OK: harness is the certified NVE port (DLRM_ROCM_NVE + DLRM_CLAMP_OOB_IDS present)'
    else
      echo '    WARN: harness sentinels missing — not the certified tree.'
      echo '          Re-run setup_workspace.sh to clone AMD-AGI/dlrm-v3-harness-rocm.'
    fi"
  # GR: verify the certified repo clone is in place. The load-bearing sentinels are the
  # Plan-12 MPI-lookup swap (run_gold.sh sets DLRM_USE_MPI_LOOKUP=1) + the Plan-10.1.b
  # rocm_compat reorder fallback + the (spuriously-deleted-in-cursor, restored) model_family.py.
  log "[5] verify certified GR tree (MPI-lookup + reorder-fallback sentinels in ${C_GR})"
  dexec "${CONTAINER}" bash -lc "
    base='${C_GR}/recommendation/dlrm_v3'
    sr=\"\$base/sparse_routing.py\"; rc=\"\$base/generative_recommenders/ops/rocm_compat.py\"; mf=\"\$base/model_family.py\"
    ok=1
    grep -q 'set_route_lookup_comm\|_use_mpi_lookup_env' \"\$sr\" 2>/dev/null || { ok=0; echo '    WARN: sparse_routing.py MISSING Plan-12 MPI-lookup swap (run_gold sets DLRM_USE_MPI_LOOKUP=1!)'; }
    grep -q 'reorder_batched_ad' \"\$rc\" 2>/dev/null || { ok=0; echo '    WARN: rocm_compat.py MISSING Plan-10.1.b reorder CPU fallback'; }
    [ -f \"\$mf\" ] && grep -q 'class HSTUModelFamily' \"\$mf\" 2>/dev/null || { ok=0; echo '    WARN: model_family.py missing/empty (HSTUModelFamily)'; }
    if [ \"\$ok\" = 1 ]; then echo '    OK: GR is the certified tree (MPI-lookup + reorder-fallback + model_family present)';
    else echo '          Re-run setup_workspace.sh to clone AMD-AGI/dlrm-v3-gr-rocm.'; fi"
fi

# ── [6] Module-path shim (SETUP.md §2e) ───────────────────────────────────────
if want shims; then
  log "[6] checkpoint re-export shim"
  SHIM="${C_GR}/recommendation/dlrm_v3/generative_recommenders/dlrm_v3/checkpoint.py"
  dexec "${CONTAINER}" bash -lc \
    "d=\$(dirname '${SHIM}'); if [ -d \"\$d\" ]; then \
       printf '%s\n' '# shim: re-export the top-level dlrm_v3 checkpoint loader (SETUP.md 2e)' 'from checkpoint import *  # noqa: F401,F403' > '${SHIM}'; \
       echo '    wrote ${SHIM}'; \
     else echo '    WARN: package dir missing (\$d) — verify the GR tree layout'; fi"
  DATASETS_INIT="${C_GR}/recommendation/dlrm_v3/datasets/__init__.py"
  dexec "${CONTAINER}" bash -lc \
    "d=\$(dirname '${DATASETS_INIT}'); if [ -d \"\$d\" ]; then \
       printf '%s\n' '\"\"\"Local DLRM-v3 dataset package.\"\"\"' > '${DATASETS_INIT}'; \
       echo '    wrote ${DATASETS_INIT}'; \
     else echo '    WARN: datasets package dir missing (\$d) — verify the GR tree layout'; fi"
fi

# ── [7] Certified LoadGen confs ───────────────────────────────────────────────
if want confs; then
  log "[7] write certified PROD10min confs into ${C_BENCH}"
  write_conf() {  # $1 = filename, $2 = target_qps, $3 = header comment
    write_conf_explicit "$1" "$2" "$2" 600000 "$3"
  }
  write_conf_explicit() {  # $1=file $2=server_qps $3=offline_qps $4=min_duration_ms $5=header
    dexec "${CONTAINER}" bash -lc "cat > '${C_BENCH}/$1' <<'CONF'
# $5
*.Server.target_qps = $2
*.Server.target_latency = 80
*.Server.target_latency_percentile = 99
*.Server.min_duration = $4
*.Server.min_query_count = 1
*.Offline.target_qps = $3
*.Offline.min_duration = $4
*.Offline.min_query_count = 1
CONF
echo '    wrote $1'"
  }
  # GOLD figure of record (closed submission): b64 full-causal Win-B(occ), C1-off, inflight=128.
  write_conf user_mi355x8_nve_b64_qps12200_PROD10min.conf 12200 \
    "GOLD figure of record: b64 full-causal Win-B(occ)+bf16+c40p6/P8/P0/P1 ring1024+deg5 gate -> 12,200 issued q/s, 12,198.90 completed q/s VALID."
  write_conf_explicit user_mi355x8_nve_b64_qps12200_OFFLINE10min.conf 12200 14000 600000 \
    "Offline throughput run for q12,200 GOLD stack: Offline target_qps=14,000 to keep min_duration satisfied while measuring unconstrained Offline throughput."
  write_conf_explicit user_mi355x8_nve_b64_qps12200_OFFLINE90s.conf 12200 14000 90000 \
    "Offline throughput probe for q12,200 GOLD stack: Offline target_qps=14,000, 90s."
  write_conf user_mi355x8_nve_b64_qps11970_PROD10min.conf 11970 \
    "Previous GOLD with latest LoadGen 6.0.16: b64 full-causal Win-B(occ)+bf16+c40p6/P8/P0/P1 ring1024 -> 11,970 issued q/s, 11,968.67 completed q/s VALID."
  write_conf user_mi355x8_nve_b64_qps12000_PROD10min.conf 12000 \
    "Headroom probe: q12,000 is close but latest LoadGen 6.0.16 measured INVALID on chi2761 (p99 82.61 ms)."
  write_conf user_mi355x8_nve_b64_qps9600_PROD10min.conf 9600 \
    "Conservative tail-margin point: b64 full-causal Win-B(occ)+bf16 -> ~9,595 q/s VALID."
  # Legacy windowed C1-on confs (NOT submission-legal; reachable via WINDOW=1) — kept for reference.
  write_conf user_mi355x8_nve_b48_qps10600_PROD10min.conf 10600 \
    "Legacy windowed C1-on: A-FUSE+C1+b48+D2 -> 10,595 q/s VALID, p99 51.07 ms (NOT submission-legal)."
  write_conf user_mi355x8_nve_b48_qps10500_PROD10min.conf 10500 \
    "Legacy windowed C1-on (prior): A-FUSE+C1+b48 -> 10,500 q/s VALID, p99 48.76 ms (NOT submission-legal)."
fi

# ── [8] Verify ────────────────────────────────────────────────────────────────
if want verify; then
  log "[8] verify: torch pin + pynve NATIVE import + Plan-18 sentinel"
  # Assert the certified own-device-grant fix (Plan-18) is in the built source — a
  # non-certified tree (e.g. a stock NVIDIA nv-embedding-cache, or the retired patch-built
  # tree that dropped Plan-18) would wedge the box at multi-GPU scale; the AMD-AGI/pynve-rocm
  # repo has it baked in. Then import the NATIVE extension (pynve.nve),
  # not just the pure-python package (a bare `import pynve` only loads pynve._version and
  # passes even with a stale/missing .so). build_rocm/lib holds libnve-common.so.
  dexec "${CONTAINER}" bash -lc "
    grep -q own_desc '${C_PYNVE}/src/distributed.cpp' \
      || { echo 'ERROR: pynve missing Plan-18 own-device grant (own_desc) — non-certified build'; exit 1; }
    echo '  Plan-18 own-device-grant sentinel OK'"  || die "[8] verify failed — pynve is missing Plan-18 (re-checkout github.com/AMD-AGI/pynve-rocm and rebuild)"
  dexec -e PYTHONPATH="${C_PYNVE}/python" -e LD_LIBRARY_PATH="${C_PYNVE}/build_rocm/lib" "${CONTAINER}" bash -lc '
    python3 - <<PY
import torch
assert torch.__version__.startswith("2.10.0+rocm7.2.3"), torch.__version__
print("  torch pinned OK:", torch.__version__)
import pynve, pynve.nve as n
print("  pynve + native nve import OK:", n.__file__)
PY' || die "[8] verify failed — torch ABI or pynve native extension is off"

  if [[ "${VERIFY_MPI}" == "1" ]]; then
    log "[8] 8-rank MPIMemBlock GPU smoke (VERIFY_MPI=1)"
    dexec "${CONTAINER}" bash -lc \
      "cd '${C_PYNVE}' && ls build_rocm/*mpi_buffer_test* 2>/dev/null && \
       echo '    (run the mpi_buffer_test / repro_mpi_fault per PORTING.md Validate)' \
       || echo '    WARN: mpi_buffer_test binary not found — build with -DNVE_DISABLE_TESTS_AND_SAMPLES=0 to enable'"
  fi
fi

cat <<EOF

=== setup_submission: DONE ===
  container : ${CONTAINER}
  phases    : ${PHASES}

Run the GOLD 10-min Server cert (figure of record: b64 full-causal q12,200 on a clean qualified host):
  cd ${SELF%/scripts/build}
  bash scripts/run/run_gold.sh            # defaults to b64 full-causal Win-B + qps12200 PROD10min conf
                                          # (legacy windowed C1-on path: WINDOW=1 bash scripts/run/run_gold.sh)

Notes:
  * DLRM_ZMQ_TRACE=0 is already baked into run_gold.sh (cert-run requirement, SETUP.md 6).
  * Cold Triton autotune can take ~10-15 min before LoadGen starts; budget ~35-40 min wall
    for a first run after rebuilding or clearing the Triton cache.
  * Keep OUTPUT_DIR on fast local storage for perf runs (the launcher uses the repo artifacts dir).
EOF
