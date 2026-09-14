#!/usr/bin/env bash
# setup_workspace.sh — REFERENCE prerequisite step: obtain the 3rd-party source
# trees at their PINNED commits, on the host, before running setup_submission.sh.
#
# This is the "phase 0" that setup_submission.sh deliberately does NOT do (it only
# preflights the trees' presence). The certified port repos are PRIVATE, but release
# zips also vendor source snapshots under vendor/. If a private clone is unavailable,
# this script copies the vendored snapshot into the workspace automatically.
#
# After this completes, the trees live under WORKSPACE_HOST and you can run:
#   bash scripts/build/setup_submission.sh
#
# The three certified ROCm/gfx950 ports each live in their OWN private repo (the
# per-plan patch history that produced them lives in those repos). We clone them DIRECTLY:
#  * HARNESS — github.com/AMD-AGI/dlrm-v3-harness-rocm, cloned as the dlrm-v3-harness-rocm/
#    tree (repo contents at the root; launchers cd into its benchmarks/).
#  * GR      — github.com/AMD-AGI/dlrm-v3-gr-rocm, the certified recommendation/dlrm_v3,
#    cloned INTO the mlcommons/inference baseline at recommendation/dlrm_v3 (the hard-coded
#    PYTHONPATH target). The baseline is still cloned — slimmed to a sparse checkout of
#    loadgen/ — because loadgen lives there and is pip-installed at runtime.
#  * PYNVE   — github.com/AMD-AGI/pynve-rocm, built in place by build_pynve_rocm_full.sh.
# All of {harness,gr,pynve} repos are PRIVATE: cloning needs git/gh auth. setup_submission.sh
# does NOT patch any tree — it only VERIFIES the certified sentinels.
#
# ── Configuration (override via env) ──────────────────────────────────────────
#   WORKSPACE_HOST       where the trees live   [parent of this repo checkout]
#   HARNESS_REPO_REMOTE / HARNESS_REPO_REF  certified harness repo + pinned commit
#   GR_REPO_REMOTE      / GR_REPO_REF       certified GR repo + pinned commit
#   USE_VENDOR_SNAPSHOTS copy vendor/* instead of cloning private repos when available [1]
set -uo pipefail

# WORKSPACE_HOST defaults to the folder CONTAINING this repo checkout (the execution
# workspace) — not a node-specific path. Override by exporting WORKSPACE_HOST.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${WORKSPACE_HOST:-}" ]]; then
  _RR="$(git -C "${SELF}" rev-parse --show-toplevel 2>/dev/null || true)"
  WORKSPACE_HOST="$(cd "${_RR:-${SELF}/../..}/.." && pwd)"
fi
RUNNER_ROOT="$(cd "${SELF}/../.." && pwd)"
VENDOR_ROOT="${VENDOR_ROOT:-${RUNNER_ROOT}/vendor}"
USE_VENDOR_SNAPSHOTS="${USE_VENDOR_SNAPSHOTS:-1}"

log()  { echo "[setup-workspace $(date -u +%H:%M:%S)] $*"; }
note() { echo "    $*"; }

# dir | git url | pinned commit | note | [sparse cone path]
# The 5th field, when present, requests a FILTERED CONE sparse-checkout limited to that
# path. mlcommons/inference is kept ONLY for loadgen/ (the certified GR tree is cloned into
# recommendation/dlrm_v3 from its OWN repo, below), so we sparse-checkout just loadgen — far
# smaller than the full ~1.4G monorepo, and it dodges any ':'-in-filename result files.
TREES=(
  "mlcommons-inference|https://github.com/mlcommons/inference|393d8ef71190f32bf08544c64514af517b17f157|latest LoadGen source (6.0.16, pip-installed at runtime); GR cert tree is cloned into recommendation/dlrm_v3 from AMD-AGI/dlrm-v3-gr-rocm (below)|loadgen"
  "pynve-rocm|https://github.com/AMD-AGI/pynve-rocm.git|d34a5fe5c891af224f2b9e89209beaf48e298168|certified ROCm/gfx950 NVE port (PRIVATE; needs auth); pinned to Plan 55 cache metrics + odd-set geometry on main for the C1-off optimization line; built in place by build_pynve_rocm_full.sh; cuembed is vendored in-tree"
  "FBGEMM|https://github.com/pytorch/fbgemm|5beb3e6e0ef5ec830461ce163c012864677647a9|built from source for gfx950:sramecc+ (build_fbgemm_gfx950_sramecc.sh)"
)
PYNVE_SUB="pynve-rocm"
# Build deps the pynve core .so needs (plugins disabled). cuembed is VENDORED in the
# repo (not a submodule); we init ONLY these to avoid the heavy unused plugin trees.
PYNVE_SUBMODULES=(third_party/pybind11 third_party/json third_party/dlpack)

# Certified harness port repo — cloned directly as the harness tree (contents at the root).
HARNESS_SUB="dlrm-v3-harness-rocm"
HARNESS_REPO_REMOTE="${HARNESS_REPO_REMOTE:-https://github.com/AMD-AGI/dlrm-v3-harness-rocm.git}"
# Pinned to the q12,200 degree-5 GOLD harness commit (q12200 confs plus prior Plan 64/Plan 4
# q11,900/q12,000/q11,970 confs and the P0/P1 host-output
# reuse plumbing: per-rank Triton cache epoch, worker pinned-output reuse, and LoadGen response
# buffer ring).
# The Plan 63 q11,800 confs + vectorized response are the
# parent). The pre-promotion main is preserved at frozen/main-20260614-pre-fullcausal; the old dev
# branch full_causal_optimization (cert base ec130e37) is fully contained in main. Override with
# HARNESS_REPO_REF=<branch|tag|commit>.
HARNESS_REPO_REF="${HARNESS_REPO_REF:-5088d7c902c790b31fc0b39ae8c80ff0ba67c0ee}"

# Certified GR port repo — cloned INTO the mlcommons baseline at recommendation/dlrm_v3
# (the hard-coded PYTHONPATH target across the launchers; the baseline still supplies loadgen).
GR_BASELINE_SUB="mlcommons-inference"
GR_CERT_SUBPATH="recommendation/dlrm_v3"
GR_REPO_REMOTE="${GR_REPO_REMOTE:-https://github.com/AMD-AGI/dlrm-v3-gr-rocm.git}"
# Pinned to the q12,200 degree-5 GOLD GR commit (DLRM_HSTU_GATE_POLY_DEG selector on top of
# Plan 62 output-LN fast-inference path and Plan 61 P3 lnaddfold,
# plus the local datasets package marker needed by the harness import path). main carries the Triton gfx950 buffer-ops
# *_multirow routing fix; the old dev branch full_causal_optimization is fully contained
# in main, and the pre-promotion main is preserved at frozen/main-20260614-pre-fullcausal.
# Override with GR_REPO_REF; cert C1-on base is 4c0ace45d9d62a3bbb4fc0cda8b4cfbc1555981b.
GR_REPO_REF="${GR_REPO_REF:-7eb52e96f9bb8dcba748b310b28c9824b752abed}"

# Clone a certified PRIVATE port repo to <dest> at <ref> (used for harness + GR). Leaves an
# existing checkout's local work intact (fetch+checkout); refuses to clobber a pre-existing
# non-repo dir (e.g. an old baseline/overlay) with a clear hint.
copy_vendor_snapshot() {  # $1=dest  $2=vendor-name  $3=label
  local dest="$1" vendor_name="$2" label="$3"
  local src="${VENDOR_ROOT}/${vendor_name}"
  [[ "${USE_VENDOR_SNAPSHOTS}" == "1" && -d "${src}" ]] || return 1
  if [[ -e "${dest}" && -n "$(ls -A "${dest}" 2>/dev/null)" ]]; then
    note "vendor ${label}: ${dest} exists; leaving it intact"
    return 1
  fi
  mkdir -p "$(dirname "${dest}")"
  cp -a "${src}" "${dest}"
  note "vendor ${label}: copied ${src} -> ${dest}"
  return 0
}

clone_cert_repo() {  # $1=dest  $2=remote  $3=ref  $4=label  $5=vendor-name
  local dest="$1" remote="$2" ref="$3" label="$4"
  local vendor_name="${5:-}"
  log "=== ${label} (certified repo) ==="
  note "url: ${remote}${ref:+ @ ${ref}}  (PRIVATE; needs git/gh auth)"
  note "-> ${dest}"
  if [[ -d "${dest}/.git" ]]; then
    note "present — fetching (leaving local work intact)"
    git -C "${dest}" fetch --all --tags 2>/dev/null || note "WARN: fetch failed — using on-disk state"
    [[ -n "${ref}" ]] && { git -C "${dest}" checkout -q "${ref}" 2>/dev/null \
      && note "checked out ${ref}" || note "WARN: checkout ${ref} failed"; }
  elif [[ -e "${dest}" && -n "$(ls -A "${dest}" 2>/dev/null)" ]]; then
    note "WARN: ${dest} exists and is non-empty but is not a git checkout of the ${label} repo"
    note "      (likely an OLD baseline/snapshot overlay) — move/remove it, then re-run to clone the repo."
  elif [[ -n "${vendor_name}" ]] && copy_vendor_snapshot "${dest}" "${vendor_name}" "${label}"; then
    note "${label} now at CERTIFIED state (vendored snapshot)"
  else
    mkdir -p "$(dirname "${dest}")"
    if git clone "${remote}" "${dest}" 2>/dev/null; then
      [[ -n "${ref}" ]] && { git -C "${dest}" checkout -q "${ref}" 2>/dev/null \
        && note "checked out ${ref}" || note "WARN: checkout ${ref} failed"; }
      note "${label} now at CERTIFIED state (cloned)"
    else
      note "WARN: clone failed — the repo is PRIVATE (needs git/gh auth) or not yet created."
      if [[ -n "${vendor_name}" && -d "${VENDOR_ROOT}/${vendor_name}" ]]; then
        note "      vendor snapshot exists at ${VENDOR_ROOT}/${vendor_name}; move/remove ${dest} if it is a stale non-repo dir, then re-run."
      else
        note "      create+push the ${label} repo, provide vendor/${vendor_name}, or set ${label^^}_REPO_REMOTE; then re-run."
      fi
    fi
  fi
}

command -v git >/dev/null || { echo "ERROR: git not on PATH" >&2; exit 1; }
mkdir -p "${WORKSPACE_HOST}"
log "workspace: ${WORKSPACE_HOST}"

for row in "${TREES[@]}"; do
  IFS='|' read -r dir url pin desc sparse <<<"${row}"
  dest="${WORKSPACE_HOST}/${dir}"
  log "=== ${dir} @ ${pin} ==="
  note "${desc}"
  note "url: ${url}"
  if [[ -d "${dest}/.git" ]]; then
    note "present: ${dest} (will fetch+checkout the pin, leaving local work intact)"
    git -C "${dest}" fetch --all --tags 2>/dev/null || note "WARN: fetch failed (offline / access-blocked) — using on-disk state"
  elif [[ "${dir}" == "${PYNVE_SUB}" ]] && copy_vendor_snapshot "${dest}" "pynve-rocm" "pynve"; then
    note "pynve now at CERTIFIED state (vendored snapshot)"
  elif [[ -n "${sparse}" ]]; then
    note "cloning ${url} -> ${dest} (filtered cone sparse-checkout: ${sparse})"
    if ! git clone --filter=blob:none --no-checkout "${url}" "${dest}" 2>/dev/null; then
      note "WARN: clone failed (access-blocked upstream?) — obtain out-of-band (private-repo auth), then re-run"
      continue
    fi
    git -C "${dest}" sparse-checkout init --cone 2>/dev/null
    git -C "${dest}" sparse-checkout set ${sparse} 2>/dev/null \
      || note "WARN: sparse-checkout set ${sparse} failed"
  else
    note "cloning ${url} -> ${dest}"
    if ! git clone "${url}" "${dest}" 2>/dev/null; then
      note "WARN: clone failed (access-blocked upstream?) — obtain out-of-band (private-repo auth), then re-run"
      continue
    fi
  fi
  if git -C "${dest}" cat-file -e "${pin}^{commit}" 2>/dev/null; then
    git -C "${dest}" checkout -q "${pin}" 2>/dev/null \
      && note "checked out ${pin}" \
      || note "WARN: checkout ${pin} failed (uncommitted local changes? stash/inspect manually)"
  else
    note "WARN: pin ${pin} not found in ${dir} — fetch the right remote/ref, then 'git checkout ${pin}'"
  fi

  # pynve-rocm: init ONLY the build-deps submodules (cuembed is vendored in-tree), so the
  # checkout is build-ready. build_pynve_rocm_full.sh also inits these, so this is belt-and-
  # suspenders; skipped silently if the clone above was access-blocked.
  if [[ "${dir}" == "${PYNVE_SUB}" && -e "${dest}/.gitmodules" ]]; then
    git -C "${dest}" submodule update --init "${PYNVE_SUBMODULES[@]}" 2>/dev/null \
      && note "pynve build submodules ready: ${PYNVE_SUBMODULES[*]}" \
      || note "WARN: pynve submodule init failed — build_pynve_rocm_full.sh will retry"
  fi
done

# ── Certified port repos cloned directly (harness standalone; GR into the baseline) ────
harness_dest="${WORKSPACE_HOST}/${HARNESS_SUB}"
gr_dest="${WORKSPACE_HOST}/${GR_BASELINE_SUB}/${GR_CERT_SUBPATH}"
clone_cert_repo "${harness_dest}" "${HARNESS_REPO_REMOTE}" "${HARNESS_REPO_REF}" "harness" "dlrm-v3-harness-rocm"
clone_cert_repo "${gr_dest}"      "${GR_REPO_REMOTE}"      "${GR_REPO_REF}"      "gr"      "dlrm-v3-gr-rocm"

_short() { git -C "$1" rev-parse --short HEAD 2>/dev/null; }
cat <<EOF

=== setup_workspace: DONE (reference run) ===
  workspace : ${WORKSPACE_HOST}
  pins      : mlcommons-inference@393d8ef (loadgen 6.0.16, sparse)   FBGEMM@5beb3e6e
              pynve-rocm@d34a5fe (Plan 55 cache geometry)   harness@${HARNESS_REPO_REF:-<default>}   gr@${GR_REPO_REF:-<default>}
              (full-causal q12,200 degree-5 GOLD solution SHA-pinned for harness + gr + pynve)
  harness   : $( [[ -d "${harness_dest}/.git" ]] && echo "certified (cloned: ${HARNESS_REPO_REMOTE} @ $(_short "${harness_dest}"))" || echo "MISSING — clone failed (private-repo auth?) or pre-existing non-repo dir" )
  gr        : $( [[ -d "${gr_dest}/.git" ]] && echo "certified (cloned: ${GR_REPO_REMOTE} @ $(_short "${gr_dest}"))" || echo "MISSING — clone failed (private-repo auth?) or pre-existing non-repo dir" )
Next:
  * harness + gr + pynve = certified repo clones; setup_submission.sh does NOT patch them,
    it only verifies the certified sentinels.
  * Then:  bash scripts/build/setup_submission.sh
EOF
