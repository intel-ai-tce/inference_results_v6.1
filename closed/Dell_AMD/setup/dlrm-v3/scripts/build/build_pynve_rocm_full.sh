#!/usr/bin/env bash
# build_pynve_rocm_full.sh — build the ROCm/gfx950 pynve port from the certified
# AMD-AGI/pynve-rocm repo.
#
# SOURCE OF TRUTH is now the git repo github.com/AMD-AGI/pynve-rocm — the certified,
# already-ported ROCm/gfx950 NVE tree (Plan-14.x kernels/toolchain + the Plan-18
# own-device-grant fix are baked into src/). This REPLACES the two old reconstruction
# paths, both now DEPRECATED and removed:
#   * the vendored post-patch snapshot (vendor/pynve-port-src/*.tar.{gz,zst}), and
#   * cloning the NVIDIA base + applying the nve-rocm--*/nve-cuembed-rocm--* patch series.
# The patch series could not faithfully rebuild the cert (it silently dropped Plan-18,
# THE multi-GPU fault fix), which is exactly why the repo exists — so we no longer
# reconstruct; we build the repo directly.
#
# Behavior:
#   [1] If PYNVE_TREE already holds the certified source (CMakeLists.txt +
#       src/distributed.cpp with the Plan-18 `own_desc` sentinel), build it IN PLACE.
#       Otherwise clone PYNVE_REPO_REMOTE @ PYNVE_REPO_REF into PYNVE_TREE.
#   [2] Ensure the required third-party submodules are populated (cuembed is VENDORED
#       in-tree, not a submodule — only checked for, never fetched).
#   [3] cmake configure (-DNVE_ROCM=ON) + build, then install the python extension.
#   [4] import smoke (pynve + native pynve.nve).
#
# RUN inside the pynve build container (HIP 7.2.x, cmake >= 3.27, MPI dev headers,
# python3.12 with the *pinned* atom torch). Example:
#   docker exec <ctr> bash .../scripts/build/build_pynve_rocm_full.sh
#
# IDEMPOTENT: an already-checked-out / already-built tree rebuilds in place.
#
# ── Configuration (override via env) ──────────────────────────────────────────
#   PYNVE_TREE         working tree to build      [${PWD}/pynve-rocm]
#   PYNVE_REPO_REMOTE  git URL of the cert port   [https://github.com/AMD-AGI/pynve-rocm.git]
#   PYNVE_REPO_REF     branch/tag/commit          [pinned cert commit; see below]
#   PYNVE_ARCH         gfx arch                   [gfx950]
#   BUILD_JOBS         parallel build jobs        [8]
#   SKIP_SUBMODULES=1     skip third-party submodule init
#   SKIP_IMPORT_SMOKE=1   skip the final import check (no torch in env)
#
# Exit non-zero on any failure; the clone step prints a clear auth/remediation hint.

set -uo pipefail

# ── Resolve defaults ──────────────────────────────────────────────────────────
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYNVE_ARCH="${PYNVE_ARCH:-gfx950}"
PYNVE_REPO_REMOTE="${PYNVE_REPO_REMOTE:-https://github.com/AMD-AGI/pynve-rocm.git}"
# PINNED to the promoted full-causal + bf16 commit on main (cert base 558d544 + license/
# CODEOWNERS + bf16 embedding-table gather + full_causal NVE working state = 6c50a39): the
# self-clone fallback must land on this commit (not the moving branch tip) for reproducible
# builds. Matches the setup_workspace.sh Plan 55 pin. Override with PYNVE_REPO_REF=<ref>.
PYNVE_REPO_REF="${PYNVE_REPO_REF:-d34a5fe5c891af224f2b9e89209beaf48e298168}"
BUILD_JOBS="${BUILD_JOBS:-8}"
PYNVE_TREE="${PYNVE_TREE:-${PWD}/pynve-rocm}"

die()  { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">>> $*"; }

# Required third-party deps for the core .so (plugins disabled below). We init ONLY
# these — NOT --recursive over everything — to avoid pulling the heavy, unused
# rocksdb/abseil/redis plugin trees. third_party/cuembed is VENDORED in-tree (the port
# hipifies it in place), so it is always present and is only checked for, never fetched.
SUBMODULES_TO_INIT=(third_party/pybind11 third_party/json third_party/dlpack)
REQUIRED_PRESENT=("${SUBMODULES_TO_INIT[@]}" third_party/cuembed)

submodule_populated() { [[ -n "$(ls -A "$1" 2>/dev/null)" ]]; }
is_cert_tree() {  # $1 = tree root: certified iff CMakeLists + Plan-18 own-device grant
  [[ -f "$1/CMakeLists.txt" && -f "$1/src/distributed.cpp" ]] \
    && grep -q "own_desc" "$1/src/distributed.cpp" 2>/dev/null
}

# ── [1/4] Obtain the certified port source into PYNVE_TREE ─────────────────────
if is_cert_tree "${PYNVE_TREE}"; then
  note "[1/4] building EXISTING certified port tree in place: ${PYNVE_TREE}"
  note "    (Plan-18 own-device grant present — no clone needed)"
elif [[ -e "${PYNVE_TREE}" && -n "$(ls -A "${PYNVE_TREE}" 2>/dev/null)" ]]; then
  # Non-empty but not (yet) recognized as the cert tree. If it's a checkout of the port
  # the sentinel is asserted after submodule init below; refuse to clobber regardless.
  note "[1/4] PYNVE_TREE exists and is non-empty: ${PYNVE_TREE} — reusing in place"
  note "    (the Plan-18 sentinel is verified after the submodule check)"
else
  note "[1/4] clone certified port: ${PYNVE_REPO_REMOTE}${PYNVE_REPO_REF:+ @ ${PYNVE_REPO_REF}}"
  note "    -> ${PYNVE_TREE}"
  git clone "${PYNVE_REPO_REMOTE}" "${PYNVE_TREE}" || die "clone failed: ${PYNVE_REPO_REMOTE}
  -> the AMD-AGI/pynve-rocm repo is private; this needs git/gh auth in the build env. Either:
       * gh auth login   (or)   export GH_TOKEN=<pat>   then re-run, or
       * pre-checkout the repo at ${PYNVE_TREE} (it is then built in place), or
       * set PYNVE_REPO_REMOTE to an accessible mirror."
  if [[ -n "${PYNVE_REPO_REF}" ]]; then
    git -C "${PYNVE_TREE}" checkout "${PYNVE_REPO_REF}" \
      || die "ref ${PYNVE_REPO_REF} not found in ${PYNVE_REPO_REMOTE}"
  fi
fi

cd "${PYNVE_TREE}" || die "cannot cd ${PYNVE_TREE}"

# ── [2/4] Ensure the required third-party submodules are populated ─────────────
if [[ "${SKIP_SUBMODULES:-0}" == "1" ]]; then
  note "[2/4] SKIP_SUBMODULES=1 — skipping submodule init"
elif git -C "${PYNVE_TREE}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  need_init=()
  for d in "${SUBMODULES_TO_INIT[@]}"; do
    submodule_populated "${PYNVE_TREE}/${d}" || need_init+=("${d}")
  done
  if (( ${#need_init[@]} )); then
    note "[2/4] git submodule update --init: ${need_init[*]}"
    git -C "${PYNVE_TREE}" submodule update --init "${need_init[@]}" \
      || note "    (submodule init reported an issue — verifying population next)"
  else
    note "[2/4] required submodules already populated"
  fi
else
  note "[2/4] not a git checkout — skipping submodule init (expecting vendored deps)"
fi

# Hard-verify the required deps are present, however they got here.
missing=()
for d in "${REQUIRED_PRESENT[@]}"; do
  submodule_populated "${PYNVE_TREE}/${d}" || missing+=("${d}")
done
(( ${#missing[@]} )) && die "missing/empty deps: ${missing[*]}
  -> from a git checkout run:  git submodule update --init ${SUBMODULES_TO_INIT[*]}  (needs network)"

# Cert gate: refuse to build a non-certified tree. A stock NVIDIA tree (no Plan-18
# own-device grant) would wedge the box at multi-GPU ~1 TB scale.
is_cert_tree "${PYNVE_TREE}" \
  || die "src/distributed.cpp lacks the Plan-18 own-device grant (own_desc) — ${PYNVE_TREE}
  is NOT the certified ROCm port. Check out github.com/AMD-AGI/pynve-rocm."
note "    cert OK: Plan-18 own-device grant present; deps: ${REQUIRED_PRESENT[*]}"

# ── [3/4] Configure + build ────────────────────────────────────────────────────
note "[3/4] cmake configure (-DNVE_ROCM=ON, arch=${PYNVE_ARCH}) + build (-j${BUILD_JOBS})"
cmake -B build_rocm \
  -DNVE_ROCM=ON \
  -DNVE_DISABLE_PLUGINS=ON \
  -DNVE_DISABLE_TESTS_AND_SAMPLES=1 \
  -DCMAKE_HIP_ARCHITECTURES="${PYNVE_ARCH}" \
  -DCMAKE_BUILD_TYPE=Release --fresh \
  || die "cmake configure failed (check HIP toolchain / cmake>=3.27 / MPI dev headers in container)"

cmake --build build_rocm -j"${BUILD_JOBS}" \
  || die "build failed — inspect the first 'error:' above"

# Install the freshly-built python extension into the package dir. cmake emits it to
# build_rocm/lib (CMAKE_LIBRARY_OUTPUT_DIRECTORY); the package convention is
# python/pynve/nve.cpython*.so (its libnve-common.so dep stays in build_rocm/lib and is
# found via LD_LIBRARY_PATH at runtime). Skipping this leaves a STALE extension in place.
BUILT_EXT="$(ls -1 "${PYNVE_TREE}/build_rocm/lib/"nve.cpython*.so 2>/dev/null | head -1 || true)"
if [[ -n "${BUILT_EXT}" ]]; then
  cp -f "${BUILT_EXT}" "${PYNVE_TREE}/python/pynve/$(basename "${BUILT_EXT}")"
  note "    installed extension -> python/pynve/$(basename "${BUILT_EXT}")"
else
  note "    WARNING: no built nve.cpython*.so under build_rocm/lib — package extension not refreshed"
fi

# ── [4/4] Python import smoke ──────────────────────────────────────────────────
PKG_DIR="${PYNVE_TREE}/python"
export LD_LIBRARY_PATH="${PYNVE_TREE}/build_rocm/lib:${LD_LIBRARY_PATH:-}"
note "[4/4] import smoke (PYTHONPATH=${PKG_DIR}, LD_LIBRARY_PATH+=build_rocm/lib)"
if [[ "${SKIP_IMPORT_SMOKE:-0}" == "1" ]]; then
  note "    SKIP_IMPORT_SMOKE=1 — skipping"
elif [[ -d "${PKG_DIR}" ]]; then
  # Import the NATIVE extension (pynve.nve), not just the pure-python package — a bare
  # `import pynve` only loads pynve._version and would pass even with a stale/missing .so.
  PYTHONPATH="${PKG_DIR}:${PYTHONPATH:-}" python3 -c 'import pynve, pynve.nve as n; print("  pynve + native nve import OK:", n.__file__)' \
    || die "import pynve.nve failed — confirm the freshly-built extension landed in ${PKG_DIR}/pynve, LD_LIBRARY_PATH includes build_rocm/lib, and torch is the PINNED build (ABI)"
else
  note "    (no ${PKG_DIR} — verify the pynve python package path)"
fi

_SRC_URL="$(git -C "${PYNVE_TREE}" remote get-url origin 2>/dev/null || echo 'in-place (no git)')"
_SRC_REF="$(git -C "${PYNVE_TREE}" rev-parse --short HEAD 2>/dev/null || true)"
cat <<EOF

=== build_pynve_rocm_full: DONE ===
  tree     : ${PYNVE_TREE}
  source   : ${_SRC_URL}${_SRC_REF:+ @ ${_SRC_REF}}
  build    : ${PYNVE_TREE}/build_rocm
  python   : export PYTHONPATH=${PKG_DIR}

Next (need GPUs; see PORTING.md §Validate):
  1) single-GPU corruptness + 8-rank repro_mpi_fault (pynve scripts)
  2) gtest oracle: mpi_buffer_test at W=8 (ReadWrite 8/8)
  3) full harness smoke (SETUP_NVE.md §4 / §4c)
EOF
