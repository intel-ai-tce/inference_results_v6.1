#!/usr/bin/env bash
# build_rocm.sh — self-contained ROCm/gfx950 build of this pynve port.
#
# This repo IS the certified ROCm port source (the gfx950 fixes — Plan 14.8 warp-size
# cache-fill, 14.9 teardown device-sync, 14.10 cub temp-storage alignment, and the
# Plan-18 own-device `hipMemSetAccess` grant — are already baked into src/). So unlike
# the harness reconstruction script, there is NOTHING to clone or patch here: this
# script just initializes the third-party submodules (so it works from a fresh/empty
# `git clone` with no submodules checked out), configures with HIP, builds, installs the
# Python extension, and runs an import smoke.
#
# RUN inside a ROCm build container (HIP 7.2.x, cmake >= 3.27, MPI, python3.12 with the
# pinned torch ROCm build). Example:
#   docker exec <ctr> bash /path/to/pynve-rocm/build_rocm.sh
#
# Produces:
#   build_rocm/lib/libnve-common.so          (native shared lib)
#   build_rocm/lib/nve.cpython-*.so          (pybind extension)
#   python/pynve/nve.cpython-*.so            (installed copy on PYTHONPATH)
#
# Then use it with:
#   export PYTHONPATH=<repo>/python  LD_LIBRARY_PATH=<repo>/build_rocm/lib
#
# ── Configuration (override via env) ──────────────────────────────────────────
#   PYNVE_ARCH    gfx arch                              [gfx950]
#   BUILD_JOBS    parallel build jobs                   [8]
#   SKIP_SUBMODULES=1   skip `git submodule update`     [unset]
#   SKIP_IMPORT_SMOKE=1 skip the final import check     [unset]
set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYNVE_ARCH="${PYNVE_ARCH:-gfx950}"
BUILD_JOBS="${BUILD_JOBS:-8}"

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">>> $*"; }

cd "${SELF}" || die "cannot cd to repo root ${SELF}"

# ── [1/4] Initialize third-party submodules ────────────────────────────────────
# With plugins disabled (below) the core .so needs only these. We init just them
# (NOT --recursive over everything) to avoid pulling the heavy, unused rocksdb/abseil/
# redis plugin trees. Header-only deps (dlpack) live here too. NOTE: third_party/cuembed
# is VENDORED in-tree (the port hipifies it in place), so it is NOT a submodule — it is
# always present and is only checked for, never fetched.
SUBMODULES_TO_INIT=(third_party/pybind11 third_party/json third_party/dlpack)
REQUIRED_PRESENT=("${SUBMODULES_TO_INIT[@]}" third_party/cuembed)

submodule_populated() { [[ -n "$(ls -A "$1" 2>/dev/null)" ]]; }

if [[ "${SKIP_SUBMODULES:-0}" != "1" ]]; then
  if git -C "${SELF}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    # Only init the ones that aren't already populated (a fresh clone has them empty;
    # an already-checked-out tree skips this so we don't error on non-empty dirs).
    need_init=()
    for d in "${SUBMODULES_TO_INIT[@]}"; do
      submodule_populated "${SELF}/${d}" || need_init+=("${d}")
    done
    if (( ${#need_init[@]} )); then
      note "[1/4] git submodule update --init: ${need_init[*]}"
      git -C "${SELF}" submodule update --init "${need_init[@]}" \
        || note "    (submodule init reported an issue — continuing; will verify population next)"
    else
      note "[1/4] required submodules already populated — skipping 'git submodule update'"
    fi
  else
    note "[1/4] not a git checkout — skipping 'git submodule update' (expecting submodules already vendored)"
  fi
else
  note "[1/4] SKIP_SUBMODULES=1 — skipping submodule init"
fi

# Hard-verify the required deps are actually present, however they got here.
missing=()
for d in "${REQUIRED_PRESENT[@]}"; do
  submodule_populated "${SELF}/${d}" || missing+=("${d}")
done
if (( ${#missing[@]} )); then
  die "missing/empty submodules: ${missing[*]}
  -> from a git checkout run:  git submodule update --init ${SUBMODULES_TO_INIT[*]}
     (needs network the first time), then re-run this script."
fi
note "    deps OK: ${REQUIRED_PRESENT[*]}"

# Guard: this must be the certified port source (own-device grant present), not a
# stock NVIDIA tree that would wedge the box at multi-GPU 1 TB scale.
if [[ -f "${SELF}/src/distributed.cpp" ]] && ! grep -q "own_desc" "${SELF}/src/distributed.cpp"; then
  die "src/distributed.cpp lacks the Plan-18 own-device grant (own_desc) — this is not the certified ROCm port source."
fi

# ── [2/4] Configure ─────────────────────────────────────────────────────────────
note "[2/4] cmake configure (-DNVE_ROCM=ON, plugins+tests off, arch=${PYNVE_ARCH})"
cmake -B build_rocm \
  -DNVE_ROCM=ON \
  -DNVE_DISABLE_PLUGINS=ON \
  -DNVE_DISABLE_TESTS_AND_SAMPLES=1 \
  -DCMAKE_HIP_ARCHITECTURES="${PYNVE_ARCH}" \
  -DCMAKE_BUILD_TYPE=Release --fresh \
  || die "cmake configure failed (check HIP toolchain / cmake>=3.27 in the container)"

# ── [3/4] Build ─────────────────────────────────────────────────────────────────
note "[3/4] build (-j${BUILD_JOBS})"
cmake --build build_rocm -j"${BUILD_JOBS}" \
  || die "build failed — inspect the first 'error:' above"

# Install the freshly-built extension next to the package (its libnve-common.so dep is
# resolved from build_rocm/lib via LD_LIBRARY_PATH at runtime).
BUILT_EXT="$(ls -1 "${SELF}/build_rocm/lib/"nve.cpython*.so 2>/dev/null | head -1 || true)"
[[ -n "${BUILT_EXT}" ]] || die "no nve.cpython*.so under build_rocm/lib — build did not produce the extension"
cp -f "${BUILT_EXT}" "${SELF}/python/pynve/$(basename "${BUILT_EXT}")"
note "    installed extension -> python/pynve/$(basename "${BUILT_EXT}")"

# ── [4/4] Import smoke ──────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="${SELF}/build_rocm/lib:${LD_LIBRARY_PATH:-}"
if [[ "${SKIP_IMPORT_SMOKE:-0}" == "1" ]]; then
  note "[4/4] SKIP_IMPORT_SMOKE=1 — skipping"
else
  note "[4/4] import smoke (PYTHONPATH=python, LD_LIBRARY_PATH+=build_rocm/lib)"
  PYTHONPATH="${SELF}/python:${PYTHONPATH:-}" python3 -c \
    'import pynve, pynve.nve as n; print("  pynve + native nve import OK:", n.__file__)' \
    || die "import pynve.nve failed — check torch is the pinned ROCm build (ABI) and LD_LIBRARY_PATH includes build_rocm/lib"
fi

cat <<EOF

=== build_rocm: DONE ===
  repo   : ${SELF}
  build  : ${SELF}/build_rocm
  use    : export PYTHONPATH=${SELF}/python LD_LIBRARY_PATH=${SELF}/build_rocm/lib

Try it:
  python3 examples/inference/minimal_nve_example.py
  python3 examples/training/train_nve_smoke.py
EOF
