#!/usr/bin/env bash
# Build fbgemm-gpu from source for gfx950:sramecc+ on the ROCm 7.2.3 stack.
#
# WHY THIS EXISTS
# ---------------
# MI350-series GPUs (gfx950) ship with SRAM-ECC enabled, so the runtime
# device ISA is `gfx950:sramecc+:xnack-`. The PUBLIC prebuilt fbgemm-gpu
# wheels (both 1.5.0+rocm7.0 and 1.6.0+rocm7.2) embed gfx950 code objects
# built WITHOUT the `sramecc+` feature. On a sramecc+ host HIP then reports
#   "No compatible code objects found for: gfx950:sramecc+:xnack-"
# and every fbgemm GPU op SIGSEGVs (asynchronous_complete_cumsum,
# jagged_to_padded_dense, TBE lookups, ...). Triton JIT is immune because
# it compiles for the exact device ISA at runtime — which is why the HSTU
# kernel microbench ran fine while the harness sparse path crashed.
#
# Building fbgemm-gpu from source with PYTORCH_ROCM_ARCH=gfx950:sramecc+
# emits a matching code object and the ops run on the host (verified: e2e
# Mode B W=8 b=6 qps=290 VALID, 291.31 q/s p99=76.26 ms — on par with the
# rocm7.0 headline). See SETUP.md §2c and STATUS.md.
#
# REQUIREMENTS
#   * Running inside the ROCm 7.2.3 atom container (torch 2.10.0+rocm7.2.3,
#     Triton 3.6.0, full ROCm dev toolchain incl. hipcc + composable_kernel).
#   * FBGEMM source bind-mounted at $FBGEMM_DIR (default /FBGEMM).
#   * The dlrmv3-rocm patch tree reachable for fbgemm_rocm7.patch.
#
# Build is a cross-compile (offload-arch) — no GPU needed at build time.
set -euo pipefail

FBGEMM_DIR="${FBGEMM_DIR:-/FBGEMM}"
FBGEMM_COMMIT="${FBGEMM_COMMIT:-5beb3e6e0ef5ec830461ce163c012864677647a9}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx950:sramecc+:xnack-}"
BUILD_ROCM_VERSION="${BUILD_ROCM_VERSION:-7.2}"

# Patch lives in this repo; allow override.
_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FBGEMM_PATCH="${FBGEMM_PATCH:-${_self}/../../patches/fbgemm_rocm7.patch}"

log() { echo "[build-fbgemm-sramecc $(date -u +%H:%M:%S)] $*"; }

[[ -d "${FBGEMM_DIR}/.git" ]] || { echo "FBGEMM repo not at ${FBGEMM_DIR}; mount it (-v .../FBGEMM:/FBGEMM)" >&2; exit 1; }
[[ -f "${FBGEMM_PATCH}" ]]    || { echo "Missing patch: ${FBGEMM_PATCH}" >&2; exit 1; }

# Record the exact torch build string so dependency installs below can pin
# it and NEVER let pip pull a CUDA torch from PyPI (fairscale/tensordict do).
TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')"
log "host torch=${TORCH_VER}  (will pin to avoid CUDA-wheel clobber)"
echo "torch==${TORCH_VER}" > /tmp/fbgemm_torch_constraint.txt
CONSTRAINT=/tmp/fbgemm_torch_constraint.txt

# Fast idempotency: if fbgemm_gpu already imports from the INSTALLED location
# (site-packages, not the in-source shadow that ships no compiled .so) and the
# GPU op runs, the build is already good — skip the ~7 min rebuild. Run from /
# so the source-tree fbgemm_gpu/ package can never shadow the installed one.
if ( cd / && python3 - <<'PY'
import torch, fbgemm_gpu
assert "site-packages" in (fbgemm_gpu.__file__ or ""), fbgemm_gpu.__file__
torch.ops.fbgemm.asynchronous_complete_cumsum(
    torch.randint(0, 5, (8,), device="cuda", dtype=torch.int64))
torch.cuda.synchronize()
PY
) 2>/dev/null; then
  log "fbgemm-gpu already installed + GPU op verified — skipping rebuild"
  exit 0
fi

log "=== checkout FBGEMM @ ${FBGEMM_COMMIT} ==="
cd "${FBGEMM_DIR}"
if ! git cat-file -e "${FBGEMM_COMMIT}^{commit}" 2>/dev/null; then
  git fetch --depth 1 origin "${FBGEMM_COMMIT}" 2>/dev/null || git fetch origin
fi
git checkout -f "${FBGEMM_COMMIT}"

# NOTE: a shallow checkout lacks the base blobs for `git apply --3way`, so
# we apply with --reject. The only hunk that may reject is a benchmark
# helper (tbe_data_config_bench_helper.py) that is neither compiled nor used
# at runtime; the critical cmake + kernel_launcher.cuh hunks apply cleanly.
log "=== apply fbgemm_rocm7.patch (reject-tolerant) ==="
if git apply --check --reverse "${FBGEMM_PATCH}" 2>/dev/null; then
  log "patch already applied"
else
  git apply --reject --whitespace=nowarn "${FBGEMM_PATCH}" || true
  if ! grep -q "ROCM_VERSION" cmake/modules/GpuCppLibrary.cmake 2>/dev/null && \
     ! git diff --stat cmake/ | grep -q cmake; then
    echo "ERROR: critical cmake hunks did not apply" >&2; exit 1
  fi
  find . -name '*.rej' -printf '[patch] rejected (non-critical): %p\n' || true
fi

git submodule update --init --recursive

log "=== fbgemm_gpu build deps (torch-free; fairscale excluded) ==="
pip install --upgrade pip wheel setuptools
grep -v -iE '^fairscale' fbgemm_gpu/requirements.txt > /tmp/fbgemm_reqs.txt
pip install --no-input -c "${CONSTRAINT}" -r /tmp/fbgemm_reqs.txt

log "=== build + install fbgemm_gpu (PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}) ==="
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_ROOT_DIR="${ROCM_PATH}"
export CMAKE_PREFIX_PATH="${ROCM_PATH}/lib/cmake:${CMAKE_PREFIX_PATH:-}"
export PYTORCH_ROCM_ARCH
export BUILD_ROCM_VERSION
export package_name=fbgemm_gpu_rocm
export python_tag=py312
export MAX_JOBS="${MAX_JOBS:-32}"

cd "${FBGEMM_DIR}/fbgemm_gpu"
rm -rf _skbuild build 2>/dev/null || true
python setup.py install \
  --build-variant=rocm \
  -DHIP_ROOT_DIR="${ROCM_PATH}" \
  -DCMAKE_C_FLAGS=-DTORCH_USE_HIP_DSA \
  -DCMAKE_CXX_FLAGS=-DTORCH_USE_HIP_DSA

log "=== verify torch intact + fbgemm GPU op runs (no segfault) ==="
# Run from / so the in-source fbgemm_gpu/ package can't shadow the installed one.
cd /
python3 - <<'PY'
import torch, fbgemm_gpu
assert torch.version.hip, "torch lost its ROCm build!"
o = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.randint(0, 5, (64,), device="cuda", dtype=torch.int64))
torch.cuda.synchronize()
print(f"[ok] torch {torch.__version__}  fbgemm cumsum GPU op -> {tuple(o.shape)}")
PY

log "fbgemm-gpu gfx950:sramecc+ build OK."
