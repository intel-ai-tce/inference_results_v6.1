#!/usr/bin/env bash
# One-shot setup of the canonical ROCm 7.2.3 DLRMv3 inference stack.
#
# Run INSIDE the rocm/atom:rocm7.2.3_..._pytorch_release_2.10.0_atom* container
# with FBGEMM bind-mounted at /FBGEMM and the mlcommons-inference tree at
# /work/mlcommons-inference.
#
# Stack produced (validated 2026-05-29, e2e Mode B W=8 b=6 qps=290 VALID
# 291.31 q/s p99=76.26 ms — on par with the rocm7.0 headline):
#   torch 2.10.0+rocm7.2.3 (atom)   Triton 3.6.0 (gfx950-patched)
#   fbgemm-gpu  built from source for gfx950:sramecc+   (see build script)
#   torchrec==1.4.0  tensordict==0.12.4  pyvers==0.2.2  torchmetrics==1.0.3  mpi4py  pyzmq
#   (the torchrec/tensordict/pyvers versions are pinned to the certified-run set;
#    --no-deps still protects the atom torch from a CUDA-wheel clobber.)
#   mlcommons_loadgen (from mounted tree)
#
# The torch pin (-c) is load-bearing: fairscale/tensordict declare a bare
# `torch` dep and will otherwise drag a CUDA torch wheel from PyPI, silently
# replacing the atom ROCm torch. Every install below is --no-deps or
# constrained accordingly.
set -euo pipefail
_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log() { echo "[setup-rocm723 $(date -u +%H:%M:%S)] $*"; }

TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')"
echo "torch==${TORCH_VER}" > /tmp/rocm723_constraint.txt
C=/tmp/rocm723_constraint.txt
log "pinned torch=${TORCH_VER}"

# 1) fbgemm-gpu from source for gfx950:sramecc+ (the whole reason for 7.2.3).
log "=== build fbgemm-gpu (gfx950:sramecc+) ==="
bash "${_self}/build_fbgemm_gfx950_sramecc.sh"

# 2) Triton gfx950: buffer ops are SAFE — no compiler patch.
#    The 2D-jagged concat/split kernels are routed to their mask-based *_multirow
#    variants on HIP (generative_recommenders/ops/triton/triton_jagged_tensors.py:
#    _prefer_multirow_concat_split). Those select sources with masks over a single
#    base pointer, so they compile cleanly under the AMDGPU canonicalize-pointers /
#    convert-buffer-ops passes that the basic 3-way-scf.if kernels used to crash. We
#    therefore leave Triton UNPATCHED so buffer ops (buffer_load/buffer_store) stay ON
#    for every kernel — notably _hstu_attn_fwd, the dominant inference kernel (C1-off
#    b40: VALID 7,395 q/s p99 58.0 ms with buffer ops, vs 59.9 ms patched/off).
#    patch_triton_compiler.sh is retained only as a fallback (blanket-disables buffer
#    ops on gfx950) if a future kernel ever re-trips the pass.
log "=== Triton gfx950: buffer ops ON (no patch; GR *_multirow routing) ==="

# 3) Harness Python deps — all torch-pinned / --no-deps to protect atom torch.
log "=== torchrec + sparse-path deps ==="
pip install --no-input -c "$C" --no-deps torchrec==1.4.0
pip install --no-input -c "$C" --no-deps tensordict==0.12.4 torchmetrics==1.0.3
# pyvers is a newer tensordict transitive dep; the rest are torch-free.
pip install --no-input -c "$C" \
    pyvers==0.2.2 lightning-utilities cloudpickle orjson packaging \
    pyre-extensions iopath portalocker tqdm gin_config==0.5.0 pandas tensorboard
MPICC="$(command -v mpicc)" pip install --no-input -c "$C" mpi4py pyzmq

# 4) loadgen from the mounted mlcommons tree.
#    Always reinstall from the pinned sparse checkout so advancing
#    mlcommons-inference actually updates the in-container wheel. Use tar instead
#    of `cp -a`: some container/tmpfs combinations reject permission preservation
#    and make cp fail before pip can build the wheel.
log "=== build mlcommons loadgen ==="
rm -rf /tmp/loadgen-build
mkdir -p /tmp/loadgen-build
(cd /work/mlcommons-inference/loadgen && tar cf - .) | (cd /tmp/loadgen-build && tar xf -)
CFLAGS="-std=c++14 -O3" pip install --force-reinstall --no-cache-dir /tmp/loadgen-build

# 5) `datasets` shadows dlrm_v3/datasets/ — must be absent.
pip uninstall -y datasets 2>/dev/null || true

log "=== sanity ==="
python3 -c "import torch, triton, fbgemm_gpu, torchrec, tensordict, mpi4py, mlcommons_loadgen; \
print('torch', torch.__version__, '| triton', triton.__version__, '| torchrec', torchrec.__version__)"
log "ROCm 7.2.3 stack ready."
