#!/usr/bin/env bash
# Build libfp8tuned_<arch>.so (no-torch C ABI) with hipcc, like the probes. The output is
# tagged with the GPU target (e.g. libfp8tuned_gfx950.so) so binaries built for different
# archs are never mixed up / loaded on the wrong device.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM="${ROCM_PATH:-/opt/rocm}"

# Target arch: GPU_ARCH env > amdgpu-arch > rocminfo. Strip any feature suffix (e.g.
# "gfx950:sramecc+:xnack-" -> "gfx950").
ARCH="${GPU_ARCH:-}"
[ -z "$ARCH" ] && ARCH="$("$ROCM/llvm/bin/amdgpu-arch" 2>/dev/null | head -1 || true)"
[ -z "$ARCH" ] && ARCH="$(amdgpu-arch 2>/dev/null | head -1 || true)"
[ -z "$ARCH" ] && ARCH="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1 || true)"
ARCH="${ARCH%%:*}"
if [ -z "$ARCH" ]; then
  echo "[build_lib] could not detect GPU arch; set GPU_ARCH=gfxNNNN" >&2
  exit 1
fi

OUT="$HERE/libfp8tuned_${ARCH}.so"
hipcc -O3 -std=c++17 -fPIC -shared --offload-arch="$ARCH" \
  "$HERE/fp8tuned_lib.cpp" -o "$OUT" \
  -I"$ROCM/include" -L"$ROCM/lib" -lhipblaslt -lamdhip64
echo "built: $OUT"
