#!/usr/bin/env bash
# Patch installed ROCm Triton AMD backend: gfx950 skips canonicalize-pointers / buffer-op
# passes that crash on _concat_2D_jagged. Supports Triton 3.3.x and 3.6.x layouts.
set -euo pipefail

COMPILER_PY="$(python3 -c "import triton.backends.amd.compiler as c; print(c.__file__)")"
echo "Triton compiler: ${COMPILER_PY}"
python3 -c "import triton; print('Triton version:', triton.__version__)"

if grep -q 'gfx950_skip_buffer_ops' "${COMPILER_PY}"; then
  echo "Already patched. Nothing to do."
  exit 0
fi

python3 - "${COMPILER_PY}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

helper = '''
def gfx950_skip_buffer_ops(arch: str) -> bool:
    """Skip AMDGPU buffer-op passes on gfx950 unless AMDGCN_USE_BUFFER_OPS_GFX950=1."""
    import os
    if arch != "gfx950":
        return False
    return os.environ.get("AMDGCN_USE_BUFFER_OPS_GFX950", "0") != "1"

'''

if "def gfx950_skip_buffer_ops" not in text:
    anchor = "class HIPBackend(BaseBackend):"
    if anchor not in text:
        raise SystemExit("HIPBackend class not found")
    text = text.replace(anchor, helper + anchor, 1)

# Triton 3.3.x
old33 = "        if HIPBackend.use_buffer_ops():"
new33 = "        if HIPBackend.use_buffer_ops() and not gfx950_skip_buffer_ops(options.arch):"
if old33 in text:
    text = text.replace(old33, new33, 1)
elif "if knobs.amd.use_buffer_ops:" in text:
    old36 = "        if knobs.amd.use_buffer_ops:"
    new36 = "        if knobs.amd.use_buffer_ops and not gfx950_skip_buffer_ops(options.arch):"
    text = text.replace(old36, new36, 1)
else:
    raise SystemExit("No known buffer-ops guard found in compiler.py")

path.write_text(text)
print("patched", path)
PY

rm -f "$(dirname "${COMPILER_PY}")/__pycache__/compiler."*.pyc 2>/dev/null || true
echo "Patched OK. gfx950 skips buffer-op passes unless AMDGCN_USE_BUFFER_OPS_GFX950=1"
