#!/usr/bin/env bash
# setup_fp4_route_a.sh — reconstruct the fp4 (MXFP4) "Route A" toolchain that was
# NODE-ONLY on mi350x-106 (lost in the 2026-06-13 GPU wedge). This is the durability
# backstop the repo did not have before: every node-only fp4 asset is now either a
# committed patch (against a PINNED upstream commit) or committed source under
# this repo's fp4/, and this script puts them back in place.
#
# Companion to scripts/build/setup_workspace.sh (which obtains harness/gr/pynve/FBGEMM/
# mlcommons). Run setup_workspace.sh FIRST so the GR tree exists, then this.
#
# What this restores (see fp4/README.md and the jiaweichen-amd FP4_STATUS_AND_RECOVERY.md):
#   1. hipBLASLt @ b9d79701 + the MXFP4 fused-pack epilogue patch
#      (patches/hipblaslt_mxfp4_pack_epilogue.patch)  — the producer codegen.
#   2. route_a_probe/ source toolchain (probes, tuners, library builders, the
#      torch custom-op mxfp4_ext, winning tuning configs) — fp4/route_a_probe/.
#   3. GR fp4-QK attention diff into the GR tree
#      (patches/gr_fp4_qk_attention.patch) — the consumer + Plan-37 fused plumbing.
#
# This is intentionally NOT airtight: hipBLASLt is a large upstream and the GR tree is
# private. On any failure it prints guidance and moves on — treat it as the canonical
# record of WHICH commit / WHICH patch / WHICH build each fp4 asset is pinned at.
#
# ── Configuration (override via env) ──────────────────────────────────────────
#   WORKSPACE_HOST        where the trees live   [parent of this repo checkout]
#   HIPBLASLT_REMOTE / HIPBLASLT_PIN   upstream hipBLASLt + pinned commit
#   APPLY_GR_PATCH=0      skip patching the GR tree (default: apply if tree present)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORKSPACE_HOST:-}" ]]; then
  _RR="$(git -C "${REPO}" rev-parse --show-toplevel 2>/dev/null || echo "${REPO}")"
  WORKSPACE_HOST="$(cd "${_RR}/.." && pwd)"
fi

# hipBLASLt source of record: ROCm 7.2.3 / gfx950 dev base (2026-03-23).
HIPBLASLT_REMOTE="${HIPBLASLT_REMOTE:-https://github.com/ROCm/hipBLASLt.git}"
HIPBLASLT_PIN="${HIPBLASLT_PIN:-b9d797017b4d4d7c3692ccc5ac099ff9777ace0f}"
HIPBLASLT_DEST="${WORKSPACE_HOST}/hipBLASLt"

# GR (consumer) tree — cloned by setup_workspace.sh INTO the mlcommons baseline.
GR_DEST="${WORKSPACE_HOST}/mlcommons-inference/recommendation/dlrm_v3"
APPLY_GR_PATCH="${APPLY_GR_PATCH:-1}"

# route_a_probe toolchain destination (where the build scripts hard-code their paths).
RAP_SRC="${REPO}/fp4/route_a_probe"
RAP_DEST="${WORKSPACE_HOST}/route_a_probe"

HIPBLASLT_PATCH="${REPO}/patches/hipblaslt_mxfp4_pack_epilogue.patch"
GR_PATCH="${REPO}/patches/gr_fp4_qk_attention.patch"

log()  { echo "[setup-fp4 $(date -u +%H:%M:%S)] $*"; }
note() { echo "    $*"; }

command -v git >/dev/null || { echo "ERROR: git not on PATH" >&2; exit 1; }
mkdir -p "${WORKSPACE_HOST}"
log "workspace: ${WORKSPACE_HOST}"

# Apply <patch> into <tree> idempotently: skip if already applied (reverse-check), else
# apply, else warn (and leave the tree untouched — never half-apply).
apply_patch() {  # $1=tree  $2=patch  $3=label
  local tree="$1" patch="$2" label="$3"
  if [[ ! -d "${tree}/.git" ]]; then
    note "WARN: ${label}: ${tree} is not a git checkout — skip (run setup_workspace.sh first?)"; return
  fi
  if [[ ! -f "${patch}" ]]; then
    note "WARN: ${label}: patch missing: ${patch}"; return
  fi
  if git -C "${tree}" apply --check -R -p1 "${patch}" 2>/dev/null; then
    note "${label}: already applied (reverse-check clean) — nothing to do"; return
  fi
  if git -C "${tree}" apply --check -p1 "${patch}" 2>/dev/null; then
    git -C "${tree}" apply -p1 "${patch}" && note "${label}: applied ${patch##*/}"
  else
    note "WARN: ${label}: patch does NOT apply cleanly onto $(git -C "${tree}" rev-parse --short HEAD 2>/dev/null)"
    note "      tree may be off the pinned base, or already partially modified — inspect manually:"
    note "      git -C ${tree} apply --3way -p1 ${patch}"
  fi
}

# ── 1. hipBLASLt @ pin + fused-pack epilogue patch ────────────────────────────
log "=== hipBLASLt (producer) @ ${HIPBLASLT_PIN:0:8} ==="
if [[ -d "${HIPBLASLT_DEST}/.git" ]]; then
  note "present: ${HIPBLASLT_DEST} (fetch + checkout pin, leaving local work intact)"
  git -C "${HIPBLASLT_DEST}" fetch --all --tags 2>/dev/null || note "WARN: fetch failed — using on-disk state"
else
  note "cloning ${HIPBLASLT_REMOTE} -> ${HIPBLASLT_DEST} (blob:none — full history is large)"
  git clone --filter=blob:none "${HIPBLASLT_REMOTE}" "${HIPBLASLT_DEST}" 2>/dev/null \
    || note "WARN: clone failed (network / access) — obtain hipBLASLt out-of-band, then re-run"
fi
if git -C "${HIPBLASLT_DEST}" cat-file -e "${HIPBLASLT_PIN}^{commit}" 2>/dev/null; then
  git -C "${HIPBLASLT_DEST}" checkout -q "${HIPBLASLT_PIN}" 2>/dev/null \
    && note "checked out ${HIPBLASLT_PIN:0:8}" \
    || note "WARN: checkout ${HIPBLASLT_PIN:0:8} failed (uncommitted local changes? stash/inspect)"
  apply_patch "${HIPBLASLT_DEST}" "${HIPBLASLT_PATCH}" "hipBLASLt"
else
  note "WARN: pin ${HIPBLASLT_PIN:0:8} not present — fetch the right ref, then re-run"
fi

# ── 2. route_a_probe toolchain source ─────────────────────────────────────────
log "=== route_a_probe (toolchain source) ==="
if [[ ! -d "${RAP_SRC}" ]]; then
  note "WARN: source missing in this repo: ${RAP_SRC}"
else
  mkdir -p "${RAP_DEST}"
  # Copy source in WITHOUT clobbering node-built artifacts (libs/probes) if present.
  rsync -a "${RAP_SRC}/" "${RAP_DEST}/" \
    && note "restored toolchain source -> ${RAP_DEST} ($(find "${RAP_SRC}" -type f | wc -l) files)" \
    || cp -rn "${RAP_SRC}/." "${RAP_DEST}/" 2>/dev/null
  note "NOTE: build scripts hard-code ${WORKSPACE_HOST}/{route_a_probe,hipBLASLt}; keep this layout."
fi

# ── 3. GR fp4-QK attention diff (consumer) ────────────────────────────────────
log "=== GR fp4-QK attention (consumer) ==="
if [[ "${APPLY_GR_PATCH}" == "1" ]]; then
  apply_patch "${GR_DEST}" "${GR_PATCH}" "GR fp4-QK"
  note "this diff (triton_addmm/_hstu_attention/_preprocess) belongs on the GR repo's"
  note "full_causal_optimization branch — commit+push it there once validated to close the gap."
else
  note "APPLY_GR_PATCH=0 — skipped (patch: ${GR_PATCH})"
fi

# ── Build steps (run inside the ROCm 7.2.3 / gfx950 container: dlrmv3-e2e723) ──
cat <<EOF

=== setup_fp4_route_a: DONE (restore reference) ===
  workspace   : ${WORKSPACE_HOST}
  hipBLASLt   : $( [[ -d "${HIPBLASLT_DEST}/.git" ]] && echo "@ $(git -C "${HIPBLASLT_DEST}" rev-parse --short HEAD 2>/dev/null) (+ fused-pack patch)" || echo "MISSING" )
  route_a_probe: $( [[ -d "${RAP_DEST}" ]] && echo "${RAP_DEST}" || echo "MISSING" )
  GR consumer : $( [[ -d "${GR_DEST}/.git" ]] && echo "$(git -C "${GR_DEST}" rev-parse --short HEAD 2>/dev/null) (+ fp4-QK patch if applied)" || echo "MISSING (run setup_workspace.sh)" )

BUILD (in the ROCm 7.2.3 / gfx950 container — e.g. dlrmv3-e2e723):
  # a) rocisa C++/nanobind ext (TensileCreateLibrary needs it on PYTHONPATH):
  cmake -S ${RAP_DEST}/rocisa_super -B ${RAP_DEST}/rocisa_super/build
  cmake --build ${RAP_DEST}/rocisa_super/build -j
  # b) fused MXFP4-pack library (the multi-solution winner -> lib_mxpack_multi/):
  cd ${RAP_DEST} && python3 build_multi_mxpack.py
  # c) GEMM-level probes (mirror mxfp4_ext flags):
  cd ${RAP_DEST} && for p in probe_mxfp4_perf probe_mxfp4_vary probe_pack_axis; do \\
    hipcc -O3 --offload-arch=gfx950 -I/opt/rocm/include \$p.cpp -o \$p \\
      -L/opt/rocm/lib -lhipblaslt -lamdhip64 ; done
  # d) torch custom op (Plan 37 item H):
  cd ${RAP_DEST}/mxfp4_ext && python3 -c "from build import load_ext; load_ext()"
  # e) verify (LIBPATH -> the fused lib), then the bit-exact gate:
  HIPBLASLT_TENSILE_LIBPATH=${RAP_DEST}/lib_mxpack_multi/library \\
    ${RAP_DEST}/probe_mxfp4_vary 12000 2048 512   # expect MXFP4_VARY_RESULT: PASS

STABILITY GATE before ANY fp4 perf re-run (the 2026-06-13 wedge):
  single GPU (--gpus-per-node 1, one shard), small batch, short probe, with a
  'timeout' watchdog; flag-bisect DLRM_HSTU_FP4_GEMM -> _FP4_QK -> fused pack;
  re-run the bit-exact gate AT the b40 shapes first. Do NOT repeat the 8-GPU run.
  Ref: jiaweichen-amd .../dlrmv3-rocm/FP4_STATUS_AND_RECOVERY.md §1a, WORK_LOG.md §40.
EOF
