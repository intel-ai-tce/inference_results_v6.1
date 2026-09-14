#!/usr/bin/env bash
# Create a lightweight DLRM-v3 ROCm GOLD quickstart zip.
#
# The package is intentionally code/manifests only. It vendors source snapshots
# of the private harness/GR/pynve repos, but it does not include the preprocessed
# dataset, checkpoint, Docker layers, build outputs, benchmark artifacts, or
# Triton caches. Large inputs are fetched/staged by the unpacked runner using the
# pinned resources recorded in manifests/resources.lock.yaml.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SELF}/../.." && pwd)"

PACKAGE_NAME="${PACKAGE_NAME:-dlrm-v3-rocm-gold-quickstart}"
OUT_DIR="${OUT_DIR:-${ROOT}/dist}"
ZIP_PATH=""
KEEP_STAGE=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --output PATH      Write zip to PATH. Default: dist/<name>_<timestamp>.zip
  --name NAME        Package root directory name. Default: ${PACKAGE_NAME}
  --keep-stage       Keep the temporary staging directory for inspection.
  -h, --help         Show this help.

Environment:
  OUT_DIR            Output directory when --output is not supplied.
  PACKAGE_NAME       Package root directory name.
  WORKSPACE_HOST     Workspace containing sibling repos; defaults to runner parent.
  VENDOR_PRIVATE_REPOS
                     Include private harness/GR/pynve source snapshots. [1]

The zip contains tracked runner files plus vendored private source repos
(`vendor/dlrm-v3-harness-rocm`, `vendor/dlrm-v3-gr-rocm`, `vendor/pynve-rocm`)
when their workspace checkouts are available.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      ZIP_PATH="${2:?--output requires a path}"
      shift 2
      ;;
    --name)
      PACKAGE_NAME="${2:?--name requires a value}"
      shift 2
      ;;
    --keep-stage)
      KEEP_STAGE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[package] ERROR: unknown option: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

command -v git >/dev/null || { echo "[package] ERROR: git not found" >&2; exit 1; }
command -v python3 >/dev/null || { echo "[package] ERROR: python3 not found" >&2; exit 1; }

if [[ -z "${WORKSPACE_HOST:-}" ]]; then
  WORKSPACE_HOST="$(cd "${ROOT}/.." && pwd)"
fi
VENDOR_PRIVATE_REPOS="${VENDOR_PRIVATE_REPOS:-1}"

STAMP="$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "${OUT_DIR}"
if [[ -z "${ZIP_PATH}" ]]; then
  ZIP_PATH="${OUT_DIR}/${PACKAGE_NAME}_${STAMP}.zip"
fi
ZIP_PATH="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${ZIP_PATH}")"

TMP_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/dlrmv3_pkg.XXXXXX")"
STAGE="${TMP_PARENT}/${PACKAGE_NAME}"
mkdir -p "${STAGE}"

cleanup() {
  if [[ "${KEEP_STAGE}" != "1" ]]; then
    rm -rf "${TMP_PARENT}"
  else
    echo "[package] kept staging dir: ${STAGE}"
  fi
}
trap cleanup EXIT

include_path() {
  local p="$1"
  if [[ "${p}" == submission/* ]]; then
    return 0
  fi
  case "$p" in
    README.md|AGENTS.md|HANDOFF.md|HANDOFF_*.md|run.sh)
      return 0
      ;;
    scripts/build/*|scripts/run/*|scripts/package/*)
      return 0
      ;;
    docs/*|patches/*)
      return 0
      ;;
    plans/plan_1_packaging.md|plans/plan_2_test08_determinism.md|plans/plan_3_perf_then_fp8_broadcast.md|plans/plan_5_submission_assembly.md)
      return 0
      ;;
    results/fullcausal_c1off/*.md)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

copy_file() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "${dst}")"
  cp "${src}" "${dst}"
}

echo "[package] staging runner files from ${ROOT}"
while IFS= read -r -d '' rel; do
  if include_path "${rel}"; then
    copy_file "${ROOT}/${rel}" "${STAGE}/${rel}"
  fi
done < <(git -C "${ROOT}" ls-files -z)

# Ensure the packaging entry point is present even when this script is being
# tested before its first commit (git ls-files will not list untracked files).
copy_file "${SELF}/create_minimal_zip.sh" "${STAGE}/scripts/package/create_minimal_zip.sh"

# Include the configs needed for GOLD perf, TEST08, and common smoke/probe runs.
HARNESS="${WORKSPACE_HOST}/dlrm-v3-harness-rocm"
CONFIGS=(
  user_mi355x8_smoke.conf
  user_mi355x8_short.conf
  user_mi355x8_nve_b64_qps10500_PROD10min.conf
  user_mi355x8_nve_b64_qps10500_PROF90s.conf
  user_mi355x8_nve_b64_qps11970_PROD10min.conf
  user_mi355x8_nve_b64_qps11970_PROF90s.conf
  user_mi355x8_nve_b64_qps12000_PROD10min.conf
  user_mi355x8_nve_b64_qps12000_PROF90s.conf
  user_mi355x8_nve_b64_qps12200_OFFLINE10min.conf
  user_mi355x8_nve_b64_qps12200_OFFLINE90s.conf
  user_mi355x8_nve_b64_qps12200_PROD10min.conf
  user_mi355x8_nve_b64_qps12200_PROF90s.conf
  user_mi355x8_nve_b64_qps9600_PROD10min.conf
)
mkdir -p "${STAGE}/configs"
if [[ -d "${HARNESS}/benchmarks" ]]; then
  for cfg in "${CONFIGS[@]}"; do
    if [[ -f "${HARNESS}/benchmarks/${cfg}" ]]; then
      copy_file "${HARNESS}/benchmarks/${cfg}" "${STAGE}/configs/${cfg}"
    else
      echo "[package] WARN: missing harness config: ${cfg}" >&2
    fi
  done
else
  echo "[package] WARN: harness checkout not found at ${HARNESS}; configs/ will be incomplete" >&2
fi

if [[ "${VENDOR_PRIVATE_REPOS}" == "1" ]]; then
  echo "[package] vendoring private source repos from ${WORKSPACE_HOST}"
  python3 - "${WORKSPACE_HOST}" "${STAGE}" <<'PY'
import os
import shutil
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
stage = Path(sys.argv[2])

repos = {
    "dlrm-v3-harness-rocm": workspace / "dlrm-v3-harness-rocm",
    "dlrm-v3-gr-rocm": workspace / "mlcommons-inference" / "recommendation" / "dlrm_v3",
    "pynve-rocm": workspace / "pynve-rocm",
}

exclude_dirs = {
    ".git",
    ".github",  # runner docs record GitHub metadata; not needed for quickstart execution
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".rocprofv3",
    "artifacts",
    "dist",
    "build",
    "build_rocm",
    "cmake-build-debug",
    "cmake-build-release",
    "CMakeFiles",
    "node_modules",
    ".venv",
    "venv",
}
exclude_suffixes = (
    ".pyc",
    ".pyo",
    ".o",
    ".a",
    ".so",
    ".dylib",
    ".dll",
    ".distcp",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
)

def skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & exclude_dirs:
        return True
    name = path.name
    if name.startswith(".triton_cache"):
        return True
    if name.endswith(exclude_suffixes):
        return True
    return False

missing = [name for name, src in repos.items() if not src.is_dir()]
if missing:
    raise SystemExit(
        "[package] ERROR: missing private repo checkout(s): " + ", ".join(missing)
    )

for name, src in repos.items():
    dst = stage / "vendor" / name
    if dst.exists():
        shutil.rmtree(dst)
    count = 0
    bytes_copied = 0
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel_root = root_path.relative_to(src)
        dirs[:] = [
            d for d in dirs
            if not skip(rel_root / d)
        ]
        for file_name in files:
            rel = rel_root / file_name
            if skip(rel):
                continue
            src_file = root_path / file_name
            dst_file = dst / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            count += 1
            bytes_copied += src_file.stat().st_size
    print(
        f"[package] vendored {name}: {count} files, "
        f"{bytes_copied / (1024 * 1024):.2f} MiB"
    )
PY
fi

RUNNER_REF="$(git -C "${ROOT}" rev-parse HEAD)"
RUNNER_SHORT="$(git -C "${ROOT}" rev-parse --short HEAD)"
RUNNER_BRANCH="$(git -C "${ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
DIRTY=0
git -C "${ROOT}" diff --quiet -- . || DIRTY=1
git -C "${ROOT}" diff --cached --quiet -- . || DIRTY=1

python3 - "${ROOT}" "${STAGE}" "${RUNNER_REF}" "${RUNNER_BRANCH}" "${DIRTY}" "${WORKSPACE_HOST}" "${VENDOR_PRIVATE_REPOS}" <<'PY'
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
stage = Path(sys.argv[2])
runner_ref = sys.argv[3]
runner_branch = sys.argv[4]
dirty = sys.argv[5] == "1"
workspace = Path(sys.argv[6])
vendor_private_repos = sys.argv[7] == "1"
setup = (root / "scripts/build/setup_workspace.sh").read_text()

def default_var(name: str) -> str:
    m = re.search(rf'{name}="\$\{{{name}:-([^}}]+)\}}"', setup)
    return m.group(1) if m else "UNKNOWN"

def tree_pin(tree: str) -> str:
    for line in setup.splitlines():
        if line.strip().startswith(f'"{tree}|'):
            parts = line.strip().strip('"').split("|")
            return parts[2]
    return "UNKNOWN"

def git_value(path: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNKNOWN"

def git_dirty(path: Path) -> str:
    try:
        subprocess.check_call(
            ["git", "-C", str(path), "diff", "--quiet", "--"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["git", "-C", str(path), "diff", "--cached", "--quiet", "--"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        untracked = subprocess.check_output(
            ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return "true" if untracked else "false"
    except subprocess.CalledProcessError:
        return "true"
    except Exception:
        return "unknown"

harness_path = workspace / "dlrm-v3-harness-rocm"
gr_path = workspace / "mlcommons-inference" / "recommendation" / "dlrm_v3"
pynve_path = workspace / "pynve-rocm"

def private_repo_block(name: str, url: str, pinned_ref: str, path: Path) -> str:
    vendored_path = f"vendor/{name}" if vendor_private_repos else "not_vendored"
    return f"""    url: {url}
    ref: {pinned_ref}
    vendored: {str(vendor_private_repos).lower()}
    vendored_path: {vendored_path}
    packaged_ref: {git_value(path, 'rev-parse', 'HEAD')}
"""

manifest = f"""# Generated by scripts/package/create_minimal_zip.sh
package:
  name: dlrm-v3-rocm-gold-quickstart
  benchmark: dlrm-v3
  scenario: Server
  target_qps: 12200
  checksum_manifest: manifests/checksums.sha256
  generated_utc: {dt.datetime.now(dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}

repos:
  runner:
    url: https://github.com/AMD-AGI/dlrm-v3-rocm-runner.git
    ref: {runner_ref}
    branch: {runner_branch}
  harness:
{private_repo_block('dlrm-v3-harness-rocm', 'https://github.com/AMD-AGI/dlrm-v3-harness-rocm.git', default_var('HARNESS_REPO_REF'), harness_path).rstrip()}
  gr:
{private_repo_block('dlrm-v3-gr-rocm', 'https://github.com/AMD-AGI/dlrm-v3-gr-rocm.git', default_var('GR_REPO_REF'), gr_path).rstrip()}
  pynve:
{private_repo_block('pynve-rocm', 'https://github.com/AMD-AGI/pynve-rocm.git', tree_pin('pynve-rocm'), pynve_path).rstrip()}
  mlcommons_inference:
    url: https://github.com/mlcommons/inference.git
    ref: {tree_pin('mlcommons-inference')}
    sparse:
      - loadgen
      - compliance/TEST08
  fbgemm:
    url: https://github.com/pytorch/fbgemm.git
    ref: {tree_pin('FBGEMM')}

container:
  image: rocm/atom:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom20260511
  digest: sha256:399cc8d9a003c92c30d64eec80b3589d8595eb3ba64e4edb573ef6899cd62fc4
  image_id: sha256:b257e9a833c8baa972089aba1bf789f4b8df6b417aa8274735181626ca37fb99
  created: 2026-05-11T17:00:41.044780581Z
  size_bytes: 47270334239
  runtime:
    python: 3.12.3
    torch: 2.10.0+rocm7.2.3.git1a270074
    triton: 3.6.0

data:
  dataset:
    destination: dlrmv3_preprocessed_full
    artifact_name: dlrmv3_preprocessed_full
    source: https://inference.mlcommons-storage.org/metadata/dlrm-v3-dataset.uri
    required_layout: metadata.json plus ts_90 through ts_99 directories
    validation_size_bytes: 149999298411
    validation_file_count: 75
    total_samples: 349823
    timestamps: [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    samples_per_timestamp: [34944, 35115, 34890, 35061, 35045, 34955, 34887, 34992, 35061, 34873]
    top_level_files: [metadata.json, offset.csv, requests_per_ts.csv, requests_per_ts_offset.csv, users_cumsum_per_ts.csv]
    sha256_manifest: MLCommons R2 metadata/checksum handled by downloader
  checkpoint:
    destination: dlrmv3_trained_checkpoint/dlrm-v3-checkpoint
    artifact_name: dlrm-v3-checkpoint
    source: https://inference.mlcommons-storage.org/metadata/dlrm-v3-checkpoint.uri
    required_layout: non_sparse.ckpt plus sparse/.metadata and sparse/__0_0.distcp through sparse/__7_0.distcp
    validation_size_bytes: 1034440601125
    validation_file_count: 12
    non_sparse_ckpt_size_bytes: 200434092
    sparse_metadata_size_bytes: 4176
    sparse_shards:
      - {{name: __0_0.distcp, size_bytes: 130560003206}}
      - {{name: __1_0.distcp, size_bytes: 130560003206}}
      - {{name: __2_0.distcp, size_bytes: 130560003206}}
      - {{name: __3_0.distcp, size_bytes: 130560003206}}
      - {{name: __4_0.distcp, size_bytes: 128000035974}}
      - {{name: __5_0.distcp, size_bytes: 128000035974}}
      - {{name: __6_0.distcp, size_bytes: 128000035974}}
      - {{name: __7_0.distcp, size_bytes: 128000035974}}
    auxiliary_files: [LICENSE.txt, dlrm-v3-checkpoint.md5]
    sha256_manifest: MLCommons R2 metadata/checksum handled by downloader
"""

(stage / "manifests").mkdir(parents=True, exist_ok=True)
(stage / "manifests/resources.lock.yaml").write_text(manifest)
(stage / "VERSION").write_text(
    f"package=dlrm-v3-rocm-gold-quickstart\n"
    f"generated_utc={dt.datetime.now(dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
    f"runner_ref={runner_ref}\n"
    f"runner_branch={runner_branch}\n"
    f"vendor_private_repos={str(vendor_private_repos).lower()}\n"
)
PY

cat > "${STAGE}/QUICKSTART.md" <<'EOF'
# DLRM-v3 ROCm GOLD Quickstart Package

This zip contains runner orchestration, pinned resource manifests, selected configs, support docs,
and vendored source snapshots of the private harness/GR/pynve repos under `vendor/`.
It intentionally does not include the dataset, checkpoint, Docker image layers, or build artifacts.

Typical internal flow after unpacking:

```bash
bash run.sh                         # data -> workspace -> submission -> run
STAGES=run bash run.sh              # rerun the GOLD Server cert after setup
bash scripts/run/_test08_chain.sh   # run TEST08 compliance chain
```

See `README.md` and `manifests/resources.lock.yaml` for the exact GOLD recipe and pinned resources.
EOF

echo "[package] writing checksums"
(
  cd "${STAGE}"
  find . -type f ! -path './manifests/checksums.sha256' -print0 \
    | sort -z \
    | xargs -0 sha256sum > manifests/checksums.sha256
)

echo "[package] creating ${ZIP_PATH}"
python3 - "${STAGE}" "${ZIP_PATH}" <<'PY'
import sys
import zipfile
from pathlib import Path

stage = Path(sys.argv[1])
zip_path = Path(sys.argv[2])
zip_path.parent.mkdir(parents=True, exist_ok=True)
root_name = stage.name
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(p for p in stage.rglob("*") if p.is_file()):
        zf.write(path, Path(root_name) / path.relative_to(stage))
PY

python3 - "${ZIP_PATH}" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
print(f"[package] wrote {p} ({p.stat().st_size / (1024 * 1024):.2f} MiB)")
PY
