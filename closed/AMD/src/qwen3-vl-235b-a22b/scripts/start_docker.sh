#!/usr/bin/env bash
# =====================================================================================
# Launch the AMD Qwen3-VL-235B (Q3VL) MLPerf v6.1 submission image with the
# src/qwen3-vl-235b-a22b/ directory mounted at /work. AMD / ROCm only.
#
# Flow (see README.md): 1) build the image, 2) run this launcher, 3) inside the container run the
# quantize + benchmark commands (README sections 2-3), 4) package + validate — either in-container
# (submission/ is mounted at /submission) or on the host (section 4).
#
# Requires: HF_CACHE = host Hugging Face cache dir (mounted read-write; holds the base model, the
# quantized checkpoint that section 2 writes, and the datasets). Optional: HF_TOKEN. Override the
# image with IMAGE=... ; start detached (for `docker exec`) with DETACH=1.
# =====================================================================================
set -euo pipefail

CONTAINER_NAME="${1:-q3vl-submission}"
IMAGE="${IMAGE:-amd-mlperf6.1-qwen3-vl-235b-a22b}"

# The src/qwen3-vl-235b-a22b/ dir (this script lives in scripts/, so go up one: holds scripts/,
# configs/, docker/) -> /work in-container.
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The submission root (two levels up: holds packager.py, systems/, measurements/, documentation/)
# -> /submission in-container, so package + validate can run inside the container, not only on the host.
SUBMISSION_DIR="$(cd "${BENCHMARK_DIR}/../.." && pwd)"

if [[ -z "${HF_CACHE:-}" ]]; then
    echo "error: set HF_CACHE to your Hugging Face cache dir, e.g. export HF_CACHE=/data/hf" >&2
    exit 1
fi
mkdir -p "${HF_CACHE}" "${SUBMISSION_DIR}/outputs"

echo "launching '${CONTAINER_NAME}': ${IMAGE}"
echo "  src/qwen3-vl-235b-a22b ${BENCHMARK_DIR} -> /work (WORKDIR)"
echo "  submission ${SUBMISSION_DIR} -> /submission (runs land in /submission/outputs; packager.py + systems/measurements/documentation)"
echo "  HF_CACHE ${HF_CACHE} -> /root/.cache/huggingface"

# umask 000 so outputs/ written as root inside are editable/deletable by any host user afterwards.
if [[ "${DETACH:-0}" == "1" ]]; then
    RUN_TTY=(-d);  RUN_CMD=(-c 'umask 000; exec sleep infinity')
else
    RUN_TTY=(-it); RUN_CMD=(-c 'umask 000; exec /bin/bash')
fi

# AMD/ROCm GPU access (matches the tested config): KFD + DRI devices, video group, unconfined seccomp
# (so NUMA-pinned workers can call set_mempolicy).
docker run \
    "${RUN_TTY[@]}" \
    --name "${CONTAINER_NAME}" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --network=host \
    --shm-size=16g \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}" \
    -e "GPU_NAME=${GPU_NAME:-}" \
    -w /work \
    -v "${BENCHMARK_DIR}:/work" \
    -v "${SUBMISSION_DIR}:/submission" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "${IMAGE}" \
    "${RUN_CMD[@]}"
