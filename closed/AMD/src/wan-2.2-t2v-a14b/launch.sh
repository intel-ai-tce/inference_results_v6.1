#!/usr/bin/env bash
# Thin docker launcher for the wan-harness image.
#
# Usage:
#   ./launch.sh --build                  # build the image
#   ./launch.sh                          # drop into an interactive shell
#   ./launch.sh wan-harness run ...      # run a command inside the container
#
# Environment overrides:
#   WAN_HARNESS_IMAGE     tag for the built image (default: wan-harness:dev)
#   WAN_HARNESS_NAME      container name (default: wan-harness)
#   WAN_HARNESS_GPUS      GPU spec passed to docker (default: all on AMD or NVIDIA)
#   WAN_HARNESS_HF_CACHE  host dir bind-mounted to /hf_cache inside the
#                         container (default: /hf_cache). Persists HF model
#                         downloads -- and, via the VBENCH_CACHE_DIR=/hf_cache/
#                         vbench env var baked into the image, the ~few-GB
#                         VBench checkpoint set -- across container
#                         recreations.
#   WAN_VBENCH_PRETRAINED host dir bind-mounted to /hf_cache/vbench inside
#                         the container, overriding the default location
#                         derived from WAN_HARNESS_HF_CACHE. Use only when the
#                         VBench checkpoints need to live on a separate
#                         filesystem from the rest of /hf_cache.
#   WAN_HARNESS_RUNS_DIR  host dir bind-mounted over runs/ inside the
#                         container (default: <repo>/runs). Point this at a
#                         large/fast disk to keep experiment trees off the
#                         repo filesystem without changing run_all.sh paths.

set -euo pipefail

IMAGE="${WAN_HARNESS_IMAGE:-wan-harness:dev}"
NAME="${WAN_HARNESS_NAME:-wan-harness}"
HF_CACHE="${WAN_HARNESS_HF_CACHE:-/hf_cache}"
VBENCH_PRETRAINED="${WAN_VBENCH_PRETRAINED:-${HF_CACHE}/vbench}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${WAN_HARNESS_RUNS_DIR:-${REPO_ROOT}/runs}"

cmd_build() {
    docker build \
        -f "${REPO_ROOT}/docker/Dockerfile" \
        -t "${IMAGE}" \
        "${REPO_ROOT}"
}

detect_gpu_flags() {
    if [[ -n "${WAN_HARNESS_GPUS:-}" ]]; then
        echo "${WAN_HARNESS_GPUS}"
        return
    fi
    if [[ -e /dev/kfd ]] || [[ -d /dev/dri ]]; then
        # ROCm device exposure (AMD GPUs).
        echo "--device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "--gpus=all"
        return
    fi
    echo ""  # no GPU available
}

ensure_hf_cache_dir() {
    if [[ ! -d "${HF_CACHE}" ]]; then
        echo "[launch] creating HF cache dir on host: ${HF_CACHE}"
        # Try sudo if direct mkdir fails (e.g. /hf_cache at filesystem root).
        mkdir -p "${HF_CACHE}" 2>/dev/null || sudo mkdir -p "${HF_CACHE}"
    fi
}

ensure_vbench_pretrained_dir() {
    # VBench downloads its checkpoint set (~few GB) on first run, keyed by
    # the VBENCH_CACHE_DIR env var the image sets to /hf_cache/vbench (see
    # docker/Dockerfile). Bind-mounting from the host keeps that download
    # persistent across container recreations. Soft-create only -- if the
    # dir cannot be made (e.g. parent is read-only), skip silently rather
    # than blocking the launch.
    if [[ ! -d "${VBENCH_PRETRAINED}" ]]; then
        mkdir -p "${VBENCH_PRETRAINED}" 2>/dev/null \
            || sudo mkdir -p "${VBENCH_PRETRAINED}" 2>/dev/null \
            || true
    fi
}

ensure_runs_dir() {
    if [[ ! -d "${RUNS_DIR}" ]]; then
        echo "[launch] creating runs dir on host: ${RUNS_DIR}"
        mkdir -p "${RUNS_DIR}" 2>/dev/null || sudo mkdir -p "${RUNS_DIR}"
    fi
}

cmd_run() {
    local gpu_flags
    gpu_flags="$(detect_gpu_flags)"
    ensure_hf_cache_dir
    ensure_vbench_pretrained_dir
    ensure_runs_dir

    # Only add an explicit /hf_cache/vbench mount when the operator chose a
    # path that is NOT already inside ${HF_CACHE} -- the default case
    # (${HF_CACHE}/vbench) is already reachable through the /hf_cache mount
    # above, so a second bind here would just be redundant and would also
    # shadow the parent's permissions/ownership.
    local vbench_mount=()
    if [[ -d "${VBENCH_PRETRAINED}" ]] \
            && [[ "${VBENCH_PRETRAINED}" != "${HF_CACHE}/vbench" ]]; then
        vbench_mount=(-v "${VBENCH_PRETRAINED}:/hf_cache/vbench")
    fi

    docker run --rm -it \
        --name "${NAME}-$$" \
        --hostname "$(hostname)-docker" \
        ${gpu_flags} \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --shm-size=32g \
        -v "${REPO_ROOT}:/workspace/wan-harness" \
        -v "${RUNS_DIR}:/workspace/wan-harness/runs" \
        -v "${HF_CACHE}:/hf_cache" \
        "${vbench_mount[@]}" \
        -w /workspace/wan-harness \
        -e WAN_HARNESS_LOG_LEVEL="${WAN_HARNESS_LOG_LEVEL:-INFO}" \
        "${IMAGE}" \
        "$@"
}

case "${1:-}" in
    --build)
        cmd_build
        ;;
    "")
        cmd_run bash
        ;;
    *)
        cmd_run "$@"
        ;;
esac
