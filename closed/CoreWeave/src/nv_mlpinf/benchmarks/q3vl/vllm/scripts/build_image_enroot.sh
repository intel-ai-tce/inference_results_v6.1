#!/bin/bash
# Build the q3vl/vllm container image using enroot (no Docker daemon required).
# All sources are cloned from git inside the container — no local checkouts needed.
#
# Usage:
#   bash build_image_enroot.sh [options]
#   sbatch build_image_enroot.sh [options]
#
# The script dispatches into itself: the host runs the default path,
# enroot re-invokes it with "build-vllm", "dynamo", or "mlperf" as $1.

set -eux
set -o pipefail

# ── Architecture detection (shared by functions and host orchestration) ───────
# Sets DETECTED_UNAME (raw uname -m), DETECTED_ARCH (docker-style: arm64/amd64),
# and DETECTED_UUARCH (gdrcopy/etcd-style: aarch64/x64).
DETECTED_UNAME=$(uname -m)
case "${DETECTED_UNAME}" in
    aarch64|arm64) DETECTED_ARCH=arm64;  DETECTED_UUARCH=aarch64 ;;
    x86_64)        DETECTED_ARCH=amd64;  DETECTED_UUARCH=x64 ;;
    *)             echo "Warning: Unknown architecture ${DETECTED_UNAME}, defaulting to amd64"
                   DETECTED_ARCH=amd64;  DETECTED_UUARCH=x64 ;;
esac

# ── Helpers ───────────────────────────────────────────────────────────────────
# Convert a git URL to a slug for use in image tags/filenames.
_repo_slug() { echo "$1" | sed -e 's|https://github.com/||' -e 's|\.git$||' -e 's|/|_|g'; }

# ═════════════════════════════════════════════════════════════════════════════
# In-container: build vllm from source
# ═════════════════════════════════════════════════════════════════════════════
function _build_vllm() {
    local vllm_repo=$1 vllm_revision=$2 cuda_version=$3

    export HOME=/root
    export DEBIAN_FRONTEND=noninteractive

    local python_version=${PYTHON_VERSION:-3.12}
    local cuda_major=$(echo "${cuda_version}" | cut -d. -f1)
    local cuda_major_minor=$(echo "${cuda_version}" | cut -d. -f1,2 | tr -d '.')
    local cuda_major_minor_dot=$(echo "${cuda_version}" | cut -d. -f1,2)
    local cuda_version_dash=$(echo "${cuda_version}" | cut -d. -f1,2 | tr '.' '-')

    export MAX_JOBS=${MAX_JOBS:-256}
    export NVCC_THREADS=${NVCC_THREADS:-2}
    export TORCH_CUDA_ARCH_LIST='9.0 10.0+PTX 10.3'

    local pytorch_index="https://download.pytorch.org/whl/cu${cuda_major_minor}"
    local workspace=/workspace
    mkdir -p "${workspace}"

    # ── base stage: system packages + Python via deadsnakes ─────────────
    echo 'tzdata tzdata/Areas select America' | debconf-set-selections
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections
    apt-get update -y
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ccache git curl sudo python3-pip \
        ffmpeg libsm6 libxext6 libgl1 \
        libibverbs-dev gcc-10 g++-10
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-10

    # Python 3.12 via deadsnakes (ubuntu 22.04 ships 3.10)
    for i in 1 2 3; do
        add-apt-repository -y ppa:deadsnakes/ppa && break || \
        { echo "Attempt $i failed, retrying..."; sleep 5; }
    done
    apt-get update -y
    apt-get install -y --no-install-recommends \
        "python${python_version}" \
        "python${python_version}-dev" \
        "python${python_version}-venv"

    update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python${python_version}" 1
    update-alternatives --set python3 "/usr/bin/python${python_version}"
    ln -sf "/usr/bin/python${python_version}-config" /usr/bin/python3-config
    curl -sS https://bootstrap.pypa.io/get-pip.py | "python${python_version}"

    # uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:${PATH}"
    export UV_HTTP_TIMEOUT=500
    export UV_INDEX_STRATEGY="unsafe-best-match"
    export UV_LINK_MODE=copy

    # CUDA compat
    echo "/usr/local/cuda-${cuda_major_minor_dot}/compat/" \
        > /etc/ld.so.conf.d/00-cuda-compat.conf
    ldconfig

    # CUDA runtime packages needed for JIT compilation
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        cuda-nvcc-${cuda_version_dash} \
        cuda-cudart-${cuda_version_dash} \
        cuda-nvrtc-${cuda_version_dash} \
        cuda-cuobjdump-${cuda_version_dash} \
        libcurand-dev-${cuda_version_dash} \
        libcublas-${cuda_version_dash}
    local nccl_version
    nccl_version=$(dpkg -l libnccl2 2>/dev/null | awk '/^ii/{print $3}')
    if [ -n "${nccl_version}" ]; then
        apt-get install -y --no-install-recommends "libnccl-dev=${nccl_version}" || \
            echo "libnccl-dev pin failed, skipping"
    else
        apt-get install -y --no-install-recommends libnccl-dev || \
            echo "libnccl-dev install failed, skipping"
    fi
    rm -rf /var/lib/apt/lists/*

    # ── Clone vllm source ─────────────────────────────────────────────────
    git clone "${vllm_repo}" "${workspace}/vllm"
    cd "${workspace}/vllm"
    git checkout "${vllm_revision}"

    # ── Install PyTorch + CUDA requirements ───────────────────────────────
    uv pip install --system \
        -r requirements/cuda.txt --extra-index-url "${pytorch_index}"
    uv pip install --system \
        -r requirements/build.txt --extra-index-url "${pytorch_index}"

    # ── DeepGEMM (optional) ───────────────────────────────────────────────
    mkdir -p /tmp/deepgemm/dist
    VLLM_DOCKER_BUILD_CONTEXT=1 TORCH_CUDA_ARCH_LIST="9.0a 10.0a" \
        bash tools/install_deepgemm.sh \
            --cuda-version "${cuda_version}" \
            --ref 594953acce41793ae00a1233eb516044d604bcb6 \
            --wheel-dir /tmp/deepgemm/dist \
        || echo "DeepGEMM build skipped"

    # ── pplx-kernels + DeepEP (optional) ──────────────────────────────────
    mkdir -p /tmp/ep_kernels_workspace/dist
    TORCH_CUDA_ARCH_LIST="9.0a 10.0a" \
        bash tools/ep_kernels/install_python_libraries.sh \
            --workspace /tmp/ep_kernels_workspace \
            --mode wheel \
            --pplx-ref 12cecfd \
            --deepep-ref 73b6ea4 \
        || echo "EP kernels build skipped"
    export TORCH_CUDA_ARCH_LIST='9.0 10.0+PTX 10.3'

    # ── Compile vllm wheel ────────────────────────────────────────────────
    mkdir -p .deps dist
    export VLLM_DOCKER_BUILD_CONTEXT=1
    export VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1
    python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38

    # ── Install vllm wheel ────────────────────────────────────────────────
    uv pip install --system dist/*.whl --verbose \
        --extra-index-url "${pytorch_index}"

    if ls /tmp/deepgemm/dist/*.whl 2>/dev/null; then
        uv pip install --system /tmp/deepgemm/dist/*.whl
    fi
    if ls /tmp/ep_kernels_workspace/dist/*.whl 2>/dev/null; then
        uv pip install --system /tmp/ep_kernels_workspace/dist/*.whl --verbose \
            --extra-index-url "${pytorch_index}"
    fi

    # ── FlashInfer (CentML fork with cuDNN FP8 support) ─────────────────
    local flashinfer_version=0.5.3
    local flashinfer_repo=https://github.com/CentML/flashinfer.git
    local flashinfer_branch=mlperf-inf-mm-q3vl-v6.0-rc2
    git clone --recursive -b "${flashinfer_branch}" "${flashinfer_repo}" /tmp/flashinfer
    cd /tmp/flashinfer
    uv pip install --system --no-build-isolation -v .
    cd "${workspace}/vllm"
    rm -rf /tmp/flashinfer
    uv pip install --system "flashinfer-cubin==${flashinfer_version}"
    uv pip install --system "flashinfer-jit-cache==${flashinfer_version}" \
        --extra-index-url "https://flashinfer.ai/whl/cu${cuda_major_minor}"
    flashinfer show-config

    # ── gdrcopy ───────────────────────────────────────────────────────────
    bash tools/install_gdrcopy.sh "Ubuntu22_04" "12.8" "${DETECTED_UUARCH}" \
        || echo "gdrcopy install skipped"

    # ── vllm-openai extras ────────────────────────────────────────────────
    uv pip install --system \
        accelerate hf_transfer modelscope \
        "bitsandbytes>=0.42.0" "timm>=1.0.17" \
        "runai-model-streamer[s3,gcs]>=0.15.3"
    uv pip install --system "nixl-cu${cuda_major}"

    echo "vllm build complete."
}

# ═════════════════════════════════════════════════════════════════════════════
# In-container: install dynamo + dependencies
# ═════════════════════════════════════════════════════════════════════════════
function _dynamo() {
    local dynamo_repo=$1 dynamo_revision=$2 cuda_version=$3

    export HOME=/root
    export DEBIAN_FRONTEND=noninteractive

    local cuda_major_version=$(echo "${cuda_version}" | cut -d. -f1)
    local workdir=/vllm-workspace
    mkdir -p "${workdir}"

    # ── System dependencies ───────────────────────────────────────────────
    apt-get update && \
    apt-get -y install \
        build-essential ca-certificates cmake curl libclang-dev libhwloc-dev \
        libopenmpi-dev libudev-dev numactl openmpi-bin pkg-config \
        protobuf-compiler python3-dev tmux vim git \
    && rm -rf /var/lib/apt/lists/*

    # ── Rust ──────────────────────────────────────────────────────────────
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

    # ── Python packages ───────────────────────────────────────────────────
    uv pip install --system --no-cache --verbose \
        'triton>=3.5.1' maturin nixl "nixl-cu${cuda_major_version}" pip

    export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
    export TRITON_PTXAS_BLACKWELL_PATH=${TRITON_PTXAS_PATH}

    # ── Dynamo ────────────────────────────────────────────────────────────
    cd "${workdir}"
    git clone "${dynamo_repo}" dynamo
    cd "${workdir}/dynamo"
    git checkout "${dynamo_revision}"

    cd "${workdir}/dynamo/lib/bindings/python"
    . "${HOME}/.cargo/env" && maturin build --release -i python3
    uv pip install target/wheels/*.whl --system

    cd "${workdir}/dynamo"
    uv pip install . --system

    # ── ETCD ──────────────────────────────────────────────────────────────
    local etcd_ver=v3.6.6
    mkdir -p /opt/etcd
    curl -L "https://storage.googleapis.com/etcd/${etcd_ver}/etcd-${etcd_ver}-linux-${DETECTED_ARCH}.tar.gz" \
        -o /tmp/etcd.tar.gz
    tar xzvf /tmp/etcd.tar.gz -C /opt/etcd --strip-components=1 --no-same-owner
    rm /tmp/etcd.tar.gz

    # ── NATS ──────────────────────────────────────────────────────────────
    cd /opt
    curl -fsSL https://binaries.nats.dev/nats-io/nats-server/v2@v2.11.6 | sh

    # Persist runtime environment variables
    echo 'PATH="/opt/:/opt/etcd/:${PATH}"' >> /etc/environment
    echo 'TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"' >> /etc/environment
    echo 'TRITON_PTXAS_BLACKWELL_PATH="/usr/local/cuda/bin/ptxas"' >> /etc/environment

    echo "Dynamo install complete."
}

# ═════════════════════════════════════════════════════════════════════════════
# In-container: install mlperf packages
# ═════════════════════════════════════════════════════════════════════════════
function _mlperf() {
    local mlperf_inf_mm_q3vl_install_url=$1 mlperf_inf_mm_q3vl_nv_install_url=$2

    export HOME=/root
    export DEBIAN_FRONTEND=noninteractive
    export PATH="/opt/:/opt/etcd/:${PATH}"

    # For local paths mounted read-only, copy to a writable temp dir first
    # (setuptools needs to write egg-info into the source tree).
    # Allowlist only the files pip needs to build/install the package, so we
    # don't drag build artifacts, VCS state, or a stray output sqsh into the
    # final image. Extend _pkg_items if a package legitimately needs more.
    _copy_src() {
        local src=$1 dst=$2
        mkdir -p "${dst}"
        local item
        for item in pyproject.toml README.md src; do
            cp -r "${src}/${item}" "${dst}/"
        done
    }
    _install_url="${mlperf_inf_mm_q3vl_install_url}"
    _install_nv_url="${mlperf_inf_mm_q3vl_nv_install_url}"
    if [[ "${_install_url}" == /tmp/q3vl* ]]; then
        _copy_src "${_install_url}" /tmp/_q3vl_build
        _install_url="/tmp/_q3vl_build"
    fi
    if [[ "${_install_nv_url}" == /tmp/q3vl* ]]; then
        _copy_src "${_install_nv_url}" /tmp/_q3vl_nv_build
        _install_nv_url="/tmp/_q3vl_nv_build"
    fi

    # ── mlperf-inf-mm-q3vl ────────────────────────────────────────────────
    uv pip install --system --no-cache --verbose "${_install_url}"

    # ── mlperf-inf-mm-q3vl-nv ────────────────────────────────────────────
    uv pip install --system --no-cache --verbose "${_install_nv_url}"

    echo "MLPerf install complete."
}

# ═════════════════════════════════════════════════════════════════════════════
# Dispatch: if called with a subcommand, run the corresponding function
# ═════════════════════════════════════════════════════════════════════════════
case "${1:-}" in
    build-vllm) shift; _build_vllm "$@"; exit 0 ;;
    dynamo)     shift; _dynamo "$@";      exit 0 ;;
    mlperf)     shift; _mlperf "$@";      exit 0 ;;
esac

# ═════════════════════════════════════════════════════════════════════════════
# Host-side orchestration (default entry point)
# ═════════════════════════════════════════════════════════════════════════════

# ── Defaults (mirror ../build_image.sh) ───────────────────────────────────────
vllm_repo=https://github.com/CentML/vllm.git
vllm_revision=mlperf-inf-mm-q3vl-v6.0
dynamo_repo=https://github.com/CentML/dynamo.git
dynamo_revision=mlperf-inf-mm-q3vl-v6.0
mlperf_inf_mm_q3vl_install_url="git+https://github.com/mlcommons/inference.git#subdirectory=multimodal/qwen3-vl/"
mlperf_inf_mm_q3vl_nv_install_url=$(pwd)
cuda_version=13.0.1
result_image_repo=gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv
sqsh_output_dir=$(pwd)/build
result_image_tag=""
vllm_base_sqsh=""
cache_vllm_base=false
container_name=""

function _usage() {
    cat <<EOF
Build the q3vl/vllm enroot container image.

Usage: ${BASH_SOURCE[0]} [options]

Options:
  -h, --help                                                Print this help message.
  --vllm-repo <vllm_repo>                                   The repository to use for vLLM (default: ${vllm_repo}).
  --vllm-revision <vllm_revision>                            The revision to use for vLLM (default: ${vllm_revision}).
  --vllm-build-cuda-version <vllm_build_cuda_version>        The CUDA version to use for vLLM build (default: ${cuda_version}).
  --result-image-repo <result_image_repo>                    The docker image repository for the result image (default: ${result_image_repo}).
  --result-image-tag <result_image_tag>                      Full result image tag (overrides repo+tag).
  --dynamo-repo <dynamo_repo>                                The repository to use for Dynamo (default: ${dynamo_repo}).
  --dynamo-revision <dynamo_revision>                        The revision to use for Dynamo (default: ${dynamo_revision}).
  --mlperf-inf-mm-q3vl-install-url <url>                     The URL to use for mlperf-inf-mm-q3vl (default: ${mlperf_inf_mm_q3vl_install_url}).
  --mlperf-inf-mm-q3vl-nv-install-url <url>                  The URL or path for mlperf-inf-mm-q3vl-nv (default: local project dir).
  --vllm-base-sqsh <path>                                    Pre-built vllm sqsh to skip vllm compilation.
  --cache-vllm-base                                         Export vllm sqsh to cache dir after building.
  --sqsh-output-dir <dir>                                    Output directory for sqsh file (default: cwd/build).
  --container-name <name>                                    enroot container name (default: q3vl-build-<jobid|pid>).
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)                          _usage 0 ;;
        --vllm-repo)                        vllm_repo=$2; shift 2 ;;
        --vllm-revision)                    vllm_revision=$2; shift 2 ;;
        --vllm-build-cuda-version)          cuda_version=$2; shift 2 ;;
        --result-image-repo)                result_image_repo=$2; shift 2 ;;
        --result-image-tag)                 result_image_tag=$2; shift 2 ;;
        --dynamo-repo)                      dynamo_repo=$2; shift 2 ;;
        --dynamo-revision)                  dynamo_revision=$2; shift 2 ;;
        --mlperf-inf-mm-q3vl-install-url)   mlperf_inf_mm_q3vl_install_url=$2; shift 2 ;;
        --mlperf-inf-mm-q3vl-nv-install-url) mlperf_inf_mm_q3vl_nv_install_url=$2; shift 2 ;;
        --vllm-base-sqsh)                   vllm_base_sqsh=$2; shift 2 ;;
        --cache-vllm-base)                  cache_vllm_base=true; shift ;;
        --sqsh-output-dir)                  sqsh_output_dir=$2; shift 2 ;;
        --container-name)                   container_name=$2; shift 2 ;;
        *) echo "Unknown option: $1"; _usage 1 ;;
    esac
done

vllm_build_arch="${DETECTED_ARCH}"

# Compute the result image tag (same scheme as ../build_image.sh).
if [ -z "${result_image_tag}" ]; then
    _dynamo_slug=$(_repo_slug "${dynamo_repo}")
    _vllm_slug=$(_repo_slug "${vllm_repo}")
    result_tag="${vllm_build_arch}_cuda${cuda_version}_${_dynamo_slug}-${dynamo_revision}_${_vllm_slug}-${vllm_revision}"
    if [ ${#result_tag} -gt 128 ]; then result_tag="${result_tag:0:128}"; fi
    result_image_tag="${result_image_repo}:${result_tag}"
fi

# The script mounts itself into the container as a single file.
# Under sbatch BASH_SOURCE points to the spool copy (named slurm_script),
# but the content is this script, so we mount it with an explicit target name.
SELF=$(realpath "${BASH_SOURCE[0]}")

# ── Intermediate result caching ───────────────────────────────────────────────
_cache_dir="${sqsh_output_dir}/cache"
mkdir -p "${_cache_dir}"

cuda_base_sqsh="${_cache_dir}/cuda-${cuda_version}-devel-ubuntu22.04-${vllm_build_arch}.sqsh"

# ── enroot container name ─────────────────────────────────────────────────────
container_name="${container_name:-q3vl-build-${SLURM_JOB_ID:-$$}}"

# Ensure the writable container is removed on any exit (success, failure, ^C).
trap 'enroot remove -f "${container_name}" 2>/dev/null || true' EXIT

echo "Building result image: ${result_image_tag}"
echo "enroot container:      ${container_name}"

if [ -n "${vllm_base_sqsh}" ]; then
    # Fast path: use provided vllm sqsh, skip steps 1-3
    echo "Using provided vllm sqsh: ${vllm_base_sqsh}"
    enroot create --name "${container_name}" "${vllm_base_sqsh}"
else
    # ── Step 1: import the CUDA devel base image (cached) ─────────────────
    if [ ! -f "${cuda_base_sqsh}" ]; then
        echo "Downloading CUDA base image..."
        enroot import \
            --arch "${DETECTED_UNAME}" \
            -o "${cuda_base_sqsh}" \
            "docker://nvidia/cuda:${cuda_version}-devel-ubuntu22.04"
    else
        echo "Reusing cached CUDA base sqsh: ${cuda_base_sqsh}"
    fi

    # ── Step 2: create a writable container from the base ─────────────────
    enroot create --name "${container_name}" "${cuda_base_sqsh}"

    # ── Step 3: build vllm from source (cloned inside container) ──────────
    enroot start \
        --root \
        --rw \
        --mount "${SELF}:/build_image_enroot.sh" \
        "${container_name}" \
        -- bash /build_image_enroot.sh build-vllm \
            "${vllm_repo}" "${vllm_revision}" "${cuda_version}"

    # Optionally cache the vllm image for future runs
    if [ "${cache_vllm_base}" = true ]; then
        _vllm_cache_slug=$(_repo_slug "${vllm_repo}")
        _vllm_cache_sqsh="${_cache_dir}/vllm-${_vllm_cache_slug}-${vllm_revision}-cuda${cuda_version}-${vllm_build_arch}.sqsh"
        echo "Caching vllm image to: ${_vllm_cache_sqsh}"
        enroot export --force -o "${_vllm_cache_sqsh}" "${container_name}"
    fi
fi

# ── Step 4: install dynamo ─────────────────────────────────────────────────────
enroot start \
    --root \
    --rw \
    --mount "${SELF}:/build_image_enroot.sh" \
    "${container_name}" \
    -- bash /build_image_enroot.sh dynamo \
        "${dynamo_repo}" "${dynamo_revision}" "${cuda_version}"

# ── Step 5: install mlperf packages ───────────────────────────────────────────
# If URLs are local paths, mount them into the container.
_mounts=("--mount" "${SELF}:/build_image_enroot.sh")
_q3vl_url="${mlperf_inf_mm_q3vl_install_url}"
_q3vl_nv_url="${mlperf_inf_mm_q3vl_nv_install_url}"

# For local paths, mount with x-create=dir so enroot creates the target.
if [[ "${mlperf_inf_mm_q3vl_install_url}" == /* ]]; then
    _mounts+=("--mount" "${mlperf_inf_mm_q3vl_install_url}:/tmp/q3vl:x-create=dir,bind,ro")
    _q3vl_url="/tmp/q3vl"
fi
if [[ "${mlperf_inf_mm_q3vl_nv_install_url}" == /* ]]; then
    _mounts+=("--mount" "${mlperf_inf_mm_q3vl_nv_install_url}:/tmp/q3vl_nv:x-create=dir,bind,ro")
    _q3vl_nv_url="/tmp/q3vl_nv"
fi

enroot start \
    --root \
    --rw \
    "${_mounts[@]}" \
    "${container_name}" \
    -- bash /build_image_enroot.sh mlperf \
        "${_q3vl_url}" "${_q3vl_nv_url}"

# ── Step 6: export the finished container to a sqsh file ─────────────────────
sqsh_name=$(echo "${result_image_tag}" | sed 's|/|+|g; s|:|+|g')
sqsh_file="${sqsh_output_dir}/${sqsh_name}.sqsh"
enroot export --force -o "${sqsh_file}" "${container_name}"
echo "sqsh written to: ${sqsh_file}"

# Step 7 (cleanup) handled by the EXIT trap above.
