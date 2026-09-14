# syntax=docker/dockerfile:1.7
#
# MLPerf inference v6.1 harness for wan-2.2-t2v-a14b.
#
# Two principles:
#   1. Do not override anything the base image already ships. Adding new
#      pinned versions of torch / xDiT / diffusers / transformers risks
#      shadowing the curated stack in amdsiloai/pytorch-xdit:v26.6.
#   2. Clone mlcommons/inference cheaply. We pin to a single SHA, do a
#      no-history + no-blobs fetch, then sparse-checkout only the paths
#      we actually need (`loadgen/` for the install + `compliance/` for
#      TEST04 audit configs). Other benchmarks stay out of the image.
#
# Layer ordering: the slow-changing MLPerf checkout and VBench venv are
# installed FIRST, above the AITER / xDiT / Iris / DistVAE layers we
# iterate on far more often. Both are self-contained (MLPerf installs
# loadgen into system Python; VBench lives in an isolated /opt/vbench_env),
# so editing an AITER or xDiT SHA below no longer busts their (expensive)
# cache.

FROM amdsiloai/pytorch-xdit:v26.6

# Build-time-essential ENV only. Runtime configuration (HF_HOME,
# TORCH_HOME, WAN_FFMPEG_BIN, ...) is set at the bottom of this file so
# changes to it do not bust the cache of the expensive AITER / xDiT /
# VBench install layers below.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Only what we actually need on top of the base image:
#   - git: clone mlcommons/inference
#   - wget, unzip: occasional ad-hoc downloads from scripts
#   - ca-certificates: required by git over https
#   - ffmpeg: full Ubuntu build with libx264; the base image's ffmpeg is
#     a "GPL-free" build that omits libx264 (libx264 is GPLv2+).
#     Accuracy-mode video encoding (src/wan_harness/video_encoder.py)
#     defaults to libx264 to match the MLPerf reference; we then point
#     WAN_FFMPEG_BIN at /usr/bin/ffmpeg below so a conda-installed
#     ffmpeg earlier on PATH cannot shadow it.
# We run as root inside the container, so sudo is not needed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git wget unzip ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 # Fail the build immediately if the apt-installed ffmpeg somehow
 # doesn't include libx264 (e.g. mirror serving a stripped package);
 # better to learn at build time than at the first accuracy run.
 # Path is inlined (rather than via WAN_FFMPEG_BIN) so this RUN does
 # not depend on a top-of-file ENV that would otherwise need to live
 # up here for build correctness; see the runtime ENV block near the
 # end of this file.
 && /usr/bin/ffmpeg -hide_banner -encoders 2>/dev/null \
        | grep -qE '^[[:space:]]*V[^[:space:]]*[[:space:]]+libx264[[:space:]]' \
    || (echo "ERROR: /usr/bin/ffmpeg does not provide libx264 encoder" >&2; exit 1)

# Cheap, SHA-pinned MLPerf checkout. Three tricks make this small and fast:
#   1. `fetch --depth 1 origin <SHA>` skips history.
#   2. `--filter=blob:none` does a partial clone -- blob objects are fetched
#      lazily, only for paths we actually check out.
#   3. `sparse-checkout` (cone mode) restricts which paths are materialised,
#      so the partial clone above only ever pulls blobs for those paths.
# Net effect on disk vs. a full clone: ~250 MB -> ~20 MB.
#
# Override the SHA at build time with `--build-arg MLPERF_INFERENCE_COMMIT=<sha>`
# once the v6.1 seed lands.
ARG MLPERF_INFERENCE_COMMIT=2a3cb460f5edb9e3b173739c2937f081b4c861e9
ENV MLPERF_INFERENCE_DIR=/opt/mlperf-inference
RUN git init "${MLPERF_INFERENCE_DIR}" \
 && git -C "${MLPERF_INFERENCE_DIR}" remote add origin https://github.com/mlcommons/inference \
 && git -C "${MLPERF_INFERENCE_DIR}" config core.sparseCheckout true \
 && git -C "${MLPERF_INFERENCE_DIR}" sparse-checkout init --cone \
 && git -C "${MLPERF_INFERENCE_DIR}" sparse-checkout set loadgen compliance tools/submission \
 && git -C "${MLPERF_INFERENCE_DIR}" -c protocol.version=2 \
        fetch --depth 1 --filter=blob:none origin "${MLPERF_INFERENCE_COMMIT}" \
 && git -C "${MLPERF_INFERENCE_DIR}" checkout --detach FETCH_HEAD \
 && pip install --no-cache-dir "${MLPERF_INFERENCE_DIR}/loadgen"

# ---------------------------------------------------------------------------
# VBench evaluator: isolated venv at /opt/vbench_env.
#
# VBench pulls a large, opinionated dependency stack (its own torch/CUDA
# combo, plus transformers/timm/decord/openai-clip/...) that conflicts
# hard with the curated xDiT/AITER stack the base image ships (and that
# we customise below). We solve this
# the same way the v6.0 AMD-AGI reference did: a parallel venv, opaque to
# the main inference path, that the `with-vbench` shim activates.
#
# Inference (wan22 backend) runs from the system Python.
# VBench scoring (tools/run_vbench, `wan-harness vbench`) runs from
# /opt/vbench_env. The two never share a process.
#
# Override the VBench SHA via --build-arg if a future fix is required.
# ---------------------------------------------------------------------------

ARG VBENCH_DIR=/opt/vbench
ARG VBENCH_COMMIT=45e79ec14e69a2187202c675d2dbce1a71843d53
ARG ROCM_INDEX_URL=https://rocm.nightlies.amd.com/v2-staging/gfx950-dcgpu/
# Pin the VBench venv's ROCm runtime to the same major.minor as
# amdsiloai/pytorch-xdit:v26.6 (HIP 7.13.x). The earlier 7.12.0 pin was
# operationally fine for single-rank invocations (torch.cuda basic ops
# work despite the skew) but segfaulted during torch.distributed
# init_process_group with backend="nccl" on 8 MI355Xs: the bundled
# `_rocm_sdk_libraries_gfx950_dcgpu/lib/librccl.so.1` could not hand off
# correctly to the host's newer KFD driver. The 7.13.0 pin tracks the
# host runtime closely enough for RCCL to come up. There is no published
# `torch==2.9.1+rocm7.13.<host-exact-build>` wheel, so we lock to the
# 7.13.0 stream's last validated date stamp rather than a moving alias.
ARG ROCM_VERSION=7.13.0a20260515
ENV WAN_VBENCH_DIR=${VBENCH_DIR}
RUN python3 -m venv /opt/vbench_env \
 && /opt/vbench_env/bin/pip install --no-cache-dir --upgrade pip wheel \
 # ROCm-pinned torch stack, matching the v6.0 AMD-AGI reference.
 && /opt/vbench_env/bin/pip install --no-cache-dir \
        --index-url "${ROCM_INDEX_URL}" \
        "rocm[libraries,devel]==${ROCM_VERSION}" \
        "torch==2.9.1+rocm${ROCM_VERSION}" \
        "torchaudio==2.9.0+rocm${ROCM_VERSION}" \
        "torchvision==0.24.0+rocm${ROCM_VERSION}" \
 # Minimal VBench deps for the 6 reference dimensions
 # (subject_consistency, background_consistency, motion_smoothness,
 # dynamic_degree, appearance_style, scene). Non-reference dimensions
 # (e.g. imaging_quality, object_class) would need pyiqa / detectron2 /
 # lvis on top of this; intentionally not installed.
 && /opt/vbench_env/bin/pip install --no-cache-dir \
        "transformers==4.46.0" \
        decord \
        easydict \
        "fairscale>=0.4.4" \
        imageio \
        omegaconf \
        openai-clip \
        opencv-python \
        "setuptools==78.1.1" \
        scipy \
        "timm>=0.9,<=1.0.12" \
 # SHA-pinned VBench checkout. Shallow + blobless to keep the layer small.
 && git init "${VBENCH_DIR}" \
 && git -C "${VBENCH_DIR}" remote add origin https://github.com/Vchitect/VBench \
 && git -C "${VBENCH_DIR}" -c protocol.version=2 \
        fetch --depth 1 --filter=blob:none origin "${VBENCH_COMMIT}" \
 && git -C "${VBENCH_DIR}" checkout --detach FETCH_HEAD \
 # Expose ${VBENCH_DIR} on sys.path via a .pth file in the venv's
 # site-packages, instead of `pip install -e ${VBENCH_DIR}`.
 #
 # VBench's setup.py calls `import torch` then rejects anything where
 # `torch.version.cuda` is not in {"11.6", "11.7", "11.8", "12.1"}. Our
 # ROCm wheel reports `torch.version.cuda is None` (HIP is in
 # `torch.version.hip` instead), so the install always aborts. Even with
 # `--no-build-isolation`, the CUDA-version check would fail on ROCm.
 #
 # The MLPerf reference docker/Dockerfile in mlcommons/inference solves
 # this exact problem the same way: it sets PYTHONPATH to point at the
 # VBench checkout rather than installing it. We do the equivalent via a
 # .pth file so the path is wired into the venv permanently and works
 # for `with-vbench`, `source .../activate`, and direct
 # `/opt/vbench_env/bin/python` invocations alike.
 && /opt/vbench_env/bin/python -c "import sys, sysconfig, pathlib; \
        sp = pathlib.Path(sysconfig.get_paths()['purelib']); \
        (sp / 'vbench.pth').write_text('${VBENCH_DIR}\n')" \
 # Sanity check: `import vbench` must work straight out of the box.
 && /opt/vbench_env/bin/python -c "import vbench; print('vbench OK from', vbench.__file__)"

# Wrapper that flips into the VBench venv. Any VBench evaluation should be
# invoked as `with-vbench <cmd>` so the wan-harness inference path is
# unaffected by the VBench dep stack.
RUN printf '%s\n' \
        '#!/usr/bin/env bash' \
        'source /opt/vbench_env/bin/activate' \
        'exec "$@"' \
    > /usr/local/bin/with-vbench \
 && chmod +x /usr/local/bin/with-vbench

# ---------------------------------------------------------------------------
# Customisations on top of the base image's curated stack.
#
# The base image ships AITER and xDiT pre-installed as editable git
# checkouts (under /app/external/aiter and /app/xDiT respectively). We
# bring in changes that aren't yet upstream, including support for quantized
# dense attention.
# ---------------------------------------------------------------------------

ARG AITER_DIR=/app/external/aiter
ARG AITER_BRANCH=i8fp8_fmha_gfx950_sparse
ARG AITER_COMMIT=039f6fe8c128bafa0bce6cc81818b68846c974e3
# Avoid AITER's setup.py uninstalling triton and pulling a wheel from pypi.amd.com (see
# https://github.com/ROCm/aiter/pull/3161).
ENV AITER_USE_SYSTEM_TRITON=1
RUN test -d "${AITER_DIR}/.git" \
        || (echo "ERROR: ${AITER_DIR} is not a git checkout" >&2; exit 1) \
 && cd "${AITER_DIR}" \
 # Fetch directly from the canonical ROCm/aiter remote rather than relying
 # on whatever `origin` happens to be in the base image.
 && git fetch --tags https://github.com/ROCm/aiter.git "${AITER_BRANCH}" \
 && git checkout --detach "${AITER_COMMIT}" \
 && git submodule sync --recursive \
 && git submodule update --init --recursive \
 # Drop the pre-installed AITER (whatever form it took -- wheel or
 # editable) and reinstall in editable mode pointing at the new checkout.
 && pip uninstall -y amd-aiter \
 && pip install --no-cache-dir -e .

ARG XDIT_FORK_URL=https://github.com/ksikiric/xDiT.git
ARG XDIT_FORK_BRANCH=aiter_asm_kernels
ARG XDIT_COMMIT=efa43ba0c1f9c696b0fc7a0a04b62ef1fade29f4
RUN XDIT_DIR=$(pip show xfuser 2>/dev/null \
        | awk -F': +' '/^Editable project location:/ {print $2}') \
 && test -n "${XDIT_DIR}" -a -d "${XDIT_DIR}/xfuser" -a -d "${XDIT_DIR}/.git" \
        || (echo "ERROR: xDiT must be an editable git checkout (got '${XDIT_DIR}')" >&2; exit 1) \
 && cd "${XDIT_DIR}" \
 && echo "Repointing xDiT in ${XDIT_DIR} (was HEAD=$(git rev-parse --short HEAD))" \
 # The base image's /app/xDiT typically has uncommitted modifications
 # in the working tree (AMD-applied tweaks on top of the published
 # HEAD). Stash them so the checkout can proceed; the stash is kept
 # in the reflog for audit but never reapplied -- we want the fork's
 # state, not a hybrid.
 && git -c user.email=docker-build@local -c user.name=docker-build \
        stash push --include-untracked --message "pre-repoint base-image working tree" \
        || echo "(no local changes to stash)" \
 # Shallow fetch -- the four commits we care about plus their parent
 # all sit within the last 5 commits of aiter_asm_sparge.
 && git fetch --no-tags --depth=20 "${XDIT_FORK_URL}" "${XDIT_FORK_BRANCH}" \
 && git checkout --detach "${XDIT_COMMIT}" \
 && git submodule sync --recursive \
 && git submodule update --init --recursive \
 # Re-run the editable install so any new entry points / package
 # metadata in the fork's setup.py are picked up. --no-deps because
 # the base image already pins the full xDiT dependency closure at
 # the versions the surrounding stack expects, and we don't want pip
 # to second-guess it.
 && pip install --no-deps --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Iris + DistVAE for the ring-attention path.
#
#   1. Iris (ROCm's Triton RMA framework) is not in the base image. Install
#      nsakkine/iris@feature/ring-attention in editable mode so the checkout
#      stays inspectable and patchable inside the container.
#   2. The base image pulls DistVAE from PyPI as a transitive xfuser
#      dependency. Replace that wheel with nsakkine/DistVAE@
#      feature/wan-iris-ring-attention, which carries the Iris-backed ring
#      attention used by the distributed VAE path.
#
# Order matters: Iris is installed first because the DistVAE fork imports it.
# ---------------------------------------------------------------------------

ARG IRIS_FORK_URL=https://github.com/nsakkine/iris.git
ARG IRIS_FORK_BRANCH=feature/ring-attention
ARG IRIS_COMMIT=470c1cc4745be7618d8abb689bb94b90473aec12
ARG IRIS_DIR=/app/external/iris
RUN git init "${IRIS_DIR}" \
 && git -C "${IRIS_DIR}" remote add origin "${IRIS_FORK_URL}" \
 && git -C "${IRIS_DIR}" -c protocol.version=2 \
        fetch --depth 1 origin "${IRIS_COMMIT}" \
 && git -C "${IRIS_DIR}" checkout --detach FETCH_HEAD \
 && pip install --no-cache-dir -e "${IRIS_DIR}"

ARG DISTVAE_FORK_URL=https://github.com/nsakkine/DistVAE.git
ARG DISTVAE_FORK_BRANCH=feature/wan-iris-ring-attention
ARG DISTVAE_COMMIT=43159d6fca166008a5ee1fad9ce18b7ffb03a34a
RUN pip uninstall -y distvae \
 && pip install --no-cache-dir "git+${DISTVAE_FORK_URL}@${DISTVAE_COMMIT}"

# Intentionally a no-op for the first iteration; see docker/requirements.txt.
COPY docker/requirements.txt /tmp/requirements.txt
RUN if [ -s /tmp/requirements.txt ] && grep -qE '^[^#[:space:]]' /tmp/requirements.txt; then \
        pip install --no-cache-dir -r /tmp/requirements.txt ; \
    fi && rm -f /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Runtime configuration. Kept near the bottom so changes here only bust
# the cheap layers below (harness install + final COPY) instead of the
# expensive AITER / xDiT / mlperf / VBench install layers above.
#
# Set HF_HOME only -- do NOT also set HF_HUB_CACHE or (legacy)
# TRANSFORMERS_CACHE. `huggingface_hub` derives `${HF_HOME}/hub` itself,
# which matches the canonical on-host layout other tooling uses (most
# importantly the existing /hf_cache/hub/ tree on this host).
# Setting HF_HUB_CACHE=/hf_cache (or TRANSFORMERS_CACHE=/hf_cache) makes
# downloads land at /hf_cache/models--* instead of /hf_cache/hub/models--*,
# which silently duplicates anything the host already cached under the
# standard layout -- observed in practice as a 118 GB second copy of
# Wan-AI/Wan2.2-T2V-A14B-Diffusers next to the canonical one.
#
# VBENCH_CACHE_DIR controls where VBench stores its per-dimension
# checkpoints (AMT, RAFT, DINO, tag2text/scene, CLIP, ViCLIP, ...).
# Upstream reads this env var in `vbench/utils.py` and defaults to
# `~/.cache/vbench` when unset -- which inside our root container is
# `/root/.cache/vbench`, a non-persistent path that forces every fresh
# container to re-download multiple GB. Pointing it under /hf_cache
# instead keeps it bind-mounted via the same /hf_cache mount used for
# Hugging Face downloads, so VBench scoring pays the download cost
# exactly once per host. Do NOT point this at /opt/vbench/pretrained:
# that directory is only mentioned in VBench's bundled AMT third-party
# README and is never written to by the actual init_submodules flow.
#
# launch.sh bind-mounts the host's /hf_cache into /hf_cache, so the
# in-container path matches the host path one-to-one for easy cache
# introspection.
# ---------------------------------------------------------------------------

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache \
    VBENCH_CACHE_DIR=/hf_cache/vbench \
    TORCH_HOME=/workspace/.cache/torch \
    TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
    WAN_FFMPEG_BIN=/usr/bin/ffmpeg

# Disable direct solution methods slow for large convs.
ENV MIOPEN_DEBUG_CONV_DIRECT=0

# Soft-create the runtime cache dirs so fresh containers without the
# host bind-mount still have valid paths. Mounted containers ignore
# these (the host dir wins).
RUN mkdir -p /hf_cache /hf_cache/vbench \
        /workspace/.cache/torch /workspace/.cache/torch_extensions

# Install the harness itself.
WORKDIR /workspace/wan-harness
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Bring in the rest of the repo (configs, scripts, data dir, tests, etc.).
COPY . /workspace/wan-harness

COPY docker/entrypoint.sh /usr/local/bin/wan-harness-entrypoint
RUN chmod +x /usr/local/bin/wan-harness-entrypoint

# safe (aiter-native) + fast (vendored) fmha_v3 kernels; run_scenario.sh installs the per-scenario pick.
RUN mkdir -p /opt/aiter-kernels/safe/fmha_v3_fwd /opt/aiter-kernels/fast/fmha_v3_fwd \
 && cp /app/external/aiter/hsa/gfx950/fmha_v3_fwd/fwd_hd128_fp8.co \
       /app/external/aiter/hsa/gfx950/fmha_v3_fwd/fwd_hd128_mxfp4.co \
       /opt/aiter-kernels/safe/fmha_v3_fwd/
COPY docker/aiter-kernels/gfx950/fmha_v3_fwd/fwd_hd128_fp8.co \
     /opt/aiter-kernels/fast/fmha_v3_fwd/fwd_hd128_fp8.co
COPY docker/aiter-kernels/gfx950/fmha_v3_fwd/fwd_hd128_mxfp4.co \
     /opt/aiter-kernels/fast/fmha_v3_fwd/fwd_hd128_mxfp4.co

ENTRYPOINT ["/usr/local/bin/wan-harness-entrypoint"]
CMD ["bash"]
