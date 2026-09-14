# =====================================================================================
# Qwen3-VL (Q3VL) MLPerf v6.1 -- IMG-S serving image.
# Self-contained IMG-S serving image (frozen from the repo's vllm-rocm-vllm025-imgS.Dockerfile).
# COPY paths are relative to THIS directory (src/qwen3-vl-235b-a22b/); build with it as the context:
#   docker build -f docker/Dockerfile --network=host -t amd-mlperf6.1-qwen3-vl-235b-a22b .
# See README.md section 1.
# =====================================================================================
ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.25.1
FROM ${BASE_IMAGE}

ENV VLLM_TARGET_DEVICE=rocm

ARG Q3VL_QUARK=1
ARG Q3VL_FUSIONS=1
ARG Q3VL_UATTN=1
ARG Q3VL_FP8VIT=1
ARG Q3VL_FLYDSL_MROPE=1

# ---- System tools + harness deps ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends git tmux vim build-essential haproxy numactl && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir uv hydra-core mlflow-skinny databricks-sdk

# =====================================================================================
# vLLM 0.25 -- patch the PREBUILT wheel in place (pure-Python patches; no source rebuild).
# =====================================================================================
COPY docker/patches_vllm25/0023-quark-q3vl-scale-name-vllm025.patch /tmp/0023.patch
COPY docker/patches_vllm25/0024-vllm025-native-w4a8-flydsl-afp8-wfp4.patch    /tmp/0024.patch
COPY docker/patches_vllm25/0025-vllm025-native-w4a16-mxfp4mxfp4-fp4act.patch  /tmp/0025.patch
COPY docker/patches_vllm25/0001-rocm-prefer-aiter-fa-decoder.patch  /tmp/0001.patch
COPY docker/patches_vllm25/0011-q3vl-vit-fmha-hd128-asm.patch       /tmp/0011.patch
COPY docker/patches_vllm25/0012-q3vl-vit-fmha-binned.patch          /tmp/0012.patch
COPY docker/patches_vllm25/0033-q3vl-aiter-rmsnorm-flexible-layout.patch /tmp/0033.patch

RUN VLLM_SITE=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") && \
    echo "patching vLLM at ${VLLM_SITE}" && \
    patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0023.patch && \
    grep -q "_quark_scale_rename" "${VLLM_SITE}/vllm/model_executor/models/utils.py" && \
    echo "OK 0023 (loader scale rename)"

RUN VLLM_SITE=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") && \
    patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0024.patch && \
    grep -q "F-patch" "${VLLM_SITE}/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py" && \
    echo "OK 0024 (native W4A8 afp8_wfp4)" && \
    patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0025.patch && \
    grep -q "F2-patch" "${VLLM_SITE}/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py" && \
    echo "OK 0025 (native W4A16 afp4_wfp4)"

RUN VLLM_SITE=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") && \
    patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0001.patch && echo "OK 0001 (prefer aiter FA decoder)" && \
    patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0011.patch && \
    grep -q "VLLM_Q3VL_VIT_FMHA_ASM" "${VLLM_SITE}/vllm/v1/attention/ops/vit_attn_wrappers.py" && \
    echo "OK 0011 (ViT-FMHA hd128 ASM)" && \
    patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0012.patch && \
    grep -q "VLLM_Q3VL_VIT_FMHA_BINNED" "${VLLM_SITE}/vllm/v1/attention/ops/vit_attn_wrappers.py" && \
    echo "OK 0012 (ViT-FMHA binned)" && \
    patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0033.patch && \
    grep -q "flexible_layout" "${VLLM_SITE}/vllm/_aiter_ops.py" && \
    echo "OK 0033 (aiter rmsnorm flexible-layout copy-elim)"

COPY docker/patches_vllm25/0007-vit-add-layernorm-fuse.patch        /tmp/0007.patch
COPY docker/patches_vllm25/0009-q3vl-fuse-attn-prologue.patch       /tmp/0009.patch
COPY docker/patches_vllm25/0014-q3vl-vit-gelu-fc1-epilogue.patch    /tmp/0014.patch
COPY docker/patches_vllm25/0016-q3vl-fused-router-sort-vllm.patch   /tmp/0016-vllm.patch
RUN if [ "${Q3VL_FUSIONS}" = "1" ]; then \
        VLLM_SITE=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") && \
        patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0007.patch      && echo "OK 0007 (ViT add+LN)" && \
        patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0009.patch      && echo "OK 0009 (attn prologue)" && \
        patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0014.patch      && echo "OK 0014 (ViT GELU->fc1)" && \
        patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0016-vllm.patch && echo "OK 0016-vllm (router-sort)"; \
    fi

# quark quantization support
RUN if [ "${Q3VL_QUARK}" = "1" ]; then \
        python3 -m pip install --no-cache-dir "datasets>=4.0" openai pydantic && \
        echo "quark $(python3 -c 'import quark;print(quark.__version__)') ready (native tf>=5 MoE unfuse; no 0004)"; \
    fi

# AITER
ARG AITER_COMMIT=d7c405147
ARG AITER_GPU_ARCHS="gfx950"

COPY docker/patches_vllm25/0026-aiter-unified-attn-triton-opt.patch /tmp/0026.patch
COPY docker/patches_vllm25/0013-q3vl-moe-prefill-fuse-quant.patch      /tmp/0013.patch
COPY docker/patches_vllm25/0015-q3vl-moe-reduce-atomic-epilogue.patch  /tmp/0015.patch
COPY docker/patches_vllm25/0016-q3vl-fused-router-sort-aiter.patch     /tmp/0016-aiter.patch
COPY docker/patches_vllm25/0008-aiter-mrope-flash-layout.patch         /tmp/0008.patch

COPY docker/patches_vllm25/q3vl_sqn12_tuned_fmoe.csv /tmp/q3vl_sqn12_tuned_fmoe.csv
RUN if [ -n "${AITER_COMMIT}" ]; then \
        pip uninstall -y aiter amd-aiter || true && \
        git clone https://github.com/ROCm/aiter.git /opt/aiter && \
        cd /opt/aiter && \
        git checkout "${AITER_COMMIT}" && \
        git submodule update --init --depth 1 3rdparty/composable_kernel && \
        if [ "${Q3VL_UATTN}" = "1" ]; then \
            git apply --verbose /tmp/0026.patch && \
            grep -q "force_2d_decode" /opt/aiter/aiter/ops/triton/attention/unified_attention.py && \
            echo "OK 0026 (unified-attn opt baked, Q3VL_UATTN=1)"; \
        else echo "SKIP 0026 (Q3VL_UATTN=0 -- no unified-attn opt)"; fi && \
        if [ "${Q3VL_FUSIONS}" = "1" ]; then \
            (git apply --verbose /tmp/0013.patch || echo "WARN: 0013 (moe prefill fuse-quant) did not seat on d7c405147 -- re-seat (plan §6)") && \
            (git apply --verbose /tmp/0015.patch || echo "WARN: 0015 (moe reduce->atomic) did not seat on d7c405147 -- re-seat (plan §6)") && \
            git apply --verbose /tmp/0016-aiter.patch && echo "OK 0016-aiter (router-sort, re-seated for d7c405147)" && \
            git apply --verbose /tmp/0008.patch && echo "OK 0008 (mrope flash layout, re-seated for d7c405147)"; \
        fi && \
        mkdir -p /opt/aiter/aiter/configs/model_configs && \
        cp /tmp/q3vl_sqn12_tuned_fmoe.csv /opt/aiter/aiter/configs/model_configs/q3vl_sqn12_fp4_tuned_fmoe.csv && \
        echo "staged sqn12 tuned fmoe -> model_configs/q3vl_sqn12_fp4_tuned_fmoe.csv (PRE-build: FlyDSL AOT will bake it)" && \
        pip install -r requirements.txt && \
        AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS="${AITER_GPU_ARCHS}" PREBUILD_KERNELS=1 MAX_JOBS=128 python3 setup.py develop; \
    fi

# 0027: fp8 ViT-FMHA ASM branch. Applied LATE for AITER-layer cache warmth.
COPY docker/patches_vllm25/0027-q3vl-vit-fmha-fp8-asm.patch /tmp/0027.patch
RUN if [ "${Q3VL_FP8VIT}" = "1" ]; then \
        VLLM_SITE=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") && \
        patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0027.patch && \
        grep -q "_q3vl_fp8_asm_seg" "${VLLM_SITE}/vllm/v1/attention/ops/vit_attn_wrappers.py" && \
        echo "OK 0027 (ViT-FMHA fp8 ASM, Q3VL_FP8VIT=1)"; \
    else echo "SKIP 0027 (Q3VL_FP8VIT=0 -- no fp8-ViT; IMG-P/uattn-only build)"; fi

# =====================================================================================
# shuffle-KV + FlyDSL mrope writer (aiter#4371), on top of everything above.
# =====================================================================================
COPY docker/patches_vllm25/0030-q3vl-shuffle-kv-hip.patch                /tmp/0030.patch
COPY docker/patches_vllm25/0031-aiter-uattn-shuffle-fasttile-other.patch /tmp/0031.patch
COPY docker/patches_vllm25/0029-flydsl-fused-qk-norm-mrope-optest.patch  /tmp/0029.patch
COPY docker/patches_vllm25/0032-q3vl-mrope-flydsl-select.patch           /tmp/0032.patch
RUN if [ "${Q3VL_FLYDSL_MROPE}" = "1" ]; then \
        VLLM_SITE=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") && \
        patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0030.patch && \
        grep -q "return (2, num_blocks, num_kv_heads, head_size, block_size)" "${VLLM_SITE}/vllm/v1/attention/backends/rocm_aiter_unified_attn.py" && \
        echo "OK 0030 (Path B shuffle-KV wiring)" && \
        K=/opt/aiter/aiter/ops/triton/_triton_kernels/attention/unified_attention.py && \
        patch -p1 --fuzz=3 -d /opt/aiter < /tmp/0031.patch && \
        grep -q "other=fast_other" "$K" && echo "OK 0031 (aiter fast-tile shuffle-read fix)" && \
        patch -p1 --fuzz=3 -d /opt/aiter < /tmp/0029.patch && \
        test -f /opt/aiter/aiter/ops/flydsl/kernels/fused_qk_norm_mrope_3d_cache_pts_quant_shuffle.py && \
        grep -q "flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle" /opt/aiter/aiter/ops/flydsl/__init__.py && \
        echo "OK 0029 (FlyDSL kernel: file+export verified)" && \
        patch -p1 --fuzz=3 -d "${VLLM_SITE}" < /tmp/0032.patch && \
        grep -q "VLLM_Q3VL_MROPE_FLYDSL" "${VLLM_SITE}/vllm/model_executor/models/qwen3_moe.py" && \
        python3 -c "import py_compile; py_compile.compile('${VLLM_SITE}/vllm/model_executor/models/qwen3_moe.py', doraise=True); py_compile.compile('${VLLM_SITE}/vllm/v1/attention/backends/rocm_aiter_unified_attn.py', doraise=True); py_compile.compile('/opt/aiter/aiter/ops/flydsl/kernels/fused_qk_norm_mrope_3d_cache_pts_quant_shuffle.py', doraise=True)" && \
        echo "OK 0032 (FlyDSL selector) -- IMG-S flydsl stack baked"; \
    else echo "SKIP flydsl/pathb (Q3VL_FLYDSL_MROPE=0 -- identical to IMG-R)"; fi

# ---- Q3VL runtime env defaults ----
ENV AITER_FLYDSL_FORCE=1
ENV VLLM_Q3VL_VIT_FMHA_BINNED=1
ENV VLLM_Q3VL_VIT_FMHA_FP8=${Q3VL_FP8VIT}
ENV AITER_BF16_FP8_MOE_BOUND=0
ENV VLLM_Q3VL_FUSE_ATTN_PROLOGUE=${Q3VL_FUSIONS}
ENV VLLM_Q3VL_FUSE_VIT_ADD_LN=${Q3VL_FUSIONS}
ENV VLLM_Q3VL_FUSE_MOE_PREFILL_QUANT=${Q3VL_FUSIONS}
ENV VLLM_Q3VL_FUSE_VIT_GELU_FC1=0
ENV VLLM_Q3VL_FUSE_ROUTER_SORT=${Q3VL_FUSIONS}
ENV VLLM_Q3VL_MROPE_FLYDSL=${Q3VL_FLYDSL_MROPE}

# ---- MLPerf v6.1 endpoints client, isolated in /opt/endpoints/.venv ----
# mlcommons/endpoints main @ 2026-07-28. Bumped from fad6ff25: the v6.1 checker reads
# latency.early_stopping_percentiles for Q3VL Server latency, which endpoints only emits since it
# added metrics/early_stopping.py (on by default) -- absent in fad6ff25 AND in the reference pin 381d13b.
ARG ENDPOINTS_COMMIT=9d8abdfb05c77a857170cb6948a1f3c2ec606c79
ENV MLPERF_ENDPOINTS_COMMIT=${ENDPOINTS_COMMIT}
RUN git clone https://github.com/mlcommons/endpoints.git /opt/endpoints && \
    cd /opt/endpoints && \
    git checkout "${ENDPOINTS_COMMIT}" && \
    uv sync --frozen
RUN printf '#!/bin/bash\nexec uv run --no-sync --project /opt/endpoints inference-endpoint "$@"\n' \
        > /usr/local/bin/inference-endpoint && \
    chmod +x /usr/local/bin/inference-endpoint

ENTRYPOINT ["/bin/bash"]
