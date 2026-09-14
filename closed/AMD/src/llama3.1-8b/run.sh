#!/usr/bin/env bash

#unset FLATMM_HIP_CLANG_PATH
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

# Re-apply in-container vLLM patches (idempotent). vLLM lives in the image's
# site-packages and is reset on every fresh container; this restores the
# ASM paged-attention high_precision=0 tuning. See code/patches/README.md.
bash "$(dirname -- "$0")/patches/apply_vllm_patches.sh" || true

# Verbose vLLM logging so engine-startup details (CUDA-graph capture sizes,
# piecewise/full-graph mode, any capture fallbacks) are printed to stdout. The
# run below is tee'd to results/engine_run_<ts>.log so we can confirm whether the
# fused path achieves FULL cudagraph capture or silently falls back (root-cause
# check for the fluctuating <~50% GPU utilization).
#export VLLM_LOGGING_LEVEL=DEBUG

# Pre-compile the new fused MXFP4 Triton kernels (fused_rms_mxfp4_quant +
# act_mul_and_mxfp4_quant) for ALL cudagraph capture sizes at engine startup.
# Without this they JIT lazily per-shape on each of the 8 engines during the
# timed run -> engines desync -> rotating-idle GPUs (0<->100% sawtooth). No-op
# on the old image (fused kernels absent -> import is skipped).
#export HARNESS_WARMUP_FUSED_MXFP4_KERNELS=1
#export VLLM_ROCM_USE_AITER_CUSTOM_ALL_REDUCE=1
#export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8
#export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
#export VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=2560
# the following one is likely for FP8 model, from Nico's config for AWS
#export VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE=1
#GITHUB_WORKSPACE=./
#mkdir -p /workspace/profile
#mkdir ./profile
#export VLLM_TORCH_PROFILER_DIR=./profile/
#export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
#export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1
#export RCCL_MSCCL_ENABLE=0
#export AMD_LOG_LEVEL=4
#export AMD_LOG_LEVEL=5
#export AMD_LOG_LEVEL_FILE=debug
#export AMD_LOG_LEVEL_FILE=out
#export DEBUG_HIP_GRAPH_DOT_PRINT=1

# Kill any orphaned engine/server processes from a previous (killed) run that
# may still be holding GPUs and would cause a persistent straggler on this run.
#pkill -9 -f 'vllm serve'  2>/dev/null || true
#pkill -9 -f 'EngineCore'  2>/dev/null || true
#pkill -9 -f 'main.py'     2>/dev/null || true
#sleep 2
#rocm-smi --showpids   # verify no KFD pids remain before launching

# Torch profiler (RPD is unavailable in this image). Run with
#   ENABLE_TORCH_PROFILE=1 bash run.sh
# to capture a vLLM torch trace of the server steady state on one GPU. Tune the
# window via VLLM_PROFILE_DEVICES / VLLM_PROFILE_DELAY_SEC / VLLM_PROFILE_DURATION_SEC.
if [ "${ENABLE_TORCH_PROFILE:-0}" = "1" ]; then
    mkdir -p /lab-mlperf-inference/code/traces
    export VLLM_TORCH_PROFILER_DIR=/lab-mlperf-inference/code/traces
    export VLLM_PROFILE_DEVICES=${VLLM_PROFILE_DEVICES:-0}
    export VLLM_PROFILE_DELAY_SEC=${VLLM_PROFILE_DELAY_SEC:-30}
    export VLLM_PROFILE_DURATION_SEC=${VLLM_PROFILE_DURATION_SEC:-60}
    # Offline (async): anchor the capture to the pure-decode tail by starting once
    # this fraction of the shard's samples have completed (overrides DELAY_SEC).
    # Set to 0 to fall back to the wall-clock delay instead.
    export VLLM_PROFILE_START_FRAC=${VLLM_PROFILE_START_FRAC:-0.6}
    echo "[run.sh] Torch profiler ON: dir=$VLLM_TORCH_PROFILER_DIR devices=$VLLM_PROFILE_DEVICES start_frac=$VLLM_PROFILE_START_FRAC delay=${VLLM_PROFILE_DELAY_SEC}s dur=${VLLM_PROFILE_DURATION_SEC}s"
fi

# Disk guard: this is a shared, near-full box and Offline (cudagraph) rebuilds a
# large per-size inductor/triton compile cache (/root/.cache/vllm_* + /root/.triton,
# ~25-100GB). To avoid ENOSPC mid-run, prune those regenerable caches BEFORE the run
# ONLY when free space is below CACHE_FREE_GB_MIN (so normal runs keep their warm
# cache and don't pay recompile cost). Set CACHE_FREE_GB_MIN=0 to disable.
CACHE_FREE_GB_MIN=${CACHE_FREE_GB_MIN:-150}
if [ "${CACHE_FREE_GB_MIN}" -gt 0 ] 2>/dev/null; then
    avail_gb=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ -n "${avail_gb}" ] && [ "${avail_gb}" -lt "${CACHE_FREE_GB_MIN}" ]; then
        echo "[run.sh] Low disk (${avail_gb}GB < ${CACHE_FREE_GB_MIN}GB) -> pruning regenerable vLLM/Triton compile caches"
        rm -rf /root/.cache/vllm_* /root/.triton/* 2>/dev/null || true
        echo "[run.sh] after prune: $(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')GB free"
    fi
fi

# Clean up stale shared memory from prior runs (safe: not a compile cache).
rm -rf /dev/shm/benchmark_cli_benchmark_*

# NOTE: Do NOT wipe the kernel/compile caches below. The new docker image ships
# 3 modified Triton kernels (activation.py, quant/fused_mxfp4_quant.py,
# quant/quant.py) that are JIT-compiled on first use. Wiping ~/.triton/cache
# every run forces them to recompile DURING the timed run -> the GPU stalls
# mid-run -> 0<->100% utilization sawtooth. Triton/vLLM hash source+config into
# their cache keys, so stale entries are ignored (never misused); keeping the
# caches lets these kernels compile once and stay warm.
#rm -rf ~/.triton/cache*
#rm -rf ~/.config/miopen_*
#rm -rf ~/.cache/torch/kernels_*
#rm -rf ~/.cache/vllm_*
#for i in $(seq 0 7); do
#    rm -rf ~/.cache/vllm_${i}
#    rm -rf ~/.cache/torch_inductor_${i}
#done

python3 main.py --config-path llama3.1-8b --config-name offline_mi355x --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_1_8b_offline_performance 
#python3 main.py --config-path llama3.1-8b --config-name server_mi355x_dpx --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_1_8b_server_performance 
#python3 main.py --config-path llama3.1-8b --config-name server_mi355x --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_1_8b_server_performance 
#python3 main.py --config-path llama3.1-8b --config-name interactive_mi355x_dpx --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_1_8b_interactive_performance  
#python3 main.py --config-path llama3.1-8b --config-name interactive_mi355x --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_1_8b_interactive_performance  
#python3 main.py --config-path llama3.1-405b --config-name offline_mi355x --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_offline_performance 
#python3 main.py --config-path llama3.1-405b --config-name offline_mi355x_pruned --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_offline_pruned_performance 
#python3 main.py --config-path harness_llm/models/llama3_1-405b --config-name offline_mi355x --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_offline_performance 
#python3 main.py --config-path harness_llm/models/llama3_1-405b --config-name server_mi355x --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_server_performance
#python3 main.py --config-path llama3.1-405b --config-name interactive_mi355x --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_interactive_performance
#python3 main.py --config-path harness_llm/models/llama2-70b --config-name server_mi355x --backend vllm test_mode=performance harness_config.output_log_dir=results/llama2_server_performance
