#!/bin/bash

patches_run_script() {
    local patch_name="$1" patch_path=""
    case "${patch_name}" in
        gc_gen2|allreduce_interval|decode_step_pacing|dp_engine_id|engine_core_env|nixl_notification_buffer|nixl_zmq_timeout)
            patch_path="${SCRIPT_DIR}/src/patches/runtime/${patch_name}.py"
            ;;
        extract_kv_scales|kv_scale_override|kv_scale_override_nvidia|nhd_mode|nixl_gpu_sync|rocm_unified_kv_scale_override|shuffle_kv|shuffle_to_nhd)
            patch_path="${SCRIPT_DIR}/src/patches/cross_platform/${patch_name}.py"
            ;;
        aiter_a8w4_block_n|aiter_a16w16_lds_cap|aiter_reduce_grouped_chunk)
            patch_path="${SCRIPT_DIR}/src/patches/vendor/${patch_name}.py"
            ;;
        *)
            echo "ERROR: unsupported patch '${patch_name}'" >&2
            return 1
            ;;
    esac
    [ -f "${patch_path}" ] || { echo "ERROR: missing patch ${patch_path}" >&2; return 1; }
    python3 "${patch_path}"
}

patches_apply_list() {
    local patch
    for patch in "$@"; do
        [ -z "${patch}" ] || patches_run_script "${patch}"
    done
}

patches_apply_runtime() {
    [ -z "${RUNTIME_PATCHES:-}" ] || patches_apply_list ${RUNTIME_PATCHES}
}

patches_apply_vendor() {
    [ "${IS_MOE:-0}" = "1" ] || return 0
    python3 -c 'import aiter' >/dev/null 2>&1 || return 0
    patches_apply_list aiter_a8w4_block_n aiter_a16w16_lds_cap aiter_reduce_grouped_chunk
}

patches_apply_cross_platform() {
    [ "${CROSS_VENDOR:-false}" = "true" ] || return 0
    if [ "${LOCAL_PLATFORM}" = "nvidia" ] && [ "${ROLE}" = "decode" ]; then
        export VLLM_SHUFFLE_TO_NHD="${VLLM_SHUFFLE_TO_NHD:-1}"
    fi
    [ -z "${HW_PATCHES:-}" ] || patches_apply_list ${HW_PATCHES}
}

patches_ensure_kv_scales() {
    [ -z "${KV_SCALE_SOURCE:-}" ] && return 0
    local scale_output="${KV_SCALE_OUTPUT:-/tmp/fp8_kv_scales.json}"
    if [ ! -f "${scale_output}" ]; then
        python3 "${SCRIPT_DIR}/src/patches/cross_platform/extract_kv_scales.py" "${KV_SCALE_SOURCE}" -o "${scale_output}"
    fi
    export VLLM_KV_SCALE_OVERRIDE="${VLLM_KV_SCALE_OVERRIDE:-${scale_output}}"
}

patches_apply_pd_support() {
    [ "${ROLE}" = "standalone" ] && return 0
    patches_apply_list nixl_notification_buffer nixl_zmq_timeout
    if [ "${CROSS_VENDOR:-false}" = "true" ]; then
        patches_run_script nixl_gpu_sync
        if [ "${ROLE}" = "decode" ]; then
            export NIXL_PRE_HANDSHAKE_HOST="${NIXL_PRE_HANDSHAKE_HOST:-${PREFILL_IP}}"
            export NIXL_PRE_HANDSHAKE_PORT="${NIXL_PRE_HANDSHAKE_PORT:-${VLLM_NIXL_SIDE_CHANNEL_PORT}}"
            export NIXL_PRE_HANDSHAKE_DP="${NIXL_PRE_HANDSHAKE_DP:-${PREFILL_ENGINE_COUNT:-${DP_SIZE}}}"
        fi
    fi
}

patches_apply_all() {
    patches_apply_cross_platform
    patches_ensure_kv_scales
    patches_apply_pd_support
    patches_apply_vendor
    patches_apply_runtime
}
