#!/bin/bash
# Start a fleet of vLLM workers for distributed inference (mp fan-out).
#
# Roles:
#   prefill      N independent prefill engines (one per servers.prefill entry)
#   decode       M independent decode engines (one per servers.decode entry)
#   standalone   N independent engines doing full prefill+decode (no PD)
#
# Usage:
#   ./start_server.sh --role prefill    --hardware h200   --model gptoss_120b
#   ./start_server.sh --role decode     --hardware mi350x --model gptoss_120b --remote-hardware h200
#   ./start_server.sh --role standalone --hardware mi350x --model gptoss_120b

set -eu
ulimit -l unlimited
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY 2>/dev/null || true

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-${SCRIPT_DIR}/config/deployment.env}"
if [ ! -f "${DEPLOYMENT_ENV}" ]; then
    echo "ERROR: deployment environment file not found: ${DEPLOYMENT_ENV}" >&2
    exit 1
fi
set -a
source "${DEPLOYMENT_ENV}"
set +a
: "${WORK_DIR:?Set WORK_DIR in config/deployment.env}"
: "${MODEL_ROOT:?Set MODEL_ROOT in config/deployment.env}"
: "${DATA_ROOT:?Set DATA_ROOT in config/deployment.env}"
VLLM_CACHE_BASE="${VLLM_CACHE_BASE:-${WORK_DIR}/.cache/vllm-cache}"
export VLLM_CACHE_BASE
mkdir -p "${VLLM_CACHE_BASE}"


# ================================================================
# Logging - wrap entire script in tee to logs/<role>_<ts>.log.
# ================================================================
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
if [ -z "${_LOGGING_WRAPPED:-}" ]; then
    _ROLE_HINT=""
    for _a in "$@"; do
        if [ "${_prev:-}" = "--role" ]; then _ROLE_HINT="$_a"; break; fi
        _prev="$_a"
    done
    LOGFILE="${LOG_DIR}/${_ROLE_HINT:-server}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to ${LOGFILE}"
    export _LOGGING_WRAPPED=1
    exec > >(tee -a "${LOGFILE}") 2>&1
fi

# ================================================================
# CLI argument parsing
# ================================================================
ROLE=""
HARDWARE=""
MODEL=""
REMOTE_HARDWARE=""
MLPERF_SCENARIO=""
PROFILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --role)            ROLE="$2";              shift 2 ;;
        --hardware)        HARDWARE="$2";          shift 2 ;;
        --model)           MODEL="$2";             shift 2 ;;
        --remote-hardware) REMOTE_HARDWARE="$2";   shift 2 ;;
        --scenario)        MLPERF_SCENARIO="$2";    shift 2 ;;
        --profile)         PROFILE="$2";            shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$ROLE" ] || [ -z "$HARDWARE" ] || [ -z "$MODEL" ]; then
    cat <<EOF
Usage: $0 --role <prefill|decode|standalone> --hardware <hw> --model <name> [options]

Roles:
  prefill      N prefill engines (kv_producer, forwards to decode peers)
  decode       M decode engines (kv_consumer, finishes generation)
  standalone   N engines doing full prefill+decode (no PD disaggregation)

Options:
  --scenario   offline | server | interactive
  --profile    named complete model profile (for example standalone_mi350x)
  
EOF
    exit 1
fi

case "$ROLE" in
    prefill|decode|standalone) ;;
    *) echo "ERROR: --role must be 'prefill', 'decode', or 'standalone'"; exit 1 ;;
esac

# ================================================================
# Resolve configuration from YAML (exports MODEL_PATH, TP_SIZE, DP_SIZE,
# VISIBLE_DEVICES, MP_ENDPOINTS, MP_PEER_ENDPOINTS, KV_SCALE_SOURCE,
# HW_PATCHES, RUNTIME_PATCHES, etc. -- see src/resolve_config.py).
# ================================================================
eval "$(python3 "${SCRIPT_DIR}/src/resolve_config.py" \
    --role "${ROLE}" \
    --hardware "${HARDWARE}" \
    --model "${MODEL}" \
    ${MLPERF_SCENARIO:+--scenario "${MLPERF_SCENARIO}"} \
    ${REMOTE_HARDWARE:+--remote-hardware "${REMOTE_HARDWARE}"} \
    ${GPUS_OVERRIDE:+--gpus "${GPUS_OVERRIDE}"} \
    ${PROFILE:+--profile "${PROFILE}"} \
)"
case "${MLPERF_SCENARIO}" in
    ""|offline|server|interactive) ;;
    *) echo "ERROR: --scenario must be offline, server, or interactive" >&2; exit 1 ;;
esac

# ================================================================
# Apply patches (restore stock vLLM, vendor, cross-vendor, runtime,
# DP allreduce, KV scales, NIXL ZMQ tweaks). See scripts/lib/patches.sh.
# ================================================================
source "${SCRIPT_DIR}/scripts/lib/patches.sh"
patches_apply_all

# ================================================================
# Runtime env (NIXL timeout, cache roots, drain limit).
# ================================================================
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DG_JIT_USE_NVRTC="${DG_JIT_USE_NVRTC:-0}"

if [ "$ROLE" != "standalone" ]; then
    export VLLM_NIXL_ABORT_REQUEST_TIMEOUT="${VLLM_NIXL_ABORT_REQUEST_TIMEOUT:-3600}"
else
    export VLLM_NIXL_ABORT_REQUEST_TIMEOUT="${VLLM_NIXL_ABORT_REQUEST_TIMEOUT:-15}"
fi

# ================================================================
# Validate MP_ENDPOINTS and compute slab geometry.
# ================================================================
if [ -z "${MP_ENDPOINTS:-}" ]; then
    if [ "${ROLE}" = "standalone" ]; then
        cat >&2 <<EOF
ERROR: servers.standalone is empty in the model YAML.
       Add a single base host:port; the engine count auto-resolves to
         harness_config.device_count // (standalone.tp_size * standalone.dp_size)
       e.g.
         servers:
           standalone:
             - "127.0.0.1:8200"
EOF
    else
        cat >&2 <<EOF
ERROR: servers.${ROLE} is empty in the model YAML.
       Add e.g.
         servers:
           ${ROLE}:
             - "<role-host>:8200"
             - "<role-host>:8201"
             - ...
EOF
    fi
    exit 1
fi

IFS=',' read -ra _DEVS <<< "${VISIBLE_DEVICES}"
IFS=' '  read -ra _EPS  <<< "${MP_ENDPOINTS}"
N_ENGINES=${#_EPS[@]}
SLAB_SIZE=$(( TP_SIZE * DP_SIZE ))
REQUIRED_GPUS=$(( N_ENGINES * SLAB_SIZE ))
if [ "${REQUIRED_GPUS}" -gt "${#_DEVS[@]}" ]; then
    if [ "${ROLE}" = "standalone" ]; then
        echo "ERROR: need ${REQUIRED_GPUS} GPUs (${N_ENGINES} engines x TP=${TP_SIZE} x DP=${DP_SIZE}) but only ${#_DEVS[@]} visible." >&2
        echo "       The standalone engine count auto-resolves from harness_config.device_count;" >&2
        echo "       check that harness_config.device_count matches the number of GPUs you actually have." >&2
    else
        echo "ERROR: need ${REQUIRED_GPUS} GPUs (${N_ENGINES} engines x TP=${TP_SIZE} x DP=${DP_SIZE}) but only ${#_DEVS[@]} visible" >&2
    fi
    exit 1
fi

# Peer endpoints (decode side) for prefill role's DECODE_FORWARD_ADDRS.
PEER_LIST=""
if [ "$ROLE" = "prefill" ]; then
    if [ -z "${MP_PEER_ENDPOINTS:-}" ]; then
        echo "WARNING: prefill role but MP_PEER_ENDPOINTS is empty." >&2
        echo "         servers.decode should list the decode-node engines." >&2
        echo "         Prefill will fall back to SUT-relayed decode (slow)." >&2
    else
        IFS=' ' read -ra _PEER_EPS <<< "${MP_PEER_ENDPOINTS}"
        for pep in "${_PEER_EPS[@]}"; do PEER_LIST+="${pep},"; done
        PEER_LIST="${PEER_LIST%,}"
    fi
fi

NIXL_BASE="${VLLM_NIXL_SIDE_CHANNEL_PORT:-5600}"

# ================================================================
# Announce
# ================================================================
echo "=========================================="
echo "Starting ${ROLE} server (${HARDWARE}) [mp x${N_ENGINES}]"
echo "=========================================="
echo "Model:       ${MODEL_PATH}"
echo "Per engine:  TP=${TP_SIZE}  DP=${DP_SIZE}  slab=${SLAB_SIZE}"
echo "Batch:       max_num_seqs=${MAX_NUM_SEQS}  max_batched_tokens=${MAX_BATCHED_TOKENS}  gpu_mem_util=${GPU_MEM_UTIL}  max_model_len=${MAX_MODEL_LEN}"
echo "Engines:     ${N_ENGINES}"
echo "GPUs:        ${VISIBLE_DEVICES}"
echo "Endpoints:   ${MP_ENDPOINTS}"
[ "$ROLE" = "prefill" ] && echo "Peer decode: ${MP_PEER_ENDPOINTS:-(none)}"
[ "$ROLE" != "standalone" ] && echo "NIXL base:   ${VLLM_NIXL_SIDE_CHANNEL_HOST}:${NIXL_BASE}  (+1 per engine)"
[ -n "${REMOTE_HARDWARE}" ] && echo "Remote: ${REMOTE_HARDWARE} (patches=${APPLY_PATCHES})"
echo "=========================================="

# ================================================================
# Common per-engine engine config (shared exports consumed by workers).
# ================================================================
export MODEL_PATH TP_SIZE DP_SIZE MAX_MODEL_LEN MAX_NUM_SEQS
export MAX_BATCHED_TOKENS GPU_MEM_UTIL BLOCK_SIZE KV_CACHE_DTYPE DTYPE
export CALCULATE_KV_SCALES="${CALCULATE_KV_SCALES}"
export QUANTIZATION="${QUANTIZATION}"
export SERVED_MODEL_NAME="${MODEL_NAME}"
export KV_ROLE
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}"
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL}"
export ENFORCE_EAGER="${ENFORCE_EAGER}"
export DISABLE_SLIDING_WINDOW="${DISABLE_SLIDING_WINDOW}"
[ -n "${DISABLE_HYBRID_KV_CACHE_MANAGER+x}" ] && export DISABLE_HYBRID_KV_CACHE_MANAGER
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL}"
export ENABLE_DBO="${ENABLE_DBO}"
export DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD}"
export ENABLE_EPLB="${ENABLE_EPLB}"
export ALL2ALL_BACKEND="${ALL2ALL_BACKEND}"
export LINEAR_BACKEND="${LINEAR_BACKEND:-auto}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-}"
export SPECULATIVE_MODEL="${SPECULATIVE_MODEL:-}"
export SPECULATIVE_MODEL_REFERENCE="${SPECULATIVE_MODEL_REFERENCE:-}"
export SPECULATIVE_NUM_TOKENS="${SPECULATIVE_NUM_TOKENS:-0}"
export SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-0}"
export SPECULATIVE_DRAFT_SAMPLE_METHOD="${SPECULATIVE_DRAFT_SAMPLE_METHOD:-}"
export MLPERF_SCENARIO
export MOE_BACKEND="${MOE_BACKEND:-}"
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE}"
[ -n "${COMPILATION_MODE:-}" ] && export COMPILATION_MODE="${COMPILATION_MODE}"
export ASYNC_SCHEDULING="${ASYNC_SCHEDULING}"
export NIXL_USE_UCCL="${CROSS_VENDOR}"
[ -n "${COMPILE_SIZES:-}" ] && export COMPILE_SIZES="${COMPILE_SIZES}"
[ -n "${CUDAGRAPH_CAPTURE_RANGE:-}" ] && export CUDAGRAPH_CAPTURE_RANGE="${CUDAGRAPH_CAPTURE_RANGE}"
[ -n "${DISABLE_CUSTOM_ALL_REDUCE:-}" ] && export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE}"
export DECODE_MAX_TOKENS DECODE_MIN_TOKENS DECODE_TEMPERATURE DECODE_TOP_K DECODE_TOP_P
export USE_GENERATION_STOP_TOKEN_IDS

# ================================================================
# Fan out N engines (see scripts/lib/spawn.sh).
# ================================================================
source "${SCRIPT_DIR}/scripts/lib/spawn.sh"
spawn_all_engines

# Wait for any child to exit; tear the rest down via the EXIT trap.
wait -n
rc=$?
echo "A ${ROLE} engine exited (rc=${rc}); shutting down siblings"
exit "${rc}"
