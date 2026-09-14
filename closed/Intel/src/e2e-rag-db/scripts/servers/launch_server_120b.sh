#!/bin/bash
#
# Load config (repo root is two levels up); safe if absent.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "${_ROOT}/config.sh" ] && source "${_ROOT}/config.sh"
[ -f "${_ROOT}/config.default.sh" ] && source "${_ROOT}/config.default.sh"

MODEL_PATH=${MODEL_PATH:-${SERVER_120B_MODEL:-/data/gpt-oss-120b-mxfp4}}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-${SERVER_120B_MAX_NUM_SEQS:-1024}}   # concurrent decode slots
CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}
MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-${SERVER_120B_MAX_CUDAGRAPH_CAPTURE_SIZE:-784}}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-${SERVER_120B_GPU_MEM_UTIL:-0.90}}
PORT=${PORT:-${SERVER_120B_PORT:-8123}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-${SERVER_120B_MAX_MODEL_LEN:-131072}}

kill_lingering_gpu_procs() {
    local patterns=("vllm.entrypoints" "VLLM::" "EngineCore" "vllm serve"
                    "multiprocessing.spawn" "multiprocessing.resource_tracker")
    local deadline=$(( $(date +%s) + 60 ))

    # helper: list PIDs (excluding self) that either match a vLLM pattern OR
    # hold an open fd to a GPU render node (/dev/dri/*). fd scan is the backstop.
    _gpu_pids() {
        local pids="" p
        for pat in "${patterns[@]}"; do
            pids+=" $(pgrep -f "$pat" 2>/dev/null)"
        done
        # fd backstop: any process holding /dev/dri/* (renderD*/card*)
        for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
            [ "$p" = "$$" ] && continue
            if ls -l /proc/$p/fd 2>/dev/null | grep -q '/dev/dri/'; then
                pids+=" $p"
            fi
        done
        # dedupe, drop self and this shell's ancestry
        echo $pids | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u \
            | grep -vx "$$" | grep -vx "$PPID"
    }

    local victims
    victims=$(_gpu_pids)
    if [ -z "$victims" ]; then
        echo "[guard] no lingering GPU/vLLM processes found"
        return 0
    fi

    echo "[guard] found lingering processes, killing:"
    ps -o pid,cmd -p $(echo $victims | tr '\n' ',' | sed 's/,$//') 2>/dev/null | tail -n +2 | sed 's/^/[guard]   /'

    # SIGTERM first (graceful), escalate to SIGKILL
    kill $victims 2>/dev/null
    sleep 3
    kill -9 $(_gpu_pids) 2>/dev/null

    # poll until fully gone (GPU memory releases on process exit)
    while [ "$(date +%s)" -lt "$deadline" ]; do
        victims=$(_gpu_pids)
        [ -z "$victims" ] && { echo "[guard] all lingering processes cleared"; return 0; }
        kill -9 $victims 2>/dev/null
        sleep 2
    done

    echo "[guard] WARNING: processes still present after 60s: $(_gpu_pids | tr '\n' ' ')"
    echo "[guard] proceeding anyway -- launch may OOM if they hold GPU memory"
    return 1
}

# ---------------------------------------------------------------------------
# Health check: poll the OpenAI /v1/models endpoint until the server is ready.
# ---------------------------------------------------------------------------
health_check() {
    echo "Waiting for server on port ${PORT} to become ready..."
    local no_proxy_bak="$no_proxy" NO_PROXY_bak="$NO_PROXY"
    export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
    local retries=0 max_retries=180 code   # 180 * 5s = 15 min (120B load+capture is slow)
    while true; do
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null)
        [ "$code" = "200" ] && break
        # bail early if the server process died
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server process ($SERVER_PID) exited before becoming ready -- check log"
            exit 1
        fi
        retries=$((retries + 1))
        if [ "$retries" -gt "$max_retries" ]; then
            echo "Server failed to become ready within timeout (last status: ${code:-none})"
            exit 1
        fi
        sleep 5
    done
    export no_proxy="$no_proxy_bak" NO_PROXY="$NO_PROXY_bak"
    echo "Server ready on port ${PORT} (HTTP 200)"
}

kill_lingering_gpu_procs

# ---------------------------------------------------------------------------
# Runtime env
# ---------------------------------------------------------------------------
export TRITON_INTEL_DEVICE_ARCH=bmg          # B70 arch-parser fix (triton-xpu 3.6.0)
export VLLM_USE_TRITON_XPU_ATTN=1

export VLLM_XPU_ENABLE_XPU_GRAPH=1

export VLLM_XPU_FP8_ALLREDUCE=1
export VLLM_XPU_FP8_ALLREDUCE_MIN_TOKENS=513 # for B70

export VLLM_ENABLE_DIST_SAMPLE=1             # vocab-parallel Gumbel-max
export VLLM_ENABLE_DIST_SAMPLE_FP32_REDUCE=1 # fp32 SUM reduce

export SAMPLING_TEMPERATURE=1.0
export SAMPLING_TOP_P=1.0
export SAMPLING_TOP_K=-1
export KV_CACHE_DTYPE=fp8

HOST_CORES=${HOST_CORES:-${SERVER_120B_HOST_CORES:-40-42,83}}
TASKSET_PREFIX=""
if [ -n "$HOST_CORES" ]; then
    TASKSET_PREFIX="taskset -c $HOST_CORES"
    echo "[120b] host pinned to cores: $HOST_CORES"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
# Launch unattended; log to repo root.
LOG="${_ROOT:-.}/log-server-120b.log"
echo "[120b] port=${PORT} -> ${LOG}"
nohup $TASKSET_PREFIX python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --dtype=bfloat16 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=${GPU_MEMORY_UTIL} \
    --enable-prefix-caching \
    --max-num-batched-tokens=3072 \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --max-model-len=${MAX_MODEL_LEN} \
    --block-size 64 \
    --port ${PORT} \
    --async_scheduling \
    --kv_cache_dtype fp8 \
    --compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"max_cudagraph_capture_size\":${MAX_CUDAGRAPH_CAPTURE_SIZE}}" \
    -tp 4 > "${LOG}" 2>&1 &
    #--enforce-eager \
    # (no --enforce-eager: XPU graph decode capture is enabled via the config above)
SERVER_PID=$!
echo "[120b] started PID ${SERVER_PID} (unattended)"

health_check
# health_check returns once ready; server keeps running under nohup.
