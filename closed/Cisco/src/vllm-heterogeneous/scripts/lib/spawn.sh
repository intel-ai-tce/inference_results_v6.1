# scripts/lib/spawn.sh -- engine fan-out for prefill/decode/standalone.
#
# Collapses what used to be three nearly-identical role branches in
# start_server.sh into a single loop. Per-role differences are limited
# to:
#   - KV_ROLE env var:
#       prefill   -> kv_producer + DECODE_FORWARD_ADDRS to peer decodes
#       decode    -> kv_consumer
#       standalone-> (unset; worker ignores PD env vars)
#   - VLLM_NIXL_SIDE_CHANNEL_PORT: set for prefill/decode, omitted for standalone.
#   - The Python entry point: src/workers/${ROLE}.py.
#
# All other per-engine environment (GPU visibility, cache roots, ZMQ
# ports, NUM_WORKERS) is identical across roles.
#
# Required globals (set by start_server.sh):
#   ROLE                  prefill | decode | standalone
#   SCRIPT_DIR            repo root
#   _EPS[]                array of "host:port" endpoints (worker pull ports)
#   _DEVS[]               array of physical GPU indices (HIP_VISIBLE_DEVICES split)
#   SLAB_SIZE             TP * DP per engine (>= 1)
#   NIXL_BASE             base NIXL side-channel port (per-engine = NIXL_BASE + i)
#   PEER_LIST             comma-separated host:port for decode peers (prefill only)
#   MAX_NUM_SEQS          consumed by worker as NUM_WORKERS
#   DECODE_FORWARD_PORT   optional, default 5557 (prefill fallback)
#
# Outputs:
#   PIDS[]                array of background pipeline PIDs (sed processes)
#   Sets EXIT/INT/TERM trap to tear down siblings on first death.

source "${SCRIPT_DIR}/scripts/lib/numa.sh"

# spawn_one_engine <idx>  -- launches the i-th worker in the background.
spawn_one_engine() {
    local i="$1"
    local ep="${_EPS[$i]}"
    local host="${ep%%:*}"
    local pull_port="${ep##*:}"
    local push_port=$((pull_port + 1000))

    # TP*DP slab of physical GPUs.
    local slab_start=$(( i * SLAB_SIZE ))
    local slab_devs="" k
    for ((k=0; k<SLAB_SIZE; k++)); do
        slab_devs+="${_DEVS[$((slab_start + k))]},"
    done
    slab_devs="${slab_devs%,}"
    local slab_tag="${slab_devs//,/_}"

    # Optional per-physical-GPU UCX NIC selection. This is intentionally opt-in:
    # normal launch paths keep inheriting UCX_NET_DEVICES unchanged. The value is
    # a whitespace-separated array indexed by physical GPU ID, for example:
    #   UCX_NET_DEVICES_BY_GPU="gpu0_rdma:1 gpu1_rdma:1 ..."
    local worker_ucx_devices="" ucx_net_devices_env=""
    if [ "${UCX_NET_DEVICES_AUTO_SELECT:-0}" = "1" ]; then
        ucx_net_devices_env="-u UCX_NET_DEVICES"
        worker_ucx_devices="auto"
    elif [ -n "${UCX_NET_DEVICES_BY_GPU:-}" ]; then
        local physical_gpu="${_DEVS[$slab_start]}"
        local -a ucx_devices_by_gpu
        read -r -a ucx_devices_by_gpu <<< "${UCX_NET_DEVICES_BY_GPU}"
        if [ "${physical_gpu}" -ge "${#ucx_devices_by_gpu[@]}" ] ||
           [ -z "${ucx_devices_by_gpu[$physical_gpu]:-}" ]; then
            echo "ERROR: UCX_NET_DEVICES_BY_GPU has no NIC for physical GPU ${physical_gpu}" >&2
            return 1
        fi
        worker_ucx_devices="${ucx_devices_by_gpu[$physical_gpu]}"
        ucx_net_devices_env="UCX_NET_DEVICES=${worker_ucx_devices}"
    fi

    # Per-engine cache dirs so JIT outputs never collide. Keep all caches on
    # storage in our containers and can grow into hundreds of GB.
    local cache_base="${VLLM_CACHE_BASE:?VLLM_CACHE_BASE must be set by start_server.sh}"
    local cache="${cache_base}/mp_${ROLE}_${i}"
    local cache_large="${cache}"
    mkdir -p "${cache}/vllm" "${cache}/triton" \
             "${cache_large}/deep_gemm" "${cache_large}/torchinductor"
    # Stale triton tmp.* dirs can cause profile_run to hang silently.
    find "${cache}/triton" -maxdepth 1 -type d -name 'tmp.*' \
        -exec rm -rf {} + 2>/dev/null || true

    local nixl_port=$(( NIXL_BASE + i ))
    local vllm_port_base="${VLLM_PORT_BASE:-${VLLM_PORT:-37600}}"
    local engine_vllm_port=$(( vllm_port_base + i * 100 ))

    # NUMA pinning based on the FIRST physical GPU of this engine's slab.
    # Empty (no prefix) if topology / numactl unavailable.
    local numa_prefix
    numa_prefix="$(numa_prefix_for_gpu "${_DEVS[$slab_start]}")"

    # Per-role extras. Building them as space-separated KEY=VAL lets bash
    # treat the empty case as a no-op and the non-empty case as additional
    # env assignments before the command.
    local kv_env="" peer_env="" nixl_env=""
    case "${ROLE}" in
        prefill)
            kv_env="KV_ROLE=kv_producer"
            # The default fan-out sends each prefill engine to every decoder.
            # For equal-sized MP PD deployments, paired routing preserves a
            # stable producer/consumer relationship (engine i -> peer i),
            # avoiding an 8x8 NIXL stream fan-in while retaining SUT-level
            # round-robin load balancing across all prefill engines.
            local decode_peer_list="${PEER_LIST}"
            if [ "${PD_DECODE_FORWARD_MODE:-all_to_all}" = "paired" ]; then
                local peer_count=${#_PEER_EPS[@]}
                if [ "${peer_count}" -eq 0 ]; then
                    echo "ERROR: paired PD routing requires decode peer endpoints" >&2
                    return 1
                fi
                local peer_idx=$(( i % peer_count ))
                decode_peer_list="${_PEER_EPS[$peer_idx]}"
                echo "  engine[$i]: paired decode peer=${decode_peer_list}"
            fi
            peer_env="DECODE_FORWARD_ADDRS=${decode_peer_list} DECODE_FORWARD_PORT=${DECODE_FORWARD_PORT:-5557}"
            nixl_env="VLLM_NIXL_SIDE_CHANNEL_PORT=${nixl_port}"
            ;;
        decode)
            kv_env="KV_ROLE=kv_consumer"
            nixl_env="VLLM_NIXL_SIDE_CHANNEL_PORT=${nixl_port}"
            ;;
        standalone)
            : # no PD env
            ;;
    esac

    local vllm_port_env vllm_port_label
    if [ "${DP_SIZE:-1}" -gt 1 ] 2>/dev/null; then
        # vLLM DP config asks for multiple open ports. If VLLM_PORT is fixed,
        # network_utils.get_open_ports_list() repeatedly returns the same port.
        vllm_port_env="-u VLLM_PORT"
        vllm_port_label="auto"
    else
        vllm_port_env="VLLM_PORT=${engine_vllm_port}"
        vllm_port_label="${engine_vllm_port}"
    fi

    echo "  engine[$i]: GPUs=${slab_devs}  pull=${pull_port} push=${push_port}  nixl=${nixl_port}  vllm_port=${vllm_port_label}  host=${host}  numa=${numa_prefix:-none}  ucx=${worker_ucx_devices:-inherited}"

    # The full command line. ${numa_prefix} is either empty or
    # "numactl --cpunodebind=N --membind=N" -- bash handles both.
    HIP_VISIBLE_DEVICES="${slab_devs}" \
    CUDA_VISIBLE_DEVICES="${slab_devs}" \
    VLLM_CACHE_ROOT="${cache}/vllm" \
    TRITON_CACHE_DIR="${cache}/triton" \
    DG_JIT_CACHE_DIR="${cache_large}/deep_gemm" \
    TORCHINDUCTOR_CACHE_DIR="${cache_large}/torchinductor" \
    ZMQ_PULL_PORT="${pull_port}" \
    ZMQ_PUSH_PORT="${push_port}" \
    WORKER_IDX="${i}" \
    NUM_WORKERS="${MAX_NUM_SEQS}" \
    MODEL_SEED="${MODEL_SEED:-0}" \
    env ${ucx_net_devices_env} ${vllm_port_env} ${nixl_env} ${kv_env} ${peer_env} \
        ${numa_prefix} "${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/src/workers/${ROLE}.py" \
            2>&1 | sed -u "s/^/[gpu${slab_tag}] /" &
}

# spawn_all_engines -- fan out N engines and install the teardown trap.
spawn_all_engines() {
    PIDS=()
    cleanup_mp() {
        echo "Stopping ${#PIDS[@]} ${ROLE} engines (PIDs: ${PIDS[*]})"
        # PIDS tracks the log pipeline tail; also reap worker wrappers and
        # EngineCore children that can survive as PID 1 orphans.
        for pid in "${PIDS[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
        pkill -TERM -f "src/workers/${ROLE}.py" 2>/dev/null || true
        pkill -TERM -f "VLLM::EngineCore" 2>/dev/null || true
        sleep 5
        for pid in "${PIDS[@]}"; do kill -KILL "${pid}" 2>/dev/null || true; done
        pkill -KILL -f "src/workers/${ROLE}.py" 2>/dev/null || true
        pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
    }
    trap cleanup_mp EXIT INT TERM

    local i
    for i in "${!_EPS[@]}"; do
        spawn_one_engine "${i}"
        PIDS+=($!)
    done
    echo "Spawned ${#PIDS[@]} ${ROLE} engines"
}
