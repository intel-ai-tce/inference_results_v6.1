# scripts/lib/numa.sh -- NUMA pinning helper.
#
# Each engine is pinned to the NUMA node of its first physical GPU, so
# host-side work (Python, msgpack, ZMQ, PCIe DMA staging) stays on the
# memory channel closest to the GPU. On dual-socket EPYC + 8-GPU MI3xx
# hosts this typically gives 10-30% throughput vs the OS-default
# scheduler placement, and -- more importantly -- eliminates the
# "one engine drops to 40% utilization" pattern caused by an engine
# accidentally landing on the wrong NUMA node.
#
# How the mapping works:
#   1. Enumerate /sys/class/drm/card* that are driven by amdgpu (vendor
#      0x1002). On many hosts the card numbering is non-contiguous
#      (card1, card9, card17, ...) because each GPU exposes multiple
#      DRM nodes per connector.
#   2. Sort the survivors by their associated renderD minor; that is
#      the same ordering HIP uses for HIP_VISIBLE_DEVICES (PCI BDF).
#   3. Read numa_node out of each card's device sysfs.
#   4. Cache the result so we only scan sysfs once.
#
# Usage:
#   source scripts/lib/numa.sh
#   prefix=$(numa_prefix_for_gpu 4)
#   # prefix is either "numactl --cpunodebind=N --membind=N" or empty.
#   ${prefix} python my_worker.py

declare -gA _NUMA_NODE_CACHE 2>/dev/null || true
_NUMA_MAP_BUILT=""

# _numa_build_map -- populates _NUMA_NODE_CACHE[<hip_gpu_idx>] = <numa_node>.
# Robust against:
#   - non-contiguous DRM card numbering
#   - connector-suffixed names (cardN-DP-1)
#   - non-AMD cards mixed into /sys/class/drm
#   - numa_node = -1 (kernel says "not NUMA aware")
_numa_build_map() {
    [ -n "${_NUMA_MAP_BUILT}" ] && return
    _NUMA_MAP_BUILT=1
    local card name drv vendor numa minor r rname dev1 dev2
    local -a entries=()
    for card in /sys/class/drm/card*; do
        [ -e "${card}" ] || continue
        name="$(basename "${card}")"
        case "${name}" in *-*) continue ;; esac  # skip connectors
        drv="$(readlink -f "${card}/device/driver" 2>/dev/null | xargs -r basename)"
        [ "${drv}" = "amdgpu" ] || continue
        vendor="$(cat "${card}/device/vendor" 2>/dev/null)"
        [ "${vendor}" = "0x1002" ] || continue

        # Find this card's renderD minor.
        minor=""
        dev1="$(readlink -f "${card}/device" 2>/dev/null)"
        for r in /sys/class/drm/renderD*; do
            [ -e "${r}" ] || continue
            rname="$(basename "${r}")"
            case "${rname}" in *-*) continue ;; esac
            dev2="$(readlink -f "${r}/device" 2>/dev/null)"
            if [ -n "${dev1}" ] && [ "${dev1}" = "${dev2}" ]; then
                minor="${rname#renderD}"
                break
            fi
        done
        [ -z "${minor}" ] && continue

        numa="$(cat "${card}/device/numa_node" 2>/dev/null | tr -d '[:space:]')"
        [ "${numa}" = "-1" ] && numa=""

        entries+=("${minor}:${numa}")
    done

    # Sort by renderD minor ascending = HIP_VISIBLE_DEVICES order.
    local sorted idx=0 line _node
    sorted="$(printf '%s\n' "${entries[@]}" | sort -t: -k1,1n)"
    while IFS= read -r line; do
        [ -z "${line}" ] && continue
        _node="${line#*:}"
        _NUMA_NODE_CACHE[$idx]="${_node}"
        idx=$((idx + 1))
    done <<< "${sorted}"
}

# numa_node_for_gpu <hip_gpu_idx>
# Echoes the NUMA node (integer >=0) or empty if topology unavailable.
numa_node_for_gpu() {
    _numa_build_map
    local gpu="$1"
    echo "${_NUMA_NODE_CACHE[$gpu]:-}"
}

# numa_prefix_for_gpu <hip_gpu_idx>
# Echoes "numactl --cpunodebind=N --membind=N" when topology + numactl
# are available; otherwise echoes empty. One-shot warning if numactl
# is missing.
numa_prefix_for_gpu() {
    local gpu="$1"
    local node
    node="$(numa_node_for_gpu "${gpu}")"
    [ -z "${node}" ] && { echo ""; return; }
    if ! command -v numactl >/dev/null 2>&1; then
        if [ -z "${_NUMACTL_WARNED:-}" ]; then
            echo "WARNING: numactl not found in PATH; engines will run without NUMA pinning." >&2
            export _NUMACTL_WARNED=1
        fi
        echo ""
        return
    fi
    echo "numactl --cpunodebind=${node} --membind=${node}"
}

# numa_print_map -- diagnostic helper. Prints HIP idx -> NUMA node.
numa_print_map() {
    _numa_build_map
    local i
    echo "HIP GPU -> NUMA node mapping:"
    for i in "${!_NUMA_NODE_CACHE[@]}"; do
        echo "  GPU ${i}: node=${_NUMA_NODE_CACHE[$i]:-unknown}"
    done | sort
}
