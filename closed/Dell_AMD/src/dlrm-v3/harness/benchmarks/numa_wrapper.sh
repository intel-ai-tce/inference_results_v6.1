#!/bin/bash
# NUMA binding wrapper for MPI ranks. Pins each rank's CPU + memory to one NUMA
# node, spreading ranks evenly across the node's NUMA domains. This isolates the
# latency-sensitive ZMQ dispatch / worker poll loops from cross-NUMA bouncing and
# from noisy neighbors on shared nodes (without binding, mpirun --bind-to none
# lets the scheduler migrate these threads, wrecking the batch cadence under load).
# Enabled by BIND_NUMA=1 in MI355_run_performance_harness.sh. Requires PYTHON_ARGS.

RANK=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}
SIZE=${OMPI_COMM_WORLD_SIZE:-9}

NODES=$(numactl --hardware 2>/dev/null | awk '/^available:/{print $2}')
NODES=${NODES:-1}

if command -v numactl >/dev/null 2>&1 && [ "${NODES}" -gt 1 ]; then
    NODE=$(( RANK * NODES / SIZE ))
    [ "$NODE" -ge "$NODES" ] && NODE=$(( NODES - 1 ))
    exec numactl --cpunodebind="$NODE" --membind="$NODE" python3 -u $PYTHON_ARGS
else
    exec python3 -u $PYTHON_ARGS
fi
