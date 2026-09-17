#!/usr/bin/env bash

export VLLM_CPU_KVCACHE_SPACE=300
export NUM_NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $NF}')
export PORT=8192
export RESERVED_CPUS_PER_NUMA="${RESERVED_CPUS_PER_NUMA:-2}"
export VLLM_CPU_NUM_OF_RESERVED_CPU="${VLLM_CPU_NUM_OF_RESERVED_CPU:-$((NUM_NUMA_NODES * RESERVED_CPUS_PER_NUMA))}"
export MAX_NUM_SEQS=1536
export MAX_NUM_BATCHED_TOKENS=32768
export TP_SIZE=1
export MODEL_PATH="/model/Llama-3.1-8B-Instruct_calibrated-cpu"
if [ -z "${VLLM_CPU_OMP_THREADS_BIND:-}" ]; then
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export VLLM_CPU_OMP_THREADS_BIND
    VLLM_CPU_OMP_THREADS_BIND="$(
        "${SCRIPT_DIR}/generate_omp_threads_bind.sh" "${RESERVED_CPUS_PER_NUMA}"
    )"
fi

if [ "${NUM_CORES}" == "172" ] || [ "${NUM_CORES}" == "192" ]; then
    export VLLM_CPU_KVCACHE_SPACE=200
fi

if [ "${SCENARIO}" = "Server" ]; then
    export TP_SIZE=2
    if [ "${NUM_CORES}" == "192" ]; then
        export MAX_NUM_SEQS=32
        export MAX_NUM_BATCHED_TOKENS=1024
    else
	export MAX_NUM_SEQS=32
        export MAX_NUM_BATCHED_TOKENS=256
    fi
fi

echo "Using vLLM with TP $TP_SIZE and KV $VLLM_CPU_KVCACHE_SPACE"
echo "VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND}"

vllm serve $MODEL_PATH \
    --dtype bfloat16 \
    --port $PORT \
    --host 0.0.0.0 \
    --no-enable-prefix-caching \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-model-len 2880 \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --tensor-parallel-size $TP_SIZE \
    --data-parallel-size $(($NUM_NUMA_NODES/$TP_SIZE)) \
    --kv-cache-dtype fp8 > run_output/offline.log 2>&1 &
    # --served-model-name $SERVED_MODEL_NAME \
    # --api-server-count $NUM_VISIBLE_MEMORY_NODES \

function health_check_servers() {
    echo "Waiting for servers to start and checking health..."
    echo "Checking server at: $PORT"

    RETRY_COUNT=0
    MAX_RETRIES=100
    while ! curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -gt $MAX_RETRIES ]; then
            echo "Server failed to start"
            exit 1
        fi
        sleep 5
    done
    echo "Server ready"
}

# Check if servers were launched
health_check_servers
