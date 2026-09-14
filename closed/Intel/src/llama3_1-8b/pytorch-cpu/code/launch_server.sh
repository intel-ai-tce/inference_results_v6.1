#!/bin/bash

export NUM_SOCKETS=$(lscpu | grep "Socket(s):" | awk '{print $2}')
export NUM_NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $NF}')
export NUM_CORES=$(($(lscpu | grep "Socket(s):" | awk '{print $2}') * $(lscpu | grep "Core(s) per socket:" | awk '{print $4}')))
export NUMA_PER_SOCKET=$(($NUM_NUMA_NODES/$NUM_SOCKETS))
NUM_INSTS=${NUM_INSTS:-$NUM_NUMA_NODES}
CORES_PER_INST=${CORES_PER_INST:-$(($NUM_CORES / $NUM_NUMA_NODES))}

if [ "${NUM_CORES}" == "288" ]; then
    export VLLM_CPU_KVCACHE_SPACE="25"
    export KV_DTYPE=auto
    export MAX_NUM_BATCHED_TOKENS=178880
    if [ "${SCENARIO}" == "Offline" ]; then
        export BS=128
    else
        export BS=64
    fi
else
    export VLLM_CPU_KVCACHE_SPACE="75"
    export KV_DTYPE=fp8
    export MAX_NUM_BATCHED_TOKENS=357760
    if [ "${SCENARIO}" == "Offline" ]; then
        export BS=256
    else
       export BS=128
    fi
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export START_PORT=8000
export OMP_NUM_THREADS=$(($CORES_PER_INST-2))
export KV_BUFFER_DEVICE="cpu"
export DECODER_KV_LAYOUT="HND"
export UCX_TLS=sm,cma,self,tcp
export UCX_LOG_LEVEL=error
export UCX_NET_DEVICES="all"
export MODEL_PATH=${1:-"/model/Llama-3.1-8B-Instruct_calibrated-cpu"}

# Function to launch servers with specific port and CPU affinity list
function launch_decode() {
    local port=$1
    local start_core=$2
    local end_core=$3
    local index=$4
    local model_path=${MODEL_PATH}
    
    export KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":30001,\"http_ip\":\"127.0.0.1\",\"http_port\":$port,\"send_type\":\"PUT_ASYNC\",\"num_threads\":1},\"kv_buffer_device\":\"cpu\"}"
    export VLLM_NIXL_SIDE_CHANNEL_PORT=$(($index+5660))
    export VLLM_CPU_OMP_THREADS_BIND="$start_core-$(($start_core+$OMP_NUM_THREADS-1))"
    echo "OMP cores $VLLM_CPU_OMP_THREADS_BIND"
    echo "Launching decode on port $port with CPU affinity "$start_core-$end_core""
    taskset -c "$start_core-$end_core" vllm serve "$model_path" \
        --host 0.0.0.0 \
        --port $port \
        --dtype bfloat16 \
        --no-enable-prefix-caching \
        --max-model-len 2668 \
        --max-num-seqs $BS \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --gpu-memory-utilization 0.95 \
        --kv-cache-dtype $KV_DTYPE \
        --kv-transfer-config $KV_CONFIG > ${RUN_LOGS}/decode_$index.log 2>&1 &
}

function launch_prefill() {
    local port=$1
    local start_core=$2
    local end_core=$3
    local index=$4
    local model_path=${MODEL_PATH}
    export KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":30001,\"http_ip\":\"127.0.0.1\",\"http_port\":$port,\"send_type\":\"PUT_ASYNC\",\"num_threads\":2},\"kv_buffer_device\":\"cpu\"}"
    export VLLM_NIXL_SIDE_CHANNEL_PORT=$(($index+5660))
    export VLLM_CPU_OMP_THREADS_BIND="$start_core-$(($start_core+$OMP_NUM_THREADS))"
    echo "OMP cores $VLLM_CPU_OMP_THREADS_BIND"
    echo "Launching prefill on port $port with CPU affinity "$start_core-$end_core""
    taskset -c "$start_core-$end_core" vllm serve "$model_path" \
        --host 0.0.0.0 \
        --port $port \
        --dtype bfloat16 \
        --no-enable-prefix-caching \
        --max-model-len 2668 \
        --max-num-seqs $BS \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.95 \
        --kv-cache-dtype $KV_DTYPE \
        --kv-transfer-config $KV_CONFIG > ${RUN_LOGS}/prefill_$index.log 2>&1 &
}

function health_check_servers() {
    echo "Waiting for prefill servers to start and checking health..."
    for port in "${PREFILL_PORTS[@]}"; do
        echo "Checking server at: $port"
        
        # Wait up to 120 seconds for server to become ready
        RETRY_COUNT=0
        MAX_RETRIES=100
        # Wait for prefill server
        while ! curl -s http://localhost:$port/v1/models > /dev/null 2>&1; do
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -gt $MAX_RETRIES ]; then
                echo "Prefill server failed to start"
                exit 1
            fi
            sleep 5
        done
        echo "Prefill server ready"
    done

    echo "Waiting for decode servers to start and checking health..."
    for port in "${DECODE_PORTS[@]}"; do
        echo "Checking server at: $port"
        
        # Wait up to 120 seconds for server to become ready
        RETRY_COUNT=0
        MAX_RETRIES=100
        # Wait for decode server
        while ! curl -s http://localhost:$port/v1/models > /dev/null 2>&1; do
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -gt $MAX_RETRIES ]; then
                echo "Decode server failed to start"
                exit 1
            fi
            sleep 5
        done
        echo "Decode server ready"
    done
}

# List of launched servers
DECODE_PORTS=()
PREFILL_PORTS=()
# Launch servers on numa nodes with different CPU affinities
NUM_LAUNCHED_SERVERS=0
# Loop through each NUMA node starting from START_NODE
for ((i=0; i<$NUM_INSTS; i++)); do
        # Get the start and end cores for this NUMA node. Get the physical cores only.  
        numa_info=$(lscpu | grep "NUMA node$i CPU(s):")
        numa_cores_list=$(echo $numa_info | awk '{print $4}' | tr ',' ' ')

        start_core=$(echo $numa_cores_list | awk '{print $1}' | cut -d'-' -f1)
        end_core=$(echo $numa_cores_list | awk '{print $1}' | cut -d'-' -f2)

        # start_core=$(($i * $CORES_PER_INST))
        # end_core=$(($start_core + $CORES_PER_INST - 1))

        # Launch server with the calculated CPU affinity
        port=$(($START_PORT+$NUM_LAUNCHED_SERVERS))
        if (( $(($i+1)) % 3 == 0 )); then
            launch_decode $port $start_core $end_core $i
            DECODE_PORTS+=("$port")
        else
            launch_prefill $port $start_core $end_core $i
            PREFILL_PORTS+=("$port")
        fi

        NUM_LAUNCHED_SERVERS=$(($NUM_LAUNCHED_SERVERS+1))
        echo -e "Finished launching servers on NUMA node $i\n"
done

# Check if servers were launched
health_check_servers

# Launch proxy server
echo "Launching proxy server for prefill ${PREFILL_PORTS[@]} decode ${DECODE_PORTS[@]}"
python code/proxy.py \
    --port 8192 \
    --prefiller-ports ${PREFILL_PORTS[@]} \
    --decoder-ports ${DECODE_PORTS[@]} > ${RUN_LOGS}/proxy.log 2>&1 &
echo "Proxy server ready"
