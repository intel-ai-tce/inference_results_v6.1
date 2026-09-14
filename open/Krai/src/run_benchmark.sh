# Run this script from open/Krai
# EXPORT HF_TOKEN before running
set -eo pipefail

MODEL="Qwen/Qwen3-VL-235B-A22B-Instruct"

RESULTS_LOCATION="/root/mlperf_bench" # wherever endpoints is configured to store results by default
MLPERF_MOUNTS=" \
    -v $(pwd)/results/h200/qwen3-vl-235b-a22b/Offline:${RESULTS_LOCATION} \
    -v $(pwd)/src:/src"

PORT=30000

CONTAINER_IMG=ainikolai/mlperf_vlm_v6.1:latest
VLLM_IMAGE=vllm/vllm-openai:v0.24.0-cu129-ubuntu2404

echo "STARTING"

server_running() {
    local url="http://localhost:${PORT}/health"
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo "True"
    else
        echo "False"
    fi
}

wait_for_server() {
    local url="http://localhost:${PORT}/health"
    local max_attempts=${1:-1200}
    local attempt=0
    echo "Waiting for LLM server to be ready..."
    while true; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo "Server is ready."
            return 0
        fi
        attempt=$((attempt + 1))
        echo -ne "Progress: $attempt/${max_attempts}\r"
        if [ $attempt -ge $max_attempts ]; then
            echo "ERROR: Server did not become ready in time."
            return 1
        fi
        sleep 1
    done
}

cleanup() {
    echo "Cleaning up: stopping model server container..."
    docker rm -f gentic-mlperf-vision-model 2>/dev/null || true
}

trap cleanup EXIT

echo "======"
echo "Is server already running?"
is_server=$(server_running)
if [ $is_server = "False" ]; then
    echo "No Server Running. Attempting to start server..."
    echo "STARTING SERVER with $MODEL"

    docker pull "$VLLM_IMAGE"
    docker run -d --rm \
        --name gentic-mlperf-vision-model \
        --network host \
        -p ${PORT}:${PORT} \
        --ipc=host \
        --gpus '"device=0,1,2,3,4,5,6,7"' \
        ${MLPERF_MOUNTS} \
        -e VLLM_USE_FLASHINFER_SAMPLER=1 \
        -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
        -e VLLM_FLASHINFER_MOE_BACKEND=latency \
        -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((6 * 256 * 1024 * 1024)) \
        -e VLLM_USE_TRITON_POS_EMBED=1 \
        -e VLLM_MM_ENCODER_FP8_ATTN=1 \
        -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
        --entrypoint bash \
        ${VLLM_IMAGE} \
        -c "patch -p0 -d / < /src/integration.patch && \
            vllm serve $MODEL \
            --tensor-parallel-size 1 \
            --data-parallel-size 8 \
            --attention-backend TRITON_ATTN \
            --mm-encoder-attn-backend=FLASHINFER \
            --no-enable-prefix-caching \
            --kv-cache-dtype=fp8 \
            --max-model-len=32768 \
            --max-num-seqs=1024 \
            --max-num-batched-tokens=4864 \
            --async-scheduling \
            --limit-mm-per-prompt.video 0 \
            --mm-processor-cache-gb=0 \
            --kv-events-config='{\"publisher\":\"null\"}' \
            --compilation-config='{
                \"max_cudagraph_capture_size\": 4864,
                \"cudagraph_capture_sizes\": [
                    1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
                    136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248,
                    256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480,
                    496, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928,
                    960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344,
                    1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760,
                    1792, 1824, 1856, 1888, 1920, 1952, 1984, 2016, 2048, 2080, 2112, 2144, 2176,
                    2208, 2240, 2272, 2304, 2336, 2368, 2400, 2432, 2464, 2496, 2528, 2560, 2592,
                    2624, 2656, 2688, 2720, 2752, 2784, 2816, 2848, 2880, 2912, 2944, 2976, 3008,
                    3040, 3072, 3104, 3136, 3168, 3200, 3232, 3264, 3296, 3328, 3360, 3392, 3424,
                    3456, 3488, 3520, 3552, 3584, 3616, 3648, 3680, 3712, 3744, 3776, 3808, 3840,
                    3872, 3904, 3936, 3968, 4000, 4032, 4064, 4096, 4128, 4160, 4192, 4224, 4256,
                    4288, 4320, 4352, 4384, 4416, 4448, 4480, 4512, 4544, 4576, 4608, 4640, 4672,
                    4704, 4736, 4768, 4800, 4832, 4864
                ]
            }' \
            --override-generation-config='{\"max_new_tokens\": 150}' \
            --host 0.0.0.0 \
            --port ${PORT}"
    wait_for_server

else
  echo "Server is already running."
fi

echo "==========================="
echo "STARTING MLPERF BENCHMARK"

docker pull "$CONTAINER_IMG"

docker run --rm \
  --name gentic-mlperf-vision-bench \
  --network host \
  -p ${PORT}:${PORT} \
  --ipc=host \
  -e HF_TOKEN=$HF_TOKEN \
  ${MLPERF_MOUNTS} \
  --entrypoint bash \
  ${CONTAINER_IMG} \
  -c "inference-endpoint benchmark from-config \
    -c /src/mlperf_small_endpoint.yaml"