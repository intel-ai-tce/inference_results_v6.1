#!/bin/bash
set -x

OUTPUT_DIR=${RUN_LOGS}
SCENARIO=${SCENARIO}
export MODEL=${MODEL}

MODEL_NAME="/model/Llama-3.1-8B-Instruct_calibrated-xpu"
DATASET_PATH="/data/cnn_eval.json"
TOTAL_SAMPLE_COUNT=13368
TP=1
PP=1

export MAX_MODEL_LEN=3072
export VLLM_USE_TRITON_XPU_ATTN=1
export VLLM_XPU_USE_W4A8=1
export XPU_COUNT=$(python -c "import torch; count = 0; count = torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0; print(count)")

if (( XPU_COUNT > 0 )); then
    # XPU-specific
    export VLLM_USE_V1=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export XPU_DEVICE_ID=$(python -c "import torch; print(torch.xpu.get_device_properties(0).device_id)")

    export NUM_INSTS=$((XPU_COUNT / TP))
    if (( XPU_DEVICE_ID == 57891 )); then
        # B70
        export BATCH_SIZE=384
        export GPU_MEMORY_UTILIZATION=0.90
        if [ "${SCENARIO}" == "Server" ]; then export MAX_NUM_BATCHED_TOKENS=512; else export MAX_NUM_BATCHED_TOKENS=2048; fi
    elif (( XPU_DEVICE_ID == 57873 )); then
        # B60
        export BATCH_SIZE=256
        if [ "${SCENARIO}" == "Server" ]; then export MAX_NUM_BATCHED_TOKENS=768; else export MAX_NUM_BATCHED_TOKENS=1536; fi
        if [ "${SCENARIO}" == "Server" ]; then export GPU_MEMORY_UTILIZATION=0.9; else export GPU_MEMORY_UTILIZATION=0.88; fi
    elif (( XPU_DEVICE_ID == 57874 )); then
        # B50
        export BATCH_SIZE=128
        export GPU_MEMORY_UTILIZATION=0.85
        if [ "${SCENARIO}" == "Server" ]; then export MAX_NUM_BATCHED_TOKENS=256; else export MAX_NUM_BATCHED_TOKENS=768; fi
    else
        echo "Error: XPU device ID $XPU_DEVICE_ID is not in allowed set})."
        return 1
    fi
else
    echo "Error: No XPU device ID found})."
    return 1
fi

run_cmd="python -u /workspace/code/${MODEL}/main.py 
         --mlperf-conf mlperf.conf 
	 --model-path ${MODEL_NAME}
	 --workload-name ${MODEL}
	 --dataset-path ${DATASET_PATH}
	 --scenario ${SCENARIO}
	 --total-sample-count ${TOTAL_SAMPLE_COUNT}
	 --batch-size ${BATCH_SIZE}
	 --num-workers ${NUM_INSTS}
	 --tensor-parallel ${TP}
	 --pipeline-parallel ${PP}
	 --output-log-dir ${OUTPUT_DIR}
	 --warmup
	 --user-conf user.conf"

if [ "${MODE}" == "Accuracy" ]; then
    run_cmd+=" --accuracy "
fi

echo "RUN_COMMAND: ${run_cmd}"
sleep 5

$run_cmd 2>&1 | tee ${OUTPUT_DIR}/run.log

if [ "${MODE}" == "Accuracy" ]; then
	python3 /workspace/code/${MODEL}/evaluate-accuracy.py \
	--checkpoint-path ${MODEL_NAME} \
        --mlperf-accuracy-file ${OUTPUT_DIR}/mlperf_log_accuracy.json \
        --dataset-file ${DATASET_PATH} \
        --dtype int64 2>&1 | tee ${OUTPUT_DIR}/accuracy.txt
fi
