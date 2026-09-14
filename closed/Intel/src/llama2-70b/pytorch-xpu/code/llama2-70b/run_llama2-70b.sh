#!/bin/bash
set -x

OUTPUT_DIR=${RUN_LOGS}
SCENARIO=${SCENARIO}
export MODEL=${MODEL}

MODEL_NAME="/model/Llama-2-70b-chat-hf_calibrated-xpu"
DATASET_PATH="/data/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
TOTAL_SAMPLE_COUNT=24576
TP=1

export MAX_MODEL_LEN=2048
export VLLM_USE_TRITON_XPU_ATTN=1
export VLLM_XPU_USE_W4A8=1
export XPU_COUNT=$(python -c "import torch; count = 0; count = torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0; print(count)")

if (( XPU_COUNT > 0 )); then
    # XPU-specific
    export VLLM_USE_V1=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export XPU_DEVICE_ID=$(python -c "import torch; print(torch.xpu.get_device_properties(0).device_id)")

    if (( XPU_DEVICE_ID == 57891 )); then
        # B70
        export PP=2
        export NUM_INSTS=$((XPU_COUNT / (TP*PP)))
        export BATCH_SIZE=1024
        export GPU_MEMORY_UTILIZATION=0.99
        if (( XPU_COUNT > 2 )); then
            export MAX_NUM_BATCHED_TOKENS=320
        else
            export MAX_NUM_BATCHED_TOKENS=256
        fi
    elif (( XPU_DEVICE_ID == 57873 )); then
        # B60
	if [ "${SCENARIO}" == "Server" ]; then export PP=2; else export PP=4; fi
        export NUM_INSTS=$((XPU_COUNT / (TP*PP)))
        export BATCH_SIZE=1024
        export GPU_MEMORY_UTILIZATION=0.93
	if [ "${SCENARIO}" == "Server" ]; then export GPU_MEMORY_UTILIZATION=0.99; else export GPU_MEMORY_UTILIZATION=0.93; fi
        if [ "${SCENARIO}" == "Server" ]; then export MAX_NUM_BATCHED_TOKENS=192; else export MAX_NUM_BATCHED_TOKENS=320; fi
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

