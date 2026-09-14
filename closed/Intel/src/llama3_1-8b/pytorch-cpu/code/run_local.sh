#!/bin/bash

# Launch servers
export OUTPUT_DIR=${RUN_LOGS:-run_output}
mkdir -p ${OUTPUT_DIR}

if [ "${SCENARIO}" = "Server" ] && [ "${NUM_CORES}" == "256" ]; then
    source code/launch_server.sh
else
    source code/launch_offline.sh
fi

# Start workers with parameters
export TOTAL_SAMPLE_COUNT=13368 # Total number of samples in the dataset

BATCH_SIZE=${BATCH_SIZE:-128} # Batch size for each worker

export DATASET_PATH=/data/cnn_eval.json
cmd="python3 -u /workspace/code/main.py --dataset-path ${DATASET_PATH} \
    --scenario ${SCENARIO} \
    --mode ${MODE} \
    --workload-name llama3_1-8b \
    --model-path ${MODEL_PATH} \
    --total-sample-count ${TOTAL_SAMPLE_COUNT} \
    --batch-size ${BATCH_SIZE} \
    --device cpu \
    --user-conf user.conf \
    --output-log-dir ${OUTPUT_DIR} 
    ${EXTRA_ARGS} 2>&1 | tee ${OUTPUT_DIR}/run.log"

# Run the command
echo "Running command: $cmd"
eval $cmd

# If run was successful, evaluate the accuracy
if [ "${MODE}" = "Accuracy" ]; then
    echo "Run completed successfully. Evaluating accuracy..."
    if [ -f ${OUTPUT_DIR}/mlperf_log_accuracy.json ]; then
        python3 -u code/evaluation.py --mlperf-accuracy-file ${OUTPUT_DIR}/mlperf_log_accuracy.json \
                                      --dataset-file ${DATASET_PATH} \
                                      --model-name ${MODEL_PATH} \
                                      --dtype int32 \
                                      2>&1 | tee ${OUTPUT_DIR}/accuracy.txt
    else
        echo "[NOTE] Accuracy log file not found: ${OUTPUT_DIR}/mlperf_log_accuracy.json"
    fi
fi

pkill -TERM -f 'VLLM::|vllm serve|code/proxy\.py|code/main\.py'; sleep 5
pkill -KILL -f 'VLLM::|vllm serve|code/proxy\.py|code/main\.py'
