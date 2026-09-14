#!/bin/bash
set -x

run_cmd="python -u code/main.py 
         --mlperf-conf mlperf.conf 
		 --model-path ${MODEL_NAME}
		 --workload-name ${WORKLOAD}
		 --dataset-path ${DATASET_PATH}
		 --total-sample-count ${TOTAL_SAMPLE_COUNT}
		 --batch-size ${BATCH_SIZE}
		 --num-workers ${NUM_INSTS}
		 --tensor-parallel ${TP}
		 --pipeline-parallel ${PP}
		 --output-log-dir ${OUTPUT_DIR}
		 --scenario ${SCENARIO}
		 --warmup
		 --user-conf user.conf "

if [[ "$MODE" == "Accuracy" ]]; then
	run_cmd+="--accuracy "
fi

echo $run_cmd

$run_cmd 2>&1 | tee ${OUTPUT_DIR}/run.log

if [[ "$MODE" == "Accuracy" ]]; then
	python3 /workspace/code/eval_mlperf_accuracy.py \
        --mlperf-log ${OUTPUT_DIR}/mlperf_log_accuracy.json \
        --reference-data ${DATASET_PATH} \
        --tokenizer openai/gpt-oss-120b 2>&1 | tee ${OUTPUT_DIR}/accuracy.txt
fi
