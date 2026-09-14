#!/usr/bin/env bash

CHECKPOINT_PATH=/model/
DATASET_PATH="${DATASET_PATH:-/data/cnn_eval.json}"
GPU_COUNT="${GPU_COUNT:-1}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "run_outputs"

python3 -u main.py --scenario Offline \
	--model-path "${CHECKPOINT_PATH}" \
	--batch-size 4096 \
	--accuracy \
	--dtype bfloat16 \
        --tensor-parallel-size "${GPU_COUNT}" \
	--user-conf user.conf \
	--total-sample-count 13368 \
	--dataset-path "${DATASET_PATH}" \
	--output-log-dir offline_accuracy_loadgen_logs \
	--vllm 2>&1 | tee offline_accuracy_log.log
