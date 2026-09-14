#!/usr/bin/env bash

CHECKPOINT_PATH="${CHECKPOINT_PATH}"
DATASET_PATH="${DATASET_PATH}"

python -u main.py --scenario Offline \
	--model-path "${CHECKPOINT_PATH}" \
	--batch-size 8192 \
	--dtype float16 \
	--user-conf user.conf \
	--total-sample-count 13368 \
	--dataset-path "${DATASET_PATH}" \
	--output-log-dir output \
	--tensor-parallel-size "${GPU_COUNT}" \
	--vllm 2>&1 | tee offline.log
