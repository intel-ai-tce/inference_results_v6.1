#!/usr/bin/env bash

#CHECKPOINT_PATH="${CHECKPOINT_PATH}"
#DATASET_PATH="${DATASET_PATH}"

DATASET_PATH="/data/cnn_eval.json"

mkdir -p "run_outputs"

python3 evaluation.py \
	--mlperf-accuracy-file mlperf_log_accuracy.json \
	--dataset-file "${DATASET_PATH}" \
	--dtype int32
