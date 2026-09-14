#!/bin/bash

mkdir -p /model
mkdir -p /data
curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh > /workspace/mlc-r2-downloader.sh

# Inference dataset
cd /data;  bash /workspace/mlc-r2-downloader.sh https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri
cd /data;  bash /workspace/mlc-r2-downloader.sh https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-dailymail-calibration.uri

# Inference model
cd /model;  bash /workspace/mlc-r2-downloader.sh https://llama3-1.mlcommons-storage.org/metadata/llama3-1-8b-instruct.uri
cd /workspace; python code/calibration/quantize_model.py --model-name /model/Llama-3.1-8B-Instruct --dataset-path /data/cnn_dailymail_calibration.json --quant-recipe code/calibration/recipe.yaml
