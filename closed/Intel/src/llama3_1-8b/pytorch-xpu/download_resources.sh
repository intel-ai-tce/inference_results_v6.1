#!/bin/bash

mkdir -p /model
mkdir -p /data
curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh > /workspace/mlc-r2-downloader.sh

export MODEL="${MODEL:-llama3_1-8b}"

# Inference dataset
if [ "${MODEL}" == "llama3_1-8b" ]; then
    cd /model; bash /workspace/mlc-r2-downloader.sh https://inference.mlcommons-storage.org/metadata/llama3-1-8b-instruct_calibrated-xpu.uri
    cd /data;  bash /workspace/mlc-r2-downloader.sh https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri
elif [ "${MODEL}" == "llama2-70b" ]; then
    cd /model; bash /workspace/mlc-r2-downloader.sh https://inference.mlcommons-storage.org/metadata/llama-2-70b-chat-hf_calibrated-xpu.uri
    cd /data;  bash /workspace/mlc-r2-downloader.sh https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.uri
    gunzip /data/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
fi

