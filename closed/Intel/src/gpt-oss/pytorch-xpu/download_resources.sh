#!/bin/bash

mkdir -p /model
mkdir -p /data
curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh > /workspace/mlc-r2-downloader.sh

# Inference dataset
cd /model;  bash /workspace/mlc-r2-downloader.sh https://inference.mlcommons-storage.org/metadata/gpt-oss-model.uri
cd /data;  bash /workspace/mlc-r2-downloader.sh https://inference.mlcommons-storage.org/metadata/gpt-oss-data.uri
