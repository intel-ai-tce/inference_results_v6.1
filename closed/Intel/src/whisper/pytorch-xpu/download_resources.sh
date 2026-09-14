#!/bin/bash

mkdir -p /model
mkdir -p /data
curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh > /workspace/mlc-r2-downloader.sh

# Inference model
cd /model
python /workspace/code/calibration/download_model.py
python /workspace/code/calibration/calibrate_whisper.py

# Inference dataset
cd /data; bash /workspace/mlc-r2-downloader.sh https://inference.mlcommons-storage.org/metadata/whisper-dataset.uri; mv /data/dataset/dev-all-repack* /data/
sed -i 's|./data/dev-all-repack|/data/dev-all-repack|g' /data/dev-all-repack.json
