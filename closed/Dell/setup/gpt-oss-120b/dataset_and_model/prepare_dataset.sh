#!/bin/bash

DOWNLOAD_DIR="/data/gpt-oss-120b"

mlcr get-dataset-mlperf-inference-gpt-oss,_mlc,_r2-downloader --outdirname=gpt-oss-120b -j

mkdir -p $DOWNLOAD_DIR
cp /lab-mlperf-inference/gpt-oss-120b/gpt-oss-dataset/acc/acc_eval_compliance_gpqa.parquet $DOWNLOAD_DIR/
cp /lab-mlperf-inference/gpt-oss-120b/gpt-oss-dataset/acc/acc_eval_ref.parquet $DOWNLOAD_DIR/
cp /lab-mlperf-inference/gpt-oss-120b/gpt-oss-dataset/perf/perf_eval_ref.parquet $DOWNLOAD_DIR/
cp /lab-mlperf-inference/gpt-oss-120b/gpt-oss-dataset/acc/calibration_unique_sampled1024.parquet $DOWNLOAD_DIR/
