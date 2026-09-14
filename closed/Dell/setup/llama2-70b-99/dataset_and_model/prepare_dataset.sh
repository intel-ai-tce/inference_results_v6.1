#!/bin/bash

DOWNLOAD_DIR="/data/processed-openorca"

echo "0" | mlcr get,dataset,preprocessed,openorca,_validation,_r2-downloader,_mlc --outdirname=llama2-dataset -j
mlcr get,dataset,preprocessed,openorca,_calibration,_r2-downloader,_mlc --outdirname=llama2-dataset -j

mkdir -p $DOWNLOAD_DIR
cp /lab-mlperf-inference/llama2-dataset/llama-2-70b-open-orca-dataset.uri/open_orca_gpt4_tokenized_llama.sampled_24576.pkl $DOWNLOAD_DIR/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
cp /lab-mlperf-inference/llama2-dataset/llama-2-70b-open-orca-dataset.uri/open_orca_gpt4_tokenized_llama.calibration_1000.pkl $DOWNLOAD_DIR/open_orca_gpt4_tokenized_llama.calibration_1000.pkl 
