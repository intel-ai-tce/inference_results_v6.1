#!/bin/bash

# set -x

if [ -e /lab-mlperf-inference/code/scripts/setup_llama2_accuracy_env.sh ]
then
    python3 -m venv /lab-mlperf-inference/code/llama2_accuracy_venv
    source /lab-mlperf-inference/code/llama2_accuracy_venv/bin/activate
    pip install transformers google nltk==3.8.1 evaluate==0.4.0 absl-py==1.4.0 rouge-score==0.1.2 sentencepiece==0.1.99 accelerate==0.21.0 numpy==1.26.4 protobuf==3.20.0
    deactivate
else
    echo "ERROR: Please enter the MLPerf container before running this script"
    exit 1
fi
