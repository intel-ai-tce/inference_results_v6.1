#!/bin/bash

# set -x

PRM800K_PATH="/lab-mlperf-inference/prm800k"
LIVECODEBENCH_PATH="/lab-mlperf-inference/LiveCodeBench"

if [ -e /lab-mlperf-inference/code/scripts/setup_deepseek_accuracy_env.sh ]
then
    if [ ! -d $PRM800K_PATH ];then
        pip install pylatexenc
        git clone https://github.com/openai/prm800k $PRM800K_PATH
    fi
    if [ ! -d $LIVECODEBENCH_PATH ]
    then
        git clone https://github.com/LiveCodeBench/LiveCodeBench $LIVECODEBENCH_PATH
    fi
else
    echo "ERROR: Please enter the MLPerf container before running this script"
    exit 1
fi
