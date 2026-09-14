#!/bin/bash

# set -x
set -e

PYTHON3_PATH=/usr/bin/python3
DATASET_FILE_PATH=/data/gpt-oss-120b/acc_eval_ref.parquet
TEST07_DATASET_FILE_PATH=/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet
TEST07_DATASET_SIZE=990
TEST09_DATASET_FILE_PATH=/data/gpt-oss-120b/perf_eval_ref.parquet
TEST09_DATASET_SIZE=6396
MODEL_PATH=/model/gpt-oss-120b/fp4_quantized/
EVAL_SCRIPT_PATH=/lab-mlperf-inference/mlperf_inference/language/gpt-oss-120b/eval_mlperf_accuracy.py
MODEL_OVERRIDE_PATH=${MODEL_OVERRIDE_PATH:-""}

ARCH=${ARCH:-$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')}

if [ -n "$MODEL_OVERRIDE_PATH" ]; then
    echo "Overriding model path with ${MODEL_OVERRIDE_PATH}"
    MODEL_PATH=${MODEL_OVERRIDE_PATH}
fi

if [ ! -f ${PYTHON3_PATH} ]; then
    echo "Wrong python3 path"

    echo "Fallback to 'which python3' path"
    PYTHON3_PATH=$(which python3 2>/dev/null)

    if [ ! -f ${PYTHON3_PATH} ]; then
        echo "Fallback to 'which python' path"
        PYTHON3_PATH=$(which python 2>/dev/null)
    fi

    if [ -z "$PYTHON3_PATH" ]; then
        echo "Error: No Python interpreter found."
        exit 1
    fi
fi

if [ ! -f ${DATASET_FILE_PATH} ]; then
    echo "dataset not found, check the README.md how to download it"
    exit 1
fi

if [ ! -d ${MODEL_PATH} ]; then
    echo "model not found, check the README.md how to download it"
    exit 1
elif [ -z "$(ls -A ${MODEL_PATH})" ]; then
    echo "model dir is empty, check the README.md how to download it"
    exit 1
fi

if [ ! -f ${EVAL_SCRIPT_PATH} ]; then
    echo "Accuracy eval script not found: ${EVAL_SCRIPT_PATH}"
    exit 1
fi

ACCURACY_JSON=${1}

if [ -z ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    exit 1
fi

if [ ! -f ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    exit 1
fi

OUTPUT_DIR=$(dirname ${ACCURACY_JSON})
RESULT_TXT=${OUTPUT_DIR}/accuracy.txt
RESULT_JSON=${OUTPUT_DIR}/accuracy_results.json

${PYTHON3_PATH} -u ${EVAL_SCRIPT_PATH} --reference-data ${DATASET_FILE_PATH} \
                                       --mlperf-log ${ACCURACY_JSON} \
                                       --tokenizer ${MODEL_PATH} \
                                       --output-file ${RESULT_JSON} | tee 1> ${RESULT_TXT}

echo "Check $RESULT_TXT for the accuracy scores"
