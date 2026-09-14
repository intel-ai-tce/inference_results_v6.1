#!/bin/bash

# set -x
set -e

PYTHON3_BIN_PATH=/lab-mlperf-inference/code/moe_accuracy_venv/bin
PYTHON3_PATH=${PYTHON3_BIN_PATH}/python3
ACTIVATE_PATH=${PYTHON3_BIN_PATH}/activate
DATASET_FILE_PATH=/data/mixtral-8x7b/mlperf_mixtral8x7b_dataset_15k.pkl
MODEL_PATH=/model/mixtral-8x7b/fp8_quantized/
EVAL_SCRIPT_PATH=/lab-mlperf-inference/mlperf_inference/language/mixtral-8x7b/evaluate-accuracy.py
MODEL_OVERRIDE_PATH=${MODEL_OVERRIDE_PATH:-""}

if [ -n "$MODEL_OVERRIDE_PATH" ]; then
    echo "Overriding model path with ${MODEL_OVERRIDE_PATH}"
    MODEL_PATH=${MODEL_OVERRIDE_PATH}
fi

# Initialize environments for accuracy evaluation
eval "$(rbenv init - --no-rehash bash)"
export PATH="$HOME/.rbenv/bin:$PATH"
eval "$(rbenv init -)"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion


if [ ! -f ${PYTHON3_PATH} ]; then
    echo "venv not found, run bash ./scripts/setup_mixtral_accuracy_env.sh"
    exit 1
fi

if [ ! -f ${DATASET_FILE_PATH} ]; then
    echo "dataset not found, check the README.md how to download it"
    exit 1
fi

if [ ! -f ${EVAL_SCRIPT_PATH} ]; then
    echo "tools/evaluate-accuracy.py not found"
    exit 1
fi

source ${ACTIVATE_PATH}

if [ ${PYTHON3_PATH} != `which python3` ]; then
    echo "incorrect python3 is used"
    exit 1
fi

ACCURACY_JSON=${1}

if [ -z ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    deactivate
    exit 1
fi

if [ ! -f ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    deactivate
    exit 1
fi

# Pre-download datasets to avoid multithreading issues
python -c 'import evaluate; evaluate.load("rouge"); import nltk; nltk.download("punkt"); nltk.download("punkt_tab")'

OUTPUT_DIR=$(dirname ${ACCURACY_JSON})
RESULT_TXT=${OUTPUT_DIR}/accuracy.txt

python -u ${EVAL_SCRIPT_PATH} --checkpoint-path ${MODEL_PATH} \
                              --mlperf-accuracy-file ${ACCURACY_JSON} \
                              --dataset-file ${DATASET_FILE_PATH} \
                              --dtype int32 \
                              --n_workers 8 > ${RESULT_TXT}
mv evaluated_test.json ${OUTPUT_DIR}

deactivate

echo "Check $RESULT_TXT for the accuracy scores"
