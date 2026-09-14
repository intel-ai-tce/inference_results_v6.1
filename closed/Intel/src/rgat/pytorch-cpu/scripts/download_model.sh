#!/bin/bash

CHECKSUMS=(
            /model/RGAT.pt,81225f862db03e6042f68b088b84face
          )

echo "Downloading model..."
export MODEL_DIR=/model
mlc pull repo mlcommons@mlperf-automations --branch=dev
mlcr get,ml-model,rgat --outdirname=$MODEL_DIR
mv $MODEL_DIR/RGAT/RGAT.pt $MODEL_DIR/RGAT.pt

echo "Verifying md5sum of the model"
pushd /workspace
    for ITEM in "${CHECKSUMS[@]}"; do
        FILENAME=$(echo ${ITEM} | cut -d',' -f1)
        CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
        bash run_validation.sh ${FILENAME} ${CHECKSUM}
    done
popd
