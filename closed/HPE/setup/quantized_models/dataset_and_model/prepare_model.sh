#!/bin/bash

MODEL=$1
MODEL_PATH=$2
HF_TOKEN=$3

hf download $MODEL --token $HF_TOKEN --local-dir $MODEL_PATH
