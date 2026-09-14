# SPDX-FileCopyrightText: Copyright (c) 2026 Intel Corporation & Affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datasets import Dataset
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", default="/model/Llama-3.1-8B-Instruct_calibrated-cpu"
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to calibration dataset"
    )
    parser.add_argument("--quant-recipe", type=str, required=True, help="Path to quantization recipe")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataframe = pd.read_json(args.dataset_path)


    ds = Dataset.from_dict({"input_ids": dataframe["tok_input"].tolist()})

    NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 4096
    SAVE_DIR = args.model_name + "_calibrated-cpu"
    oneshot(
        model=model,
        dataset=ds,
        recipe=args.quant_recipe,
        shuffle_calibration_samples=True,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        # output_dir=SAVE_DIR,
    )

    # Save to disk compressed.
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)

if __name__ == "__main__":
    main()
