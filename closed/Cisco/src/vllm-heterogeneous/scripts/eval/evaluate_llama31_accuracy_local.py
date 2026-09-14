#!/usr/bin/env python3
"""Local Llama 3.1 8B accuracy scorer.

Mirrors the MLCommons llama3.1-8b evaluator but avoids Hugging Face
`evaluate.load("rouge")`, which may require a network/module cache. It uses the
same underlying `rouge_score` package and local tokenizer/model path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nltk
import numpy as np
from rouge_score import rouge_scorer
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-accuracy-file", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--total-sample-count", type=int, default=13368)
    parser.add_argument("--dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument("--nltk-data", default="/inference/nltk_data")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def postprocess_text(items: list[str]) -> list[str]:
    processed = []
    for item in items:
        stripped = item.strip()
        processed.append("\n".join(nltk.sent_tokenize(stripped)))
    return processed


def load_targets(dataset_file: Path, total_sample_count: int) -> list[str]:
    with dataset_file.open() as f:
        data = json.load(f)
    if len(data) < total_sample_count:
        raise ValueError(f"dataset has {len(data)} rows, expected {total_sample_count}")
    targets = []
    for row in data[:total_sample_count]:
        if "output" not in row:
            raise KeyError("dataset row missing output field")
        targets.append(row["output"])
    return targets


def main() -> None:
    args = parse_args()
    nltk.data.path.insert(0, args.nltk_data)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=128000,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    targets = load_targets(Path(args.dataset_file), args.total_sample_count)
    with Path(args.mlperf_accuracy_file).open() as f:
        raw_results = json.load(f)

    seen = set()
    qsl_indices = []
    pred_token_ids = []
    eval_dtype = np.int32 if args.dtype == "int32" else np.int64
    for result in raw_results:
        qsl_idx = result["qsl_idx"]
        if qsl_idx in seen:
            continue
        seen.add(qsl_idx)
        qsl_indices.append(qsl_idx)
        pred_token_ids.append(np.frombuffer(bytes.fromhex(result["data"]), eval_dtype))

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
        split_summaries=True,
    )
    sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    gen_len = 0
    count = 0
    for start in range(0, len(pred_token_ids), args.batch_size):
        end = start + args.batch_size
        decoded = tokenizer.batch_decode(pred_token_ids[start:end], skip_special_tokens=True)
        preds = postprocess_text(decoded)
        refs = postprocess_text([targets[idx] for idx in qsl_indices[start:end]])
        for pred, ref in zip(preds, refs):
            scores = scorer.score(ref, pred)
            for key in sums:
                sums[key] += scores[key].fmeasure
        gen_len += sum(len(pred) for pred in preds)
        count += len(preds)
    result = {key: f"{round((value / count) * 100, 4)}" for key, value in sums.items()}
    result["gen_len"] = gen_len
    result["gen_num"] = count
    print("\nResults\n")
    print(result)


if __name__ == "__main__":
    main()
