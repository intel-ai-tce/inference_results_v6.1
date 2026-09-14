"""Offline accuracy evaluation for Llama2-70B (no internet required)."""
import argparse
import os
import json
import numpy as np
from multiprocessing import Pool, cpu_count

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import nltk
nltk_data = os.environ.get("NLTK_DATA")
if nltk_data:
    nltk.data.path.insert(0, nltk_data)
nltk.download = lambda *a, **k: True

from transformers import AutoTokenizer
from rouge_score import rouge_scorer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--mlperf-accuracy-file", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--dtype", default="int32", choices=["int32", "int64", "float"])
    return parser.parse_args()


def get_groundtruth(dataset_file):
    import pandas as pd
    return pd.read_pickle(dataset_file)["output"]


def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]
    try:
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]
    except LookupError:
        pass
    return preds, targets


def compute_rouge_chunk(chunk):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    preds, targets = chunk
    results = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}
    for pred, target in zip(preds, targets):
        score = scorer.score(target, pred)
        for key in results:
            results[key].append(score[key].fmeasure)
    return results


def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, model_max_length=2048, padding_side="left", use_fast=False,
    )

    targets = get_groundtruth(args.dataset_file)

    eval_dtype = {"int32": np.int32, "int64": np.int64, "float": np.float32}[args.dtype]

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    target_required = []
    preds_token_ids = []
    gen_tok_len = 0

    for pred in results:
        qsl_idx = pred["qsl_idx"]
        if qsl_idx in seen:
            continue
        seen.add(qsl_idx)
        target_required.append(targets[qsl_idx])
        token_ids = np.frombuffer(bytes.fromhex(pred["data"]), eval_dtype)
        gen_tok_len += len(token_ids)
        preds_token_ids.append(token_ids)

    print(f"Loaded {len(preds_token_ids)} predictions, {gen_tok_len} total tokens")

    preds_decoded_text = tokenizer.batch_decode(preds_token_ids, skip_special_tokens=True)
    preds, targets_post = postprocess_text(preds_decoded_text, target_required)

    num_chunks = min(cpu_count(), len(preds))
    chunk_size = len(preds) // num_chunks + (len(preds) % num_chunks > 0)
    chunks = [(preds[i:i+chunk_size], targets_post[i:i+chunk_size])
              for i in range(0, len(preds), chunk_size)]

    print(f"Computing ROUGE with {num_chunks} workers...")
    with Pool(num_chunks) as pool:
        results_list = pool.map(compute_rouge_chunk, chunks)

    aggregated = {}
    for result in results_list:
        for k, v in result.items():
            aggregated.setdefault(k, []).extend(v)

    final = {k: round(np.mean(v) * 100, 4) for k, v in aggregated.items()}
    final["gen_num"] = len(preds)
    final["gen_tok_len"] = gen_tok_len
    final["tokens_per_sample"] = round(gen_tok_len / len(preds), 1)

    print("\nResults\n")
    print(final)


if __name__ == "__main__":
    main()
