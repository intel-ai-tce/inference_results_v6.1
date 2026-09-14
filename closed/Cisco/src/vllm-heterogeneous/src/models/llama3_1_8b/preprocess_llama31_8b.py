#!/usr/bin/env python3
"""Prepare Llama 3.1 8B CNN/DailyMail JSON for this harness.

The downloaded MLCommons JSON already includes tokenized inputs in `tok_input`.
This helper validates the schema and optionally emits the generic npy directory
format used by `src/sut/dataset.py`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_records(path):
    with open(path, "r") as f:
        records = json.load(f)
    required = {"input", "tok_input", "output"}
    missing = required - set(records[0])
    if missing:
        raise ValueError(f"{path} is missing required fields: {sorted(missing)}")
    for i, row in enumerate(records):
        if not isinstance(row["tok_input"], list):
            raise ValueError(f"{path} row {i} has non-list tok_input")
    return records


def write_outputs(records, output_dir, prefix, pad_token_id):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    pkl_path = output_dir / f"{prefix}.pkl"
    df.to_pickle(pkl_path)

    lengths = np.asarray([len(row["tok_input"]) for row in records],
                         dtype=np.int32)
    max_len = int(lengths.max())
    padded = np.full((len(records), max_len), pad_token_id, dtype=np.int32)
    for i, row in enumerate(records):
        toks = np.asarray(row["tok_input"], dtype=np.int32)
        padded[i, :len(toks)] = toks

    ids_path = output_dir / f"{prefix}_input_ids_padded.npy"
    lens_path = output_dir / f"{prefix}_input_lens.npy"
    np.save(ids_path, padded)
    np.save(lens_path, lengths)

    if prefix == "cnn_eval":
        np.save(output_dir / "input_ids_padded.npy", padded)
        np.save(output_dir / "input_lens.npy", lengths)

    print(f"{prefix}: samples={len(records)} max_input_len={max_len}")
    print(f"  {pkl_path}")
    print(f"  {ids_path}")
    print(f"  {lens_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--calibration-json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pad-token-id", type=int, default=128001)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    eval_records = load_records(Path(args.eval_json))
    write_outputs(eval_records, output_dir, "cnn_eval", args.pad_token_id)

    if args.calibration_json:
        calib_records = load_records(Path(args.calibration_json))
        write_outputs(
            calib_records, output_dir, "cnn_dailymail_calibration",
            args.pad_token_id)


if __name__ == "__main__":
    main()
