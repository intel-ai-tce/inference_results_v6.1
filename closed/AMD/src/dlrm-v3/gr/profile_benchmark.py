#!/usr/bin/env python3
"""Profile DLRMv3 inference batches without LoadGen (timing_stats enabled)."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

from configs import get_embedding_table_config, get_hstu_configs
from datasets.dataset import Samples
from generative_recommenders.common import set_dev_mode, set_verbose_level
from inference_modules import set_is_inference
from model_family import HSTUModelFamily
from timing_stats import enabled, format_summary, maybe_report, reset, summarize
from utils import SUPPORTED_DATASETS, get_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("profile_benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile DLRMv3 forward path (no LoadGen).")
    p.add_argument("--dataset", default="sampled-streaming-100b", choices=SUPPORTED_DATASETS)
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset-path-prefix", default="/data/dlrmv3_dataset/")
    p.add_argument("--batchsize", type=int, default=10)
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "8")))
    return p.parse_args()


def main() -> None:
    if not enabled():
        logger.error("Set DLRM_TIMING=1 before running.")
        sys.exit(1)

    args = parse_args()
    os.environ["WORLD_SIZE"] = str(args.world_size)
    set_verbose_level(1)
    set_dev_mode(False)
    set_is_inference(is_inference=True)

    hstu_config = get_hstu_configs(args.dataset)
    hstu_config.max_num_candidates = hstu_config.max_num_candidates_inference
    table_config = get_embedding_table_config(args.dataset)
    is_streaming = "streaming" in args.dataset
    dataset_cls, kwargs = get_dataset(args.dataset, args.dataset_path_prefix)
    ds = dataset_cls(
        hstu_config=hstu_config,
        embedding_config=table_config,
        is_inference=True,
        **kwargs,
    )
    if is_streaming:
        from main import StreamingQuerySampler  # noqa: PLC0415

        ds = StreamingQuerySampler(
            ds=ds,
            dataset_percentage=0.0001,
            input_queries=args.num_batches * args.batchsize,
            compute_eval=False,
            scenario_name="Server",
            offline_target_qps=1000,
            target_duration=600_000,
        )

    model = HSTUModelFamily(
        hstu_config=hstu_config,
        table_config=table_config,
    )
    try:
        logger.info("Loading checkpoint from %s ...", args.model_path)
        model.load(args.model_path)
        logger.info("Model load complete.")

        def _run_batch(batch_ids: list[int]) -> None:
            if is_streaming:
                ds.init_sut()  # pyre-ignore [16]
            result = ds.get_samples(batch_ids)
            if isinstance(result, Samples):
                model.predict(result)
            elif isinstance(result, list):
                for sample, _, _ in result:
                    model.predict(sample)
            else:
                for s in result:
                    model.predict(s)

        autotune_iters = int(os.environ.get("DLRM_AUTOTUNE_ITERS", "2"))
        extra_warmup = int(os.environ.get("DLRM_PROFILE_WARMUP_BATCHES", "0"))
        warmup_ids = list(range(args.batchsize))
        logger.info(
            "Warmup: autotune_iters=%d extra_batches=%d batchsize=%d",
            autotune_iters,
            extra_warmup,
            args.batchsize,
        )
        ds.load_query_samples(warmup_ids)
        for _ in range(autotune_iters):
            _run_batch(warmup_ids)
        for i in range(extra_warmup):
            _run_batch(warmup_ids)
            if (i + 1) % 5 == 0:
                logger.info("Extra warmup batch %d/%d", i + 1, extra_warmup)
        ds.unload_query_samples(None)

        if enabled():
            reset()
            logger.info("Timing stats reset after warmup")

        logger.info(
            "Timed run: %d batches of size %d (DLRM_HSTU_KERNEL=%s)",
            args.num_batches,
            args.batchsize,
            os.environ.get("DLRM_HSTU_KERNEL", "(unset/triton)"),
        )
        ids = list(range(args.batchsize))
        ds.load_query_samples(ids)
        for b in range(args.num_batches):
            _run_batch(ids)
            if (b + 1) % 10 == 0:
                maybe_report(force=True)
        ds.unload_query_samples(None)
        maybe_report(force=True)

        summary = summarize()
        print("\n=== PROFILE SUMMARY ===")
        print(format_summary(summary))
        print("\n=== JSON ===")
        import json

        print(json.dumps(summary, indent=2))
    finally:
        logger.info("Shutting down model (dense workers) ...")
        model.shutdown()


if __name__ == "__main__":
    main()
