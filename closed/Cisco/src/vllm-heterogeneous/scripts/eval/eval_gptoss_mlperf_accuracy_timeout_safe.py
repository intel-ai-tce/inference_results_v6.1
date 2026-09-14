#!/usr/bin/env python3
"""Run the MLCommons GPT-OSS scorer with timeout-safe LCB cleanup.

This wrapper imports the official GPT-OSS accuracy scorer and delegates to its
main function. The only behavior changed here is LiveCodeBench future
collection: if the official 1200s batch timeout is reached, unfinished samples
are marked incorrect and the worker pool is force-cleaned instead of hanging
forever in ProcessPoolExecutor.shutdown(wait=True).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from concurrent.futures import TimeoutError, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
def _scorer_path() -> Path:
    override = os.environ.get("MLPERF_GPTOSS_SCORER")
    if override:
        return Path(override)
    inference_dir = os.environ.get("MLPERF_INFERENCE_DIR")
    if not inference_dir:
        raise RuntimeError(
            "Set MLPERF_GPTOSS_SCORER or MLPERF_INFERENCE_DIR before running the GPT-OSS scorer."
        )
    return Path(inference_dir) / "language/gpt-oss-120b/eval_mlperf_accuracy.py"



class TimeoutSafeProcessPoolExecutor(_ProcessPoolExecutor):
    """Process pool that can avoid waiting forever after an LCB batch timeout."""

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        if not getattr(self, "_mlperf_timeout_safe_timed_out", False):
            return super().shutdown(wait=wait, cancel_futures=cancel_futures)

        processes = list((getattr(self, "_processes", None) or {}).values())
        for process in processes:
            if process.is_alive():
                process.terminate()

        result = super().shutdown(wait=False, cancel_futures=True)

        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

        return result


def _load_official_scorer():
    scorer_path = _scorer_path()
    if not scorer_path.exists():
        raise FileNotFoundError(
            f"Official GPT-OSS scorer not found at {scorer_path}. "
            "Set MLPERF_GPTOSS_SCORER to override."
        )

    import sys

    sys.path.insert(0, str(scorer_path.parent))
    spec = importlib.util.spec_from_file_location(
        "mlperf_gptoss_eval_mlperf_accuracy", scorer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import scorer from {scorer_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_lcb_future(module, future, idx: int):
    try:
        question_id, is_correct, detailed_reason = future.result(timeout=80)
        return is_correct, detailed_reason
    except TimeoutError:
        module.logger.warning(
            "Timeout evaluating sample %s: Test execution exceeded 80s timeout",
            idx,
        )
        return False, "Timeout: Test execution exceeded time limit"
    except Exception as exc:  # noqa: BLE001 - scorer records evaluator failures.
        module.logger.error("Error evaluating sample %s: %s", idx, exc)
        return False, f"Error: {exc}"


def _timeout_safe_process_livecodebench_batch(module):
    def process_livecodebench_batch(
        entries: List[Dict[str, Any]],
        reference_df: pd.DataFrame,
        tokenizer,
        evaluator: Dict[str, Any],
        lcb_executor: TimeoutSafeProcessPoolExecutor,
        dataset_name: str,
        args,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        work_items = []
        entry_metadata = []

        module.logger.info("Parsing %s %s entries...", len(entries), dataset_name)
        for entry in tqdm(entries, desc=f"Parsing {dataset_name}", unit="entry"):
            qsl_idx = entry["qsl_idx"]
            ref_row = reference_df.iloc[qsl_idx]
            ground_truth = ref_row.get("ground_truth", None)
            token_ids = module.decode_hex_to_tokens(entry["data"])
            model_output = module.detokenize(token_ids, tokenizer)
            extracted_code = evaluator["parse"](model_output)

            entry_metadata.append(
                {
                    "entry": entry,
                    "qsl_idx": qsl_idx,
                    "ref_row": ref_row,
                    "token_ids": token_ids,
                    "model_output": model_output,
                    "extracted_code": extracted_code,
                    "ground_truth": ground_truth,
                }
            )

            if extracted_code is not None and not pd.isna(ground_truth):
                work_items.append((extracted_code, ground_truth))
            else:
                work_items.append(None)

        active_items = len([work for work in work_items if work is not None])
        module.logger.info(
            "Evaluating %s %s code samples with parallel workers...",
            active_items,
            dataset_name,
        )

        future_to_idx = {}
        for idx, work_item in enumerate(work_items):
            if work_item is not None:
                future = lcb_executor.submit(
                    module.evaluate_livecodebench_worker, work_item
                )
                future_to_idx[future] = idx

        eval_results = [None] * len(work_items)

        try:
            futures = as_completed(future_to_idx.keys(), timeout=1200)
            for future in tqdm(
                futures,
                total=len(future_to_idx),
                desc=f"Evaluating {dataset_name}",
                unit="sample",
            ):
                idx = future_to_idx[future]
                eval_results[idx] = _collect_lcb_future(module, future, idx)
        except TimeoutError as exc:
            pending = [
                future
                for future, idx in future_to_idx.items()
                if eval_results[idx] is None
            ]
            module.logger.warning(
                "%s LiveCodeBench futures did not complete before the 1200s "
                "batch timeout: %s",
                len(pending),
                exc,
            )
            setattr(lcb_executor, "_mlperf_timeout_safe_timed_out", True)

            for future in pending:
                idx = future_to_idx[future]
                if future.done():
                    eval_results[idx] = _collect_lcb_future(module, future, idx)
                else:
                    future.cancel()
                    eval_results[idx] = (
                        False,
                        "Timeout: LiveCodeBench future did not complete before "
                        "1200s batch timeout",
                    )

        results_list = []
        outputs_list = []
        for idx, metadata in enumerate(entry_metadata):
            entry = metadata["entry"]
            qsl_idx = metadata["qsl_idx"]
            token_ids = metadata["token_ids"]
            model_output = metadata["model_output"]
            extracted_code = metadata["extracted_code"]
            ground_truth = metadata["ground_truth"]

            if extracted_code is None or pd.isna(ground_truth):
                is_correct = False
                eval_details = (
                    "No code extracted from model output"
                    if extracted_code is None
                    else "No ground truth available"
                )
            else:
                is_correct, eval_details = eval_results[idx] or (
                    False,
                    "Timeout: LiveCodeBench result missing after timeout-safe collection",
                )

            result = {
                "seq_id": entry["seq_id"],
                "qsl_idx": qsl_idx,
                "dataset": dataset_name,
                "is_correct": is_correct,
                "extracted_answer": (
                    str(extracted_code)[:200] if extracted_code is not None else None
                ),
                "ground_truth": (
                    str(ground_truth) if not pd.isna(ground_truth) else None
                ),
                "evaluation_details": eval_details,
                "token_count": len(token_ids),
                "model_output_preview": model_output[:200]
                if args.verbose
                else None,
            }
            results_list.append(result)

            if args.save_outputs:
                outputs_list.append(
                    {
                        "qsl_idx": qsl_idx,
                        "seq_id": entry["seq_id"],
                        "dataset": dataset_name,
                        "ground_truth": ground_truth,
                        "model_output": model_output,
                        "output_token_ids": token_ids,
                        "extracted_answer": extracted_code,
                        "is_correct": is_correct,
                        "evaluation_details": eval_details,
                    }
                )

        return results_list, outputs_list

    return process_livecodebench_batch


def main():
    module = _load_official_scorer()
    module.ProcessPoolExecutor = TimeoutSafeProcessPoolExecutor
    module.process_livecodebench_batch = _timeout_safe_process_livecodebench_batch(
        module
    )
    module.main()


if __name__ == "__main__":
    main()
