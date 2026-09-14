#!/usr/bin/env python3
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Accuracy evaluation script for RAG-QnA loadgen results.
Evaluates both retrieval accuracy and answer quality using LLM judge.
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import aiohttp
import pandas as pd
import requests


# OpenRouter configuration
DEFAULT_JUDGE_URL = "http://127.0.0.1:8125/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# Masked API key (set OPENROUTER_API_KEY environment variable to use OpenRouter)
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY',
    'sk-or-v1-****')


JUDGE_PROMPT = """You are grading whether an LLM answer is correct against a ground truth answer.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

LLM ANSWER: {llm_answer}

Grade in two steps.

STEP 1 - If the LLM answer is empty, "Unknown", "I don't know", "cannot be determined", or otherwise does not commit to an answer, then it is WRONG: output correct=false immediately and do not go to step 2.

STEP 2 - Otherwise compare it to the ground truth by meaning, not wording. correct=true only if it supplies every fact the ground truth requires and each clearly matches; if you are unsure or the match is only partial, output correct=false. Rules:
- If the ground truth is a list or has multiple parts, an answer missing any of them is correct=false.
- Every number, date, and name must match the ground truth; a different or differently-rounded value is correct=false, a different name is correct=false.
- Do NOT penalize harmless extras or omissions when the required facts match: a missing suffix like "Inc.", an added state/country, a full middle name, missing units when the number is right, or a briefer/longer phrasing.

Return your evaluation in JSON format:
{{
    "correct": true/false,
    "reasoning": "brief explanation"
}}
"""


def _judge_params(model_name):
    """Per-model token budget + timeout (Llama-8B is non-thinking; else thinking)."""
    is_llama_8b = "llama-3.1-8b" in model_name.lower()
    return (1024, 1200) if is_llama_8b else (4096, 1200)


def _build_payload(question, ground_truth, llm_answer, model_name):
    max_tokens, _ = _judge_params(model_name)
    prompt = JUDGE_PROMPT.format(question=question, ground_truth=ground_truth,
                                 llm_answer=llm_answer)
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }


def _parse_verdict(result):
    """Extract {correct, reasoning} from an OpenAI-style judge response."""
    message = result['choices'][0]['message']
    content = (message.get('content') or message.get('reasoning_content') or "").strip()
    if "```" in content:
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if m:
            content = m.group(1).strip()
    m = re.search(r'\{.*\}', content, re.DOTALL)
    if not m:
        return {"correct": False, "reasoning": "No JSON found in judge response"}
    return json.loads(m.group(0))


def call_judge(question: str, ground_truth: str, llm_answer: str,
               service_url: str = DEFAULT_JUDGE_URL,
               model_name: str = DEFAULT_JUDGE_MODEL,
               api_key: str = OPENROUTER_API_KEY) -> Dict:
    """Synchronous single judge call (kept for external callers / tests)."""
    _, timeout = _judge_params(model_name)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = _build_payload(question, ground_truth, llm_answer, model_name)
    try:
        response = requests.post(service_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return _parse_verdict(response.json())
    except Exception as e:
        print(f"Error calling judge: {e}")
        return {"correct": False, "reasoning": f"Judge error: {e}"}


async def _judge_async(session, service_url, model_name, api_key,
                       question, ground_truth, llm_answer):
    """Async judge call (aiohttp). Fan-out driven; no per-request thread/conn cap."""
    _, timeout = _judge_params(model_name)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = _build_payload(question, ground_truth, llm_answer, model_name)
    try:
        async with session.post(service_url, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            resp.raise_for_status()
            return _parse_verdict(await resp.json())
    except Exception as e:
        return {"correct": False, "reasoning": f"Judge error: {type(e).__name__}: {e}"}


def calculate_retrieval_metrics(retrieved_urls: List[str], expected_urls: List[str]) -> Dict:
    """Calculate precision, recall, F1 for retrieval."""

    retrieved_set = set(retrieved_urls)
    expected_set = set(expected_urls)

    if not expected_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    correct = retrieved_set & expected_set

    precision = len(correct) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(correct) / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_results(results: Dict, dataset_path: str, num_workers: int = 4,
                    judge_service_url: str = DEFAULT_JUDGE_URL,
                    judge_model: str = DEFAULT_JUDGE_MODEL) -> Dict:
    """
    Evaluate loadgen results.

    Args:
        results: Dict mapping query_id -> result_dict
        dataset_path: Path to frames_dataset.tsv
        num_workers: Number of parallel judge workers
        judge_service_url: Judge LLM service URL
        judge_model: Judge LLM model name

    Returns:
        Dict with aggregate metrics
    """

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, sep='\t')

    # Build query -> ground truth mapping
    query_to_gt = {}
    for _, row in df.iterrows():
        query = row['Prompt']
        query_to_gt[query] = {
            'answer': row['Answer'],
            'expected_urls': []
        }
        # Extract expected URLs
        for col in df.columns:
            if col.startswith('wikipedia_link_'):
                url = row[col]
                if pd.notna(url) and url != '':
                    query_to_gt[query]['expected_urls'].append(url)

    print(f"Evaluating {len(results)} queries...")
    print(f"Using judge: {judge_model} at {judge_service_url}  (async, max_concurrent={num_workers})")

    detailed_results = []

    async def _run_all():
        sem = asyncio.Semaphore(num_workers)
        done = 0
        connector = aiohttp.TCPConnector(limit=0)  # no client-side conn cap
        async with aiohttp.ClientSession(connector=connector) as session:
            async def one(query_id, result):
                nonlocal done
                query = result.get('query', '')
                llm_answer = result.get('answer', '')
                gt_data = query_to_gt.get(query)
                if not gt_data:
                    return None
                rm = calculate_retrieval_metrics(result.get('retrieved_urls', []),
                                                 gt_data['expected_urls'])
                async with sem:
                    verdict = await _judge_async(session, judge_service_url, judge_model,
                                                 OPENROUTER_API_KEY, query,
                                                 gt_data['answer'], llm_answer)
                done += 1
                if done % 50 == 0:
                    print(f"  Evaluated {done}/{len(results)} queries...", flush=True)
                return {
                    'query_id': query_id,
                    'query': query,
                    'retrieval_precision': rm['precision'],
                    'retrieval_recall': rm['recall'],
                    'retrieval_f1': rm['f1'],
                    'answer_correct': 1 if verdict.get('correct', False) else 0,
                    'judge_reasoning': verdict.get('reasoning', ''),
                    'llm_answer': llm_answer,
                    'ground_truth': gt_data['answer'],
                }
            tasks = [one(qid, r) for qid, r in results.items()]
            return await asyncio.gather(*tasks)

    for eval_result in asyncio.run(_run_all()):
        if eval_result:
            detailed_results.append(eval_result)

    total_queries = len(detailed_results)
    total_retrieval_precision = sum(r['retrieval_precision'] for r in detailed_results)
    total_retrieval_recall = sum(r['retrieval_recall'] for r in detailed_results)
    total_retrieval_f1 = sum(r['retrieval_f1'] for r in detailed_results)
    total_answer_correct = sum(r['answer_correct'] for r in detailed_results)

    # Calculate averages
    if total_queries > 0:
        avg_metrics = {
            'total_queries': total_queries,
            'retrieval_precision': total_retrieval_precision / total_queries,
            'retrieval_recall': total_retrieval_recall / total_queries,
            'retrieval_f1': total_retrieval_f1 / total_queries,
            'answer_accuracy': total_answer_correct / total_queries,
            'detailed_results': detailed_results
        }
    else:
        avg_metrics = {
            'total_queries': 0,
            'retrieval_precision': 0.0,
            'retrieval_recall': 0.0,
            'retrieval_f1': 0.0,
            'answer_accuracy': 0.0,
            'detailed_results': []
        }

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG-QnA loadgen accuracy")
    parser.add_argument('--log_dir', required=True, help='Loadgen log directory')
    parser.add_argument('--results_file', required=True, help='SUT results JSON file')
    parser.add_argument('--dataset_path', required=True, help='Path to frames_dataset.tsv')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of parallel judge workers')
    parser.add_argument('--output', default='accuracy_results.json', help='Output file for detailed results')
    parser.add_argument('--judge_service_url', default=DEFAULT_JUDGE_URL, help='Judge LLM service URL')
    parser.add_argument('--judge_model', default=DEFAULT_JUDGE_MODEL, help='Judge LLM model name')
    args = parser.parse_args()

    try:
        base = args.judge_service_url.rsplit('/v1/', 1)[0] + '/v1/models'
        served = requests.get(base, timeout=10).json().get('data', [])
        served_ids = [m['id'] for m in served]
        if served_ids and args.judge_model not in served_ids:
            print(f"Judge model '{args.judge_model}' not served; using '{served_ids[0]}'")
            args.judge_model = served_ids[0]
    except Exception as e:
        print(f"Warning: could not verify judge model against server: {e}")

    if "llama-3.1-8b" not in args.judge_model.lower():
        print("=" * 80)
        print(f"WARNING: judging with '{args.judge_model}', but the reference judge "
              "is Llama-3.1-8B.")
        print("         Scores are NOT directly comparable to the reference.")
        print("=" * 80)

    # Load results
    print(f"Loading results from {args.results_file}...")
    with open(args.results_file, 'r') as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results")

    # Evaluate
    metrics = evaluate_results(results, args.dataset_path, args.num_workers,
                               judge_service_url=args.judge_service_url,
                               judge_model=args.judge_model)

    # Print summary
    print("\n" + "="*80)
    print("ACCURACY EVALUATION RESULTS")
    print("="*80)
    print(f"Total Queries:        {metrics['total_queries']}")
    print(f"\nRetrieval Metrics:")
    print(f"  Precision@N:        {metrics['retrieval_precision']:.3f}")
    print(f"  Recall@N:           {metrics['retrieval_recall']:.3f}")
    print(f"  F1@N:               {metrics['retrieval_f1']:.3f}")
    print(f"\nAnswer Quality:")
    print(f"  LLM Judge Accuracy: {metrics['answer_accuracy']:.3f}")
    print("="*80 + "\n")

    # Save detailed results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Detailed results saved to {args.output}")

    # Write accuracy.txt into the loadgen log dir in MLPerf format. The
    # submission checker parses the LLM judge answer accuracy (as a percentage)
    # from the "Accuracy:" line. The hash= line and log truncation are added
    # later by tools/submission/truncate_accuracy_log.py during submission prep.
    accuracy_txt_path = os.path.join(args.log_dir, "accuracy.txt")
    with open(accuracy_txt_path, 'w') as f:
        f.write(f"Accuracy: {metrics['answer_accuracy'] * 100:.4f}\n")
    print(f"Accuracy report saved to {accuracy_txt_path}")


if __name__ == "__main__":
    main()
