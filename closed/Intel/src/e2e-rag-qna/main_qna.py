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
MLPerf Loadgen entry point for RAG-QnA workload.
Initializes QSL/SUT, configures loadgen settings, and runs the test.
"""

import os
import re
import sys
import logging
import argparse
import hashlib
import subprocess
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("main_qna")

# Local server calls (health/metrics/models) must bypass any HTTP proxy
_loopback = "0.0.0.0,127.0.0.1,localhost,::1"
for _v in ("no_proxy", "NO_PROXY"):
    _cur = os.environ.get(_v, "")
    os.environ[_v] = f"{_loopback},{_cur}" if _cur else _loopback

import mlperf_loadgen as lg
from sut.SUT_qna import E2ESUT
from common.params import add_all_args
from common.config_display import print_config


def _file_md5(path, _chunk=1 << 20):
    """md5 hex of a file, or '(missing)' if it can't be read."""
    try:
        # usedforsecurity=False: this is a file-identity checksum, not a
        # security digest (silences bandit B324).
        h = hashlib.md5(usedforsecurity=False)
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(_chunk), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return "(missing)"


def _embed_server_db(embed_url):
    try:
        r = requests.get(f"{embed_url.rstrip('/')}/health", timeout=5)
        r.raise_for_status()
        j = r.json()
        return j.get("db", "(unknown)"), j.get("db_md5", "(unknown)")
    except Exception as e:
        return "(unreachable)", f"({type(e).__name__})"


def _prefix_cache_state(base_url, timeout=5):
    """Probe a vLLM server's prefix-cache warmth from /metrics. Clean = 0 prior
    queries. Returns (summary, verdict): "clean" / "warm" / "uncheckable"."""
    base = base_url.rstrip("/")
    for suf in ("/v1/chat/completions", "/chat/completions"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    base = base.rstrip("/")
    try:
        r = requests.get(base + "/metrics", timeout=timeout)
        if r.status_code >= 300:
            return "(no /metrics — cannot check)", "uncheckable"
        m = re.search(r'^vllm:prefix_cache_queries_total\{[^}]*}\s+([0-9.eE+]+)',
                      r.text, re.M)
        if m is None:
            return "(prefix caching off)", "clean"
        q = int(float(m.group(1)))
        if q == 0:
            return "CLEAN", "clean"
        return f"WARM ({q} prior queries)", "warm"
    except Exception as e:
        return f"(unreachable: {type(e).__name__} — cannot check)", "uncheckable"


def _probe(url, timeout=5):
    """GET url; return (ok, detail). ok=False on any non-2xx or connection error."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code < 300:
            return True, f"HTTP {r.status_code}"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, type(e).__name__


def preflight_async(args):
    """Verify every server the async pipeline needs is reachable BEFORE the run.
    Exits non-zero (listing what's down) if any required server is unreachable.
    Servers are always needed — even perf runs resolve model names over HTTP."""
    embed_url = os.environ.get("EMBED_URL", "http://127.0.0.1:8100")
    rerank_url = os.environ.get("RERANK_URL", "http://127.0.0.1:8101")

    def _models(u):  # LLM /v1/chat/completions -> /v1/models
        u = u.rstrip("/")
        for suf in ("/v1/chat/completions", "/chat/completions"):
            if u.endswith(suf):
                u = u[: -len(suf)]
                break
        u = u.rstrip("/")
        if u.endswith("/v1"):
            u = u[: -len("/v1")]
        return u.rstrip("/") + "/v1/models"

    query_url = getattr(args, "query_service_url", None) or args.llm_service_url
    suff_url = getattr(args, "sufficiency_service_url", None) or query_url
    # (label, url, required). The judge runs AFTER the benchmark, so it's not
    # required to start an accuracy run — probe it and warn, but don't block.
    # The vLLM /v1/score reranker (RERANK_API=score) has no /health; probe the
    # OpenAI /v1/models endpoint it does serve.
    rerank_probe = (_models(rerank_url)
                    if os.environ.get("RERANK_API", "").strip().lower() == "score"
                    else f"{rerank_url.rstrip('/')}/health")
    checks = [
        ("embed", f"{embed_url.rstrip('/')}/health", True),
        ("rerank", rerank_probe, True),
        ("grader/LLM", _models(args.llm_service_url), True),
        ("query", _models(query_url), True),
        ("sufficiency", _models(suff_url), True),
    ]
    if args.accuracy:
        checks.append(("judge", _models(args.judge_service_url), False))

    log.info("\n" + "=" * 72)
    log.info("  Preflight: server reachability")
    log.info("=" * 72)
    down = []
    for label, url, required in checks:
        ok, detail = _probe(url)
        tag = "OK  " if ok else ("DOWN" if required else "WARN")
        note = "" if (ok or required) else "  (needed after the run, not to start)"
        log.info(f"  {tag}  {label:<14} {url}  ({detail}){note}")
        if not ok and required:
            down.append(label)
    log.info("=" * 72)
    if down:
        log.info(f"ERROR: required server(s) unreachable: {', '.join(down)}. "
              f"Start them (scripts/servers/launch_servers.sh cpu + the 120B "
              f"on the GPU container) and retry.")
        import sys
        sys.exit(1)
    log.info("All required servers reachable.\n")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MLPerf Loadgen for RAG-QnA Multi-hop RAG Benchmark"
    )

    # Loadgen-specific arguments
    parser.add_argument(
        "--scenario",
        choices=["Offline", "Server"],
        default="Offline",
        help="Loadgen scenario (default: Offline)"
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Enable accuracy pass (vs. performance)"
    )
    parser.add_argument(
        "--mlperf_conf",
        default="mlperf.conf",
        help="MLPerf rules config file"
    )
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="User config for LoadGen settings (e.g., target QPS)"
    )
    parser.add_argument(
        "--audit_conf",
        default="audit.conf",
        help="Audit config for compliance runs"
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Output root. Loadgen logs (mlperf_log_*) are written here directly; "
             "results.json, accuracy_results.json and SUT logs go in <output_dir>/results."
    )

    # Add all standard e2e parameters from params.py first
    # This includes --database, --device, etc.
    add_all_args(parser)

    # E2E workload arguments (non-conflicting with params.py)
    parser.add_argument(
        "--dataset_path",
        default="data/frames_dataset.tsv",
        help="Path to frames_dataset.tsv"
    )
    parser.add_argument(
        "--perf_count",
        type=int,
        default=None,
        help="Number of queries for performance testing (None = all)"
    )

    # Multi-shot specific parameters (these are unique to multi_shot_retrieval.py)
    parser.add_argument(
        '--max-sub-queries',
        type=int,
        default=3,
        help='Maximum number of sub-queries per iteration (default: 3)'
    )
    parser.add_argument(
        '--reasoning',
        type=str,
        default='medium',
        choices=['low', 'medium', 'high'],
        help='LLM reasoning level (default: medium)'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum retrieval iterations (default: 10)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='LLM sampling temperature (default: 1.0)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=5,
        help='Max retries for LLM calls (default: 5)'
    )

    # Judge service configuration for accuracy evaluation
    parser.add_argument(
        '--judge_service_url',
        default='http://127.0.0.1:8125/v1/chat/completions',
        help='Judge LLM service URL for accuracy evaluation (default: local vLLM)'
    )
    parser.add_argument(
        '--judge_model',
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='Judge LLM model name (default: Llama-3.1-8B-Instruct)'
    )

    # Query service configuration (separate from main LLM service)
    parser.add_argument(
        '--query_service_url',
        default=None,
        help='Query generation service URL (if different from main LLM service)'
    )

    # Threading configuration for parallel query processing
    parser.add_argument(
        '--max_workers',
        type=int,
        default=10,
        help='Maximum number of worker threads for parallel query processing (default: 10)'
    )

    # --- SUT selection: async pipelined SUT (aiohttp embed/rerank + concurrent
    # LLM) is the default; --sequential opts out to the sequential SUT.
    parser.add_argument(
        '--sequential',
        dest='async_pipeline',
        action='store_false',
        default=True,
        help='Use the sequential SUT instead of the default async pipeline '
             '(async needs the embed/rerank servers running).'
    )
    parser.add_argument(
        '--server_limits',
        default=None,
        help='Per-server concurrency caps (async SUT), e.g. "LLM-120B=512,LLM-20B=256". '
             'Servers: LLM-120B,LLM-20B,embedder,reranker. Or via SERVER_LIMITS env.'
    )

    args = parser.parse_args()
    return args, parser


# Scenario mapping
scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main():
    """Main entry point."""
    args, parser = get_args()
    if not args.server_limits and os.environ.get("SERVER_LIMITS"):
        args.server_limits = os.environ["SERVER_LIMITS"]
    # Async pipeline reads a few settings from the environment (not argparse);
    # surface them in the config print so runs are self-documenting.
    if args.async_pipeline:
        embed_url = os.environ.get("EMBED_URL", "http://127.0.0.1:8100")
        db_path, db_md5 = _embed_server_db(embed_url)
        query_url = getattr(args, "query_service_url", None) or args.llm_service_url
        q_summary, q_verdict = _prefix_cache_state(query_url)
        g_summary, g_verdict = _prefix_cache_state(args.llm_service_url)
        extra = [
            ("embed db", db_path),
            ("embed db md5", db_md5),
            ("embed_url (env)", os.environ.get("EMBED_URL")),
            ("rerank_url (env)", os.environ.get("RERANK_URL")),
            ("120B prefix cache", q_summary),
            ("20B prefix cache", g_summary),
            ("trace_views (env)", os.environ.get("TRACE")),
            ("trace_dir (env)", os.environ.get("TRACE_DIR")),
        ]
    else:
        extra = [("database md5", _file_md5(args.database))]
    print_config(
        args, parser, title="RAG-QnA run configuration",
        endpoints=[
            ("grader/LLM", "llm_service_url", "llm_model"),
            ("query", "query_service_url", "query_model"),
            ("sufficiency", "sufficiency_service_url", "sufficiency_model"),
            ("judge", "judge_service_url", "judge_model"),
        ],
        extra=extra,
    )

    # Refuse a warm prefix cache 
    if args.async_pipeline:
        warm = [lbl for lbl, v in (("120B", q_verdict), ("20B", g_verdict)) if v == "warm"]
        unchk = [lbl for lbl, v in (("120B", q_verdict), ("20B", g_verdict)) if v == "uncheckable"]
        if unchk:
            log.info(f"WARNING: prefix-cache state UNCHECKABLE for {', '.join(unchk)} "
                  f"(no /metrics — e.g. OpenRouter); cannot verify a clean cache.")
        if warm:
            log.info(f"ERROR: {', '.join(warm)} prefix cache is WARM (carries prior-run KV). "
                  f"Restart the server for a clean run.")
            import sys
            sys.exit(1)

    # Fail fast if the async pipeline's servers aren't up (the SUT would
    # otherwise start and die deep inside loadgen with per-query errors).
    if args.async_pipeline:
        preflight_async(args)

    # Output layout: loadgen logs go in output_dir directly; results
    # (results.json, accuracy_results.json, SUT logs) go in output_dir/results.
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize SUT
    log.info("\n" + "="*80)
    log.info("Initializing RAG-QnA SUT...")
    log.info("="*80)

    if args.async_pipeline:
        log.info(f"Using ASYNC SUT (server_limits={args.server_limits or os.environ.get('SERVER_LIMITS') or 'unlimited'})")
        from sut.SUT_async import E2ESUTAsync
        sut = E2ESUTAsync(
            dataset_path=args.dataset_path,
            db_path=args.database,
            args=args,
        )
    else:
        sut = E2ESUT(
            dataset_path=args.dataset_path,
            db_path=args.database,
            max_sub_queries=args.max_sub_queries,
            top_k_retriever=args.top_k_retriever,
            top_k_reranking=args.top_k_reranking,
            max_iterations=args.max_iterations,
            no_rerank=args.no_rerank,
            retrieval_strategy=args.retrieval_strategy,
            reasoning_effort=args.reasoning,
            perf_count=args.perf_count,
            device=args.device,
            temperature=args.temperature,
            max_retries=args.max_retries,
            output_dir=results_dir,
            max_workers=args.max_workers,
            args=args,  # Pass full args for additional params
        )

    log.info("\n" + "="*80)
    log.info("SUT initialization complete")
    log.info("="*80 + "\n")

    # Configure loadgen settings
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]

    # Load config files
    if os.path.exists(args.user_conf):
        # Section name must match user.conf, which uses the submission checker's
        # workload name (e2e-rag-qna) rather than the older rag-qna.
        settings.FromConfig(args.user_conf, "e2e-rag-qna", args.scenario)
        log.info(f"Loaded user config from {args.user_conf}")
    else:
        log.info(f"Warning: User config not found: {args.user_conf}")
        log.info("Using default loadgen settings")

    # Set test mode
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
        log.info("Running in ACCURACY mode")
    else:
        settings.mode = lg.TestMode.PerformanceOnly
        log.info("Running in PERFORMANCE mode")

    # Configure log output
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_dir
    log_output_settings.copy_summary_to_stdout = True

    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    # Run loadgen test
    log.info("\n" + "="*80)
    log.info("Running MLPerf Loadgen test...")
    log.info("="*80 + "\n")

    lg.StartTestWithLogSettings(
        sut.sut,
        sut.qsl.qsl,
        settings,
        log_settings,
        args.audit_conf
    )

    log.info("\n" + "="*80)
    log.info("Loadgen test complete")
    log.info("="*80 + "\n")

    # Finalize SUT (save logs, cleanup)
    sut.finalize()

    # Save results
    results_path = os.path.join(results_dir, "results.json")
    sut.save_results(results_path)
    log.info(f"Results saved to {results_path}")

    # Run accuracy evaluation if in accuracy mode
    if args.accuracy:
        accuracy_output = os.path.join(results_dir, "accuracy_results.json")
        cmd = [
            "python3", "-u", "-m",
            "evaluation.accuracy_eval_qna",
            "--log_dir", args.output_dir,
            "--results_file", results_path,
            "--dataset_path", args.dataset_path,
            "--judge_service_url", args.judge_service_url,
            "--judge_model", args.judge_model,
            "--output", accuracy_output,
        ]

        # Re-probe: the judge may have been started during the (long) run.
        judge_models = args.judge_service_url.rstrip("/").rstrip("/v1/chat/completions").rstrip("/v1") + "/v1/models"
        judge_ok, _ = _probe(judge_models)
        if judge_ok:
            log.info("\n" + "="*80)
            log.info("Running accuracy evaluation...")
            log.info("="*80 + "\n")
            log.info(f"Command: {' '.join(cmd)}")
            subprocess.check_call(cmd)
        else:
            log.info("\n" + "="*80)
            log.info(f"Judge unreachable at {args.judge_service_url} — SKIPPING evaluation.")
            log.info("Results are saved. Start the judge, then run:")
            log.info("="*80)
            log.info("\n  " + " ".join(cmd) + "\n")

    log.info("\n" + "="*80)
    log.info("Done!")
    log.info("="*80)


if __name__ == "__main__":
    main()
