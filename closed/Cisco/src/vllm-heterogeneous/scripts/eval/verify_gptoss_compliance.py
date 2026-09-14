#!/usr/bin/env python3
"""Run MLPerf GPT-OSS compliance verification with stable argv handling."""

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def audit_config(compliance_dir: Path, test: str, canonical_model: str) -> Path:
    model_cfg = compliance_dir / test / canonical_model / "audit.config"
    if model_cfg.exists():
        return model_cfg
    generic_cfg = compliance_dir / test / "audit.config"
    if generic_cfg.exists():
        return generic_cfg
    raise FileNotFoundError(f"No audit.config for {test}/{canonical_model}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, choices=["TEST07", "TEST09"])
    parser.add_argument("--logs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gptoss_120b")
    parser.add_argument("--canonical-model", default="gpt-oss-120b")
    parser.add_argument("--compliance-dir")
    args = parser.parse_args()

    inference_dir = os.environ.get("MLPERF_INFERENCE_DIR")
    if args.compliance_dir:
        compliance_dir = Path(args.compliance_dir)
    elif inference_dir:
        compliance_dir = Path(inference_dir) / "compliance"
    else:
        parser.error("set --compliance-dir or MLPERF_INFERENCE_DIR")

    test_script = compliance_dir / args.test / "run_verification.py"
    if not test_script.exists():
        raise FileNotFoundError(test_script)

    audit = audit_config(compliance_dir, args.test, args.canonical_model)
    cmd = [
        "python3",
        str(test_script),
        "-c",
        args.logs,
        "-o",
        args.out,
        "--audit-config",
        str(audit),
    ]
    if args.test == "TEST07":
        try:
            source_root = Path(inference_dir) if inference_dir else None
            data_root = Path(os.environ["DATA_ROOT"] + "/gpt-oss-120b")
            model_root = Path(os.environ["MODEL_ROOT"] + "/gpt-oss-120b")
        except KeyError as exc:
            parser.error(f"missing deployment setting: {exc.args[0]}")
        if source_root is None:
            parser.error("set MLPERF_INFERENCE_DIR for TEST07")
        submission_root = Path(__file__).resolve().parents[2]
        scorer = submission_root / "scripts/eval/eval_gptoss_mlperf_accuracy_timeout_safe.py"
        accuracy_script = (
            f"PYTHONPATH={shlex.quote(str(source_root / 'language/gpt-oss-120b'))} "
            f"python3 {shlex.quote(str(scorer))} "
            "--mlperf-log {accuracy_log} "
            f"--reference-data {shlex.quote(str(data_root / 'acc/acc_eval_compliance_gpqa.parquet'))} "
            f"--tokenizer {shlex.quote(str(model_root / 'model'))}"
        )
        cmd.extend(["--accuracy-script", accuracy_script])

    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
