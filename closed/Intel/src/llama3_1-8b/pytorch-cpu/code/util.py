import subprocess
import argparse
import dataclasses
from typing import Tuple, List
import torch

@dataclasses.dataclass
class RunnerArgs:
    """Arguments for the SGLang runner."""
    batch_size: int = 1
    dataset_path: str = ""
    model_path: str = ""
    total_sample_count: int = 1000
    scenario: str = "Offline"
    workload_name: str = "llama3_1-8b"
    device: str = "cpu"
    run_name: str = "llama3_1-8b-run"
    accuracy: bool = False
    audit_conf: str = "audit.conf"
    user_conf: str = "user.conf"
    output_log_dir: str = "output-logs"
    enable_log_trace: bool = False
    mode: str = "performance"  # 'performance' or 'accuracy'

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=RunnerArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, default=RunnerArgs.batch_size
        )
        parser.add_argument(
            "--dataset-path", type=str, default=RunnerArgs.dataset_path, help="Path to the dataset file."
        )
        parser.add_argument(
            "--model-path", type=str, default=RunnerArgs.model_path, help="Path to the model."
        )
        parser.add_argument(
            "--total-sample-count", type=int, default=RunnerArgs.total_sample_count, help="Total number of samples to load."
        )
        parser.add_argument("--scenario", type=str, choices=["Offline", "Server", "offline", "server"], default="Offline", help="Scenario")
        parser.add_argument("--workload-name", type=str, default="llama3_1-8b")
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--accuracy", action="store_true", help="Run accuracy mode")
        parser.add_argument("--audit-conf", type=str, default="audit.conf", help="audit config for LoadGen settings during compliance runs")
        parser.add_argument("--user-conf", type=str, default="user.conf", help="user config for user LoadGen settings such as target QPS")
        parser.add_argument("--output-log-dir", type=str, default="output-logs", help="Where logs are saved")
        parser.add_argument("--enable-log-trace", action="store_true", help="Enable log tracing. This file can become quite large")
        parser.add_argument("--mode", type=str, choices=["performance", "accuracy", "Performance", "Accuracy"], default="performance", help="Mode of the test: performance or accuracy")
        
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )
