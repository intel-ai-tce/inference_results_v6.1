import argparse
import pandas as pd
import os

from ast import literal_eval
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from re import compile as re_compile
from sys import exit

class RunStatus(Enum):
    VALID = 1
    INVALID = 2
    UNKNOWN = 3

    @classmethod
    def from_string(cls, status_str):
        for status in cls:
            if status.name.lower() == status_str.lower():
                return status
        return RunStatus.UNKNOWN

    def __str__(self):
        return self.name

@dataclass
class InferenceMetrics:
    job_id: str = ""
    timestamp: int = 0
    run_idx: int = 0
    run_type: str = ""
    model: str = ""
    accelerator: str = ""
    system_name: str = ""
    scenario: str = ""
    base_docker_image: str = ""
    docker_image: str = ""
    harness_commit: str = ""
    vllm_commit: str = ""
    hipblaslt_commit: str = ""
    power_settings: bool = False
    perf_validity: RunStatus = RunStatus.UNKNOWN
    perf_tokens_per_sec: float = 0.0
    perf_samples_per_sec: float = 0.0
    perf_target_qps: float = 0.0
    ttft_lat_99pct_ns: float = 0.0
    tpot_lat_99pct_ns: float = 0.0
    accuracy_validity: RunStatus = RunStatus.UNKNOWN
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    gsm8k: float = 0.0
    mbxp: float = 0.0
    tokens_per_sample: float = 0.0
    exact_match: float = 0.0
    artifact_path: str = ""
    notes: str = ""
    sglang_commit: str = ""
    host_processor_model: str = ""

    def to_dict(self):
        return asdict(self)


map_accuracy_lower_limits = {
    "deepseek-r1": {'exact_match': 80.544618, 'tokens_per_sample': 3497.6046600000004},
    "llama2-70b-99": {'rouge1': 44.3867688, 'rouge2': 22.0131648, 'rougeL': 28.5875838, 'tokens_per_sample': 265.005},
    "llama3_1-405b": {'rougeL': 21.449934, 'exact_match': 89.232165, 'tokens_per_sample': 616.212},
    "mixtral-8x7b": {'rouge1': 45.14291, 'rouge2': 23.11907, 'rougeL': 30.15619, 'gsm8k': 72.9234, 'mbxp': 59.5584, 'tokens_per_sample': 130.356},
    "gpt-oss-120b": {},    # [WIP] Update after workload is finalized by MLCommons
}

map_accuracy_upper_limits = {
    "deepseek-r1": {'tokens_per_sample': 4274.8501400000005},
    "llama2-70b-99": {'tokens_per_sample': 323.895},
    "llama3_1-405b": {'tokens_per_sample': 753.148},
    "mixtral-8x7b": {'tokens_per_sample': 160.49},
    "gpt-oss-120b": {},    # [WIP] Update after workload is finalized by MLCommons
}

regex_number = " *([0-9.]+)"
regex_text_upper = " *([A-Z]+)"

map_perf_regexes = {
    "Offline": {
        "perf_samples_per_sec" : re_compile(rf"Samples per second:{regex_number}"),
        "perf_tokens_per_sec" : re_compile(rf"Tokens per second:{regex_number}"),
        "perf_validity" : re_compile(rf"Result is :{regex_text_upper}"),
        "perf_target_qps" : re_compile(rf"target_qps :{regex_number}"),
    },
    "Server": {
        "perf_samples_per_sec" : re_compile(rf"Completed samples per second    :{regex_number}"),
        "perf_tokens_per_sec" : re_compile(rf"Completed tokens per second:{regex_number}"),
        "perf_validity" : re_compile(rf"Result is :{regex_text_upper}"),
        "ttft_lat_99pct_ns" : re_compile(rf"99.00 percentile first token latency \(ns\)   :{regex_number}"),
        "tpot_lat_99pct_ns" : re_compile(rf"99.00 percentile time to output token \(ns\)   :{regex_number}"),
        "perf_target_qps" : re_compile(rf"target_qps :{regex_number}"),
    },
    "Interactive": {
        "perf_samples_per_sec" : re_compile(rf"Completed samples per second    :{regex_number}"),
        "perf_tokens_per_sec" : re_compile(rf"Completed tokens per second:{regex_number}"),
        "perf_validity" : re_compile(rf"Result is :{regex_text_upper}"),
        "ttft_lat_99pct_ns" : re_compile(rf"99.00 percentile first token latency \(ns\)   :{regex_number}"),
        "tpot_lat_99pct_ns" : re_compile(rf"99.00 percentile time to output token \(ns\)   :{regex_number}"),
        "perf_target_qps" : re_compile(rf"target_qps :{regex_number}"),
    },
}


def print_file_content(name, file_desc):
    print(name)
    print(file_desc.read())
    # Reset file pointer, allow further reading
    file_desc.seek(0)


def extract_accuracy_line(filepath, metrics_keys):
    def all_keys_in_line(line):
        for key in metrics_keys:
            if key not in line:
                return False
        return True

    with open(filepath, 'r') as file:
        print_file_content("ACCURACY:", file)
        for _, line in enumerate(file, start=1):
            line = line.strip()
            if line and all_keys_in_line(line):
                return literal_eval(line)
    return None


def process_perf_log(metrics, perf_dir):
    file_summary = Path(os.path.join(perf_dir, "mlperf_log_summary.txt"))
    if not file_summary.is_file():
        print(f"Summary file not found: {file_summary}")
        return

    with open(file_summary, "r") as f:
        print_file_content("PERFORMANCE:", f)
        for line in f.readlines():
            for key, rgx in map_perf_regexes[metrics.scenario].items():
                match_res = rgx.match(line)
                if match_res:
                    if key == "perf_validity":
                        metrics.perf_validity = RunStatus.from_string(match_res[1])
                    else:
                        setattr(metrics, key, match_res[1])


def process_accuracy_log(metrics, acc_dir):
    file_summary = Path(os.path.join(acc_dir, "accuracy.txt"))
    if not file_summary.is_file():
        print(f"Summary file not found: {file_summary}")
        return

    accuracy_lower_limits = map_accuracy_lower_limits[metrics.model]
    accuracy_metrics_keys = list(accuracy_lower_limits.keys())
    accuracy_upper_limits = map_accuracy_upper_limits[metrics.model]

    accuracy_metrics = extract_accuracy_line(file_summary, accuracy_metrics_keys)
    if not accuracy_metrics:
        print(f"Accuracy metrics not found in {file_summary}")
        return

    result_is_valid = RunStatus.VALID
    for metric in accuracy_metrics_keys:
        value = accuracy_metrics.get(metric, "N/A")
        upper_limit = accuracy_upper_limits.get(metric, float('inf'))
        if value != "N/A" and (value <= accuracy_lower_limits[metric] or value > upper_limit):
            result_is_valid = RunStatus.INVALID
        setattr(metrics, metric, value)

    metrics.accuracy_validity = result_is_valid


def process_system_info(metrics, result_dir):
    info_keys = ["timestamp", "accelerator", "system_name", "docker_image", "harness_commit",
        "hipblaslt_commit", "power_settings", "vllm_commit", "sglang_commit", "run_type",
        "notes", "host_processor_model", "artifact_path"]

    for ikey in info_keys:
        fname = result_dir / f"system_info/{ikey}.txt"
        try:
            with open(fname) as file_sysinfo:
                setattr(metrics, ikey, file_sysinfo.readline().strip())
        except FileNotFoundError:
            print(f"System info: {fname} not found")


def save_results(args, metrics_arr):
    data = []
    for metrics in metrics_arr:
        data.append({**metrics.to_dict()})

    df = pd.DataFrame(data)

    # Save the csv file
    csv_file = args.output
    df.to_csv(csv_file, index=False)


def get_subdirs(dir):
    return [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]


# Returns the highest run id in the given folder
# E.g.: Returns 3 for folders: accuracy_1, accuracy_2, accuracy_3, performance_1, performance_2
def get_maxrunidx(scenario_dir):
    scenario_dirs = get_subdirs(scenario_dir)
    run_indexes = []
    for item in scenario_dirs:
        parts = item.split('_')
        if len(parts) > 1 and parts[1].isdigit():
            run_indexes.append(int(parts[1]))
    return max(run_indexes) if run_indexes else 0


def is_there_invalid_result(metrics_arr):
    return_code = 0
    print('\nSUMMARY:')
    for idx, metrics in enumerate(metrics_arr):
        summary_msg = f"#{idx}"
        if metrics.perf_validity != RunStatus.UNKNOWN:
            summary_msg += f" Performance:{metrics.perf_validity}"

        if metrics.accuracy_validity != RunStatus.UNKNOWN:
            summary_msg += f" Accuracy:{metrics.accuracy_validity}"

        print(summary_msg)
        if metrics.perf_validity == RunStatus.INVALID or metrics.accuracy_validity == RunStatus.INVALID:
            return_code = 1

    return return_code


def process_logs(args):
    result_dir = args.result_dir

    model_dirs = get_subdirs(result_dir)
    metrics_arr = []
    available_models = list(map_accuracy_lower_limits.keys())
    for model in model_dirs:
        model_dir = os.path.join(result_dir, model)
        if model not in available_models:
            continue
        scenario_dirs = get_subdirs(model_dir)
        for scenario in scenario_dirs:
            scenario_dir = os.path.join(model_dir, scenario)
            max_run_idx = get_maxrunidx(scenario_dir) + 1
            for num in range(1, max_run_idx):
                metrics = InferenceMetrics(model=model, scenario=scenario, run_idx=num, job_id=args.job_id, base_docker_image=args.base_docker_image)
                perf_dir_name = f"performance_{num}"
                perf_dir = os.path.join(scenario_dir, perf_dir_name)
                if os.path.isdir(perf_dir):
                    process_perf_log(metrics, perf_dir)

                acc_dir_name = f"accuracy_{num}"
                acc_dir = os.path.join(scenario_dir, acc_dir_name)
                if os.path.isdir(acc_dir):
                    process_accuracy_log(metrics, acc_dir)

                if metrics.perf_validity != RunStatus.UNKNOWN or metrics.accuracy_validity != RunStatus.UNKNOWN:
                    process_system_info(metrics, result_dir)
                    metrics_arr.append(metrics)
                else:
                    break

    save_results(args, metrics_arr)
    # The exit code represents the validity of the result
    exit(is_there_invalid_result(metrics_arr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process model results, dump as .csv')
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--output", type=str, default='result.csv')

    parser.add_argument("--job-id", type=str)
    parser.add_argument("--base-docker-image", type=str)
    args = parser.parse_args()
    process_logs(args)
