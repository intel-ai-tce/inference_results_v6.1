import argparse
import logging
import optuna
import multiprocessing as mp
import sys
import time
import os
import glob
import psutil
import math
from typing import TypeVar, List
from dataclasses import dataclass
import main as benchmark

T = TypeVar('T')

@dataclass
class ServerResult:
    valid: bool = False
    tps: float = 0.0
    ttft: int = 0
    tpot: int = 0
    ssps: float = 0.0
    csps: float = 0.0


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--storage-name", type=str, default='mlperf-inference')
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--config-name", type=str, default="offline_mi355x")
    parser.add_argument("--backend", type=str, required=True)

    args = parser.parse_args()
    return args


def collect_offline_results_from_log(log_file):
    tokens_per_sec = 0.0
    log.info(f"Collecting result from {log_file=}")
    with open(log_file) as file:
        for line in file.readlines():
            if "tokens per second:" in line.lower():
                tokens_per_sec = float(line.split(":")[-1])
    return tokens_per_sec


def collect_server_results_from_log(log_file):
    result = ServerResult()
    log.info(f"Collecting result from {log_file=}")
    with open(log_file) as file:
        for line in file.readlines():
            if "Completed tokens per second" in line:
                result.tps = float(line.split(":")[-1])
            if "Completed samples per second" in line:
                result.csps = float(line.split(":")[-1])
            if "Scheduled samples per second" in line:
                result.ssps = float(line.split(":")[-1])
            if "Performance constraints satisfied : Yes" in line:
                result.valid = True
            if "99.00 percentile first token latency (ns)" in line:
                result.ttft = int(line.split(":")[-1])
            if "99.00 percentile time to output token (ns)" in line:
                result.tpot = int(line.split(":")[-1])
    return result


def server_error_function(distance):
    # positive: result is in constrain, squared error
    # negative: result is NOT in constrain, cubic error
    power = 2 if distance >= 0.0 else 3
    return math.fabs(distance) ** power


def process_server_results(result):
    ttft_latency = 2_000_000_000
    tpot_latency = 200_000_000
    ns_to_ms = lambda x: x * 1e-6

    ttft_diff = ns_to_ms(ttft_latency - result.ttft)
    tpot_diff = ns_to_ms(tpot_latency - result.tpot)
    sps_diff = result.ssps - result.csps
    se = server_error_function
    return result.tps, se(ttft_diff), se(tpot_diff), se(sps_diff)


def get_mlperf_log_location(log_dir, trial_number):
    mlperf_log_dir = f"{log_dir}/trial_{str(trial_number).zfill(3)}"
    return (f'harness_config.output_log_dir={mlperf_log_dir}', f"{mlperf_log_dir}/mlperf_log_summary.txt")


def get_int_param_override(trial, param: str, min: int, max: int, step: int):
    return f"{param}={trial.suggest_int(param, min, max, step = step)}"


def get_float_param_override(trial, param: str, min: int, max: int, step: int):
    return f"{param}={trial.suggest_float(param, min, max, step = step)}"


def get_boolean_param_override(trial, param: str, force_disabled=False):
    values = ['False'] if force_disabled else ['True', 'False']
    return f"{param}={trial.suggest_categorical(param, values)}"


def get_categorical_param_override(trial, param: str, categories: List[T]):
    return f"{param}={trial.suggest_categorical(param, categories)}"


def cleanup_files(pattern="*core*", folder="."):
    matching_files = glob.glob(f'{folder}/{pattern}')
    if matching_files:
        log.info(f"Matching files: {matching_files}")

        for file in matching_files:
            try:
                os.remove(file)
                log.info(f"Deleted file: {file}")
            except Exception as e:
                log.error(f"Error deleting file: {file}: {e}")
        return True
    return False


def cleanup(target_pid) -> None:
    log.info(f"Started cleanup process {os.getpid()}")
    log.info(f"Target process {target_pid}")

    zombie_counter = 10
    while True:
        time.sleep(20)
        found_files = cleanup_files()
        try:
            parent = psutil.Process(target_pid)
        except psutil.NoSuchProcess:
            break
        children = parent.children(recursive=True)
        found_zombies = bool(any(c.status() == psutil.STATUS_ZOMBIE for c in children))
        if found_zombies:
            # When a process send back the results, it will be in zombie status as well
            # We need to wait a bit to be sure not to kill a valid run
            zombie_counter -= 1
        if found_files or zombie_counter <= 0:
            log.info(f"Something went wrong: {found_files=} {found_zombies=}")
            try:
                for child in children:
                    os.kill(child.pid, 15)

                os.kill(target_pid, 15)
            except Exception:
                logging.exception(f"Error during killing processes")

                processes = []
                for proc in psutil.process_iter(['pid', 'name']):
                    if proc.info['pid'] >= target_pid:
                        processes.append(proc)

                for process in processes:
                    try:
                        os.kill(process, 9)
                    except Exception:
                        logging.exception(f"Error during recovery")

            break


QPS_MAPPING = {
    "llama2-70b": {"min": 70.0, "max": 80.0, "step": 0.1},
}

TUNE_ERROR = -9999

def objective(trial):
    args = get_args()
    overrides = []
    model_name = args.config_path.rstrip('/').split("/")[-1]
    mlperf_log_dir_override, mlperf_summary_file = get_mlperf_log_location(args.log_dir, trial.number)
    overrides.append(mlperf_log_dir_override)
    if 'server' in args.config_name:
        overrides.append(get_float_param_override(trial, 'harness_config.target_qps', **QPS_MAPPING[model_name]))
    overrides.append(get_categorical_param_override(trial, 'llm_config.block_size', [8, 16, 32]))
    overrides.append(get_boolean_param_override(trial, 'llm_config.enable_chunked_prefill', force_disabled=True))
    overrides.append(get_boolean_param_override(trial, 'llm_config.enforce_eager'))
    overrides.append(get_boolean_param_override(trial, 'llm_config.enable_prefix_caching', force_disabled=True))
    overrides.append(get_float_param_override(trial, 'llm_config.gpu_memory_utilization', min = 0.90, max = 0.99, step = 0.01))
    overrides.append(get_int_param_override(trial, 'llm_config.max_num_batched_tokens', min = 16384, max = 65536, step = 2048))
    overrides.append(get_int_param_override(trial, 'llm_config.max_num_seqs', min = 256, max = 4096, step = 256))
    overrides.append(get_int_param_override(trial, 'llm_config.max_seq_len_to_capture', min = 256, max = 4096, step = 256))
    overrides.append(get_int_param_override(trial, 'llm_config.num_scheduler_steps', min = 1, max = 20, step = 1))

    result = TUNE_ERROR
    try:
        cleanup_files()
        time.sleep(10)

        harness_process = mp.Process(target=benchmark.run_from_optuna,
                                      args=(args.config_path, args.config_name, args.backend, overrides))
        harness_process.start()
        log.info(f"Started harness_process with PID: {harness_process.pid}")

        cleanup_process = mp.Process(target=cleanup, args=(harness_process.pid,))
        cleanup_process.start()

        harness_process.join()
        cleanup_process.terminate()

        if args.config_name == 'server':
            result = process_server_results(collect_server_results_from_log(mlperf_summary_file))
            log.info(f"Result: {result}")
        else:
            result = collect_offline_results_from_log(mlperf_summary_file)
            log.info(f"Tokens per sec: {result}")
    except FileNotFoundError as e:
        msg = str(e)
        if "mlperf_log_summary.txt" in msg:
            logging.error(f"Something went wrong during hyperparameter tuning: {msg}")
        else:
            logging.exception(f"Error during optuna tuning")
    except:
        logging.exception(f"Error during optuna tuning")

    if args.config_name == 'server' and result == TUNE_ERROR:
        return result, math.inf, math.inf, math.inf
    return result


def main():
    args = get_args()
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # offline: tokens_per_sec (max)
    # server: tokens_per_sec (max), ttft_distance_error (min), tpot_distance_error (min), sps_distance_error (min)
    directions = ["maximize"] if args.config_name == "offline" else ["maximize", "minimize", "minimize", "minimize"]
    study = optuna.create_study(study_name=args.study_name, storage=f"sqlite:///{args.storage_name}.db", directions=directions, load_if_exists=True)
    study.optimize(objective, n_trials=args.num_trials)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    log.info(f"mp.get_context:{mp.get_context()}")
    main()
