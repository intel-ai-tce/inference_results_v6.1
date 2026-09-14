import subprocess
import mlperf_loadgen as lg
import argparse
import os
import logging
import sys
from client import SUT
from util import RunnerArgs

sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Main")

def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    }

mode_map = {
    "accuracy": lg.TestMode.AccuracyOnly,
    "performance": lg.TestMode.PerformanceOnly,
}

def main():

    print("Starting SUT for Llama-3.1-8B")
    parser = argparse.ArgumentParser()
    RunnerArgs.add_cli_args(parser)
    args = parser.parse_args()
    runner_args = RunnerArgs.from_cli_args(args)

    print(f"Setting loadgen params")
    settings = lg.TestSettings()
    settings.scenario = scenario_map[runner_args.scenario.lower()]
    # Need to update the conf
    settings.FromConfig(runner_args.user_conf, runner_args.workload_name, runner_args.scenario)

    settings.mode = mode_map[runner_args.mode.lower()]
    
    os.makedirs(runner_args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = runner_args.enable_log_trace

    print(f"Creating SUT instance for {runner_args.scenario} scenario")

    sut = SUT(runner_args)

    # Start sut before loadgen starts
    print("Starting SUT...")
    sut.start()
    print("Waiting for SUT to be ready...")
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    print("Starting Benchmark run")
    lg.StartTestWithLogSettings(lgSUT, sut.qsl, settings, log_settings, args.audit_conf)

    # Stop sut after completion
    sut.stop()

    print("Run Completed!")

    print("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()