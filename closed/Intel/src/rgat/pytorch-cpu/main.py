import argparse
import time
import ast
import os

import logging

import mlperf_loadgen as lg
from SUT import SUT

from utils import getArgs

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RGAT-Main")


SCENARIO_MAP = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    }


def main():

    args = getArgs()
    print(args)
    settings = lg.TestSettings()
    settings.scenario = SCENARIO_MAP[args.scenario.lower()]
    settings.FromConfig(args.user_conf, "rgat", args.scenario)

    if args.mode.lower()=="accuracy":
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    os.makedirs(args.output_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    sut = SUT(args.checkpoint_path,
                args.dataset_path,
                fan_out=args.fan_out,
                batch_size=args.batch_size,
                num_proc=args.num_proc,
                workers_per_proc=args.workers_per_proc,
                cpus_per_proc=args.cpus_per_proc,
                start_core=args.cores_offset,
                warmup=args.warmup,
                performance_sample_count = settings.performance_sample_count_override,
                total_sample_count=args.total_sample_count,
                use_tpp = args.use_tpp,
                use_bf16 = args.use_bf16,
                use_qint8_gemm = args.use_qint8_gemm,
                use_fused_sampler = args.fused_sampler,
                accuracy = args.mode.lower()=="accuracy"
            )
    sut.start()
    lgSUT = sut.get_sut()
    lgQSL = sut.get_qsl()
    lg.StartTestWithLogSettings(lgSUT, lgQSL, settings, log_settings) #, args.audit_conf)

    # Stop sut after completion
    sut.stop()

    log.info("Run Completed!")

    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    log.info("Destroying QSL...")
    lg.DestroyQSL(lgQSL)


if __name__=="__main__":
    main()
