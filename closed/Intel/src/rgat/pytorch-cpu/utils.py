from argparse import ArgumentParser
import ast


def getArgs():

    parser = ArgumentParser("Parses global and workload-specific arguments")
    parser.add_argument("--workload-name", type=str, default='rgat', help="Name of workload",)
    parser.add_argument("--scenario", choices=["Offline", "Server"], help="MLPerf scenario to run", default="Offline")
    parser.add_argument("--user-conf", help="Path to user.conf file containing overridden workload params")
    parser.add_argument("--mode", choices=["Accuracy", "Performance"], help="MLPerf mode to run", default="Performance")
    parser.add_argument("--num-proc", type=int, help="Number of instances/consumers", default=2)
    parser.add_argument("--cpus-per-proc", type=int, help="Number of cores per instance", default=8)
    parser.add_argument("--warmup", action="store_true", help="Whether to do warmup")
    parser.add_argument("--workers-per-proc", type=int, help="Number of workers per each proc/instance", default = 1)
    parser.add_argument("--cores-offset", type=str, help="Cpus to offset on all NUMA", default='1,43,86,128,171,214')
    parser.add_argument("--dataset-path", type=str, help="Path to dataset", required=True)
    parser.add_argument("--checkpoint-path", type=str, help="Path to fp32 checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=1)
    parser.add_argument("--total-sample-count", type=int, help="Total number of samples to sample from", default=1000)
    parser.add_argument("--output-dir", type=str, help="Output directory for mlperf logs", default="output_logs")
    parser.add_argument("--enable-log-trace", action="store_true", help="Enable log tracing. This file can become quite large")

    parser.add_argument("--use-tpp", action='store_true', help="Whether to use tpp implementation")
    parser.add_argument("--use-bf16", action="store_true", help="Use bf16")
    parser.add_argument("--use-qint8-gemm", action="store_true", help="Use int8")
    parser.add_argument('--fan-out', type=str, default='-1,-1,-1')
    parser.add_argument("--fused-sampler", action='store_true', help="Whether to use fused sampling")


    args = parser.parse_args()
    args.cores_offset = ast.literal_eval(args.cores_offset)
    args.fan_out = ast.literal_eval(args.fan_out)

    return args
