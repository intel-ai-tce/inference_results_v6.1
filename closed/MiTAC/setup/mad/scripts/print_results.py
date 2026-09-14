from re import compile
from pathlib import Path
import os, argparse

regex_number = " *([0-9.]+)"
regex_text_upper = " *([A-Z]+)"

map_perf_regexes = {
    "offline": {
        "perf_samples" : (compile(rf"Samples per second:{regex_number}"), "samples/second"),
        "perf_tokens" : (compile(rf"Tokens per second:{regex_number}"), "tokens/second"),
        "perf_validity" : (compile(rf"Result is :{regex_text_upper}"), "boolean"),
        "perf_target" : (compile(rf"target_qps :{regex_number}"), "query/second"),
    },
    "server": {
        "perf_samples" : (compile(rf"Completed samples per second    :{regex_number}"), "samples/second"),
        "perf_tokens" : (compile(rf"Completed tokens per second:{regex_number}"), "tokens/second"),
        "perf_validity" : (compile(rf"Result is :{regex_text_upper}"), "boolean"),
        "ttft_lat_99pct" : (compile(rf"99.00 percentile first token latency \(ns\)   :{regex_number}"), "nanosecond"),
        "tpot_lat_99pct" : (compile(rf"99.00 percentile time to output token \(ns\)   :{regex_number}"), "nanosecond"),
        "perf_target" : (compile(rf"target_qps :{regex_number}"), "query/second"),
    },
}

def process_perf_log(scenario, perf_dir, result_file):
    file_summary = Path(os.path.join(perf_dir, "mlperf_log_summary.txt"))
    if not file_summary.is_file():
        print(f"Summary file not found: {file_summary}")
        return

    with open(file_summary, "r") as f:
        for line in f.readlines():
            for key, item in map_perf_regexes[scenario].items():
                match_res = item[0].match(line)
                if match_res:
                    value = match_res[1]
                    if key == "perf_validity":
                        value = 1 if "VALID" == value else 0
                    result_file.write(f"{scenario}_{key},{value},{item[1]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input parameters.")
    parser.add_argument('--output-file-name', type=str, required=True, help='Name of the output file')
    parser.add_argument('--server-results', type=str, default='', help='Server results (optional)')
    parser.add_argument('--offline-results', type=str, default='', help='Offline results (optional)')
    args = parser.parse_args()

    print(f"Output File Name: {args.output_file_name}")
    print(f"Server Results: {args.server_results}")
    print(f"Offline Results: {args.offline_results}")

    with open(args.output_file_name, "w") as f:
        f.write("model,performance,metric\n")
        if args.offline_results:
            process_perf_log("offline", args.offline_results, f)
        if args.server_results:
            process_perf_log("server", args.server_results, f)
        