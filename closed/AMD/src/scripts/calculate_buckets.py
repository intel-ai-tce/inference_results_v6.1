import pandas as pd
import os
import argparse
import re
import glob
import numpy as np

VERBOSE = False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect TPS results from run directories')
    parser.add_argument('directory', help='Root directory to search for run directories')
    parser.add_argument('output', default='results.csv', help='Output CSV file to save results')
    parser.add_argument('buckets', default='buckets.csv', help='Output CSV file to save bucket results')
    parser.add_argument('--plot', action='store_true', help='Generate a plot of the results')
    parser.add_argument('--verbose', action='store_true', help='Turn on verbose mode')
    return parser.parse_args()

def log(*args):
    if VERBOSE:
        print(*args)

def extract_tps_from_log(log_path):
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(r'Tokens per second: (\d+\.?\d*)', line)
                if match:
                    return float(match.group(1))
    except FileNotFoundError:
        return None
    return None

def collect_results(root_dir):
    results = []
    pattern = os.path.join(root_dir, '*run_*_gpu_*')
    log(f"Searching for directories matching: {pattern}")

    for dir_path in glob.glob(pattern):
        log(f"Processing directory: {dir_path}")
        if os.path.isdir(dir_path):
            dir_name = os.path.basename(dir_path)
            match = re.match(r'.*run_(\d+)_gpu_(\d+)', dir_name)
            if match:
                run_num = int(match.group(1))
                gpu_id = int(match.group(2))

                log_path = os.path.join(dir_path, 'output.log')
                tps = extract_tps_from_log(log_path)

                if tps is not None:
                    results.append({'run_number': run_num, 'gpu_id': gpu_id, 'tps': tps})

    return pd.DataFrame(results)

def plot_results(df):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        print(f"Error: missing matplotlib package. Details: {e}")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique run numbers and gpu_ids for grouping
    unique_runs = sorted(df['run_number'].unique())
    unique_gpus = sorted(df['gpu_id'].unique())

    # Set up colors for each run
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_runs)))

    # Set bar width and positions
    bar_width = 0.8 / len(unique_runs)
    x_pos = np.arange(len(unique_gpus))

    # Plot bars for each run
    for i, run_num in enumerate(unique_runs):
        run_data = df[df['run_number'] == run_num]
        tps_values = []

        for gpu_id in unique_gpus:
            gpu_data = run_data[run_data['gpu_id'] == gpu_id]
            if not gpu_data.empty:
                tps_values.append(gpu_data['tps'].iloc[0])
            else:
                tps_values.append(0)

        ax.bar(x_pos + i * bar_width, tps_values, bar_width,
               label=f'Run {run_num}', color=colors[i])

    # Customize the plot
    ax.set_xlabel('GPU ID')
    ax.set_ylabel('Tokens per Second (TPS)')
    ax.set_title('TPS Results by GPU ID and Run Number')
    ax.set_xticks(x_pos + bar_width * (len(unique_runs) - 1) / 2)
    ax.set_xticklabels(unique_gpus)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tps_results.png')
    print("Plot saved as tps_results.png")

def calc_buckets(df):
    # Calculate percentage buckets for each GPU based on average TPS
    avg_tps = df.groupby('gpu_id')['tps'].mean().reset_index()
    log("Average TPS per GPU:")
    log(avg_tps)
    avg_tps['percentage'] = (avg_tps['tps'] / avg_tps['tps'].sum()) * 100
    return avg_tps['percentage']

def print_bucket_config(items):
    print(f"{' '*4}buckets: [")
    for item in items:
        print(f"{' '*6}{item},")
    print(f"{' '*4}]")

def main():
    args = parse_arguments()
    global VERBOSE
    VERBOSE = args.verbose
    df = collect_results(args.directory)

    if not df.empty:
        df = df.sort_values(['run_number', 'gpu_id'])
        if args.plot:
            plot_results(df)
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
        buckets = calc_buckets(df)
        print("Buckets calculated:")
        print_bucket_config(buckets.tolist())
        buckets.to_csv(args.buckets, index=False)
        print(f"Buckets saved to {args.buckets}")
    else:
        print("No results found.")

if __name__ == '__main__':
    main()
