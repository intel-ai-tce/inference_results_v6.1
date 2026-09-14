import argparse
import subprocess
import os
import sqlite3
import statistics
from tqdm import tqdm
import csv

def process_log(log_file: str, hipblaslt_bench: str, output: str, db_path: str):
    cmds = []

    # Ensure the output directory exists
    os.makedirs(output, exist_ok=True)

    # Keep track of generated file names to manage unique indexing
    generated_files = {}
    summary_data = []

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query to fetch durations and names
    query = """
    SELECT
        C.string AS Name,
        (A.end - A.start) / 1000.0 AS Duration_us
    FROM (
        SELECT opType_id AS name_id, start, end
        FROM rocpd_op
        WHERE description_id IN (SELECT id FROM rocpd_string WHERE string = '')
        UNION
        SELECT description_id AS name_id, start, end
        FROM rocpd_op
        WHERE description_id NOT IN (SELECT id FROM rocpd_string WHERE string = '')
    ) A
    JOIN rocpd_string C ON C.id = A.name_id
    """

    # Execute the query and process results
    cursor.execute(query)
    rows = cursor.fetchall()

    # Organize durations by kernel name
    data = {}
    for name, duration in rows:
        if name not in data:
            data[name] = []
        data[name].append(duration)

    # Compute statistics for each kernel
    stats = {}
    total_duration_us = sum(sum(durations) for durations in data.values())
    for name, durations in data.items():
        total_calls = len(durations)
        total_duration = sum(durations)
        mean_duration = total_duration / total_calls if total_calls > 0 else 0
        median_duration = statistics.median(durations)
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        percentage = (total_duration / total_duration_us) * 100
        stats[name] = {
            "TotalCalls": total_calls,
            "TotalDuration_us": total_duration,
            "TotalDuration_min": total_duration / 1000000 / 60,
            "Mean_us": mean_duration,
            "Median_us": median_duration,
            "Min_us": min_duration,
            "Max_us": max_duration,
            "Percentage": percentage
        }

    # Read commands from the log file
    with open(log_file, 'r') as f:
        for line in f.readlines():
            num_calls = int(line.split()[0])
            if num_calls > 1000:
                cmd = line.split()[2:]  # Extract the relevant part of the command
                cmd = [hipblaslt_bench] + cmd + ['--print_kernel_info', '-i', '100', '-j', '100', '--flush', '--initialization', 'trig_float']
                cmds.append(cmd)

    # Execute each command and save the output
    for cmd in tqdm(cmds):
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, env={"HIP_VISIBLE_DEVICES": "0", **os.environ})
            output_lines = result.stdout.decode('utf-8').splitlines()

            # Extract the file name from the command output
            kernel_name = None
            ave_us_from_bench = None
            for line in output_lines:
                if '--kernel name:' in line:
                    kernel_name = line.split('--kernel name:')[1].strip()
                if 'T,N' in line:
                    try:
                        ave_us_from_bench = float(line.split(',')[-1])
                    except ValueError:
                        ave_us_from_bench = None

            if not kernel_name:
                print(f"Warning: No kernel name found for command {cmd}. Skipping output file.")
                continue

            # Use the first 100 characters of the kernel_name as the base file name
            sanitized_name = kernel_name[:100]

            # Ensure the file name is unique by using the tracked files
            if sanitized_name not in generated_files:
                generated_files[sanitized_name] = 0
            else:
                generated_files[sanitized_name] += 1

            index = generated_files[sanitized_name]
            if index == 0:
                output_file = os.path.join(output, f"{sanitized_name}.txt")
            else:
                output_file = os.path.join(output, f"{sanitized_name}_{index}.txt")

            # Retrieve statistics from precomputed data
            kernel_stats = stats.get(kernel_name, {})
            ave_us = kernel_stats.get("Mean_us", "")
            median_us = kernel_stats.get("Median_us", "")
            min_us = kernel_stats.get("Min_us", "")
            max_us = kernel_stats.get("Max_us", "")
            total_duration_us = kernel_stats.get("TotalDuration_us", "")
            total_duration_min = kernel_stats.get("TotalDuration_min", "")

            # Determine if it's slower and compute the difference
            slower = median_us > ave_us_from_bench if median_us and ave_us_from_bench else False
            difference = (
                (median_us - ave_us_from_bench) / median_us * total_duration_min
                if slower and median_us else ""
            )

            # Write output to the file
            with open(output_file, 'w') as outfile:
                outfile.write(f"hipblaslt-bench command: {' '.join(cmd)}\n")
                outfile.write(result.stdout.decode('utf-8'))
                outfile.write(f"\nAve_us from rpd db: {ave_us}\n")
                outfile.write(f"Median_us from rpd db: {median_us}\n")
                outfile.write(f"Min_us from rpd db: {min_us}\n")
                outfile.write(f"Max_us from rpd db: {max_us}\n")
                if ave_us_from_bench is not None:
                    outfile.write(f"Ave_us from hipblaslt-bench: {ave_us_from_bench}\n")
                outfile.write(f"TotalDuration_us: {total_duration_us}\n")
                outfile.write(f"TotalDuration_min: {total_duration_min}\n")
                outfile.write(f"Difference: {difference}\n")

            # Collect data for the summary file
            summary_data.append((
                difference, ave_us_from_bench, median_us, ave_us, min_us, max_us,
                total_duration_us, total_duration_min, kernel_name
            ))

        except subprocess.CalledProcessError as e:
            print(f"Error executing command {cmd}: {e}")

    # Close the database connection
    conn.close()

    # Create a summary CSV file for all kernels
    summary_file = os.path.join(output, "summary.csv")
    summary_data.sort(key=lambda x: float(x[7]) if x[7] else 0, reverse=True)  # Sort by TotalDuration_us descending

    # Write summary data to the CSV file
    with open(summary_file, 'w', newline='') as sf:
        csv_writer = csv.writer(sf)
        # Write the header
        csv_writer.writerow([
            "Difference", "Ave_us (hipblaslt-bench)", "Median_us (RPD)",
            "Ave_us (RPD)", "Min_us (RPD)", "Max_us (RPD)",
            "TotalDuration_us", "TotalDuration_min",  "Kernel Name"
        ])
        # Write the data rows
        csv_writer.writerows(summary_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to the hipblaslt output log file", type=str, required=True)
    parser.add_argument("--hipblaslt-bench", help="Path to the hipblaslt_bench binary file", type=str, required=True)
    parser.add_argument("--output", help="Path to the folder to save outputs", type=str, required=True)
    parser.add_argument("--db", help="Path to the rpd database file", type=str, required=True)
    args = parser.parse_args()

    process_log(args.file, args.hipblaslt_bench, args.output, args.db)
