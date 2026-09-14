#!/bin/bash

find $1 -type f -name 'output.log' -print0 | while IFS= read -r -d '' logfile; do
    folder_name=$(basename "$(dirname "$logfile")")
    output_file="$(dirname "$logfile")/${folder_name}_batchsize.csv"
    bash scripts/convert_batchsize_inverval_to_csv.sh "$logfile" > "$output_file"
done

find $1 -type f -name 'samples_latency_data.txt' -print0 | while IFS= read -r -d '' logfile; do
    folder_name=$(basename "$(dirname "$logfile")")
    output_file="$(dirname "$logfile")/${folder_name}_latency.csv"
    bash scripts/convert_latency_to_csv.sh "$logfile" > "$output_file"
done

find $1 -type f -name 'mlperf_log_summary.txt' -print0 | while IFS= read -r -d '' logfile; do
    folder_name=$(basename "$(dirname "$logfile")")
    output_file="$(dirname "$logfile")/${folder_name}_mlperf_summary.txt"
    cp "$logfile" "$output_file"
done

echo "Processing complete!"
