#!/bin/bash

# Parser for VLLM_LOG_BATCHSIZE_INTERVAL feature

# Print CSV header
echo "batchsize,count,median_time(ms)"

# Get the last matching line
last_line=$(grep 'Batchsize forward time stats (batchsize, count, median_time(ms)):' $1 | tail -n 1)

# Extract and format the tuples as CSV
echo "$last_line" | grep -oP '\(\d+, \d+, [0-9.]+\)' | sed -e 's/[()]//g' -e 's/, /,/g'
