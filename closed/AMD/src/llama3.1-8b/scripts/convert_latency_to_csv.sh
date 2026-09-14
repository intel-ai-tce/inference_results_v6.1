#!/bin/bash

# Parser for harness_config.debug_record_sample_latencies feature

# Print CSV header
echo "id,ttft,tpot,isl,osl"

# Process each line of the input file
while read -r line; do
    # Use grep and sed to extract the values
    id=$(echo "$line" | sed -n 's/.*sample_data.id=\([0-9]*\).*/\1/p')
    ttft=$(echo "$line" | sed -n 's/.*ttft=\([0-9]*\).*/\1/p')
    tpot=$(echo "$line" | sed -n 's/.*tpot=\([0-9]*\).*/\1/p')
    isl=$(echo "$line" | sed -n 's/.*isl=\([0-9]*\).*/\1/p')
    osl=$(echo "$line" | sed -n 's/.*osl=\([0-9]*\).*/\1/p')
    # Output as CSV
    echo "$id,$ttft,$tpot,$isl,$osl"
done < $1
