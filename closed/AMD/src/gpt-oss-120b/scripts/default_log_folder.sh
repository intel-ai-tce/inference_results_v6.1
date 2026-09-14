#!/bin/bash

generate_output_log_dir() {
    local input_string="$1"
    local model_name="$2"
    local scenario="$3"
    local gpu="$4"
    
    local test_mode=$(echo "$input_string" | grep -oP '(?<=test_mode=)[^\s]+')
    local cpu=$(lscpu | grep "Model name" | awk -F: '{print $2}' | sed -E 's/^[ \t]+//; s/([0-9]+-Core.*)//; s/^AMD //; s/ /_/g')
    cpu="${cpu%_}"
    local output_dir="results/8x${gpu^^}_2x${cpu}/$model_name/${scenario^}/$test_mode"
    
    if [ "$test_mode" = "performance" ]; then
        output_dir="${output_dir}/run_1"
    fi
    
    echo "$output_dir"
}