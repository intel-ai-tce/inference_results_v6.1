#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "Usage: store_graph.sh <path-to-data-store> <dataset_size>"
  exit
fi
# Set up the environment properly
export LD_PRELOAD="/opt/conda/lib/libiomp5.so"
export DGLBACKEND="pytorch"
export LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"

# Source the GCC toolset environment
source /opt/rh/gcc-toolset-12/enable 2>/dev/null || true

dataset_size=$2
dataset="IGBH"
in_path=$1/$dataset
graph_path=$1/$dataset/$dataset_size
out_path=$in_path"/"$dataset_size
token="p"

python -u -W ignore store_graph.py --path $in_path --dataset $dataset --dataset_size $dataset_size --output $out_path --graph_struct_only
