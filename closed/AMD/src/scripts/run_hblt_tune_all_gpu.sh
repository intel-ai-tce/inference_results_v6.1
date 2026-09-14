#!/bin/bash
set -xeu

N_GEMMS=$(wc -l < "$1")
NUM_BATCHES=8
BATCH_SIZE=$((N_GEMMS / NUM_BATCHES + 1))
FILE_PREFIX="hipblaslt_tuning_workload_part_"
split -l "$BATCH_SIZE" -a 1 -d "$1" "$FILE_PREFIX"
for i in `seq 0 $((NUM_BATCHES - 1))`; do
   export HIP_VISIBLE_DEVICES=${i}
   export ITERS=100 
   export COLD_ITERS=100 
   export ROTATING=512 
   export TUNING_FILE=tuning_test_${i} 
   export ALGO_METHOD=all 
   export INPUT_FILE=${FILE_PREFIX}${i}
   ./run_hipblaslt_tuning.sh > /dev/null &  
done
#rm "$FILE_PREFIX"*
