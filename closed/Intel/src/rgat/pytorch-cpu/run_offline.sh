#!/bin/bash

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so

export num_physical_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
num_numa=$(numactl --hardware|grep available|awk -F' ' '{ print $2 }')

#python ../../user_config.py
#USER_CONF=user.conf

export DATASET_PATH=/data/IGBH
export CHECKPOINT_PATH=/model/RGAT.pt #model_2593.pt
# NUM_PROC=2 #$num_numa
# CPUS_PER_PROC=126 #$((num_physical_cores/num_numa))
# WORKERS_PER_PROC=3
TOTAL_SAMPLE_COUNT=788379
BATCH_SIZE=32678
TIMESTAMP=$(date +%m-%d-%H-%M)
HOSTNAME=$(hostname)

#export OMP_NUM_THREADS=30
export QINT8_BLOCK_SIZE=128

export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"


cmd=""
MEM_AVAILABLE=$(free -g | awk '/Mem:/ {print $7}')
if [ $MEM_AVAILABLE -lt 1500 ]; then
	echo "Memory is less than 1500GB, setting NUMA balance"
	cmd="numactl --interleave=all"
else
	echo "Memory is greater than 1500GB"
	
fi

echo "$cmd python main.py --workload-name rgat \
	--scenario Offline \
	--mode Performance \
	--num-proc ${NUM_PROC} \
	--cpus-per-proc ${CPUS_PER_PROC} \
	--checkpoint-path ${CHECKPOINT_PATH} \
	--dataset-path ${DATASET_PATH} \
	--batch-size ${BATCH_SIZE} \
	--user-conf user.conf \
	--warmup \
	--use-tpp \
	--use-bf16 \
	--cores-offset ${CORE_OFFSET} \
	--fused-sampler \
	--use-qint8-gemm \
	--fan-out '[15,10,5]' \
	--workers-per-proc ${WORKERS_PER_PROC} \
	--total-sample-count ${TOTAL_SAMPLE_COUNT} \
	--output-dir ${TMP_DIR}"

$cmd python main.py --workload-name rgat \
	--scenario Offline \
	--mode Performance \
	--num-proc ${NUM_PROC} \
	--cpus-per-proc ${CPUS_PER_PROC} \
	--checkpoint-path ${CHECKPOINT_PATH} \
	--dataset-path ${DATASET_PATH} \
	--batch-size ${BATCH_SIZE} \
	--user-conf user.conf \
	--warmup \
	--use-tpp \
	--use-bf16 \
	--cores-offset ${CORE_OFFSET} \
	--fused-sampler \
	--use-qint8-gemm \
	--fan-out '[15,10,5]' \
	--workers-per-proc ${WORKERS_PER_PROC} \
	--total-sample-count ${TOTAL_SAMPLE_COUNT} \
	--output-dir ${TMP_DIR}

echo "run_offline done"
