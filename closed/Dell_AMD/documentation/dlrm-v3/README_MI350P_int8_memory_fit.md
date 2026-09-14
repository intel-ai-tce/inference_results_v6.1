# DLRM-v3 MI350P Memory-Fit Configuration

This package adds a functional MI350P-oriented DLRM-v3 configuration. MI350P is treated as a
half-memory / lower-throughput target relative to MI350/MI355, so the goal is memory fit and accuracy
first, not record qps.

## Configuration

Use the true packed-int8 item-table path:

```bash
DLRM_NVE_INT8_GATHER=1
DLRM_NVE_PARALLEL_CKPT_LOAD=1
DLRM_NVE_GPU_CACHE_GB=1
DLRM_LOCAL_SMALL_TABLE_LOOKUP=0
BATCH=16
INFLIGHT=32
```

The included MI350P configs set the submitted operating point: Server qps `4500` and Offline qps
`5500` (true packed-int8 memory-fit path):

```text
closed/AMD/src/dlrm-v3/harness/benchmarks/user_mi350p8_nve_int8_b16_qps2200_SERVER90s.conf
closed/AMD/src/dlrm-v3/harness/benchmarks/user_mi350p8_nve_int8_b16_qps2200_PROD10min.conf
closed/AMD/results/8xMI350P_2xEPYC_9455/dlrm-v3/Offline/user.conf
closed/AMD/results/8xMI350P_2xEPYC_9455/dlrm-v3/Server/user.conf
```

## Harness Fixes Included

Two harness fixes are included in this package:

- `scripts/run/run_gold.sh` now forwards `DLRM_NVE_INT8_GATHER` into `docker exec` and prints
  `nve_int8=<value>` in the launch line.
- `scripts/run/run_gold.sh` now uses a config-aware VRAM guard. The normal bf16/GOLD path still defaults
  to `MIN_FREE_GB=210`, but the true-int8 MI350P path defaults to about `120 GB` free for
  `BATCH=16`, `INFLIGHT=32`, `DLRM_NVE_GPU_CACHE_GB=1`.
- `inference_harness/tools/model_configs.py` now constructs the MPI item memblock at packed width
  `258` when `DLRM_NVE_INT8_GATHER=1`, instead of allocating logical fp16 width `512`.

Without both fixes, the run can silently fall back to bf16-width item storage and exceed MI350P memory.

## MI355 Pre-Validation Evidence

Measured on an 8xMI355 node using the MI350P memory-fit config:

- Offline AccuracyOnly GAUC: `0.7862148605380281`
- Relative to q12,200 fp16/int8 reference `0.7862875110`: `99.990760%`
- Accuracy entries processed: `349823`
- Offline AccuracyOnly peak VRAM: `102.79 GB/GPU`
- Server q7500 smoke on MI355: VALID, p99 `23.77 ms`, peak VRAM `109.95 GB/GPU`
- Server q8000 smoke on MI355: INVALID, p99 `256.51 ms`


## Example Commands

From `closed/AMD/setup/dlrm-v3` after setup:

```bash
export DLRM_NVE_INT8_GATHER=1
export DLRM_NVE_PARALLEL_CKPT_LOAD=1
export DLRM_NVE_GPU_CACHE_GB=1
export DLRM_LOCAL_SMALL_TABLE_LOOKUP=0
export BATCH=16
export INFLIGHT=32

CONF=user_mi350p8_nve_int8_b16_qps2200_PROD10min.conf \
TAG=mi350p_int8_offline_accuracy \
bash scripts/run/run_accuracy.sh

CONF=user_mi350p8_nve_int8_b16_qps2200_SERVER90s.conf \
TAG=mi350p_int8_server_smoke \
SCENARIO=Server MODE=performance \
bash scripts/run/run_gold.sh
```

Accuracy is the gating requirement. Performance can be low for MI350P support as long as the run is
stable, under memory, and GAUC passes.
