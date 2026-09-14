# DLRM-v3 — MI350x8 End-to-End Run Walkthrough

## 1. Stage the dataset + checkpoint

```bash
DATASET_SRC=<local-dir | host:path | rclone-remote:path> \
CHECKPOINT_SRC=<...> \
  bash scripts/build/setup_data.sh
bash scripts/build/setup_data.sh   # verify only, if already staged
```

Download from MLCommons storage + validate:

```bash
mkdir -p /data/inference/model/dlrmv3 && cd /data/inference/model/dlrmv3
aria2c -x16 -s16 -c -i aria.input   # base: https://inference.mlcommons-storage.org/dlrmv3_trained_checkpoint/
md5sum -c checksums.md5

mkdir -p /data/inference/data/dlrmv3 && cd /data/inference/data/dlrmv3
aria2c -x16 -s16 -c -i aria.input   # base: https://inference.mlcommons-storage.org/dlrmv3_dataset/
md5sum -c checksums.md5
```



## 2. Build the workspace + container (HOST)

The runtime is a prebuilt ROCm base image instantiated as a long-lived container; the cert stack (fbgemm + pynve for `gfx950:sramecc+`) is compiled inside it. There is no Dockerfile.

```bash
bash scripts/build/setup_data.sh         # stage/verify data
bash scripts/build/setup_workspace.sh    # clone cert port repos + loadgen baseline
bash scripts/build/setup_submission.sh   # create container, build fbgemm + pynve (~15 min)

bash run.sh                              # or chain data -> workspace -> submission -> run
STAGES=workspace,submission,run bash run.sh   # skip data if already staged
```

Container image and launch:

```bash
IMAGE=rocm/atom:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom20260511
docker run -d --name dlrmv3-e2e723 \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add "$KFD_GROUP" \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size 32G \
  -v "$MOUNT_ROOT:$MOUNT_ROOT" \
  "$IMAGE" sleep infinity

docker ps --filter name=dlrmv3-e2e723
docker exec -it dlrmv3-e2e723 bash -lc 'python -c "import torch; print(torch.__version__)"'   # 2.10.0+rocm7.2.3
```



## 3. Preprocess the dataset (INSIDE the container)

```bash
docker exec -it dlrmv3-e2e723 bash
export PYTHONPATH=/work/mlcommons-inference/recommendation/dlrm_v3:/work/pynve-rocm/python
cd /work/dlrm-v3-harness-rocm

python tools/preprocess_data.py \
  --dataset-path /data/inference/data/dlrmv3/sampled_data/ \
  --output-dir   /work/dlrmv3_preprocessed_full \
  --dataset-percentage 1 \
  --use-multiprocessing

ls /work/dlrmv3_preprocessed_full   # ts_90 … ts_99, metadata.json (~140 GB)
exit
```



## 4. Run performance + accuracy + compliance (HOST)

Always set `TAG=` (the auto default aborts the script under `set -euo pipefail`). Run one workload at a time — each is GPU-exclusive; wait for VRAM to drain before the next.

Make sure the conf exists in the workspace harness tree the container mounts:

```bash
cp -f .../src/dlrm-v3/harness/benchmarks/user_mi350x8_9000.conf \
      .../setup/dlrm-v3-harness-rocm/benchmarks/user_mi350x8_9000.conf
```

```bash
cd .../setup/dlrm-v3
CONF=user_mi350x8_9000.conf

# performance (PerformanceOnly)
TAG=mi350_perf_offline SCENARIO=Offline CONF=$CONF bash scripts/run/run_gold.sh
TAG=mi350_perf_server  SCENARIO=Server  CONF=$CONF bash scripts/run/run_gold.sh

# accuracy (AccuracyOnly; scores GAUC)
TAG=mi350_acc_offline SCENARIO=Offline CONF=$CONF bash scripts/run/run_accuracy.sh
TAG=mi350_acc_server  SCENARIO=Server  CONF=$CONF bash scripts/run/run_accuracy.sh

# compliance (TEST08 chain: Offline AccuracyOnly ref + Server audit + verify)
CONF=$CONF TEST08_TAG_PREFIX=mi350_test08 bash scripts/run/_test08_chain.sh
```

Server pins SCLK to 2400 MHz via `rocm-smi --setperfdeterminism` (needs privilege); Offline uses auto clocks. Accuracy PASS = relative GAUC ≥ 99.9% of the fp16 reference. TEST08 PASS = `num_ne_mismatch=0`, `num_unmatched=0`.

## 5. Read the verdicts

```bash
cat artifacts/gold_mi350_perf_offline_*/mlperf_log_summary.txt        # Offline perf: Result is : VALID
cat artifacts/gold_mi350_perf_server_*/mlperf_log_summary.txt         # Server  perf: VALID (p99 <= 80 ms)
cat artifacts/gold_mi350_acc_offline_*/accuracy_metrics.txt           # Offline accuracy: lifetime GAUC
cat artifacts/gold_mi350_acc_server_*/accuracy_metrics.txt            # Server  accuracy: lifetime GAUC
cat artifacts/gold_mi350_test08_srv_perf_audit_*/verify_accuracy.txt  # TEST08 => PASS
```

