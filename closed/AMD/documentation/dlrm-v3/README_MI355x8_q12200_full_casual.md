# DLRM-v3 q12,200 — End-to-End Run Walkthrough

## 1. Stage the dataset + checkpoint

Obtain the dataset/checkpoint per the MLCommons DLRM-v3 reference, then stage or verify:

```bash
DATASET_SRC=<local-dir | host:path | rclone-remote:path> \
CHECKPOINT_SRC=<...> \
  bash scripts/build/setup_data.sh
bash scripts/build/setup_data.sh   # verify only, if already staged
```

### Download from MLCommons storage + validate

```bash
mkdir -p /data/inference/model/dlrmv3 && cd /data/inference/model/dlrmv3
aria2c -x16 -s16 -c -i aria.input   # base: https://inference.mlcommons-storage.org/dlrmv3_trained_checkpoint/
md5sum -c checksums.md5

mkdir -p /data/inference/data/dlrmv3 && cd /data/inference/data/dlrmv3
aria2c -x16 -s16 -c -i aria.input   # base: https://inference.mlcommons-storage.org/dlrmv3_dataset/
md5sum -c checksums.md5
```

## 2. Build the workspace + container (HOST)



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
```

Confirm it is up:

```bash
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

## 4. Performance runs (HOST)

Always set `TAG=`. The auto-`TAG` default aborts the script silently under `set -euo pipefail`; `TAG` only names the artifacts folder and has no effect on the run config.

```bash
cd /home/nehmathe/AMD_closed_dlrm-v3_q12200_bundle/extracted/closed/AMD/setup/dlrm-v3

TAG=server_run  SCENARIO=Server  CONF=user_mi355x8_nve_b64_qps12200_PROD10min.conf    bash scripts/run/run_gold.sh
TAG=offline_run SCENARIO=Offline CONF=user_mi355x8_nve_b64_qps12200_OFFLINE10min.conf bash scripts/run/run_gold.sh
```

Offline uses auto clocks (no latency bar); Server pins SCLK to 2400 MHz via `rocm-smi --setperfdeterminism` (needs privilege).

## 5. Accuracy runs (HOST)

`run_accuracy.sh` reuses the GOLD stack with `MODE=accuracy`, scores GAUC, and sets `TAG` + accuracy flags itself.

```bash
bash scripts/run/run_accuracy.sh                    # Offline AccuracyOnly (default)
SCENARIO=Server bash scripts/run/run_accuracy.sh    # Server AccuracyOnly
SCORE_ONLY=artifacts/gold_acc_<...> bash scripts/run/run_accuracy.sh   # re-score only
```

PASS: relative GAUC ≥ 99.9% of the fp16 reference (certified q12,200 = 0.7862875110). ~25–30 min per run; one run at a time.

## 6. Compliance (TEST08)

```bash
bash scripts/run/_test08_chain.sh
```

Offline AccuracyOnly ref + audited Server PerformanceOnly + official verifier (PASS = `num_ne_mismatch=0`, `num_unmatched=0`).

## 7. Read the verdicts

```bash
cat artifacts/gold_<TAG>_<STAMP>/mlperf_log_summary.txt   # perf: Result is : VALID
cat artifacts/gold_acc_<...>/accuracy_metrics.txt         # accuracy: lifetime GAUC
```


