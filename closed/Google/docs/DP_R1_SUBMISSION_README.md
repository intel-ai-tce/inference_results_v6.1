# DeepSeek-R1 v6.1 submission README

This README is the release procedure for the GB200 NVL72 DeepSeek-R1 Closed
Datacenter submission. It packages the selected Offline score together with
Server, validates the package using the pinned v6.1 submission checker, and
archives the release evidence to GCS.

## Release scope

| Field | Value |
| --- | --- |
| Submitter | `NVIDIA` (change consistently if a partner is the submitter) |
| Division / system type | Closed / Datacenter |
| System ID | `GB200-NVL72_GB200-186GB_aarch64x72_TRT` |
| Benchmark | `deepseek-r1` |
| Scenarios | Offline plus Server |
| Power | Not claimed |

The v6.1 checker requires Offline and at least one of Server or Interactive.
Server is the selected latency scenario. Interactive is not included in this
release package.

## Selected result set

Use one internally consistent triplet for every submitted scenario:

| Scenario | Mode | Source |
| --- | --- | --- |
| Offline | PerformanceOnly | Job `170564`: 516,507 tokens/s, 137.515 samples/s, target QPS 135, minimum query count 631,872 |
| Offline | AccuracyOnly | Job `170566`: exact match 80.879672 and 3,758.556974 tokens/sample; both accuracy gates pass under the final 135-QPS configuration |
| Offline | TEST06 | Job `170567`: Offline TEST06 verifier passed (first-token check skipped, EOS and sample-length checks true) |
| Server | PerformanceOnly | Job `170539` |
| Server | AccuracyOnly | Job `170540` |
| Server | TEST06 | Job `170541` |

Do **not** pair Offline job `170564` with Offline jobs `170535` or `170536`.
Those earlier jobs used target QPS 108 and a minimum query count of 561,664.
MLCommons requires an accuracy validation run for each submitted performance
result. The new Offline TEST06 audit must likewise be run from the final
configuration.

## 1. Preflight

Run all commands from `closed/NVIDIA`. Do not modify the shared
`MLPERF_SCRATCH_PATH` data or model store.

```bash
cd /home/alisachen_google_com/nv-mlpinf-partner/closed/NVIDIA
source .venv/bin/activate

git status --short
test -f 3rdparty/mlc-inference/tools/submission/submission_checker/constants.py
test -f build/loadgen-configs/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Offline/user.conf
```

Confirm the generated Offline configuration is the one being submitted:

```bash
cat build/loadgen-configs/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Offline/user.conf
```

It must show `target_qps = 135` and `min_query_count = 631872`.

Before external submission, have the release owner confirm that the LoadGen
revision in the logs is an approved v6.1 partner-drop revision, and confirm
that `NVIDIA` is the intended MLCommons submitter identity.

## 2. Run the missing Offline validations

Reuse the Offline SFlow recipe and final 135-QPS configuration. Do not retune
concurrency, target QPS, model, image, or topology.

For AccuracyOnly, set:

```bash
TEST_MODE=AccuracyOnly
HARNESS_EXTRA_ARGS=""
```

For TEST06, set:

```bash
TEST_MODE=PerformanceOnly
HARNESS_EXTRA_ARGS="--audit_test=TEST06 --server_target_qps_adj_factor=0.92"
```

Generate the jobs using the command pattern in
`src/nv_mlpinf/benchmarks/deepseek_r1/README.md`; inspect, patch with
`fix_sflow_batch_exit.py`, syntax-check, and then submit the generated sbatch
script. TEST06 correctly overrides its own small audit workload. For Offline,
`First token check pass: Skipped` is expected; EOS and sample-length checks
must pass.

Validate all new results:

```bash
python3 scripts/slurm_llm/deepseek_r1/check_offline_result.py \
  --expected-samples-per-second 135 \
  --min-samples 631872 \
  "$OFFLINE_PERF"
python3 scripts/slurm_llm/deepseek_r1/check_accuracy_result.py \
  --scenario Offline "$OFFLINE_ACCURACY"
python3 scripts/slurm_llm/deepseek_r1/check_test06_result.py \
  --scenario Offline "$OFFLINE_TEST06"
```

## 3. Stage the package

Create a new, empty staging root rather than changing `build/artifacts`.
After the fresh Offline runs finish, use:

```bash
SYSTEM=GB200-NVL72_GB200-186GB_aarch64x72_TRT
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
STAGE="$PWD/build/deepseek-r1-v6.1-submission-$STAMP"

OFFLINE_PERF="$PWD/sflow_output/170564-trtllm_ifb-20260716-015156-98f5c1/mlperf_harness/PerformanceOnly/$SYSTEM/deepseek-r1/Offline"
OFFLINE_ACCURACY="$PWD/sflow_output/170566-trtllm_ifb-20260716-052957-d8a586/mlperf_harness/AccuracyOnly/$SYSTEM/deepseek-r1/Offline"
OFFLINE_TEST06="$PWD/sflow_output/170567-trtllm_ifb-20260716-054912-95b44c/mlperf_harness/PerformanceOnly/$SYSTEM/deepseek-r1/Offline"
OFFLINE_CONFIG="$PWD/build/loadgen-configs/$SYSTEM/deepseek-r1/Offline"
SERVER_SOURCE="$PWD/build/artifacts/closed/NVIDIA/results/$SYSTEM/deepseek-r1/Server"

python3 scripts/slurm_llm/deepseek_r1/prepare_submission.py \
  --stage-root "$STAGE" \
  --offline-performance "$OFFLINE_PERF" \
  --offline-accuracy "$OFFLINE_ACCURACY" \
  --offline-test06 "$OFFLINE_TEST06" \
  --offline-config "$OFFLINE_CONFIG" \
  --server-source "$SERVER_SOURCE" \
  --submitter NVIDIA
```

The helper copies only Offline and Server. It keeps the historically aligned
Server `user.conf` (`110` QPS) instead of the currently generated `101.2` QPS
file.

## 4. Preserve full accuracy logs and truncate submitted accuracy logs

The raw full accuracy JSON files must be retained. The submitted regular
accuracy JSON files must be truncated; TEST06 accuracy logs are intentionally
left untruncated by the MLCommons tool.

```bash
FULL_LOG_BACKUP="$PWD/build/full_results/deepseek-r1-v6.1-$STAMP"

python3 3rdparty/mlc-inference/tools/submission/truncate_accuracy_log.py \
  --input "$STAGE" \
  --submitter NVIDIA \
  --backup "$FULL_LOG_BACKUP"
```

## 5. Export, checker, and tarball

`make export_submission` deletes the local generated `results/` directory and
the previous repository-level `build/submission` tree. Preserve those generated
trees before exporting so a failed checker run does not discard a prior package:

```bash
EXPORT_BACKUP="$PWD/build/pre-export-backup-$STAMP"
mkdir -p "$EXPORT_BACKUP"
[ -e results ] && cp -a results "$EXPORT_BACKUP/results"
[ -e ../../build/submission ] && cp -a ../../build/submission "$EXPORT_BACKUP/submission"
```

```bash
export SUBMITTER=NVIDIA

make export_submission SUBMITTER="$SUBMITTER" ARTIFACTS_DIR="$STAGE"

set -o pipefail
python3 3rdparty/mlc-inference/tools/submission/submission_checker/main.py \
  --input ../../build/submission \
  --submitter "$SUBMITTER" \
  --version v6.1 2>&1 | tee \
  ../../build/submission/closed/NVIDIA/results/submission_checker_log.txt
test "${PIPESTATUS[0]}" -eq 0
rg -n 'SUMMARY: submission looks OK|ERROR' \
  ../../build/submission/closed/NVIDIA/results/submission_checker_log.txt

make pack_submission SUBMITTER="$SUBMITTER"

cd ../../build/submission
sha1sum -c mlperf-inference-NVIDIA-submission.sha1
```

Do not use checker skip or exception flags. `make export_submission` replaces
the generated `results/` and `build/submission` directories, which is why the
isolated staging root, its full-accuracy backup, and raw SFlow output remain
the source of truth.

## 6. Archive to GCS and submit

Archive the final tarball, SHA1, checker log, staging manifest, full accuracy
backup, and raw selected job directories to a unique immutable prefix. Keep the
raw results private; they are not part of the MLCommons tarball. Do not reuse
the older `2026-07-14` prefix and do not use a delete option with GCS sync.

```bash
REPO_ROOT=$(cd ../.. && pwd)
TARBALL="$REPO_ROOT/build/submission/mlperf-inference-NVIDIA-submission.tar.gz"
SHA1_FILE="$REPO_ROOT/build/submission/mlperf-inference-NVIDIA-submission.sha1"
CHECKER_LOG="$REPO_ROOT/build/submission/closed/NVIDIA/results/submission_checker_log.txt"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BASE="gs://alisachen/mlperf-inference/deepseek-r1/submission/v6.1/2026-07-16/gb200-nvl72-offline135-server-$STAMP"

gcloud storage cp --if-generation-match=0 "$TARBALL" \
  "$BASE/release/$(basename "$TARBALL")"
gcloud storage cp --if-generation-match=0 "$SHA1_FILE" \
  "$BASE/release/$(basename "$SHA1_FILE")"
gcloud storage cp --if-generation-match=0 "$CHECKER_LOG" \
  "$BASE/verification/submission_checker_log.txt"

gcloud storage rsync --recursive --dry-run "$EVIDENCE_DIR" "$BASE/evidence"
gcloud storage rsync --recursive --no-clobber "$EVIDENCE_DIR" "$BASE/evidence"
```

`$EVIDENCE_DIR` must contain the raw selected Offline and Server job outputs,
the full-accuracy backup, generated SFlow scripts/configuration, the staging
`manifest.json`, and SHA256 manifest. Record `$BASE` in the release manifest.
Upload only after the checker reports `SUMMARY: submission looks OK`.

Use the MLCommons round-specific Submission UI and organization-specific
submitter ID to upload the final tarball. Record the portal receipt, submission
ID, SHA1, source revision, and GCS prefix in the release manifest.

## Final release checklist

- [ ] Offline `170564` has matching fresh AccuracyOnly and TEST06 runs.
- [ ] Server `170539` / `170540` / `170541` artifacts remain aligned.
- [ ] Full accuracy logs are retained outside the tarball.
- [ ] Submitted regular accuracy logs are truncated and hashed.
- [ ] The v6.1 checker reports `SUMMARY: submission looks OK` with no errors.
- [ ] Tarball SHA1 validates.
- [ ] GCS archive contains tarball, checksum, checker log, manifest, and raw evidence.
- [ ] Submitter identity, availability status, and approved LoadGen revision are confirmed.
