# DeepSeek-R1 CentML NVFP4 v2: Offline and Server results

## Submission record

| Item | Value |
| --- | --- |
| Checkpoint | [`centml/DeepSeek-R1-NVFP4-v2-mlpinf`](https://huggingface.co/centml/DeepSeek-R1-NVFP4-v2-mlpinf) |
| Immutable revision | `93947a0d7bd04f73ff98636f6b18ff5839e7aaf9` |
| MLPerf version | Inference v6.1, closed/datacenter |
| System | `GB200-NVL72_GB200-186GB_aarch64x72_TRT` |
| Hardware allocation | 18 GB200 NVL72 nodes, 72 GPUs total |
| Submitted scenarios | Offline and Server only |
| Local submission checker | PASS: `Results=2, NoResults=0` |
| Submission state | Uploaded through the MLCommons submission CLI on 2026-07-25; the submission-check request was accepted. |

Interactive is intentionally not included in this submission.

## Throughput comparison

The NVIDIA MLPerf v6.0 values below are reference throughput values, not v6.1
validity thresholds. MLPerf v6.1 has no minimum throughput requirement for
these two scenarios.

| Scenario | New result | New tokens/s | Old checkpoint tokens/s | New vs. old | NVIDIA MLPerf v6.0 tokens/s | New vs. v6.0 | New samples/s | Old checkpoint samples/s | Samples/s change | Requirement comparison |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Offline | VALID | 510,781 | 485,843 | +5.13% | 486,141 | +5.07% | 133.883 | 129.405 | +4.478 (+3.46%) | No MLPerf throughput minimum; +25.883 samples/s (+23.97%) versus the internal 108 samples/s sizing target |
| Server | VALID | 388,590 | 392,959.40 | -1.11% | 336,106 | +15.62% | 101.799 completed | 104.728 | -2.929 (-2.80%) | No MLPerf throughput minimum |

## Server latency comparison

| Metric | New checkpoint | Old checkpoint | New vs. old | v6.1 limit | New-checkpoint margin | Result |
| --- | ---: | ---: | ---:| ---: | ---: | --- |
| p99 TTFT | 1,160.617 ms | 950.001 ms | +210.615 ms (+22.17%) | <= 2,000 ms | 839.383 ms (41.97%) below limit | PASS |
| p99 TPOT | 74.036 ms | 69.163 ms | +4.873 ms (+7.05%) | <= 80 ms | 5.964 ms (7.46%) below limit | PASS |

Offline has no TTFT or TPOT constraint.

## Accuracy comparison

| Scenario | Metric | New checkpoint | Old checkpoint | New vs. old | v6.1 criterion | New-checkpoint margin | Result |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Offline | Exact match | 81.198724% | 81.175934% | +0.022790 points | >= 80.544618% | +0.654106 points (+0.81%) | PASS |
| Offline | Tokens/sample | 3,852.355971 | 3,773.608250 | +78.747721 (+2.09%) | 3,497.604660-4,274.850140 | 10.14% above lower bound; 9.88% below upper bound | PASS |
| Offline | Evaluated samples | 4,388 | 4,388 | 0 | >= 4,388 | At requirement | PASS |
| Server | Exact match | 81.016408% | 81.175934% | -0.159526 points | >= 80.544618% | +0.471790 points (+0.59%) | PASS |
| Server | Tokens/sample | 3,795.093437 | 3,725.837284 | +69.256153 (+1.86%) | 3,497.604660-4,274.850140 | 8.51% above lower bound; 11.22% below upper bound | PASS |
| Server | Evaluated samples | 4,388 | 4,388 | 0 | >= 4,388 | At requirement | PASS |

NVIDIA MLPerf v6.0 accuracy values were not provided with the v6.0 throughput
reference, so the v6.0 comparison is limited to throughput.

## Result provenance

| Artifact | Location |
| --- | --- |
| Offline PerformanceOnly | `sflow_output_centml_v2_v6_1_3_20260724T032753Z/170690-trtllm_ifb-20260724-032902-192d07/mlperf_harness/PerformanceOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Offline/` |
| Server PerformanceOnly | `sflow_output_centml_v2_v6_1_3_20260724T032753Z/170691-trtllm_ifb-20260724-045704-5de471/mlperf_harness/PerformanceOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Server/` |
| Offline AccuracyOnly | `sflow_output_centml_v2_v6_1_3_submission_20260724T181836Z/170701-trtllm_ifb-20260724-182022-6856a6/mlperf_harness/AccuracyOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Offline/` |
| Offline TEST06 | `sflow_output_centml_v2_v6_1_3_submission_20260724T181836Z/170702-trtllm_ifb-20260724-183908-b35be3/mlperf_harness/PerformanceOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Offline/TEST06/` |
| Server AccuracyOnly | `sflow_output_centml_v2_v6_1_3_submission_20260724T181836Z/170703-trtllm_ifb-20260724-185222-9d46e3/mlperf_harness/AccuracyOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Server/` |
| Server TEST06 | `sflow_output_centml_v2_v6_1_3_submission_20260724T181836Z/170704-trtllm_ifb-20260724-191109-d7ae70/mlperf_harness/PerformanceOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Server/TEST06/` |
| Submission tarball | `/home/alisachen_google_com/nv-mlpinf-partner-v6.1.3-export-20260724/build/submission/mlperf-inference-NVIDIA-submission.tar.gz` |

The tarball SHA1 is `6a25625b4893f0b53da3347030fe88903bb6a73e`.
