# DeepSeek-R1 benchmark summary for the a4x Slurm domain

Status as of 2026-07-12: the mandatory MLPerf Inference v6.1 DeepSeek-R1
closed/datacenter benchmark execution and the optional Interactive extension
are complete. Offline, Server, and Interactive each have locally validated
PerformanceOnly, AccuracyOnly, and TEST06 submission candidates.
Submission-tree packaging and the official v6.1 submission checker remain.

## Benchmark scope and platform

| Setting | Value |
| --- | --- |
| Slurm cluster | `a4xlustref`, Slurm 25.11.2 |
| Slurm partition | `a4x` |
| System | `GB200-NVL72_GB200-186GB_aarch64x72_TRT` |
| Allocation | 18 nodes, 4 GPUs per node, 72 GPUs total |
| GPU | NVIDIA GB200, 186 GB-class HBM3e; runtime detector reports 184 GiB per GPU |
| CPU architecture | `aarch64` |
| Model | DeepSeek-R1, ModelOpt FP4 checkpoint |
| Validated Offline/Server topology | Colocated TensorRT-LLM IFB |
| Validated Interactive topology | Disaggregated TensorRT-LLM: 2 context workers x 4 GPUs plus 4 generation workers x 16 GPUs, with 4 CPU frontends |
| Colocated data-parallel layout | 9 replicas, 8 GPUs per replica, 2 nodes per replica |
| Colocated intra-replica layout | TP=8, PP=1, MoE EP=8, attention DP enabled |
| Interactive context layout | 2 replicas x 4 GPUs; TP=4, PP=1, MoE EP=4 |
| Interactive generation layout | 4 replicas x 16 GPUs; TP=16, PP=1, MoE EP=16 |
| Slurm PMIx plugin | `pmix_v5` |
| Validated launch mode | PMI-2 |
| Colocated transport | MNNVL/NVLinkOneSided with NIXL UCX initialization |
| Interactive transport | MNNVL/NVLinkOneSided with a UCX KV-cache transceiver |
| NVIDIA driver | 580.173.02; driver-reported CUDA version 13.0 |
| Container | `/lustre/alisachen/containers/mlperf-inference_tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64.sqsh` |
| Raw shared storage | `/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone` |
| User preprocessing | `/lustre/alisachen/mlperf_inference_storage/preprocessed_data` |

The topology is:

```text
18 nodes x 4 GPUs/node = 72 GPUs
9 replicas x 8 GPUs/replica = 72 GPUs
1 replica = 2 nodes x 4 GPUs/node

Interactive:
(2 x 4) context GPUs + (4 x 16) generation GPUs = 72 GPUs
4 frontends use no additional GPUs
```

## Locally validated benchmark matrix

The following six jobs form the locally validated mandatory Offline-plus-Server
submission set. Final acceptance depends on the official v6.1 submission
checker and review.

| Job | Scenario and mode | Slurm result | Locally validated result |
| ---: | --- | --- | --- |
| 170528 | Offline PerformanceOnly | `COMPLETED 0:0`, `01:22:00` | `VALID`; 485,843 tokens/s; 129.405 samples/s; 561,664 completed samples; minimum duration and query gates passed |
| 170535 | Offline AccuracyOnly | `COMPLETED 0:0`, `00:24:48` | Exact match 81.175934; 3,773.608250 tokens/sample; 4,388 samples; all accuracy gates passed |
| 170536 | Offline TEST06 | `COMPLETED 0:0`, `00:13:35`; harness `00:05:10` | `TEST06_PASS`; `audit_success=true`; first-token skipped as expected for Offline; EOS and sample-length checks passed |
| 170539 | Server PerformanceOnly | `COMPLETED 0:0`, `00:50:42`; harness `00:42:42` | `VALID`; 392,959 completed tokens/s; p99 TTFT 950.001124 ms; p99 TPOT 69.162691 ms; 210,624 completed queries |
| 170540 | Server AccuracyOnly | `COMPLETED 0:0`, `00:20:02`; harness `00:11:57` | Exact match 81.175934; 3,725.837284 tokens/sample; 4,388 samples; all accuracy gates passed |
| 170541 | Server TEST06 | `COMPLETED 0:0`, `00:13:13`; harness `00:05:08` | `TEST06_PASS`; `audit_success=true`; first-token, EOS, and sample-length checks passed |

The six validated allocations used 3 hours, 24 minutes, and 20 seconds of
aggregate Slurm wall time.

### Interactive extension

Interactive PerformanceOnly job 170549 completed `0:0` in `00:59:49` and its
outside validator passed every gate. At 60 queries/s, LoadGen reported `VALID`,
224,764 completed tokens/s, 59.7792 samples/s, p99 TTFT 430.739603 ms, and p99
TPOT 13.922472 ms. Its logical scenario and result path remain Interactive,
while nv-mlpinf intentionally uses LoadGen's effective Server scheduler. The
run met its 600,000 ms duration, 144,000 query, and early-stopping requirements.

Interactive AccuracyOnly job 170557 completed `0:0` in `00:22:36`; its harness
also completed `0:0`. Exact match was 81.221513, tokens/sample was 3,721.166135,
and all 4,388 samples were evaluated, so every accuracy gate passed. All eight
HTTP workers logged clean shutdown and the tracked PRM800k setup file remained
unchanged. This clean exit replaces diagnostic job 170551, whose scientific
metrics passed before a post-result interpreter-shutdown segfault.

Interactive TEST06 job 170558 completed `0:0` in `00:18:50`; its harness
completed `0:0` in `00:02:39`. The scenario-aware validator reported
`TEST06_PASS` with `audit_success=true`; first-token, EOS, sample-length, and
verification-complete checks all passed. Together, jobs 170549, 170557, and
170558 form the locally validated Interactive result set. The three Interactive
allocations used 1 hour, 41 minutes, and 15 seconds; all nine accepted jobs used
5 hours, 5 minutes, and 35 seconds of aggregate Slurm wall time.

The TEST06 raw LoadGen mini-run is `INVALID`: its 100-query audit set did not
meet the normal performance early-stopping sample sufficiency, and p99 TTFT was
above the scored Interactive limit. That raw mini-run is not the compliance
verdict and does not replace scored Interactive PerformanceOnly job 170549.

### Acceptance thresholds

DeepSeek-R1 accuracy requires:

- exact match at least 80.544618;
- tokens/sample from 3,497.60466 through 4,274.85014, inclusive;
- at least 4,388 evaluated samples.

Server performance additionally requires:

- p99 TTFT strictly below 2,000 ms;
- p99 TPOT strictly below 80 ms;
- a valid LoadGen result with its duration, query, and early-stopping gates
  satisfied.

Interactive performance requires the same LoadGen validity gates plus:

- p99 TTFT strictly below 1,500 ms;
- p99 TPOT strictly below 15 ms.

Offline job 170528 achieved 1.198194 times its configured sizing target of
108 samples/s. Server job 170539 scheduled 110.19 samples/s and completed
104.73 samples/s while remaining inside both latency limits. Offline and
Server scores are scenario-specific and should not be treated as directly
interchangeable measurements.

Server TEST06 job 170541 has a raw LoadGen performance result of `INVALID`
because its 100-query compliance mini-run does not have enough samples for the
normal performance early-stopping calculation; LoadGen says another 359
queries would be needed. Minimum duration, minimum queries, and TTFT/TPOT
constraints passed. The authoritative compliance result is `TEST06_PASS`, so
job 170541 is valid as TEST06 and must not be used as the scored Server
performance run. Job 170539 remains the scored Server performance result.

## Result locations and retention

Local workflows are under:

```text
/home/alisachen_google_com/nv-mlpinf-partner/closed/NVIDIA/sflow_output
```

| Job | Workflow directory |
| ---: | --- |
| 170528 | `170528-trtllm_ifb-20260710-193617-0e83eb` |
| 170535 | `170535-trtllm_ifb-20260711-075046-cf4b93` |
| 170536 | `170536-trtllm_ifb-20260711-200610-c678f5` |
| 170539 | `170539-trtllm_ifb-20260711-212202-b939e3` |
| 170540 | `170540-trtllm_ifb-20260711-221453-29169d` |
| 170541 | `170541-trtllm_ifb-20260711-223711-723a4a` |
| 170549 | `170549-trtllm_disagg-20260712-044902-1dd3be` |
| 170557 | `170557-trtllm_disagg-20260712-064904-b6f9ef` |
| 170558 | `170558-trtllm_disagg-20260712-071317-29906d` |

The complete raw workflows, final parent logs, generated sflow wrappers,
LoadGen configuration, and runtime configuration for all nine accepted jobs
are archived in GCS:

| Job | GCS prefix | Objects | Logical bytes |
| ---: | --- | ---: | ---: |
| 170528 | `gs://alisachen/mlperf-inference/deepseek-r1/offline/performance-only/2026-07-10/` | 127 | 1,171,518,006 |
| 170535 | `gs://alisachen/mlperf-inference/deepseek-r1/offline/accuracy-only/2026-07-11/` | 130 | 362,334,869 |
| 170536 | `gs://alisachen/mlperf-inference/deepseek-r1/offline/test06/2026-07-11/` | 112 | 95,746,790 |
| 170539 | `gs://alisachen/mlperf-inference/deepseek-r1/server/performance-only/2026-07-11/` | 128 | 467,850,135 |
| 170540 | `gs://alisachen/mlperf-inference/deepseek-r1/server/accuracy-only/2026-07-11/` | 130 | 229,094,564 |
| 170541 | `gs://alisachen/mlperf-inference/deepseek-r1/server/test06/2026-07-11/` | 112 | 29,339,932 |
| 170549 | `gs://alisachen/mlperf-inference/deepseek-r1/interactive/performance-only/2026-07-12/` | 125 | 717,360,531 |
| 170557 | `gs://alisachen/mlperf-inference/deepseek-r1/interactive/accuracy-only/2026-07-12/` | 127 | 252,086,017 |
| 170558 | `gs://alisachen/mlperf-inference/deepseek-r1/interactive/test06/2026-07-12/` | 119 | 40,497,723 |

The nine archives contain 1,110 objects and 3,365,828,567 logical bytes in
total. Each prefix contains only its corresponding validated job ID. Failed and
exploratory attempts are absent from GCS.

## MPI, GPU, and fabric validation

| Job | Validation | Result |
| ---: | --- | --- |
| 170507 | Two-node PMI-2 plus container `mpi4py` | Passed; ranks 0 and 1 formed a two-rank communicator |
| 170521 | Repeat two-node PMI-2/`mpi4py` smoke | Passed |
| 170523/170524 | Two-node, eight-rank GPU MPI smokes | Passed; eight distinct GPUs were visible |
| 170532 | Two-node, eight-GPU MNNVL fabric smoke | Passed; every GPU reported Fabric `Completed/Success` |
| 170538 | Full 18-node/72-rank PMI-2 all-reduce recovery smoke | Passed in 55 seconds |

The host exposes Slurm `pmix_v5` and PMIx 5.0.3. The container uses Open MPI
4.1.9a1 with embedded PMIx 3.2.5a1, and its `mpi4py` 3.1.5 extension links to
the container MPI library. Direct PMIx could not bridge those generations, but
PMI-2 was validated from Slurm through container `mpi4py`; the benchmark
therefore uses `mpi: pmi2`.

Accepted Offline job 170528, Server job 170539, and Interactive job 170549 all
showed `MnnvlMemory` and `Selected communication strategy: NVLinkOneSided` in
their rank logs. Offline and Server initialized the NIXL UCX backend;
Interactive used a UCX KV-cache transceiver between context and generation
services. These markers prove the selected software path, not the underlying
UCX fabric or protocol, so they are not described as proof of RC/RDMA. No
forced NCCL or UCX-RDMA override was used.

## Failed and exploratory attempts

The following jobs are diagnostics or failed attempts, not submission results:

| Job | Outcome and resolution |
| ---: | --- |
| 170501/170502 | Pyxis container import was directed through Docker Hub and returned 401. Correct registry syntax and then the shared `.sqsh` resolved image startup. |
| 170503 | Direct PMIx failed in `MPI_Init_thread` with `OPAL ERROR: Unreachable` because host and container PMIx generations are incompatible. The abort left node cleanup in `COMPLETING`; PMI-2 replaced PMIx. |
| 170505 | `--mpi=none` container diagnostic passed and isolated the failure to MPI integration. |
| 170519 | All nine Offline server steps failed container import. An old wrapper masked the child failure and made the Slurm parent look green. It contains no valid result. |
| 170522 | LoadGen reported `CUDA_ERROR_NO_DEVICE` because its step did not receive a GPU. `--gpus-per-task=1` and failure propagation were added. |
| 170525 | Explicitly cancelled during model startup; no measurement result. |
| 170526 | Harness failed with `ModuleNotFoundError: No module named 'constants'`. The MLCommons submission-checker import path and repository preflight were fixed. |
| 170527/170533 | One-GPU harness/import smokes passed after the preflight fixes. |
| 170530/170531 | MNNVL checker assumptions did not match driver 580 output. The checker was updated to use full `nvidia-smi -q` output and tolerate omitted UUID/clique fields. |
| 170537 | Server run issued 210,624 queries and reached approximately 98% drain, but nodes 0 and 1 hit Slurmd's compiled 50-connection ceiling, missed heartbeats, and caused `NODE_FAIL`. Partial output is invalid. |
| 170545 | Interactive `PerformanceOnly` stopped before measurement when sflow's 600-second backend readiness watcher expired during normal DeepSeek-R1 checkpoint loading and CUDA/autotuner warmup. It produced no LoadGen result. Backend readiness was increased to 1,800 seconds and frontend readiness to 600 seconds. |
| 170547 | Interactive PerformanceOnly at 64 queries/s was cancelled after p99 TPOT drifted to approximately 15.48 ms, above the strict 15 ms limit. Job 170549 reduced the target to 60 queries/s and passed. |
| 170551 | Interactive AccuracyOnly produced passing scientific metrics, then the harness segfaulted during interpreter shutdown and the parent failed `1:0`. It is diagnostic only. Worker/ZMQ teardown, health-check lifecycle, and temporary PRM800k setup handling were hardened for the clean retry. |

Job 170537 left exact-job step daemons, TRT-LLM processes, and cgroups on nodes
0 and 1. They were cleaned without touching unrelated workloads. Slurmd was
restarted across idle nodes and the cluster configuration was changed to:

```text
SlurmdParameters=conmgr_max_connections=256
```

Full-rack recovery smoke 170538 passed afterward, followed by validated Server
PerformanceOnly job 170539. All 18 nodes returned to `IDLE` at the post-recovery
check before the subsequent benchmark jobs.

## Validator and documentation updates

The benchmark work added or strengthened these outside validators:

- `scripts/slurm_llm/deepseek_r1/check_offline_result.py`
- `scripts/slurm_llm/deepseek_r1/check_accuracy_result.py`
- `scripts/slurm_llm/deepseek_r1/check_server_result.py`
- `scripts/slurm_llm/deepseek_r1/check_interactive_result.py`
- `scripts/slurm_llm/deepseek_r1/check_test06_result.py`

The accuracy path now enforces both the lower and upper tokens/sample limits,
the expected scenario, `AccuracyOnly` mode, minimum sample count, and a
nonempty raw accuracy log. Its PRM800k dependency is installed through a
private temporary setup directory, so accuracy evaluation no longer edits the
tracked submodule. The Server performance validator requires a Server result
directory and matching metadata in addition to LoadGen validity, token-latency
enablement, configured duration, performance sample count, fixed seeds, finite
score, and strict p99 TTFT/TPOT limits. The TEST06 validator and in-workflow
verifier allow a skipped first-token check only for Offline; Server and
Interactive must report `True`.

The Interactive validator preserves the logical Interactive metadata and path
while requiring LoadGen's effective Server scenario, `PerformanceOnly`, fixed
seeds, minimum duration, query, and sample gates, a finite positive score, and
strict p99 TTFT below 1,500 ms and p99 TPOT below 15 ms.

The Server sflow environment now uses the validated shared image, `a4x`
partition, four GPUs per node, PMI-2, exclusive allocation, and no unsupported
`--segment` option. Repository, evaluation-submodule, dataset, and preprocessed
data checks run before starting the model servers.

The Interactive path uses the same shared image, `a4x` allocation, PMI-2
launch, and repository/data preflight. Its context and generation cache
transceivers use UCX. Runtime logs separately confirm `MnnvlMemory` and
`NVLinkOneSided` selection across ranks; a UCX initialization marker alone is
not claimed as proof of RDMA.

The endpoint client now closes worker ZMQ sockets before terminating their
contexts, gives workers a shared 10-second graceful shutdown period, and
force-reaps only survivors. Health checks are synchronous and no longer leave
a module-global uvloop daemon thread alive across process forks or interpreter
shutdown. A run is accepted only when its harness and parent Slurm job both
exit `0:0`, even if scientific result files were written first.

## Submission status and remaining work

The pinned MLPerf Inference v6.1 checker requires DeepSeek-R1 Offline plus at
least one of Server or Interactive. Each included scenario needs performance,
accuracy, and TEST06. The locally validated Offline and Server jobs therefore
complete the mandatory six-run matrix.
Interactive remains optional, and jobs 170549, 170557, and 170558 provide its
complete locally validated PerformanceOnly, AccuracyOnly, and TEST06 result
set. No additional benchmark job is required. Power is optional and is not
claimed.

Raw GCS archives are durable run backups, not the final MLPerf submission
layout. Remaining non-benchmark work is:

1. Curate the nine validated result leaves into the official Offline, Server,
   and Interactive results/compliance tree.
2. Add and review `README.md`, `user.conf`, `measurements.json`, `mlperf.conf`,
   and calibration documentation for all three scenarios.
3. Review the updated system JSON and confirm submitter identity. Keep
   driver-reported CUDA compatibility 13.0 distinct from CUDA Toolkit 13.1 in
   the exact container.
4. Preserve full accuracy logs, create checker-compliant submitted accuracy
   artifacts, and retain the required TEST06 compliance artifacts.
5. Export a curated submission without copying `sflow_output` or the sflow
   virtual environment into it.
6. Run the pinned checker with `--version v6.1` and no skip/exception flags,
   retain its successful log, then package and checksum the final submission.

## Related documentation

- [DeepSeek-R1 a4x Slurm benchmark plan](DP_R1_SLURM_BENCHMARK_PLAN.md)
- [Submission log preparation](SUBMISSION.md)
- [DeepSeek-R1 benchmark README](../src/nv_mlpinf/benchmarks/deepseek_r1/README.md)
- [System-specific Offline README](../configs/deepseek_r1/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/README.md)
