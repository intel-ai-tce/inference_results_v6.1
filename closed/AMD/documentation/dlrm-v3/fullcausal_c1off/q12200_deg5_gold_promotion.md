# q12,200 Degree-5 GOLD Promotion Proofs

Promotion date: 2026-07-16

## Production Gate

`DLRM_HSTU_GATE_POLY=1` with `DLRM_HSTU_GATE_POLY_DEG=5`.

## Accuracy

Artifact:

```text
artifacts/gold_acc_prod_deg5_20260716T054951/accuracy_metrics.txt
```

Key metric:

```text
metric/lifetime_gauc/rating: 0.7862875110162386
metric/lifetime_gauc_num_samples/rating: 349823.0
```

## Server PROD10min

Artifact:

```text
artifacts/gold_prod_deg5_q12200_PROD10min_20260716T093104/mlperf_log_summary.txt
```

Result:

```text
Result is : VALID
Completed samples per second    : 12198.90
Scheduled samples per second : 12200.90
50.00 percentile latency (ns)   : 50035573
90.00 percentile latency (ns)   : 53985264
95.00 percentile latency (ns)   : 55340415
97.00 percentile latency (ns)   : 56303682
99.00 percentile latency (ns)   : 58319253
99.90 percentile latency (ns)   : 69142894
```

## TEST08

Artifacts:

```text
artifacts/gold_test08_q12200_deg5_ref_offline_acc_20260716T095752/
artifacts/gold_test08_q12200_deg5_srv_perf_audit_20260716T100434/
```

Verifier result:

```text
num_acc_log_entries = 349823
num_perf_log_entries = 4017
num_matched = 4017
num_unmatched = 0
num_ne_mismatch = 0
tolerance = 0.10%

TEST PASS
TEST08 verification complete
```

Audited Server leg:

```text
Result is : VALID
Completed samples per second    : 12198.91
99.00 percentile latency (ns)   : 57363775
99.90 percentile latency (ns)   : 61821250
```
