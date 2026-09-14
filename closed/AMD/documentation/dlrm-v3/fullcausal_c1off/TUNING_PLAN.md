# Full-Causal (C1-off) Tuning & Certification Plan

Goal: find and certify the **best shippable full-causal (C1-off) Server point** on MI355X,
combining the last-layer target-only lever with batch-size tuning. Started 2026-06-05,
node `chi2835`, container `dlrmv3-e2e723`.

> **SUPERSEDED (2026-06-08):** this plan tuned the *slower* fp8 attention kernel and landed at
> b40 ~7,400–7,600 q/s. The **Win-B** attention kernel later lifted full-causal to **b64 @ 9,595
> q/s VALID (p99 78.0 ms, 600 s PROD)** — the current C1-off figure of record. See the Win-B
> section in `../../HANDOFF_full_causal_optimization.md` and `winb_batch_sweep_*.tsv`.

## Baseline going in
- b64 full causal: **infeasible** (p99 ~90 ms at all loads, compute/tail-bound).
- b24 full causal, D2-on, lever-off: **max VALID ~6,650 q/s** (p99 62.7 ms). [`../fullcausal_c1off/`]
- C1-on figure of record: 11,994 q/s (b64) / 10,595 (b48 cert) — the window is worth ~1.8x.

## Key constraint
The last-layer target-only lever requires `DLRM_HSTU_FUSE_EPILOGUE=0` (D2 **off**, SiLU applied
once). The 6,650 baseline used D2 **on**. So enabling the lever forfeits D2's fusion win — the
experiment must show the lever's gain **net of** the D2 loss.

Lever-on env (the -7.7% predict combo, accuracy-validated):
`DLRM_HSTU_MAX_ATTN_LEN=0 DLRM_HSTU_LASTLAYER_TARGETS_ONLY=1 DLRM_HSTU_LASTLAYER_FP8_UVQK=1
DLRM_HSTU_LASTLAYER_SPLIT_UQKV=1 DLRM_HSTU_LASTLAYER_RETURN_TARGETS_ONLY=1` with `FUSE_EPILOGUE=0`.

## Steps

### Step 1 — re-measure the lever at b24 (the feasible batch)
The -7.7% lever was measured at b64 C1-off, which is infeasible. Re-measure at b24:
- **1a** b24, D2-off, lever-**off**, C1-off — control to isolate D2's cost vs the 6,650 (D2-on) point.
- **1b** b24, D2-off, lever-**on**, C1-off — push QPS up from 6,650 to find new max VALID.
- Decision: does lever-on max VALID beat the 6,650 (D2-on lever-off) point? Log p99 + verdict per probe.

### Step 2 — certify the Step-1 winner
Run the best Step-1 config at its max-VALID QPS as a **600 s PROD** run (certifiable number,
not a 90 s probe). Log summary + detail.

### Step 3 — tune batch size
With the Step-1 winning lever config (D2 off, lever on), sweep batch **{16, 20, 32}** to find
the throughput-optimal full-causal batch (b24 was only the reference anchor). Find max VALID per batch.

### Step 4 — certify the Step-3 winner
600 s PROD run at the best (batch, QPS) from Step 3. This is the final shippable full-causal number.

## Methodology
PROF90s probes (90 s) for sweeps; PROD600s (600 s) for certification. Server, 80 ms p99 bound,
8 workers, A-FUSE fp8, `sort_by_length=True`. Verdict = `Result is VALID` AND p99 <= 80 ms.
Near the sharp knee, PROF probes are noisy — certify before quoting.

## Results log
(filled in as steps complete)

### Step 1 — DONE (2026-06-06). Lever is a net win: max VALID 6,650 -> 6,800 (+2.3%).
| step | batch | D2 | lever | offered | completed | p99 (ms) | result |
|---|---:|:--|:--|---:|---:|---:|:--|
| ref | 24 | on | off | 6650 | 6642 | 62.7 | VALID (prior baseline) |
| 1a | 24 | off | off | 6650 | 6642 | 72.5 | VALID (D2-off costs ~10ms tail) |
| 1b | 24 | off | on | 6650 | 6642 | **43.0** | VALID (lever crushes the tail) |
| 1b | 24 | off | on | 6800 | 6791 | 61.8 | **VALID (max reliable)** |
| 1b | 24 | off | on | 6850 | 6842 | 79.9 | INVALID (p99 OK; early-stopping NO, ~16k short) |
| 1b | 24 | off | on | 7000 | 6992 | 152.7 | INVALID |
| 1b | 24 | off | on | 7200 | 7190 | 549.0 | INVALID |
| 1b | 24 | off | on | 7500 | 7452 | 1344.6 | INVALID (collapse) |

Findings:
- The -7.7% predict lever, re-measured at the feasible b24 batch, lifts max VALID Server from
  6,650 (D2-on, lever-off) to **6,800 q/s (D2-off, lever-on)** = **+2.3%** shippable, despite
  forfeiting D2's epilogue fusion.
- Its bigger effect is latency headroom: at a fixed 6,650 q/s it drops p99 72.5 -> 43.0 ms
  (D2-off) — it especially crushes the long-history tail (final-layer history attention skipped),
  which is what sets p99.
- Near the knee the raw throughput ceiling rose (kept up to 7,451 completed), but the p99 tail
  now binds at ~6,850. 6,850 fails only on early-stopping (16k queries short at p99=79.9) — a
  600 s run may clear it, but it is too close to the bound to bank on; 6,800 (p99 61.8) is the
  safe certifiable max.

### Step 2 (cert) — DONE. 600 s PROD @ b24 D2-off lever-on, 6,800 q/s = **VALID**.
- Completed **6,795 q/s**, constraints **Yes**, early-stopping **Yes**.
- Latency: p50 30.9 / p90 35.1 / p95 37.3 / **p99 43.0** / p99.9 61.2 ms.
- Note: certified p99 (43.0 ms) is far under the 90 s probe (61.8 ms) — short probes over-report
  p99 near the knee (warmup transients). 6,800 is a **conservative** certified floor with ~37 ms
  of headroom; the true stable-queue ceiling is higher (90 s probes collapse at 7,000+, but those
  are noisy). [`cert_b24_d2off_leverON_q6800_VALID_*`]

### Step 3 (batch sweep) — DONE. Optimal batch is LARGER: b40 wins (max VALID 7,400).
All rows: D2-off, lever-on, C1-off. 90 s PROF probes (over-report p99 near the knee).
| batch | offered | completed | p50 | p90 | p99 (ms) | result |
|---:|---:|---:|---:|---:|---:|:--|
| 24 | 6800 | 6791 | 31.4 | 36.9 | 61.8 | VALID (cert: p99 43.0) |
| 32 | 7000 | 6991 | 40.9 | 45.5 | 52.1 | VALID |
| 32 | 7250 | 7238 | 41.4 | 48.1 | 64.1 | VALID (b32 max) |
| 32 | 7350 | 7337 | 41.7 | 50.2 | 84.0 | INVALID |
| 40 | 7400 | 7386 | 49.1 | 54.7 | **64.8** | **VALID (best overall)** |
| 40 | 7500 | 7482 | 49.3 | 55.4 | 131.4 | INVALID |
| 40 | 7600 | 7580 | — | — | 96.0 | INVALID |

Findings:
- Max VALID climbs with batch: b24 6,800 -> b32 7,250 -> **b40 7,400**. Mechanism: each batch has
  ~fixed per-batch overhead (collate ~3 ms + dispatch/ZMQ); larger batch amortizes it, raising the
  effective service rate. (Raw compute rate ~constant since predict scales ~linearly with batch.)
- The cost is a rising p99 floor (min latency b24 24 ms -> b32 31 ms -> b40 37 ms). The optimum is
  the largest batch whose p99-at-its-knee still clears 80 ms. Gains diminish (+450 then +150) and
  b40's knee is sharp (7400 p99 65 -> 7500 collapse), so b48 would be marginal and riskier.
- b16/b20 were skipped: they sit below b24, which the monotonic trend guarantees is worse.

### Step 4 (cert) — DONE. 600 s PROD @ b40 D2-off lever-on, 7,400 q/s = **VALID** (final).
- Completed **7,395 q/s**, constraints **Yes**, early-stopping **Yes**.
- Latency: p50 48.1 / p90 52.7 / p95 54.5 / **p99 59.9** / p99.9 85.4 ms.
  [`cert_b40_d2off_leverON_q7400_VALID_*`]

## Final outcome
| config | certified VALID Server | p99 | delta |
|---|---:|---:|---|
| Full causal, b24, D2-on, lever-off (start) | ~6,650 (90 s probe) | 62.7 | baseline |
| Full causal, b24, D2-off, **lever-on** (Step 2 cert) | **6,800** | 43.0 | +2.3% (lever) |
| Full causal, **b40**, D2-off, **lever-on** (Step 4 cert) | **7,400** | 59.9 | **+11.3% total** |

The lever (+2.3%) and batch retune b24->b40 (+8.8%) together lift the **certified** full-causal
Server ceiling from ~6,650 to **7,400 q/s** (+11.3%). Versus the C1-on window (12,000 q/s b64 /
10,595 cert b48), the 1024 window is still worth ~1.6x — but full causal is now meaningfully
less far behind. Recommended shippable full-causal config: **b40, D2-off, lever-on, 7,400 q/s**.
