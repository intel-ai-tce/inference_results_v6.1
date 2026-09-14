# Last-Layer Target-Only Lever

## Summary

`DLRM_HSTU_LASTLAYER_TARGETS_ONLY=1` enables an exact inference-only optimization for
DLRM-v3 HSTU inference. The model computes all STU layers over `[history || candidates]`,
but `_postprocess` keeps only candidate embeddings and discards history rows. Therefore,
on the final STU layer, history-token outputs are unused.

The lever computes final-layer attention and output projection only for candidate
queries, while still computing the K/V context required for those candidate outputs.
The base version returns candidate rows in the original full-length layout with zeroed
history rows. The follow-up target-only return path lets `_postprocess` consume the
candidate rows directly, avoiding the full-length scatter and immediate embedding split.

## Gates (formalized 2026-06-06)

The validated follow-ups were promoted: a single flag now selects the certified best path.

- `DLRM_HSTU_LASTLAYER_TARGETS_ONLY=1`: **the single switch.** Opt-in, default off (off
  preserves the certified baseline path byte-for-byte). When on, the three follow-ups below
  **default on** (they are accuracy-validated and part of the certified full-causal config).
- `DLRM_HSTU_LASTLAYER_FP8_UVQK`: keep the final-layer UVQK projection on the fp8 path.
  Default = follows the lever (on when the lever is on).
- `DLRM_HSTU_LASTLAYER_SPLIT_UQKV`: compute final-layer U/Q only for target rows, V/K for all
  rows. Default = follows the lever.
- `DLRM_HSTU_LASTLAYER_RETURN_TARGETS_ONLY`: return target rows directly and let `_postprocess`
  skip the embedding split. Default = follows the lever.
- Each follow-up is still individually overridable to `0` for A/B (e.g.
  `DLRM_HSTU_LASTLAYER_TARGETS_ONLY=1 DLRM_HSTU_LASTLAYER_SPLIT_UQKV=0`).
- The stack also requires inference mode, uniform `num_targets`, no KV-cache prefill, and
  `DLRM_HSTU_FUSE_EPILOGUE=0` (D2 off) so SiLU is applied exactly once.

## Implementation

- `generative_recommenders/modules/stu.py`
  - Adds `STULayer.forward_targets_only`.
  - Adds `STUStack.forward` gating for the final layer only.
  - Uses direct target-row gather/scatter instead of jagged split/concat, avoiding copies of discarded history rows.
  - Optionally splits the final-layer projection into target-row U/Q and all-row V/K.
- `generative_recommenders/modules/hstu_transducer.py`
  - Optionally treats final-layer output as already target-only and skips the embedding split.
- `generative_recommenders/ops/triton/triton_hstu_preprocess_and_attention.py`
  - Adds `compute_uqvk_for_delta`, a fp8-preserving UVQK helper that mirrors the fused A-FUSE projection path.
  - Adds `compute_split_uqkv_for_delta`, which packs split U/Q and V/K projection weights for the follow-up path.

## Measurements

Environment: b64, C1 off (`DLRM_HSTU_MAX_ATTN_LEN=0`), D2 off
(`DLRM_HSTU_FUSE_EPILOGUE=0`), fp8 UVQK enabled.

| config | steady predict | delta |
| --- | ---: | ---: |
| lever off | 64.19 ms | baseline |
| lever on, jagged split/concat | 61.78 ms | -3.8% |
| lever on, gather/scatter trim | 60.13 ms | -6.3% |
| lever on, split UQ/KV | 59.98 ms | -6.6% |
| lever on, split UQ/KV + target-only return | 59.26 ms | -7.7% |

AccuracyOnly Offline GAUC over 349,823 samples:

| config | lifetime GAUC | lifetime NE |
| --- | ---: | ---: |
| lever off | 0.78628724 | 0.86772957 |
| lever on, gather/scatter trim | 0.78628766 | 0.86773077 |
| lever on, split UQ/KV + target-only return | 0.78628718 | 0.86772950 |

The GAUC deltas are within about `5e-7`, which is accuracy-neutral at this scale and
comfortably passes the MLPerf 99.9%-of-reference accuracy bar.

## Interpretation

This is a real exact C1-off predict win, but it is not a replacement for C1. With C1 off,
predict is still roughly 64 ms before this lever, while C1-on predict is roughly 28 ms in
the optimized stack. The best measured variant recovers about 14% of C1's compute win
and should be treated as a stackable exact top-up.

## Certified at the feasible batch (2026-06-06)

The `-7.7%` above was measured at **b64**, but b64 is infeasible with C1 off (p99 ~90 ms at
every load — the long-history tail never clears the 80 ms bound). Re-measured and **certified
with 600 s PROD Server runs** at the feasible batch:

| config (C1 off, D2 off) | certified Server | p99 |
| --- | ---: | ---: |
| b24, lever off (baseline) | ~6,650 q/s | 62.7 ms |
| b24, **lever on** | 6,800 q/s | 43.0 ms |
| **b40, lever on** (best) | **7,400 q/s** | 59.9 ms |

The lever adds **+2.3%** VALID throughput at b24 (6,650 -> 6,800) and, more importantly, large
latency headroom (it crushes the long-history tail: p99 72.5 -> 43.0 ms at a fixed 6,650 q/s,
D2 off). It also unlocks a larger batch: with the tail tamed, the throughput-optimal batch
moves to b40, certifying **7,400 q/s (+11.3% over the un-levered baseline)**. The 1024 window
is still worth ~1.6x (12,000 q/s b64 C1-on), so this is a stackable exact top-up, not a C1
replacement. Full sweep + certs:
`dlrm-v3-rocm-runner/results/fullcausal_c1off/TUNING_PLAN.md`.
