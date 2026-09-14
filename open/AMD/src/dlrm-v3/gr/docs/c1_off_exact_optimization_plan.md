# C1-Off Exact Optimization Plan

## Context

The 1024 sliding window (`DLRM_HSTU_MAX_ATTN_LEN=1024`) is the largest performance lever
in the current DLRM-v3 stack, but it may carry MLPerf Closed-division compliance risk.
This plan tracks exact, accuracy-neutral optimizations that can help when C1 is disabled
(`DLRM_HSTU_MAX_ATTN_LEN=0`).

The validated baseline for this round is the last-layer target-only lever:

- Initial target-only path: predict improves from 64.19 ms to 60.13 ms at b64 with C1 off (`-6.3%`).
- Best follow-up path: split UQ/KV plus target-only return improves predict to 59.26 ms (`-7.7%`).
- AccuracyOnly Offline GAUC is neutral: 0.78628718 best-path on vs 0.78628724 off.

## Certified C1-off outcome (2026-06-06)

End-to-end Server characterization + certification of the C1-off envelope (full sweep:
`dlrm-v3-rocm-runner/results/fullcausal_c1off/`):

- **b64 full causal is infeasible** under the 80 ms p99 bound (p99 ~90 ms at *every* offered
  load — compute/tail-bound, no VALID knee). C1 off forces a small batch.
- The `-7.7%` lever (measured at the now-known-infeasible b64) was **re-measured at the feasible
  batch and formalized into a single flag** (`DLRM_HSTU_LASTLAYER_TARGETS_ONLY=1`, see
  `last_layer_targets_only.md`). At b24 it lifts certified Server 6,650 -> 6,800 q/s (+2.3%) and
  drops p99 72.5 -> 43.0 ms (it crushes the long-history tail).
- With the tail tamed, the throughput-optimal batch moves up; **b40 + lever certifies at
  7,400 q/s VALID, p99 59.9 ms** (600 s PROD) = **+11.3%** over the un-levered baseline.
- **Recommended shippable C1-off config:** b40, D2 off, lever on, 7,400 q/s. The 1024 window
  (C1 on) is still worth ~1.6x (12,000 q/s b64), so the lever is a stackable exact top-up, not
  a C1 replacement.

## Principles

- Exact output semantics for returned candidate scores.
- Env-gated and default-off unless promoted into the certified stack.
- Prefer removing unused predict-path work over approximating attention.
- Validate each lever with both predict timing and AccuracyOnly GAUC.

## Candidate Directions

1. **Return target-only final embeddings**
   - Current lever scatters candidate outputs back into a full-length tensor because
     `_postprocess` expects `[history || candidates]`.
   - A deeper integration could pass candidate-only final outputs into `_postprocess`
     and eliminate the final scatter.
   - Status: implemented. This was the main follow-up win, moving from `-6.6%` to `-7.7%`.

2. **Partial final-layer UVQK projection**
   - The final layer does not need history `U/Q/output`; history rows are only needed
     as K/V context for candidate queries.
   - Target-aware mask check shows candidate queries attend to history plus their own
     candidate row only; they do not attend to other candidates.
   - Therefore candidate self K/V is still required, but history U/Q can be skipped.
   - Status: implemented as split UQ/KV. Payoff was modest: `-6.3%` to `-6.6%`.

3. **Dedicated fused target-only final-layer path**
   - Current implementation composes existing fp8 UVQK, target gather, delta attention,
     output projection, and scatter.
   - A dedicated path could reduce launches/materialization around gather and delta setup.
   - Expected payoff: medium, more engineering risk.

4. **Target-aware mask exploitation (active gate)**
   - Result: candidate queries attend to history plus their own candidate row only.
   - Candidate-to-other-candidate K/V is masked, but candidate self K/V remains live.
   - This supports split UQ/KV, not dropping candidate K/V entirely.

5. **Shape specialization**
   - Inference has fixed 2048 candidates and stable b64 serving shapes.
   - Precomputed target-row indices, reduced dynamic shape work, or graph-like launch
     capture may recover small exact overheads.

## Completed Gate: Target-Aware Mask

Questions to answer:

- For each candidate query in the target region, are key positions in the target region
  masked out?
- Does the answer differ between the regular full HSTU attention kernel and the delta
  query path used by `forward_targets_only`?
- Is the mask controlled solely by `num_targets`, or by `contextual_seq_len` /
  target-aware flags as well?

Expected decision:

- Implement split final-layer projection: all-row V/K plus target-row U/Q.
- Keep candidate self K/V because each target query attends to its own diagonal row.

## Next Open Directions

1. **Fuse or specialize the split UQ/KV path**
   - The split projection saved little because two narrower GEMMs and packed-weight
     overhead offset most of the skipped history U/Q work.
   - A custom fused projection or a persistent packed-weight cache may recover more.

2. **Remove more `_postprocess` overhead**
   - Target-only embeddings now skip the embedding split, but timestamp splitting and
     interleave handling still run through generic jagged helpers.
   - Specialized fixed-2048 target handling may remove more exact overhead.

3. **C1-on stacked measurement**
   - The best path is validated with C1 off. Measure with C1 on to quantify the exact
     contribution on the shipping stack.

4. **Dedicated final-layer target-only kernel**
   - A single path that combines target gather, delta attention setup, and output
     handoff could reduce launch/materialization overhead beyond the composed ops.
