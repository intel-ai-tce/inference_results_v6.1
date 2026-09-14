# C1-off Profile — Per-Kernel GPU Breakdown (AMD TraceLens)

Profiled the **certified C1-off config** (b40, D2 off, last-layer target-only lever on,
full causal `DLRM_HSTU_MAX_ATTN_LEN=0`) at the 7,400 q/s offered load and analyzed the
PyTorch-profiler Chrome trace with **AMD TraceLens**
(`TraceLens_generate_perf_report_pytorch_inference`). Goal: see which part of the pipeline
— and which part of attention — actually dominates GPU time once the 1024 window is off.

## TL;DR

**Attention (`_hstu_attn_fwd`) is the C1-off bottleneck at 55.5% of GPU kernel time** — 3.7x
the next-largest bucket. Turning the window off makes attention quadratic over full history,
so it is *more* dominant than under the C1-on cert (~46%). The last-layer target-only lever
is visibly working: the final-layer (delta) attention is only 8.3% of attention time because
it computes target queries only.

## Run provenance

| | |
|---|---|
| artifact | `artifacts/Plan24_fullcausal_b40_PROFILE3_20260606T202621/` |
| trace (rank 0) | `torchprof/rank0_b30_skip120.json` (39 MB Chrome/kineto) |
| TraceLens report | `tracelens_csvs/` (+ `tracelens_inference.xlsx`) |
| config | C1 off (`MAX_ATTN_LEN=0`), D2 off (`FUSE_EPILOGUE=0`), lever on, b40, Server |
| achieved | 7,386 completed q/s (target 7,400); p50 50.0 ms, p97 63.1 ms |
| INVALID? | yes — **expected**: torch profiler on rank 0 inflates that worker's p99 to 100.7 ms. The trace's per-kernel GPU times are unaffected and representative. |
| sample | rank 0, 30 steady-state predict batches (skip 120), 1,009 ms total kernel time |

> **Launch note — a blocking bug in the single-flag lever (see below) means the documented
> runbook command crashes in warmup.** This run used the warmup-safe / certified equivalent
> `-e DLRM_HSTU_LASTLAYER_RETURN_TARGETS_ONLY=0` (gather/scatter-to-full-length lever path).

## Per-kernel GPU breakdown (C1 off, full causal, b40, lever on)

| bucket | ms (30 batches, rank0) | % of GPU kernel time |
|---|---:|---:|
| **ATTENTION (`_hstu_attn_fwd`)** | **559.9** | **55.5%** |
| Elementwise / SiLU / copy / cast | 165.2 | 16.4% |
| GEMM (hipBLASLt fp8 + `_scaled_mm`) | 152.1 | 15.1% |
| Jagged gather/scatter (`concat`/`split_2D_jagged`) | 62.1 | 6.2% |
| LayerNorm (`_weighted_layer_norm_fwd`, `_ln_mul_dropout_fwd`) | 41.2 | 4.1% |
| other (`query_uvm` embedding gather, misc) | 25.9 | 2.6% |
| reduce | 2.5 | 0.2% |
| **total** | **1009.0** | **100%** |

GPU-busy ≈ **33.6 ms/batch** on rank 0; busy 87%, idle 13% (TraceLens `gpu_timeline`).

## Within attention (the part the user cares about)

`_hstu_attn_fwd` is **one fused SiLU-gated flash-attention kernel** (two matmuls per KV block:
QK^T scores, then the SiLU-gated A·V context). The trace cannot split QK^T vs A·V inside the
fused kernel, but it cleanly splits the two launch sites:

| attention path | ms | calls | mean µs/call | share of attention |
|---|---:|---:|---:|---:|
| main STU layers (full causal, fused preprocess+attn) | 513.5 | 116 | 4,427 | 91.7% |
| last STU layer — **delta / targets-only (lever)** | 46.4 | 29 | 1,601 | 8.3% |

The lever shrinks the final layer ~2.8x (4,427 → 1,601 µs/call) by computing U/Q for target
rows only (K/V still over all rows). A full-causal final layer would cost ~the main-layer
rate (~128 ms over 30 batches); trimming it to ~46 ms is the ~11% attention saving that
produces the certified **+11.3% throughput** (6,650 → 7,400 q/s). Almost all remaining
attention cost is the O(L²) full-history score/context matmuls in the non-final layers —
exactly what the 1024 window (C1 on) caps.

## GEMM mix (15.1%, no single dominant GEMM)

Fragmented across hipBLASLt fp8 kernels (UVQK projection, output projection, MLP):
`F8F8S 256x256x128` 50.5 ms, `F8BS` variants 23.1 / 20.9 / 14.3 / 9.6 ms, `BBS`
(bf16) 21.6 / 6.3 ms. These are already on the optimized hipBLASLt path — consistent with
the Plan 28 NO-GO on a custom MXFP4 dense GEMM (hipBLAS beats Triton here).

## The lever bug found while launching this run

The documented single-flag command in `docs/full_causal_run.md`
(`-e DLRM_HSTU_LASTLAYER_TARGETS_ONLY=1` only) **crashes in warmup** with
`AssertionError: total_len_left + total_len_right == total_seq_len` in
`triton_split_2D_jagged`. Root cause is an inconsistency introduced by the 2026-06-06
single-flag formalization (`9dac3cb`):

- `stu.py` defaults `_LASTLAYER_RETURN_TARGETS_ONLY` to **follow the lever** (`"1"` when the
  lever is on) → `STULayer.forward_targets_only` returns *trimmed* target-only rows.
- `hstu_transducer.py` declares its **own** `_LASTLAYER_RETURN_TARGETS_ONLY` that hard-defaults
  to `"0"` and does **not** follow the lever → `_postprocess` still runs the embedding split
  on the already-trimmed tensor → row-count mismatch → assert.

The two module-level constants disagree, so the stack trims but the postprocessor splits.

**Workaround (used here, = certified gather/scatter path):** add
`-e DLRM_HSTU_LASTLAYER_RETURN_TARGETS_ONLY=0`. This makes `forward_targets_only` scatter
target outputs back to a full-length tensor, so the split is valid. This is the variant that
actually certified end-to-end at 7,400 q/s (the `-7.7%` target-only-return number in
`last_layer_targets_only.md` was a b64 steady-predict microbench, not a full harness run).

**Fixed (2026-06-06).** `hstu_transducer.py` now mirrors `stu.py`'s `_lastlayer_default` so
`RETURN_TARGETS_ONLY` follows the lever, plus a `seq_embeddings.size(0) == total_targets`
guard in `_postprocess` that falls back to the split if the rows are full-length (robust to
flag drift). The new branch is only reached when the lever is on, so the certified C1-on path
is byte-for-byte unchanged. The documented one-flag command now runs the skip-the-split
(`-7.7%`) path and **re-verified VALID at b40 7,400 q/s, p99 64.2 ms**
(`artifacts/Plan24_fullcausal_b40_FIXVERIFY_20260606T204423/`).

## Follow-up: buffer ops re-enabled on the 55.5% attention kernel (2026-06-07)

This profile flagged `_hstu_attn_fwd` as the C1-off bottleneck (55.5% of GPU). That kernel
ships with `ENABLE_BUFFER_OPS_ASSUMES` hints specifically to let Triton lower its masked K/V
loads to AMD `buffer_load`/`buffer_store` (hardware-bounded) — **but those passes were globally
disabled on gfx950** by `patch_triton_compiler.sh`, which blanket-skipped the
`canonicalize-pointers` / `convert-buffer-ops` passes because the basic `_concat_2D_jagged`
kernel (a 3-way `if/elif/else` over different base pointers) crashed them
(`TritonAMDGPUCanonicalizePointers` assertion → `PassManager::run failed`).

**Proper fix:** route the 2D-jagged concat/split to their mask-based `*_multirow` variants on
HIP (`triton_jagged_tensors.py::_prefer_multirow_concat_split`) — single base pointer, so they
compile cleanly with buffer ops **on** — and drop the compiler patch. Buffer ops are then ON for
every kernel, including `_hstu_attn_fwd`. Multirow concat/split verified **bit-exact**
(max-abs-err `0.000e+00`) vs the basic kernels; arithmetic is unchanged, so GAUC is neutral by
construction.

Re-cert with buffer ops on (same configs, same target qps):

| regime | config | baseline qps / p99 | buffer-ops qps / p99 | p99 delta |
|---|---|---:|---:|---:|
| C1-off | b40, lever on | 7,400 / 59.9 ms (64.2 re-verify) | **7,395 / 58.0 ms** | −1.9 to −6.2 ms |
| C1-on  | b48, D2 on | 10,595 / 51.07 ms | **10,595 / 45.9 ms** | **−5.2 ms (~10%)** |

Both VALID, qps target-bound by the conf, so at the original targets the win shows as **p99
headroom**. A b40 C1-off knee re-probe (600 s PROD, buffer ops on) converts that headroom into
throughput — the knee moves **7,400 → 7,600 q/s** (PROF90s over-reports p99 near the knee, so
these are full PROD certs):

| target (b40, lever, buffer ops) | result | p99 | p99.9 | note |
|---|---|---:|---:|---|
| 7,400 | VALID | 58.0 ms | 66.3 ms | old cert point (p99 59.9 ms without buffer ops) |
| 7,500 | VALID | 58.1 ms | 67.1 ms | **recommended** — p99 flat, comfortable margin |
| 7,600 | VALID | 65.7 ms | 143.7 ms | **max VALID** — tail steepening, riskier |

So buffer ops are worth **+200 q/s (+2.7%)** at the C1-off knee (7,400 → 7,600), or +100 q/s at
*better* p99 than the old 7,400 cert. Artifacts:
`Plan24_fullcausal_b40_bufops_20260607T004936/` (7,400),
`Plan24_fullcausal_b40_bufops_cert7500_20260607T020720/` (7,500),
`Plan24_fullcausal_b40_bufops_cert7600_20260607T022922/` (7,600),
`Plan24_c1on_b48_bufops_20260607T011144/` (C1-on b48).

## Re-profile with buffer ops ON — measured (2026-06-07)

The breakdown above is the **buffer-ops-OFF** PROFILE3 trace; the follow-up only *assumed*
attention stayed at 55.5%. Re-ran the **identical** profile but with buffer ops **ON**
(`AMDGCN_USE_BUFFER_OPS_GFX950=1`; multirow concat/split routing active) to measure it:
b40, C1-off, lever on (`MAX_ATTN_LEN=0`, `RETURN_TARGETS_ONLY=0`), qps7400 PROF90s, rank 0,
30 steady batches skip 120. Artifact
`artifacts/Plan24_fullcausal_b40_bufops_PROFILE_20260607T034455/` (trace
`torchprof/rank0_b30_skip120.json`, TraceLens CSVs in `tracelens_csvs/`). The multirow
concat/split kernels in the trace confirm buffer ops compiled cleanly (the basic
`_concat_2D_jagged` would have crashed the passes). **This profile run was VALID** —
7,386 q/s, p99 60.4 ms — the rank-0 profiler did not tip it over (PROFILE3 went INVALID).

Same bucketing as the table above, OLD (buffer ops off, PROFILE3) → NEW (buffer ops on):

| bucket | OLD ms (%) | NEW ms (%) | delta |
|---|---:|---:|---:|
| **ATTENTION (`_hstu_attn_fwd`)** | **559.9 (55.5%)** | **557.0 (56.6%)** | **−2.9 ms (−0.5%)** |
| GEMM (hipBLASLt fp8/bf16) | 152.2 (15.1%) | 152.5 (15.5%) | flat |
| Elementwise / SiLU / copy / cast | 114.6 (11.4%) | 119.9 (12.2%) | +5.3 |
| LayerNorm | 74.3 (7.4%) | 72.1 (7.3%) | −2.2 |
| Jagged concat/split + cat | 54.2 (5.4%) | 25.2 (2.6%) | **−29.0 ms (−53%)** |
| other (embedding gather, misc) | 50.6 (5.0%) | 53.4 (5.4%) | +2.8 |
| reduce | 3.1 (0.3%) | 3.1 (0.3%) | flat |
| **total kernel time** | **1009.0** | **983.2** | **−25.8 ms (−2.6%)** |

**Headline: under C1-off, buffer ops do _not_ measurably speed up the dominant `_hstu_attn_fwd`
kernel.** Its mean GPU time is flat (−0.5%: main layer 513.5→508.5 ms / −1.0%, last/delta layer
46.4→48.4 ms), and its *share rises* to 56.6% only because the total shrank. This fits C1-off
attention being **compute-bound** on the O(L²) score/context matmuls, where hardware-bounded
`buffer_load`/`buffer_store` (a memory-load optimization) has little to give.

The one clear per-kernel GPU win is the **2D-jagged concat/split → `*_multirow` routing**:
`_concat_2D_jagged` 33.4 → `concat_2D_jagged_multirow` 12.4 ms and `_split_2D_jagged` 12.2 →
`split_2D_jagged_multirow` 3.5 ms — −29 ms / −53% of that bucket, ~−2.6% of total kernel time.

Implication: the buffer-ops end-to-end wins (C1-on p99 −10%, C1-off knee +2.7%) are **not**
explained by a faster C1-off attention kernel in a steady-state window — under C1-off the gain is
the lighter concat/split path plus tail/overlap effects that mean per-kernel GPU time doesn't
capture. Buffer ops most likely help the **C1-on** windowed attention (memory-bound on K/V loads)
far more than C1-off full-causal attention; a C1-on re-profile would confirm.

## Hardware-counter microprofile of `_hstu_attn_fwd` — binding resource (2026-06-07)

To find *which* resource binds the 56.6%-of-GPU attention kernel (the fused trace can't split
QK^T vs SiLU vs A·V), profiled it in isolation with **rocprofv3** hardware counters. Built a
standalone replay microbench (`scripts/profile/attn_microbench.py`) that calls
`triton_hstu_attention_fwd` directly — **no checkpoint** — with the real C1-off shapes pulled from
the dataset + `configs.py`: H=4 heads, DimQ=DimV=128, fp8 (e4m3) attn, `max_attn_len=0`, batch
Z=40, per-sequence length = contextual(1) + history(uih ≈ 6.8k) + candidates(2048) ≈ **8.5k tokens**,
num_targets=2048. Microbench reproduces the kernel at **4.9 ms/call** vs the in-harness 4.43 ms/call
(faithful; the extra is the standalone fp8 pre-cast the fused prod path skips). `rocprof-compute`
(omniperf) is unusable here — missing ~12 Python deps in the container — so metrics are rocprofv3
derived counters (pre-normalized) + basis-robust counter ratios, over the steady (post-autotune)
dispatches. gfx950 = 256 CU / 1024 SIMD / 8 XCC.

Autotune winner (full grid): **BLOCK_M=128, num_warps=8 (512 threads), VGPR=60, AccVGPR=0,
LDS=0 B, scratch=0** — the HIP path streams K/V through `buffer_load` into registers, it uses **no
LDS at all**.

| metric | value | reading |
|---|---:|---|
| MemUnitStalled | **0.02%** | **not** memory-bound |
| L2 (TCC) hit rate | 63% | fine; HBM not the limiter |
| LDS bank conflicts | ~0 (no LDS) | irrelevant on this path |
| VALUBusy | **80%** | SIMD issue is busy most of the time |
| VALUUtilization | ~100% | no thread divergence; full 64-lane waves |
| MFMA-busy / VALU-active cyc | **0.95** | the matrix (MFMA) pipe is the dominant cycle consumer |
| VALU : MFMA instruction count | 17.9 : 1 | many short VALU ops, but MFMA cycles dominate (MFMA instrs are long) |
| Occupancy | 41% (13/32 waves/CU) | moderate; not the binding constraint (mem-stall ≈ 0) |

**Verdict: the C1-off attention kernel is compute-bound on the MFMA matrix pipe (~75–80% busy),
not memory-bound (mem-unit stall ≈ 0%).** This is the mechanistic reason buffer ops did nothing for
it — there is no memory stall to remove. The remaining ~20% of cycles are issue-idle: the
dependency chain QK^T → SiLU-gate/scale → A·V plus the in-kernel VALU epilogue (SiLU, scaling,
masking; `fast_expf`/`fast_dividef` already used), not fully hidden behind MFMA.

Levers that actually pay, in order:
1. **fp4 (MXFP4) QK/AV** — directly cuts the dominant MFMA cycles. Bounded to **<2×** on the kernel
   by the ~20% dependency stall + VALU overhead; accuracy-gated (GAUC sign-off).
2. **Hide the ~20% dependency stall** — raise occupancy (41% now; tune `waves_per_eu`, or smaller
   BLOCK_M to drop VGPR) and/or overlap the VALU epilogue with MFMA. Cheap, bit-exact (autotune).
3. **Trim VALU epilogue work** (the 17.9:1) — fewer casts/scale/mask ops per block.

Not worth it: memory/`buffer_load` tuning, LDS, larger blocks-for-bandwidth — the kernel is not
memory- or LDS-bound. Artifacts: `artifacts/attnprof_g{1,2,3}/` (rocprofv3 CSVs),
microbench `scripts/profile/attn_microbench.py`.

## Fine-grained re-analysis on the Win-B 64/64 kernel of record (2026-06-07)

The microprofile above was the pre-fastmask **128/32** autotune winner. After Plan 30 #3 shipped
(fastmask interior/boundary split) and Win-B re-cert moved the kernel of record to **64/64/4w/1s**,
re-analyzed `_hstu_attn_fwd` with three instruments: a config×fastmask microbench, a new
**phase-attribution probe** (`scripts/profile/attn_phase_probe.py`), and rocprofv3 counters on both
configs *fastmask-on*. (All on the standalone `triton_hstu_attention_fwd` launcher; the harness main
layers call the fused preprocess+attn variant, so absolute ms run ~5–10% above harness — the splits
and binding resource are properties of `_hstu_attn_fwd` itself.)

### 1. Mask-VALU cost + config parity (bare-kernel microbench, b40 C1-off shapes)

| config | fastmask ON | OFF | mask-VALU saved |
|---|---:|---:|---:|
| **64/64/4w/1s** (Win-B) | **4.697 ms** | 5.773 | 1.076 ms (**18.6%**) |
| 128/32/8w/1s (Win-A) | 4.677 ms | 5.566 | 0.889 ms (16.0%) |

- The interior mask VALU is **~16–19% of the kernel** — that is exactly what fastmask removes on
  interior blocks (the #3 win), now measured in isolation.
- **Bare-kernel `_hstu_attn_fwd` is a tie: 64/64 ≈ 128/32 (4.70 vs 4.68 ms).** So Win-B's harness
  1.20× is **not** a faster steady-state attention kernel — in isolation the 64/64 tile only matches
  128/32; the harness edge comes from the fused-preprocess/launch path, not the attention math.

### 2. Per-phase split (phase probe, 64/64/1s, interior block)

| phase (isolated) | ms | reading |
|---|---:|---|
| QK^T | 0.558 | fp8 dot + K load |
| A·V | 0.764 | fp8 dot + V load (≈ QK^T, equal MACs ✓) |
| SiLU gate | 0.244 | `fast_dividef/fast_expf` VALU |
| **fusion stall** | **0.942** | full(2.508) − QK^T − A·V − SiLU |

- QK^T ≈ A·V confirms the two matmuls are balanced. The **largest single bucket (~38%) is the fusion
  stall** — at `num_stages=1` the QK^T→SiLU→cast→A·V chain runs serially (no software pipeline) plus
  the gated→fp8 cast. This is the quantified, larger-than-thought version of the prior "~20% issue
  idle." (Methodology caveat: cumulative phase-differencing is unreliable — 128/32 gave a negative
  A·V — so the *isolated* per-phase + full-kernel read above is the trustworthy one.)

### 3. rocprofv3 binding resource — 64/64 vs 128/32, both fastmask-on

| metric | **64/64/4w** (Win-B) | 128/32/8w (Win-A) |
|---|---:|---:|
| OccupancyPercent | **23.3%** (7.45 w/CU) | 44.9% (14.4 w/CU) |
| VGPR | **88** | 56 |
| MemUnitStalled | 0.02% | 0.02% |
| VALUBusy | 56% | 54% |
| VALUUtilization | 99.7% | 100% |
| TCC hit | 88% | — |
| VALU:MFMA instr | 11.5:1 | (17.9:1 fastmask-off) |

- Neither config is memory- or LDS-bound (MemUnitStalled 0.02%, no LDS) — same as before.
- **The 64/64 kernel of record runs at half the occupancy of 128/32 (23% vs 45%), because its VGPR
  is much higher (88 vs 56)** — yet it ties on time, so the bigger MFMA tile compensates. But 23%
  occupancy is too low to hide the QK^T→A·V dependency latency, which is *why* the probe sees ~38%
  stall. fastmask also cut the instruction mix 17.9→11.5 VALU:MFMA (mask VALU gone on interior).

### Verdict — attention is at its bit-exact floor

Composition of the interior block: **QK^T 22% + A·V 30% (= 52% MFMA) + SiLU 10% + ~38% dependency
stall.** The remaining bit-exact headroom is all in the stall, gated by the
occupancy↔VGPR↔pipelining tension:
- `num_stages=2` would overlap the chain but raises VGPR → occupancy cliff (the #2-killed effect),
- 64/64 already trades occupancy (23%) for tile size.

Levers, honestly ranked:
1. **Cut VGPR on 64/64 (88 is high) to lift occupancy** and hide the stall — worth re-opening #2
   *specifically for 64/64*, whose VGPR/occupancy profile differs from the killed 128/32 study.
   Bit-exact (autotune `waves_per_eu` / register limit). Ceiling ≤~10%.
2. **Shrink the gated→fp8 cast / epilogue** folded into the 38% stall. Bit-exact, small.
3. Cutting the 52% MFMA itself needs lower precision (**fp4**) — KILLED (Plan 27, unfusable MXFP4).

⇒ Pure-kernel attention gains left are **≤~10% and fight the occupancy cliff**; the kernel is
effectively at its bit-exact floor. The larger C1-off throughput levers remain **algorithmic**
(window / targets-only, Plan 29), not attention-kernel arithmetic. Artifacts:
`artifacts/attnprof64_g{1,2,3}/`, `artifacts/attnprof128_g2/`; probes
`scripts/profile/attn_phase_probe.py`, `attn_microbench.py`.

## Occupancy chase + a baseline correction (2026-06-07, Plan 30 #4)

Acting on lever #1 above ("cut VGPR / lift occupancy on 64/64"). Two findings, the first a
**correction to the section above**:

### Correction: the Win-B kernel of record is 128/64, not 64/64

The section above pins **64/64/4w/1s** as "the Win-B kernel of record" (VGPR 88, 23% occ). But
the **real autotune does not select that config**. Dumped the actual full-grid (FULLGRID) pick on
the real b40 C1-off shape (`DUMP_BEST=1`): autotune chooses **BLOCK_M=128 / BLOCK_N=64 / 8w / 1s /
waves_per_eu=0** → 3.786 ms. rocprofv3 on *that* config: **VGPR 64, OccupancyPercent 40.2%**,
VALUBusy 59.6% — i.e. production Win-B already runs at ~40% occupancy, the same neighborhood as
128/32, **not** the 23% "occupancy cliff." The 23%/VGPR-88 numbers describe the *pinned*
BLOCK_M=64 tile, which autotune passes over. So the "64/64 runs at half occupancy / fights the
cliff" framing was measuring a config that never ships. (BLOCK_M is bit-exact-preserving, so this
is purely a perf/occupancy correction, not a bit-correctness one.)

### The grid never explored occupancy at all — waves_per_eu was hardcoded 0

`_get_fw_configs()` builds the HIP fwd autotune grid with **`waves_per_eu: 0` hardcoded** (it
sweeps BLOCK_M/BLOCK_N/num_stages/num_warps/matrix_instr but never the occupancy hint). So
autotune could never trade registers for occupancy, nor reach a pipelined `num_stages=2`
configuration without spilling. Added an opt-in lever **`DLRM_HSTU_ATTN_OCCTUNE=1`** (default off ⇒
grid + selection unchanged) that adds `waves_per_eu ∈ {0,3,4}` to the grid. `waves_per_eu` only
bounds register allocation, and num_warps/num_stages re-partition independent work — all three are
**bit-exact-preserving** for fixed BLOCK_N/matrix_instr/kpack (verified `torch.equal`, max|Δ|=0,
`scripts/profile/attn_waves_ab.py`), so OCCTUNE composes with Win A (no re-cert) and Win B
(already re-certified) with **no new GAUC**.

### Autotune-vs-autotune (real grid, real b40 shape, microbench)

With OCCTUNE on, autotune *does* pick a `waves_per_eu>0` config, and it is faster:

| path | OCCTUNE off pick | OCCTUNE on pick | ms off→on | bits |
|---|---|---|---:|---|
| **Win-B** (BLOCK_N=64) | 128/64/8w/1s/we0 = 3.786 | **128/64/8w/2s/we4 = 3.584** | **−5.3%** | == Win-B (no re-cert) |
| **Win-A** (BLOCK_N=32, cert) | 128/32/8w/2s/we0 = 4.138 | **128/32/4w/2s/we3 = 3.778** | **−8.7%** | == original cert (**free**) |

**Mechanism — it is a pipelining win, not an occupancy win.** rocprofv3 on the Win-B pair (both
128/64): off 8w/1s/we0 = VGPR 64 / 40.2% occ / VALUBusy 59.6%; on 8w/2s/we4 = VGPR 64 / 37.5% occ /
**VALUBusy 66.2%**. Occupancy actually *drops* slightly; the gain comes from `num_stages=2`
software-pipelining the QK^T→A·V chain (the ~38% fusion stall), which raises sustained VALU/MFMA
overlap — and `waves_per_eu=4` is what *holds VGPR at 64* so stages=2 doesn't spill. This is the
real, correctly-attributed version of the "hide the dependency stall" lever (the earlier "−21% from
waves_per_eu" was an artifact of comparing to a non-production 64/64/4w/1s baseline; against the
*actual* autotune pick the kernel win is ~5–9%).

Notably, with OCCTUNE the **free cert-bit path (Win-A, 3.778 ms) nearly catches re-cert Win-B
(3.584 ms)** — most of Win-B's attention edge is recoverable with zero GAUC.

Patch: `triton_hstu_attention.py` (`_fw_waves_per_eu()` + grid loop, `DLRM_HSTU_ATTN_OCCTUNE`).
Sweep/AB tools: `scripts/profile/attn_occ_sweep.py`, `attn_waves_ab.py`. rocprof artifacts:
`artifacts/attnocc_*`.

### End-to-end harness A/B (b64, Server, PROF90s) — paired Win-B vs Win-B+OCCTUNE

Paired same-session A/B at b64 (the latency-bound knee, where the kernel win matters most), q9000
and q8500, OCCTUNE then baseline:

| b64 point | Win-B baseline | Win-B + OCCTUNE | verdict |
|---|---|---|---|
| q8500 | VALID — p99 77.34, p99.9 83.82 | VALID — p99 76.88, p99.9 110.12 | both pass (within capacity) |
| q9000 | **INVALID — p99 529.7 (backlog collapse)** | **VALID — p99 77.37, p99.9 181.25** | **OCCTUNE flips it** |

**OCCTUNE lifts the C1-off b64 knee from ~8,500 to ~9,000 q/s, bit-exact (no re-cert).** At q8500
both kernels are inside capacity, so the kernel edge is just a small p99 delta; at q9000 the
baseline is past its knee and the queue explodes (p99 530 ms), while OCCTUNE still drains under the
80 ms bound. This is the ~5% kernel pipelining win cashing out as a knee move at the batch where
per-step latency binds. **Caveat:** PROF90s near the knee is noisy (an earlier sweep had the Win-B
q9000 at p99 84.9 vs 529.7 here — both INVALID, but the spread is large); confirm 9,000 with a 600 s
PROD cert before quoting it as a hard figure. Artifacts: `Plan24_winBoccsweep_b64_q{9000,8500}_*`,
`Plan24_winBsweep_b64_q{9000,8500}_*`; TSVs `results/fullcausal_c1off/winb_batch_sweep_2026060716*.tsv`.

## Re-diagnosis on the live GOLD kernel — VALU lever closed at the floor (2026-06-16)

Revisited the "is there a VALU lever left?" question directly on the **shipping GOLD kernel**
(`run_gold.sh`: `FASTMASK=1 FULLGRID=1 OCCTUNE=1 BUFFER_OPS=1`), real C1-off shape Z=40,
L=341,100, N=9,702, targets≈2048. Autotune winner = **`BLOCK_M=128/BLOCK_N=64/8w/num_stages=2/
waves_per_eu=4/VGPR=64`**, **3.67 ms/call** standalone (vs 4.9 ms pre-fastmask/occtune ⇒ ~25%
already banked).

### rocprofv3 counters (`_hstu_attn_fwd`, GOLD config, fastmask on)

| metric | value | reading |
|---|---:|---|
| **VALUBusy** | **66.4%** | busiest pipe — binding resource |
| **MfmaUtil** | **28.0%** | matrix pipe <½ busy |
| VALUUtilization | 99.6% | no divergence, full 64-lane waves |
| MemUnitStalled | 0.04% | **not** memory-bound |
| OccupancyPercent | 37.8% (12.1 w/CU) | moderate |
| VGPR | 64 | occtune holds it so s2 doesn't spill |
| VALU : MFMA instr | **9.16 : 1** | many short VALU per long MFMA |
| of VALU: `exp`(TRANS_F32) | **17.5%** | already `fast_expf` |
| of VALU: casts (CVT) | 4.5% | q/k/v/gated fp8 casts |
| of VALU: **other** | **78.0%** | irreducible HSTU gating/scale/index math |

**Reconciliation of the two earlier verdicts:** the 2026-06-07 "MFMA-bound" read was a different/
earlier config; at the *actual GOLD config* VALU (66%) is ~2.4× more utilized than MFMA (28%), so
the new plan's "VALU-bound" framing is the correct one. But neither pipe saturates and mem-stall≈0
with occ 38% ⇒ **~34% of cycles are dependency/latency stall** (the QK^T→SiLU→A·V chain), not raw
VALU throughput. Counters: `route_a_probe/p1_pmc/`, parser `route_a_probe/parse_attn_pmc.py`,
counter file `route_a_probe/pmc_attn.txt`.

### Residual bit-exact sub-levers — both measured dead

1. **`num_stages=3`** (bit-exact A/B, `attn_waves_ab.py`, CFG_A=`128,64,8,2,4` vs CFG_B=
   `128,64,8,3,4`): **−5.9%** (3.693→3.913 ms/call), max|Δ|=0 / `torch.equal`=True. The 3rd stage
   pushes register pressure past `waves_per_eu=4`@VGPR64 → spill/occupancy drop. **GOLD s2 optimal.**
2. **Polynomial (no-`exp`) SiLU** (`attn_silu_cost.py`, isolated qk→silu loop, GOLD tile): exact
   `fast_expf` SiLU = 0.488 ms vs no-exp fast-sigmoid `qk·(0.5+0.5·qk/(1+|qk|))` = 0.378 ms ⇒
   **saves 0.110 ms = ≤3% of the 3.67 ms kernel as an absolute upper bound** (~1–2% realistic after
   MFMA overlap / the 34% stall absorbs freed VALU). And the poly that achieves it already has
   **cos 0.9993 but density-weighted activation rel-err 12.7%, max|Δ| 0.48** — GAUC-unsafe; a
   GAUC-safe minimax poly needs more terms and eats the win. The transcendental is cheaper in
   *cycles* than its 17.5% instruction share suggests (gfx950 SFU).

### Verdict — attention-kernel arithmetic is exhausted

~25% banked (fastmask+occtune); residual ≤~5–10% and GAUC-gated; both concrete sub-levers are
net-negative or accuracy-breaking. **The kernel is at its bit-exact floor.** Remaining C1-off
throughput must come from doing **less attention work algorithmically** (the O(L²) full-causal
score/context is the cost): the **1024 window (C1-on)** caps history to a fixed span, and the
**last-layer targets-only lever** trims the final layer to candidate rows — both already-proven
levers (Plan 29 / cert), not kernel-arithmetic. fp4 (cut the 28% MFMA) is KILLED (Plan 27,
unfusable MXFP4).
