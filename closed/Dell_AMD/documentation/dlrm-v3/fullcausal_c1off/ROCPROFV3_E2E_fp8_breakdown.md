# C1-off fp8 — End-to-End rocprofv3 Kernel Breakdown (2026-06-13)

fp4 is **parked** (the MXFP4 route is blocked by an unfixed host-kernel `amdttm` LRU-eviction
NULL-deref under memory pressure — see `docs/amd_bug_amdttm_lru_null_deref.md`). This re-profiles
the **fp8** C1-off path end-to-end with **rocprofv3 `--kernel-trace`** to find where the *next*
optimization lever is, now that the big-ticket fp4 arithmetic cut is off the table.

## Method

- Config: **C1-off** (`MAX_ATTN_LEN=0`), **b40**, D2 off, last-layer targets-only lever **on**,
  fp8 attn+GEMM, buffer ops on (the `*_multirow` jagged routing) — the certified C1-off stack.
  GR tree on `full_causal_optimization` with **all `DLRM_HSTU_FP4_*` flags off** (= fp8 path).
- rocprofv3 1.1.0 `--kernel-trace` wrapping **one GPU worker (rank 1)** via a new env-gated hook
  in `MI355_run_performance_harness.sh` (`ROCPROF=1 ROCPROF_RANK=1`); other ranks run plain.
- conf `user_mi355x8_nve_b40_qps7400_PROF90s.conf`. Run is **INVALID by construction** (rocprofv3
  serializes/slows rank 1 → backlog collapse, p99 12 s); **per-kernel GPU durations are still
  accurate** — that is what we read, not the throughput.
- Trace: `artifacts/Plan24_fp8_c1off_rocprof_20260613T205221/rocprof/rank1_kernel_trace.csv`
  (1.2 GB, 1.58 M dispatches). Aggregated over the **steady tail (last 30 %, 95.2 s, 628 k
  dispatches, 82 % GPU duty)** with `scripts/profile/rocprofv3_kernel_breakdown.py`
  (per-kernel CSV: `rocprofv3_steady_per_kernel.csv`).

## Per-bucket GPU time (steady state)

| bucket | ms | % GPU | dispatches |
|---|---:|---:|---:|
| **ATTENTION** (`_hstu_attn_fwd`) | 45,887 | **58.7 %** | 9,612 |
| GEMM (hipBLASLt fp8 `F8F8S`/`F8BS` + bf16 `BBS`) | 12,300 | 15.7 % | 51,904 |
| elementwise / cast / copy | 9,128 | 11.7 % | 277,459 |
| layernorm | 5,866 | 7.5 % | 38,447 |
| embedding (`nve::query_uvm`) | 1,997 | 2.6 % | 18,310 |
| other | 1,381 | 1.8 % | 115,828 |
| jagged concat/split (`*_multirow`) | 910 | 1.2 % | 13,455 |
| reduce | 728 | 0.9 % | 102,974 |
| **total** | **78,197** | 100 % | 627,989 |

Matches the prior PyTorch-profiler + TraceLens C1-off breakdown (attention 55.5–56.6 %), so the
rocprofv3 e2e view is consistent — attention dominates at ~59 %.

### Top single kernels

| ms | % | count | µs/call | kernel |
|---:|---:|---:|---:|---|
| 45,887 | 58.7 | 9,612 | 4,774 | `_hstu_attn_fwd` |
| 4,011 | 5.1 | 7,689 | 522 | `Cijk … F8F8S … MT256x256x128` (fp8 GEMM) |
| 3,480 | 4.4 | 11,380 | 306 | `Cijk … F8BS … MT256x256x128` (fp8 GEMM) |
| 3,251 | 4.2 | 26,911 | 121 | `_weighted_layer_norm_fwd` |
| 2,372 | 3.0 | 9,613 | 247 | `_ln_mul_dropout_fwd` |
| 2,261 | 2.9 | 13,457 | 168 | `vectorized_elementwise_kernel` (CUDAF…) |
| 1,855 | 2.4 | 9,610 | 193 | `elementwise_kernel_manual_unroll<128,4>` |
| 1,367 | 1.7 | 3,844 | 356 | `nve::query_uvm` (embedding gather) |
| 916 | 1.2 | 7,734 | 119 | `Cijk … BBS … MT256x224x64` (bf16 GEMM) |
| 882 | 1.1 | 11,532 | 77 | `concat_2D_jagged_multirow` |
| 668 | 0.9 | 67,926 | 9.8 | `__amd_rocclr_copyBuffer` (small device copies) |

## Where the next lever is

**1. Attention is still ~59 %, and the one free win is not turned on.**
The attention kernel was previously characterized to its bit-exact floor (MFMA-bound,
mem-stall ≈ 0 %). The only large arithmetic cut is **fp4 — blocked**. *But* the shipped,
bit-exact **`DLRM_HSTU_ATTN_OCCTUNE=1`** lever (adds `waves_per_eu∈{3,4}` + `num_stages=2` to the
fwd autotune grid → software-pipelines the QK^T→A·V chain) was **NOT enabled in this run**
(default off), and is **not** in `run_gold.sh`. Prior microbench: **−5 to −9 % on the kernel**;
prior e2e: lifted the C1-off b64 knee 8,500→9,000 q/s, bit-exact (no re-cert). **This is the
cheapest immediate win** — A/B `EXTRA_ENV="… -e DLRM_HSTU_ATTN_OCCTUNE=1"` at b40 and, if it holds,
bake it into the C1-off launch path. Expected: ~2–5 % e2e (≈59 % × ~5–9 %).

**2. The ~19 % "glue" (elementwise 11.7 % + layernorm 7.5 %) is the under-explored bucket.**
Almost all prior tuning went into attention; this non-attention/non-GEMM tail is now the
second-biggest pool and is *fragmented across ~316 k tiny launches*:
   - **LayerNorm 7.5 %** — `_weighted_layer_norm_fwd` (4.2 %) + `_ln_mul_dropout_fwd` (3.0 %).
     fp8 LN fusion (`FP8_FUSE_LN`/`FUSE_OUTLN`) is already on; the remaining LN is the *un-fused*
     residual/dropout path. Candidate: fold residual-add + dropout into the LN kernel, or into the
     preceding GEMM epilogue.
   - **Elementwise/copy 11.7 %** — 277 k tiny ops + **`__amd_rocclr_copyBuffer` 67,926 calls
     @ 9.8 µs (668 ms)**. These are mostly fp8 cast/scale copies and staging copies between fused
     stages. Candidate: eliminate redundant fp8 cast→copy round-trips (cast in-place / fuse the
     cast into the producer), and chase the 68 k device-to-device copies to their call sites.
   Even a 20–25 % trim of this bucket is ~4–5 % of total GPU time — comparable to the attention
   lever, and orthogonal to it.

**3. GEMM (15.7 %) is effectively tapped out.** Fragmented hipBLASLt fp8 (`F8F8S`/`F8BS`) + bf16
(`BBS`), no single dominant tile, all on the optimized hipBLASLt path. Plan 28 already NO-GO'd a
custom MXFP4 dense GEMM (hipBLASLt beats Triton here). Leave it.

## Recommended next steps (ranked)

1. **Enable `DLRM_HSTU_ATTN_OCCTUNE=1`** and run a b40 C1-off A/B (PROF then a 600 s PROD cert).
   Free, bit-exact, already implemented. → fold into the C1-off launch if it holds.
2. **Attack the elementwise/LN glue (~19 %)**: (a) fuse residual-add+dropout into LayerNorm;
   (b) hunt the 68 k `copyBuffer` + fp8-cast copies and fuse/eliminate. Bit-exact-able, ~4–5 % e2e.
3. Re-profile **C1-on (b48)** the same way — the bucket mix differs (attention ~46 %, GEMM/LN
   relatively larger), so the glue lever may pay even more there.

> Reusable tooling added: harness `ROCPROF=`/`ROCPROF_RANK=` hook
> (`MI355_run_performance_harness.sh`), and `scripts/profile/rocprofv3_kernel_breakdown.py`
> (buckets a rocprofv3 kernel-trace CSV, with `--tail-frac`/`--since`/`--until` windowing).

---

# Addendum — the CORRECT config: Win-B **b64** (2026-06-13)

The breakdown above profiled the **superseded b40 / OCCTUNE-only** path. The current best is the
**Win-B b64** recipe (`FASTMASK=1 + FULLGRID=1 + OCCTUNE=1`, **BATCH=64**, **INFLIGHT=128**) — the
fast fp8 attention kernel that makes full-causal feasible at b64 (9,600 q/s on MI355X; ~7,600 on
this MI350X VF node). Re-profiled the **same way** (rocprofv3 kernel-trace, rank 1, steady tail
30 % = 90.6 s, 440 k dispatches, 84 % duty). Trace:
`artifacts/Plan24_winb_b64_rocprof_20260613T222831/`; per-kernel CSV
`rocprofv3_winb_b64_per_kernel.csv`. (Run is INVALID by construction — rocprofv3 overhead.)

| bucket | b40 OCCTUNE-only | **b64 Win-B** | shift |
|---|---:|---:|---|
| **ATTENTION** `_hstu_attn_fwd` | 58.7 % | **52.2 %** | **−6.5 pts** |
| GEMM (hipBLASLt fp8/bf16) | 15.7 % | **18.8 %** | +3.1 |
| elementwise / cast / copy | 11.7 % | 13.6 % | +1.9 |
| layernorm | 7.5 % | 8.4 % | +0.9 |
| embedding | 2.6 % | 3.2 % | +0.6 |
| jagged / other / reduce | 3.9 % | 3.9 % | flat |

**The Win-B kernel pulls attention down from ~59 % to ~52 % of GPU**, so the *accessible*
(non-fp4) headroom now lives in the **non-attention 48 %** — and GEMM has overtaken the glue as
the #2 bucket:

- **GEMM 18.8 %** (was 15.7 %) — the two `MT256x256x128` fp8 GEMMs dominate it:
  `F8BS` 6.9 % (442 µs/call ×11.8 k) + `F8F8S` 6.3 % (**899 µs/call** ×5.3 k) = **13.2 %** of all
  GPU time in just two GEMM kernels (UVQK projection + output projection). bf16 `BBS` adds ~4 %.
  This is the new second-order lever: these are stock hipBLASLt picks — worth a hipBLASLt
  tuning/heuristic pass and checking the `F8F8S` 899 µs/call tile choice at b64 shapes.
- **elementwise 13.6 % + layernorm 8.4 % = ~22 % glue** — same story as before, now a bigger
  share: `_weighted_layer_norm_fwd` 4.4 % + `_ln_mul_dropout_fwd` 3.6 %, plus 193 k tiny
  elementwise ops and `__amd_rocclr_copyBuffer` 47 k calls (721 ms). Fuse residual-add/dropout
  into LN; kill fp8 cast/copy round-trips. Bit-exact-able.
- **attention 52.2 %** is the Win-B kernel itself (FASTMASK interior/boundary split + OCCTUNE
  pipelined 128/64/8w/2s). Bit-exact floor; the only big arithmetic cut left is **fp4 — blocked**.

**Revised ranking of next levers (Win-B b64, fp4 off):**
1. **GEMM (18.8 %)** — hipBLASLt tuning for the two `256x256x128` fp8 GEMMs (esp. the 899 µs/call
   `F8F8S`); biggest single accessible pool now.
2. **Glue (~22 %)** — LN+dropout/residual fusion and fp8 cast/copy elimination.
3. **Attention (52 %)** — already Win-B; needs fp4 (blocked) for a step change.

> Note: per-call times are not comparable across batch (b64 does more work per launch than b40);
> the *shares* are the comparable quantity. Also observed: launching runs back-to-back can OOM the
> next one (b64 NVE shards) before the prior run's VRAM is released — let VRAM drain (`rocm-smi
> --showmeminfo vram`) between runs.

---

## GEMM tuning — investigation (2026-06-13)

**Exact problems** (from `triton_addmm.py` + `HIPBLASLT_LOG_MASK=32` capture). Both fp8 dense
projections are **TN e4m3** via `torch._scaled_mm`: `transA=T, transB=N`, A=`[k,m]` (weight),
B=`[k,n]` (activation), C/D=`[m,n]`, per-tensor scaleA/scaleB, `alpha=1 beta=0`, no epilogue.
hipBLASLt swaps dims vs the torch view, so **the only variable dim is hipBLASLt `n` = token count**:

| projection | m (out) | k | n | out dtype |
|---|---:|---:|---|---|
| UVQK    | 2048 | 512  | tokens (var) | bf16 (u/v) **and** f8 (q/k/v slice) |
| OUTPROJ | 512  | 1536 | tokens (var) | bf16 **and** f8 |

**Mechanism available (ROCm 7.2.3):** only `HIPBLASLT_TUNING_OVERRIDE_FILE` (load side); the
in-process `HIPBLASLT_TUNING_FILE` recorder is NOT in this lib. Override key = exact
`transA,transB,batch,m,n,k,a/b/c/compute_type → solution_index` (`UserDrivenTuningParser`), so it
matches **exact n**. `hipblaslt-bench` is not shipped.

**Headroom** — `scripts/profile/probe_fp8_sweep.cpp` enumerates all ~19 k Tensile candidates
(`hipblaslt_ext::getAllAlgos`), times the ~900 supported, and compares the best vs the default
heuristic's pick. The heuristic consistently does **not** pick the fastest solution:

| problem | out | n=4k | n=8k | n=16k | n=32k |
|---|---|---:|---:|---:|---:|
| UVQK (m2048,k512)   | bf16 | 4.9 % | 9.5 % | 10.0 % | 11.2 % |
| UVQK                | f8   | 5.9 % | 1.5 % | 0.2 %  | 3.2 %  |
| OUTPROJ (m512,k1536)| bf16 | 7.1 % | 13.3 %| 11.8 % | 17.8 % |
| OUTPROJ             | f8   | 1.6 % | 3.6 % | 5.4 %  | 8.8 %  |

**Real token counts (live Win-B b64 capture, `real_n_distribution.txt`):** n is **large and
variable** — UVQK n≈537k–547k (every forward distinct); OUTPROJ a dominant `n=131072` (CUDA-graph /
warmup capture shape, 18 k hits) plus a tail `~540k–553k` (all distinct). So **n is not a small
discrete set**.

**Headroom at the real n:**

| problem | out | n=131072 | n≈540668 | best index |
|---|---|---:|---:|---|
| UVQK (m2048,k512)   | bf16 | **15.8 %** | **9.6 %** | 453546 (both) |
| UVQK                | f8   | 2.1 %  | 0.2 %  | — |
| OUTPROJ (m512,k1536)| bf16 | **27.2 %** | **12.8 %** | 453750 / 454444 |
| OUTPROJ             | f8   | **16.0 %** | 4.6 %  | 455427 / 455429 |

Two key facts: (1) the winning solutions (453546, etc.) are **existing stock Tensile solutions** —
no new kernels needed, the **default heuristic simply mis-picks**; (2) for UVQK bf16 the **same
index 453546 wins across both large n**, so one pinned solution generalizes the large-n regime
(OUTPROJ shifts winner with n, but both beat the heuristic). Prize ≈ **10–27 % on the bf16
projections ⇒ ~1–2 % end-to-end, bit-exact.**

**Mechanism conclusion.** Since n varies per forward and the override key is exact-(m,n,k), the
`HIPBLASLT_TUNING_OVERRIDE_FILE` route is impractical, and a free-size Tensile rebuild is overkill
(the kernels already exist). Built a **pinned-solution hipBLASLt op** (`route_a_probe/fp8tuned_ext/`,
no-torch C ABI + ctypes — torch headers drag in a broken rocThrust/cub include) that forces a chosen
stock solution index via `getAlgosFromIndex`, validates it per-shape, and falls back to the heuristic.

**Reality check vs the real baseline (`test_fp8tuned.py`, 200-iter cuda-event GPU time).** The
pinned op is **bit-identical** to `torch._scaled_mm` (|ref−pin| = 0.0000), but the win is much
smaller than the probe's raw headroom suggested — because **`torch._scaled_mm` already picks a good
solution, well above the heuristic#0 baseline the probe compared against** (and the probe's large-n
timings were unreliable):

| problem | n | pin vs `torch._scaled_mm` (GPU) |
|---|---|---:|
| UVQK    | 131072 | −1.5 % (pinning **hurts**) |
| UVQK    | 540668 | −7.3 % (pinning **hurts**) |
| OUTPROJ | 131072 | **+10.3 %** |
| OUTPROJ | 540668 | +2.2 % |

**Net:** torch is already optimal for UVQK; the only real GEMM win is **OUTPROJ (~10 % at its
dominant n=131072, ~2 % on the large-n tail)**, bit-exact. Whether n=131072 is steady-state vs a
warmup/graph-capture shape is unconfirmed; if steady-state OUTPROJ is the ~540k tail, the win is
~2 %. Either way the e2e ceiling from GEMM is **<~0.5 %** — torch's selection leaves little on the
table. Recommendation: pin **OUTPROJ only** (never UVQK), or deprioritize GEMM in favor of the
glue/LN lever (#2, ~22 % of GPU).

Tooling (committed): `scripts/profile/fp8_gemm_capture.py`, `scripts/profile/probe_fp8_sweep.cpp`,
`results/fullcausal_c1off/real_n_distribution.txt`; pinned op + tests in
`route_a_probe/fp8tuned_ext/` (node-local workbench, not in the runner repo).

---

## Glue lever — investigation (2026-06-13)

GEMM is effectively tapped out (above), so the next lever is the **~22 % "glue"** (elementwise +
layernorm). Mapped every hot non-attn/non-GEMM kernel in the Win-B b64 trace to its GR source call
site (`generative_recommenders/ops/triton/{triton_layer_norm,triton_hstu_linear,triton_hstu_preprocess_and_attention,triton_addmm}.py`,
`modules/stu.py`). One forward = **~1336 iters × 5 HSTU layers** in the steady tail.

| kernel | %GPU | per-fwd | source call site |
|---|---:|---:|---|
| `_weighted_layer_norm_fwd` | 4.37 | ~14 | input LN, `preprocess:387/402` (fp8 epilogue, Φ1) |
| `_ln_mul_dropout_fwd` | 3.62 | 5 | output LN·u(+concat), `triton_hstu_linear:904` (Φ3) |
| `CUDAFunctor_add<bf16>` | 3.54 | ~7 | **residual `out + x`** in `_scaled_addmm_fp8_preq` (`triton_addmm:1152`) — 2D residual can't ride the 1D-bias `_scaled_mm` |
| `direct_copy` bf16 `<128,4>` | 2.65 | 5 | a strided bf16 copy, 1/layer/301 µs (attribution unconfirmed under FUSE_QKV; **needs ground-truth profile**) |
| `silu_kernel` bf16 | 2.16 | 4 | **standalone `F.silu(u)`**, `preprocess:483` — fires only when `FUSE_EPILOGUE=0` |
| `float8_copy` (bf16→e4m3) | 1.96 | 3 | ATen `.to(e4m3)` — mostly one-time weight quant / boundary casts |

**Production C1-off config** (`scripts/run/winb_batch_sweep.sh`): `FP8_GEMM/ATTN/FUSE_LN/FUSE_OUTLN/FUSE_QKV=1`,
`MAX_ATTN_LEN=0`, `LASTLAYER_TARGETS_ONLY=1`, Win-B (`FASTMASK+FULLGRID`), but **`FUSE_EPILOGUE=0`**.

**Ranked glue levers:**
1. ~~`FUSE_EPILOGUE=1` (D2) on C1-off~~ — **MEASURED REGRESSION, REJECTED.** MI350 fixed-load A/B
   (b64 q6000 C1-off, `batch.predict` p50, 2× baseline bracketing the fusion run):
   `EPI0`=57.9/57.5 ms vs `EPI1`=**61.3 ms (+6 %)**. Folding SiLU into the output-LN epilogue
   (`SILU_U=True`) makes that already-heavy reduction kernel materially worse at C1-off's large
   per-layer token counts — a different regime than the C1-on config where D2 is the certified
   figure-of-record. This is why `winb_batch_sweep.sh` hardcodes `FUSE_EPILOGUE=0`; the existing
   default is correct. (Lesson: D2/knee numbers from C1-on/MI355 do **not** transfer to C1-off/MI350.)
2. **Fold residual `out + x` into the output GEMM epilogue** (`beta·C` accumulate in the OUTPROJ
   hipBLASLt call via the `fp8tuned_ext` op) ⇒ removes `CUDAFunctor_add` (**3.54 %**). NOT bit-exact
   (one rounding vs two — strictly *more* accurate); needs accuracy re-cert. Synergizes with the
   OUTPROJ pin.
3. **Kill the per-layer bf16 `direct_copy`** (**2.65 %**) once ground-truth profiling pins its source.
4. LN itself (8.4 %) is already fp8-fused; residual+dropout fold is the remaining structural win.

### Ground-truth attribution (torch profiler, C1-off/MI350, `Plan24_glue_torchprof*`)

Profiled the live C1-off path (skip 200, 8 batches, `record_shapes`; GPU `record_function` regions).
Corrects the counting-based guesses above:

- **bf16 `direct_copy<128,4>` = 4.5 %** is a copy of the **fully-padded dense `[B·L, D]=[131072,512]`**
  sequence (B=64 × L=2048 × D=512 — *mostly padding*), **~6×/batch**, landing in the **model-forward
  glue around the input-preprocessor / positional-encoder / jagged↔dense boundary** (regions
  `hstu_input_preprocessor` / `hstu_positional_encoder` in `modules/hstu_transducer.py`) — **NOT** the
  per-HSTU-layer UVQK `.contiguous()`. Exact line unpinned (this torch/ROCm build did not emit
  `with_stack` frames). If these operate on padded-dense where jagged/packed would do, the padding
  ratio is the waste.
  - **PINNED (via External-id correlation):** the copy is `aten::copy_` whose parent is
    `aten::to`/`aten::_to_copy` — a **`Half → BFloat16` cast** (input type `c10::Half`, dtype 15) of
    the jagged `[N,512]` embedding (N≈410k pre-contextual). Source: **`inference_modules.py:247-249`
    `move_sparse_output_to_device`** (called from `model_family.py:936` on the dispatcher H2D path,
    `.to(device, non_blocking).to(torch.bfloat16)`). NVE/cuembed gathers embeddings in **fp16**; the
    model computes in **bf16**, so the cast is intrinsic — but it runs as a standalone GPU kernel on
    the worker GPU. The fp16→bf16 round already happens, so **relocating it is bit-exact**: cast on
    CPU before the H2D (`.to(bfloat16).to(device, non_blocking)`) or fuse into the transfer, to move
    it off the worker-GPU compute timeline. Caveat: it lives in the multi-process dispatch path
    (`_PIPELINE_H2D_TO_WORKER`), so it needs careful handling/measurement, and may already partially
    overlap.
- **Residual `out + x` add CONFIRMED real:** `CUDAFunctor_add<bf16>` in `## stu_compute_output ##`
  = 3860 µs / 4 batches (the dominant add); a second add in `hstu_input_preprocessor` (1689 µs,
  positional/timestamp-embedding add). The compute-output residual is the fold-into-GEMM-epilogue
  target (≈3.5 %).

**Net glue picture (MI350, fp8 C1-off):** the accessible glue is the **dense padded-tensor copy
(~4.5 %, structural — in the preprocessor/boundary)** and the **residual add (~3.5 %, foldable into
the OUTPROJ GEMM epilogue but not bit-exact → accuracy re-cert)**. LN is already fp8-fused; silu
fusion regresses. No trivial bit-exact config flip remains.

> **Measurement loop (validated, MI350-local):** fixed sub-knee A/B at b64 **q6000** C1-off Win-B,
> comparing `batch.predict` p50 from `timing.jsonl` (run is INVALID by construction — 90 s probe;
> only `batch.predict` is read). ~4.5 min/run with a warm autotune cache. Baseline reproduces to
> ±0.4 ms, so it cleanly resolves a ~2 % per-forward lever. **Note:** all q/s *knee* figures on
> record are MI355; the MI350 C1-off b64 knee is **<7600** (q7600 already INVALID here). Caveat: the
> embedding cast lives in the worker's `get_item_from_queue` region (H2D), **not** `dense forward`, so
> it is *not* captured by `batch.predict` — measuring its removal needs rocprofv3 worker-GPU time or
> the e2e knee, not this loop.

### Conclusion — fp8 micro-opt is near its practical floor on MI350

After mapping + MI350 measurement of every accessible (non-fp4) lever:

| lever | %GPU | status |
|---|---:|---|
| Attention (`_hstu_attn_fwd`) | ~52 % | Win-B floor; only big cut = **fp4 (blocked)** |
| GEMM (UVQK/OUTPROJ) | ~19 % | torch already well-tuned; UVQK pin *hurts*, OUTPROJ +10%@131k bit-exact ⇒ tiny e2e |
| LayerNorm | ~8 % | already fp8-fused (Φ1/Φ3) |
| silu epilogue fusion | ~2 % | **A/B'd: +6 % regression on C1-off → rejected** |
| residual `out+x` add | ~3.5 % | foldable into OUTPROJ GEMM epilogue, **needs accuracy re-cert** (not bit-exact) |
| embedding fp16→bf16 cast | ~4.5 % | **intrinsic** (fp16 NVE gather → bf16 compute); clean removal = store/gather the embedding table in **bf16** (one-time, bit-exact) — a substantial NVE-side change, not a flag |

No trivial bit-exact config flip remains. The two real structural candidates are (a) **bf16 embedding
table** (removes the 4.5 % cast, bit-exact, NVE-side work) and (b) **residual fold into the OUTPROJ
GEMM epilogue** (3.5 %, accuracy re-cert). The dominant pool (attention 52 %) is gated on fp4, which
is parked. Recommendation: pursue (a)/(b) only if the structural cost is justified; otherwise fp8 is
effectively at floor and the next step-change requires unblocking fp4.

## Residual `out+x` fold — IMPLEMENTED + measured (2026-06-14)

Lever (b) is done, behind the single switch `DLRM_HSTU_FP8_RESID` (`off`|`pin`|`heur`, default `off`).
Full plan/details in [`docs/optimization_plan_residual_out_fold.md`](../../docs/optimization_plan_residual_out_fold.md).

- **How:** extended `route_a_probe/fp8tuned_ext` (hipBLASLt) with a `beta·C` epilogue (`D = A·B + 1·C`,
  `C` = the `[M,512]` residual) and call it from `_scaled_addmm_fp8`/`_scaled_addmm_fp8_preq` (2-D
  residual branch) when the switch is on. `torch._scaled_mm` only fuses a 1-D bias, so the matrix
  residual needs the custom op. bf16-out only; try/except falls back to the separate add.
- **Numerics (not bit-exact):** vs the separate bf16 add it differs by ~1.1 bf16 ULP, but is actually
  *more* accurate (residual accumulates in fp32, rounds once: max err 0.50 vs 1.00, mean 0.044 vs
  0.059). ⇒ low-risk accuracy re-cert, **still required before GOLD**.
- **GEMM-solution trap (important):** the stock hipBLASLt top-1 heuristic mis-picks the OUTPROJ
  solution at the production token count `n≈131072` — a slow solution that *cancels* the saved add
  (microbench: heur **−4.8 %** vs baseline at 131072). A pinned solution (`454444`) is robustly fast
  across the whole token range (**+23..+40 %** on the OUTPROJ GEMM block). ⇒ default `pin`, not `heur`.
- **e2e — controlled matched-qps sweep (b64, Server, PROF90s, off vs pin, 2026-06-14):** the headline
  is that the fold's e2e benefit on MI350 b64 is **within run-to-run noise**.

  | qps | off comp/s | off p50 | off p99 | pin comp/s | pin p50 | pin p99 |
  |---|---|---|---|---|---|---|
  | 7000 | 6977 | 78.5 ms | 243 ms | 6983 | 75.2 ms | 241 ms |
  | 7250 | 7226 | 79.5 ms | 268 ms | 7220 | 82.5 ms | 272 ms |
  | 7400 | 7360 | 86.3 ms | 324 ms | 7376 | 80.1 ms | 281 ms |

  Completed-sps differs by **<0.2 %** at every matched qps; p99/p50 bounce both ways (pin better at
  7400, worse at 7250). **The earlier single A/B pair showing p99 636→293 ms was a noisy baseline
  outlier** (the controlled off@7250 is p99 268 ms, not 636) — NOT a real fold effect. All points are
  saturated/INVALID (the valid knee is < 7000), so this probe cannot resolve a sub-1 % GPU saving.
  Artifacts: `artifacts/gold_sweep_{off,pin}_q{7000,7250,7400}_*`.
- **Verdict:** the fold is correct, slightly *more* accurate, and deletes a real kernel (+23..+40 % on
  the OUTPROJ GEMM block in isolation), but the saved time is **hidden behind attention/dispatch** ⇒ no
  robust e2e gain on MI350 b64. Unlike the bf16 gather (+4.6 %, serialized path), it does not surface
  end-to-end here. Kept as a correct, default-off, opt-in lever (feature branch `residual-out-fold`).
- **Next (if pursued):** a **batch-size sweep** (smaller b ⇒ OUTPROJ relatively larger / less hidden)
  and/or longer PROD10min runs below the knee for a cleaner p99; otherwise shelve in favor of the
  attention prize (fp4, parked). Accuracy re-cert still required before any GOLD promotion.
