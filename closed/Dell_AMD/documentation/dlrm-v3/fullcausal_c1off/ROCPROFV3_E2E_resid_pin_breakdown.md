# Post-fold re-profile — RESID=pin + BF16_GATHER, rocprofv3 kernel breakdown (2026-06-14)

Re-profile of the **current best GOLD stack** (fp8 C1-off full-causal + Win-B(occ) + **bf16 gather**
+ **OUTPROJ residual fold `RESID=pin`**) to confirm the fold removed the standalone residual-add
kernel and to re-rank the next lever. Direct successor to the 2026-06-13 fp8 baseline
(`ROCPROFV3_E2E_fp8_breakdown.md`).

## Method

- GR tree on `residual-out-fold` (vendored `fp8tuned_ext`, gfx950 lib rebuilt this node), `RESID=pin`,
  `BF16_GATHER=1`, all other GOLD flags as `run_gold.sh`.
- rocprofv3 `--kernel-trace` on **rank 1** via `run_gold.sh EXTRA_ENV="-e ROCPROF=1 -e ROCPROF_RANK=1"`,
  conf `user_mi355x8_nve_b40_qps7400_PROF90s.conf` (b40, same op-point as the 6/13 baseline).
- **Note vs 6/13:** this run came back **VALID** (7,388 comp/s; rocprof overhead was tolerable here),
  whereas the 6/13 run was INVALID-by-construction (serialized rank → backlog). So per-call kernel
  durations here are *cleaner* (less serialization inflation) — compare **percentages and dispatch
  counts**, not absolute ms (windows/duty differ: 79.6 s @ 64 % duty here vs 95.2 s @ 82 % there).
- Trace: `artifacts/gold_resid_pin_b40_rocprof_20260614T220409/rocprof/rank1_kernel_trace.csv` (1.2 GB,
  1.61 M dispatches); steady tail (last 30 %, 587,841 dispatches) via `rocprofv3_kernel_breakdown.py`
  (per-kernel CSV: `…/rocprof/steady_per_kernel.csv`).

## Per-bucket GPU time (steady tail) — vs 6/13

| bucket | 6/13 fp8 (RESID off) % | **post-fold (RESID=pin) %** | post-fold ms | count |
|---|---:|---:|---:|---:|
| **attention** (`_hstu_attn_fwd`) | 58.7 | **51.8** | 26,399 | 9,190 |
| GEMM (hipBLASLt fp8/bf16) | 15.7 | **18.9** | 9,638 | 49,626 |
| elementwise / cast / copy | 11.7 | 11.8 | 5,998 | 252,451 |
| layernorm | 7.5 | 8.9 | 4,552 | 36,760 |
| embedding (`nve::query_uvm`) | 2.6 | 3.4 | 1,740 | 17,514 |
| other | 1.8 | 2.4 | 1,247 | 110,942 |
| jagged concat/split | 1.2 | 1.6 | 796 | 12,866 |
| reduce | 0.9 | 1.2 | 590 | 98,492 |
| **total** | 100 | 100 | 50,960 | 587,841 |

## The fold landed (kernel-level proof)

The standalone OUTPROJ residual-add **`vectorized_elementwise_kernel<…CUDAFunctor_add>`** kernel:

| | 6/13 (RESID off) | post-fold (RESID=pin) |
|---|---:|---:|
| dispatches | **13,457** | **3,676** |
| ms / % GPU | 2,261 / 2.9 % | 709 / 1.4 % |

The per-iteration OUTPROJ add (~9.2 k dispatches, ≈ one per attn-fwd) is **gone** — folded into the
fp8 OUTPROJ GEMM's `beta·C` epilogue, which is why **GEMM rises 15.7 %→18.9 %** (the top fp8 GEMM
`F8F8S MT256x256x128` goes 5.1 %→6.5 %). No `fell back` in the run log: pin `454444` engaged e2e.
This is exactly the "fold residual-add into the preceding GEMM epilogue" candidate the 6/13 breakdown
flagged, now realized and accuracy-cleared (GAUC 0.78628734 ≥ fp8 baseline).

## Where the next lever is (unchanged ranking, fold cashed)

1. **Attention ~52 %** — still the dominant pool, still gated on fp4 (the only large arithmetic cut);
   `DLRM_HSTU_ATTN_OCCTUNE=1` is already baked into GOLD. No new free win here without fp4.
2. **"Glue" ~20.7 % (elementwise 11.8 % + layernorm 8.9 %)** — now the clear #2 and the most
   actionable. Still fragmented across ~252 k tiny elementwise ops + **`__amd_rocclr_copyBuffer`
   64,975 calls @ 9.8 µs (636 ms)** and the un-fused LN/dropout path (`_weighted_layer_norm_fwd`
   88 µs + `_ln_mul_dropout_fwd` 227 µs). Candidates: kill redundant fp8 cast→copy round-trips
   (cast in-place / fuse into the producer); fold the remaining residual/dropout into the LN kernel.
   A 20–25 % trim of this bucket ≈ 4–5 % of total GPU time — orthogonal to attention.
