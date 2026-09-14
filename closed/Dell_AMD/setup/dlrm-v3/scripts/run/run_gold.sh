#!/usr/bin/env bash
# run_gold.sh — GOLD launcher for the closed-submission DLRM-v3 ROCm/gfx950 Server cert.
# By DEFAULT (zero args) this reproduces the GOLD figure of record = the C1-off KNEE:
#   **12,200 issued q/s VALID, 12,198.90 completed q/s, p50 50.04 ms / p99 58.32 ms / p99.9 69.14 ms (clean node + latest LoadGen 6.0.16 + clock determinism + P0/P2/P1 ring1024 + deg-5 gate)** (b64 full-causal C1-off Win-B(occ) + bf16 gather + OUTPROJ
#   [knee bumped 11,700 -> 11,800 on 2026-07-07 (Plan 63 vectorized real-output response buffer):
#    q11,800 PROF90s with DLRM_VECTORIZE_RESPONSE_BUFFER=1 was VALID @ 11,781.25 completed
#    q/s, p50 52.67ms / p99 73.75ms / p99.9 93.73ms, and q11,800 PROD10min certified
#    VALID @ 11,794.84 completed q/s, p50 52.13ms / p99 64.92ms / p99.9 98.75ms.
#    This preserves real predictions; it only replaces the PerformanceOnly per-query Python
#    response-buffer loop with a vectorized NumPy fill.]
#   [tail tightened at the q11,800 GOLD point on 2026-07-07 (Plan 64 "c40p6"): the NVE item cache
#    default moves 32->40GB AND the P6 ROCm runtime knobs (HSA_ENABLE_INTERRUPT=0,
#    HIP_FORCE_DEV_KERNARG=1, AMD_DIRECT_DISPATCH=1; P6=1) are ON by default. A same-window
#    q11,800 PROD10min A/B on a clean reserved node had c40p6 VALID p99 66.89ms / p99.9 91.91ms
#    vs the 32GB/no-P6 default VALID p99 71.28ms / p99.9 117.67ms (both ~11,795 completed q/s) — a
#    -4.4ms p99 / -26ms p99.9 tail tightening, reproducible (c40p6 PROD p99 65.93ms the prior day).
#    Figure of record was q11,800 at THIS step (q11,900 PROD10min INVALID p99 100ms on auto clocks
#    — cache/P6 alone can't move the compute-bound O(L^2) tail cliff); Plan 64 P8 clock determinism
#    then cleared q11,900 — see the next note. Bit-exact (residency + runtime wait-impl only; no
#    GAUC surface). NB q11,800 PROF90s is a coin-flip (single-90s tail variance) — settled on
#    PROD10min, not 90s. Set DLRM_NVE_GPU_CACHE_GB=32 P6=0 to revert.]
#   [knee bumped 11,800 -> 11,900 on 2026-07-07 (Plan 64 P8 GPU clock determinism): pinning SCLK
#    (rocm-smi --setperfdeterminism 2400 — now set by the CLOCK_DETERMINISM guard below) removes the
#    2075-2400MHz DVFS ripple that was stretching the O(L^2) attention burst tail. Same c40p6 stack:
#    q11,900 PROD10min was INVALID p99 100.12ms on auto clocks -> VALID p99 70.27ms / p99.9 88.60ms
#    with determinism (11,894.23 completed q/s). Promoted on this single clean-node PROD10min VALID +
#    the clear mechanism + the auto-clock INVALID reference (operator call). Bit-exact (clock policy
#    only; no GAUC). REQUIRES a clean/un-contended node (shared-box neighbors regress the tail) AND
#    clock determinism; CONF default is now qps11900. Set CONF=...qps11800... (or CLOCK_DETERMINISM=0)
#    for the safer prior knee.]
#   [tail tightened at the q11,900 GOLD point on 2026-07-08 (Plan 64 P0/P2 host hygiene):
#    REUSE_PINNED_OUTPUT=1 reuses the worker-side pinned CPU prediction buffer by shape, removing
#    per-batch pinned-host allocator/event churn; TIMING=0 disables per-batch timing_stats logging in
#    the cert path. Same-window q11,900 PROF90s moved from baseline INVALID p99 95.64ms / p99.9
#    144.32ms to P0/P2 VALID p99 62.54ms / p99.9 66.98ms. q11,900 PROD10min with P0/P2 certified
#    VALID @ 11,894.24 completed q/s, p50 50.81ms / p99 58.58ms / p99.9 75.66ms. Bit-exact (host
#    buffer reuse + instrumentation hygiene only; no GAUC surface). Set REUSE_PINNED_OUTPUT=0 or
#    TIMING=1 to revert either half.]
#   [knee bumped 11,900 -> 11,970 on 2026-07-08 (Plan 64 P1 LoadGen response-buffer ring + latest LoadGen):
#    DLRM_REUSE_LOADGEN_BUFFERS=1 reuses a shape-keyed NumPy response-buffer ring (default
#    DLRM_RESPONSE_BUFFER_RING_SIZE=1024 after the 2026-07-10 promotion) on the LoadGen result thread. This avoids per-batch
#    response-buffer allocation churn while preserving QSC pointer lifetime safety; the C++ QSC pool
#    queues raw payload pointers asynchronously, so this must be a ring, not a single buffer. With
#    latest LoadGen 6.0.16, q12,000 PROD10min is just over the tail bar (p99 82.61ms), while
#    the previous 512-slot q11,970 PROD10min certified VALID @ 11,968.80 completed q/s,
#    p50 52.38ms / p99 64.50ms / p99.9 100.28ms. Bit-exact host/output plumbing; no GAUC
#    surface. Set REUSE_LOADGEN_BUFFERS=0 to revert.]
#   [tail tightened at the q11,970 GOLD point on 2026-07-10 (Plan 4 P1 ring-size promotion):
#    RESPONSE_BUFFER_RING_SIZE=1024 is now the GOLD default. The 90s q11,970 screen was still
#    INVALID but had the best tail of the ring/cache probes (p99 102.12ms); the cert-length
#    follow-up was VALID @ 11,968.67 completed q/s, p50 51.18ms / p99 61.91ms / p99.9 84.99ms.
#    Bit-exact host-output lifetime plumbing only; no GAUC surface. Set RESPONSE_BUFFER_RING_SIZE=512
#    to reproduce the previous P1 ring default.]
#   [TEST08 compliance verified for the q11,970 model/accuracy stack: Offline AccuracyOnly reference produced
#    349,823 entries, audited Server produced 4,012 sampled entries, and the official verifier passed
#    with num_matched=4012, num_unmatched=0, num_ne_mismatch=0 at 0.10% tolerance. Ring1024 only changes
#    host response-buffer lifetime. The 2026-07-10 replication also kept audited Server performance
#    VALID @ 11,968.78 completed q/s, p99 62.07ms.]
#   [knee bumped 11,970 -> 12,200 on 2026-07-16 (production degree-5 SiLU gate):
#    DLRM_HSTU_GATE_POLY_DEG=5 selects the shorter sigmoid polynomial that first survived the
#    barrier-free route, then passed production Offline GAUC (lifetime GAUC 0.7862875110),
#    q12,200 PROF90s (VALID p99 62.76ms), q12,200 PROD10min (VALID @ 12,198.90 completed q/s,
#    p50 50.04ms / p99 58.32ms / p99.9 69.14ms), and official TEST08 (349,823 ref entries,
#    4,017 sampled audit entries, num_unmatched=0, num_ne_mismatch=0). Set GATE_POLY_DEG=9 to
#    reproduce the prior degree-9 polynomial gate.]
#   [tail tightened at the q11,700 GOLD point on 2026-07-06 (Plan 62 P3 bf16 no-op cast guard):
#    q11,700 PROF90s with DLRM_SKIP_BF16_NOOP_CAST=1 was VALID @ 11,680.37 completed q/s,
#    p50 51.72ms / p99 66.84ms / p99.9 79.37ms vs the prior output-fastinfer PROF90s
#    p50 52.60ms / p99 68.80ms / p99.9 111.25ms. q11,800 PROF90s with the guard is still
#    INVALID at p99 134.03ms / p99.9 225.92ms, so this is a tail-margin default, not a qps bump.]
#   [knee bumped 11,600 -> 11,700 on 2026-07-06 (Plan 62 output-LN fast-inference path):
#    q11,700 PROF90s with DLRM_HSTU_OUTPUT_LN_FAST_INFER=1 was VALID (p50 52.60ms,
#    p99 68.80ms / p99.9 111.25ms), and q11,700 PROD10min certified VALID @
#    11,694.83 completed q/s, p50 52.20ms / p99 69.98ms / p99.9 221.21ms.
#    Offline AccuracyOnly recert PASSED with lifetime GAUC 0.7858985604. The
#    Server bar is p99; p99.9 remains a noisy tail diagnostic.]
#   [knee bumped 11,500 -> 11,600 on 2026-07-06 (Plan 62 sort-by-length off):
#    q11,600 PROF90s with DLRM_HSTU_SORT_BY_LENGTH=0 repeated VALID (p99 67.54ms
#    then 65.86ms), and q11,600 PROD10min certified VALID @ 11,595.72 completed
#    q/s, p50 51.62ms / p99 63.68ms / p99.9 75.41ms.]
#   [tail tightened at the q11,500 GOLD point on 2026-07-06 (Plan 61 P3 preprocessor
#    LN-add fold): q11,500 PROF90s A/B improved p99 66.44->64.44ms and p99.9
#    75.63->69.41ms; q11,500 PROD10min with lnaddfold certified VALID @
#    11,495.73 completed q/s, p50 52.18ms / p99 62.17ms / p99.9 71.73ms.
#    Offline AccuracyOnly recert PASSED with lifetime GAUC 0.7858978083.]
#   [knee bumped 11,400 -> 11,500 on 2026-07-06 (Plan 61 P1 harness sync/inference cleanup):
#    q11,500 PROF90s repeated VALID (p99 75.30ms then 67.09ms), q11,500 PROD10min
#    certified VALID @ 11,495.85 completed q/s, p50 52.17ms / p99 63.35ms / p99.9 77.31ms.
#    q11,600 PROF90s is INVALID at p99 330.17ms / p99.9 416.02ms, so the
#    short-run cliff moved to 11.5k-11.6k.]
#   [knee bumped 11,100 -> 11,400 on 2026-07-05 (Plan 59 clean-node headroom sweep):
#    q11,200/q11,300/q11,400 PROF90s were VALID, q11,500 PROF90s was INVALID at
#    p99 116.2ms / p99.9 211.9ms, and q11,400 PROD10min certified VALID @
#    11,395.30 completed q/s, p50 52.2ms / p99 62.3ms / p99.9 72.4ms.]
#   [clean-node re-measure 2026-07-05 (Plan 58 A/B baseline): the SAME q11,100 GOLD config re-ran
#    VALID @ 11,095.58 completed q/s, p99 61.70ms / p99.9 66.30ms. The original promotion run's
#    p99.9 114.16ms was node/run-variance, not the config; on a healthy node the tail has ~14ms
#    margin under the 80ms Server bar (which is a p99 bar).]
#   [knee bumped 11,000 -> 11,100 on 2026-07-05: same Plan 57 local small-table lookup
#    plus 32GB odd-set NVE cache. q11,100 PROD10min is VALID @ 11,095.71 completed q/s,
#    p99 68.53ms / p99.9 114.16ms (original promotion run).]
#   [prior knee bump 10,600 -> 10,900 on 2026-07-04: local small-table lookup bypass plus
#    32GB odd-set NVE cache remove the remaining small-table fp16->bf16 copy and reduce
#    item_id cache misses after the modulo-aliasing fix. q10,900 PROD10min is VALID @
#    10,894.82 completed q/s, p99 61.50ms / p99.9 67.17ms. Accuracy re-cert with the
#    same local-small-table path: lifetime GAUC 0.7858977608 (PASS vs 99.9% fp16-ref bar).]
#   [prior knee bump 10,500 -> 10,600 on 2026-07-04: static candidate metadata plus
#    optimized embedding lookup remove enough jagged/glue overhead to clear the next rung.
#    q10,600 PROD10min is VALID @ 10,594.65 completed q/s, p99 67.66ms.
#    q10,700 PROF90s is VALID (p99 72.59ms), but q10,700 PROD10min is INVALID
#    at p99 80.28ms / p99.9 142.97ms, so the cert-length edge is 10.6k-10.7k.]
#   [prior knee bump 10,400 -> 10,500 on 2026-07-04: NVE LinearUVM cache geometry fix
#    avoids even-num_sets modulo aliasing for low-bit-aligned item IDs, allowing the item_id
#    GPU cache default to move from 10GB to 16GB. q10,500 PROD10min is VALID @ 10,495.02
#    completed q/s, p99 68.97ms; q10,600 PROF90s is INVALID @ p99 104.76ms, so the edge
#    moved to 10,500-10,600 before the glue cleanup.]
#   resid-fold (RESID=pin) + delta V/K fp8-out fold (DELTA_VK=1) + main-layer SiLU fold
#   (SILU_MAIN=1) + preprocessor LN-add fold + output-LN fast-inference path
#    + sort-by-length off + vectorized response buffer + graph-off eager dense-STU,
#    at the qps11800 PROD10min conf, inflight=128).
#   VALID *knee* progression that
#   got here: bf16-gather 9,700 -> RESID=pin 9,800 (pin@9800 p99 72.2ms VALID vs off@9800 INVALID)
#   -> delta_vk -> 10,000 (q10000 p99 76.2ms) -> SiLU fold -> 10,100 (q10100 fold VALID p99 73.9ms)
#   -> clean-node re-measure 2026-06-26 -> **10,200** (q10,200 600s PROD VALID p99 70.5ms; the old
#   "INVALID @ q10200" was a chaotic single run — on a healthy node it is solidly VALID with margin;
#   the cliff moved up to ~10,300) -> Plan 55 guarded STU graph -> **10,300** (q10,300 600s PROD
#   VALID p99 69.6ms) -> Plan 55 P2 offset reuse -> **10,400** (q10,400 PROD10min VALID
#   p99 69.87ms) -> NVE 16GB odd-set cache geometry -> **10,500** (q10,500 PROD10min
#   VALID p99 68.97ms) -> static candidate metadata + optimized embedding lookup -> **10,600**
#   (q10,600 PROD10min VALID p99 67.66ms) -> local small-table lookup + 32GB NVE cache
#   -> **10,900** (q10,900 PROD10min VALID p99 61.50ms) -> **11,000**
#   (q11,000 PROD10min VALID p99 64.11ms) -> **11,100**
#   (q11,100 PROD10min VALID p99 68.53ms promotion run; clean-node re-measure p99 61.70ms / p99.9 66.30ms)
#   -> **11,400** (q11,400 PROD10min VALID p50 52.2ms / p99 62.3ms / p99.9 72.4ms; old q11,500 PROF90s INVALID)
#   -> **11,500** (P1 q11,500 PROD10min VALID p50 52.17ms / p99 63.35ms / p99.9 77.31ms;
#      P3 lnaddfold q11,500 PROD10min VALID p50 52.18ms / p99 62.17ms / p99.9 71.73ms;
#      q11,600 PROF90s INVALID)
#   -> **11,600** (sort-by-length off q11,600 PROD10min VALID p50 51.62ms / p99 63.68ms / p99.9 75.41ms).
#   -> **11,700** (output-LN fast-inference q11,700 PROD10min VALID p50 52.20ms / p99 69.98ms / p99.9 221.21ms)
#   -> **11,800** (Plan 63 vectorized real-output response q11,800 PROD10min VALID
#      p50 52.13ms / p99 64.92ms / p99.9 98.75ms).
#   Inflight swept {96,128,192,256}: 128<->192 p99 is noise (keep 128). At the
#   knee the GPU is ~99.3% kernel-busy (rocm-smi 92.7% coarse), <=6.5pp cross-GPU imbalance ->
#   compute-bound; the gate-poly (Plan 42, GAUC-cleared) is now ON by DEFAULT and is required to hold
#   this knee (poly OFF @ q10,200 = INVALID p99 363ms — see GATE_POLY below); next roof move needs fp4
#   on attention (dead — Plan 40) or the VMEM/VALU reorder (Plan 43, dead — see Plan 43). The
#   DEFAULT conf is now **qps12200** (the degree-5 GOLD knee); set
#   CONF=user_mi355x8_nve_b64_qps9600_PROD10min.conf for the conservative tail-margin point
#   (~9,595 q/s, p99 ~70 ms, more headroom below the cliff).
#   NB: the SiLU fold's cliff is chaotic — RE-BENCHMARK it (A/B both arms) when other levers
#   land; see the SILU_MAIN block below.
#   NB: RESID=pin only engages with the GR tree on branch `residual-out-fold` (vendored
#   fp8tuned_ext) or DLRM_FP8TUNED_EXT_PATH set; otherwise it safely falls back to the bf16 add
#   (= 9,700 knee) with a one-time warning. Accuracy-cleared: GAUC 0.78628734 >= fp8 0.78624080.
#
# Windowed (1024 sliding-window) attention is NOT permitted in the closed submission, so the
# GOLD default runs FULL CAUSAL (DLRM_HSTU_MAX_ATTN_LEN=0) with the C1-off last-layer
# target-only lever and the fast Win-B fp8 causal-mask attention kernel (FASTMASK + FULLGRID +
# OCCTUNE). The legacy windowed C1-on path (10,595 q/s b48 / ~11,994 q/s b64) is retained as an
# explicit opt-in via WINDOW=1 (for reference / non-submission probing only).
#
# Stack: §6a A-FUSE+C1 precision on top of the §6 production NVE run (Plan 18 own-device grant,
# Plan 20 parallel ckpt, Plan 21 dispatch/clamp levers). Also exposes a hook for extra precision
# env (fp4 spike etc) via EXTRA_ENV.
#
# Usage (env-driven):
#   ./run_gold.sh                                                            # GOLD: b64 full-causal KNEE 12,200
#   CONF=user_mi355x8_nve_b64_qps9600_PROD10min.conf ./run_gold.sh           # conservative tail-margin point (~9,595, p99 ~70ms)
#   WINDOW=1 ./run_gold.sh                                                   # legacy windowed C1-on b48 10,600 (NOT submission-legal)
#   FP4_FUSED=1 ./run_gold.sh                                                # reproduce the CLOSED fused-MXFP4-K-pack cert (NET-NEGATIVE: INVALID @ knee, do NOT ship)
#   BATCH=64 CONF=user_mi355x8_nve_b64_qps9500_PROF90s.conf TAG=probe ./run_gold.sh   # knee probe
#
#   WINDOW        : 0 = GOLD full-causal default | 1 = legacy windowed C1-on path [default 0]
#   FUSE_EPILOGUE : 1 = D2 silu epilogue fusion on, 0 = off  [default 0 GOLD / 1 WINDOW]
#   BATCH         : server max batch size (BATCH_SIZE)        [default 64 GOLD / 48 WINDOW]
#                   b64 is the certified sweet spot: larger batches REGRESS on the 80ms-p99
#                   Server bar (b80 sits at a flat ~85ms p99 floor across 9.5-9.7k qps =
#                   INVALID; the GPU is ~92% saturated so a bigger batch only adds per-batch
#                   latency, no throughput headroom). Do NOT raise BATCH for the cert. (2026-06-14)
#   INFLIGHT      : DLRM_ZMQ_MAX_INFLIGHT                     [default 128 GOLD / 32 WINDOW]
#   CONF          : USER_CONF filename under benchmarks/      [default b64_qps12200 GOLD (deg-5 gate knee) / b48_qps10600 WINDOW]
#   MAX_ATTN_LEN  : DLRM_HSTU_MAX_ATTN_LEN (0 = full causal)  [default 0 GOLD / 1024 WINDOW]
#   BUFFER_OPS    : AMDGCN_USE_BUFFER_OPS_GFX950 (1 needed on the OLD container; no-op on a
#                   fresh patch-dropped build where buffer ops are on by default) [default 1]
#   BF16_GATHER   : store item_id NVE table in bf16 so the LinearUVM byte-copy gather emits
#                   bf16 directly, dropping the per-iteration fp16->bf16 cast (bit-exact;
#                   ~+4.6% completed-qps, 2026-06-14) [default 1; set 0 to revert]
#   DLRM_NVE_GPU_CACHE_GB
#                 : LinearUVM item_id GPU cache size [default 32 after odd-set geometry fix
#                   plus larger-cache retest; 16GB was the prior Plan 55 default]
#   DLRM_NVE_REPLACEMENT_MARGIN
#                 : full-set NVE replacement guard; evict only when new priority beats the
#                   victim counter by this margin [default 1.0 = previous behavior]
#   DLRM_NVE_SOURCE_AWARE_ADMISSION
#                 : tag item_id history as source 0 and item_candidate_id as source 1 for
#                   NVE admission/metrics; pair with DLRM_NVE_CANDIDATE_INSERT_SCALE
#   DLRM_NVE_EXT_CACHE_GB
#                 : Plan 60 P2 candidate-only extension cache size; when >0,
#                   item_candidate_id uses a separate LinearUVM cache backed by the same item table
#   DLRM_LOCAL_SMALL_TABLE_LOOKUP
#                 : bypass NVE NoCache for fully resident user_id/item_category_id tables and
#                   gather them locally as bf16 [default 1; set 0 to revert]
#   OPTIMIZED_EMBED_LOOKUP
#                 : use the direct CustomJaggedTensor embedding path that avoids merged-KJT glue
#                   [default 1; one-batch compare guard enabled by default]
#   RESID         : OUTPROJ residual-fold mode (DLRM_HSTU_FP8_RESID) — folds the `out + x`
#                   skip into the fp8 GEMM via the hipBLASLt beta*C epilogue, dropping the
#                   standalone bf16 add kernel (CUDAFunctor_add in ## stu_compute_output ##):
#                     pin  = fold + pinned OUTPROJ solution 454444 (robust +23..+32% net on the
#                            GEMM block) [default] — lifts the b64 knee 9,700->9,800
#                     off  = disabled, separate bf16 add (pre-fold path)
#                     heur = fold + stock hipBLASLt heuristic — NOT submission-safe (mis-picks at
#                            n~131072 -> net-negative; build-dependent); probing only
#                   e2e (b64, 80ms bar): pin@9800 VALID p99 72.2ms vs off@9800 INVALID; at 9700
#                   pin p99 71.0 vs off 79.6ms. NOT bit-exact (fp32 accumulate, rounds once —
#                   measured slightly MORE accurate); accuracy re-cert PASSED. Uses the vendored
#                   generative_recommenders.ops.triton.fp8tuned_ext (or DLRM_FP8TUNED_EXT_PATH).
#   DELTA_VK      : 1 = fold the UVQK V/K GEMM output cast to e4m3 (DLRM_HSTU_FP8_DELTA_VK_FP8OUT),
#                   deleting the standalone bf16->e4m3 `float8_copy` over all rows (~2% worker GPU);
#                   0 = pre-fold two-round cast. [default 1] — iso-throughput p99 -4.5ms at the b64
#                   9900 bar (79.3->74.8ms), accuracy re-cert PASSED (GAUC 0.78628749 ~ fp16 ref).
#   LN_ADD        : 1 = fold ContextualPreprocessor content final LayerNorm + side-output additions
#                   into a Triton LN epilogue (DLRM_HSTU_FUSE_PREPROCESSOR_LN_ADD); 0 = stock
#                   LayerNorm + two PyTorch bf16 adds. [default 1 after Plan 61 P3;
#                   q11,500 PROD10min VALID and Offline GAUC PASS]
#   OUTLN_FAST    : 1 = use Plan 62 GOLD-inference output LN/gate path
#                   (DLRM_HSTU_OUTPUT_LN_FAST_INFER) after output-LN fp8 scale calibration;
#                   skips mean/rstd materialization and dropout seed plumbing. [default 1]
#   SKIP_BF16_NOOP_CAST
#                 : 1 = skip NVE post-lookup `.to(torch.bfloat16)` when the sequence embedding
#                   tensor is already bf16 (DLRM_SKIP_BF16_NOOP_CAST). [default 1 after Plan 62 P3;
#                   q11,700 PROF90s p99.9 111.25ms -> 79.37ms, q11,800 remains INVALID]
#   VECTORIZE_RESPONSE_BUFFER
#                 : 1 = use Plan 63 real-output response-buffer vectorization
#                   (DLRM_VECTORIZE_RESPONSE_BUFFER), replacing the PerformanceOnly per-query
#                   Python copy loop with a vectorized NumPy fill. [default 1 after q11,800
#                   PROD10min VALID; preserves real predictions]
#   REUSE_PINNED_OUTPUT
#                 : 1 = use Plan 64 P0 worker-side pinned-output buffer reuse
#                   (DLRM_REUSE_PINNED_OUTPUT), caching the CPU prediction D2H destination by
#                   shape instead of allocating a fresh pinned tensor every batch. [default 1
#                   after q11,900 PROD10min P0/P2 VALID; bit-exact; set 0 to revert]
#   REUSE_LOADGEN_BUFFERS
#                 : 1 = use Plan 64 P1 LoadGen response-buffer ring
#                   (DLRM_REUSE_LOADGEN_BUFFERS), reusing shape-keyed NumPy response buffers
#                   through a ring so async QuerySamplesComplete never sees overwritten payloads.
#                   [default 1 after q11,970 PROD10min P0/P2+P1 VALID with LoadGen 6.0.16; set 0 to revert]
#   RESPONSE_BUFFER_RING_SIZE
#                 : number of response buffers per shape for P1 [default 1024 after q11,970 PROD10min ring-size promotion]
#   MODE          : performance | accuracy                    [default performance]
#   CKPT_MODE     : "1" Option B (default) | "dcp" Option A
#   EXTRA_ENV     : extra "-e FOO=bar -e BAZ=qux" passed verbatim (fp4 spike, etc.)
#                   TODO(cache-prefetch): Server current-batch prefetch is invalid because it
#                   doubles lookup work on the critical path; revisit for Offline with a side
#                   stream where upcoming sample keys can be warmed ahead of demand.
#   PROFILE       : 1 = enable torch profiler (uses *_PROF*; see PROF_* below)
#   TIMING        : 1 = enable timing_stats JSONL (`timing.jsonl`) on worker/loadgen hot paths;
#                   0 = cert-hygiene default after Plan 64 P2. [default 0; set 1 for profiling]
#   TAG           : artifact-dir suffix
#   CONTAINER     : docker container [default dlrmv3-e2e723]
#   WAIT          : safety cap (s) to block for completion [default 3600] — observed perf wall
#                   is ~36 min (~14 min Triton fp8 autotune warmup + 10 min LoadGen + backlog
#                   drain); 3600 leaves headroom for a cold autotune cache. Cold launches after
#                   a fresh container rebuild, LoadGen rebuild, or cleared Triton cache can sit in
#                   checkpoint load + Triton full-autotune for ~10-15 min before LoadGen starts;
#                   that is expected and should be waited out unless the log shows a real error.
#                   IMPORTANT: the FIRST measured run after a fresh container/LoadGen/cache rebuild
#                   may still be a throwaway perf run (page cache + Triton autotune + NVE cache not
#                   yet steady) and can show queue-collapse despite a healthy stack. Warm the stack
#                   once, verify clean teardown/VRAM drain, then trust the second run for perf.
#                   The launcher waits on the benchmark PROCESS finishing (not a fixed timer), so
#                   it returns the instant the run completes and never returns while it is still on
#                   the GPUs; WAIT only bounds a stuck/hung run. If the cap is hit while still
#                   running, the launcher exits non-zero (leaving the run in place) so a chaining
#                   orchestrator will NOT start a competing run.
#   RUN_RETRIES   : retry count for transient NVE/MPIMemBlock startup failures [default 1].
#                   Some hosts occasionally return `RuntimeError: invalid argument` during
#                   MPIMemBlock setup; the launcher kills leftover MPI ranks, waits for drain,
#                   and retries instead of waiting for STARTUP_GRACE on a doomed launch.
#   HIP_VISIBLE_DEVICES
#                 : full comma-separated physical GPU list exposed to every MPI rank.
#                   Required with DLRM_HIP_FULL_VISIBILITY=1 so early per-rank set_device(local_rank)
#                   happens before fbgemm/torchrec/triton touch cuda:0. [default 0,1,2,3,4,5,6,7]
#   DLRM_HIP_FULL_VISIBILITY
#                 : keep the full HIP_VISIBLE_DEVICES list visible to every rank and let the
#                   harness pin each rank via torch.cuda.set_device(local_rank) before torch-heavy
#                   imports. Prevents all ranks collapsing onto cuda:0 during early imports. [default 1]
set -euo pipefail

CONTAINER="${CONTAINER:-dlrmv3-e2e723}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
HIP_FULL_VISIBILITY="${DLRM_HIP_FULL_VISIBILITY:-1}"
case "$HIP_FULL_VISIBILITY" in 0|1) ;; *) echo "[err] DLRM_HIP_FULL_VISIBILITY must be 0|1 (got '$HIP_FULL_VISIBILITY')" >&2; exit 64;; esac
# WINDOW selects the attention regime. GOLD (default) = full-causal, the closed-submission
# recipe. WINDOW=1 = legacy 1024 sliding-window C1-on (reference / non-submission only).
WINDOW="${WINDOW:-0}"
if [ "$WINDOW" = "1" ]; then
  # ── legacy windowed C1-on path (NOT submission-legal) ──
  FUSE_EPILOGUE="${FUSE_EPILOGUE:-1}"
  BATCH="${BATCH:-48}"
  INFLIGHT="${INFLIGHT:-32}"
  CONF="${CONF:-user_mi355x8_nve_b48_qps10600_PROD10min.conf}"
  MAX_ATTN_LEN="${MAX_ATTN_LEN:-1024}"
  LASTLAYER_TARGETS_ONLY="${LASTLAYER_TARGETS_ONLY:-0}"
  ATTN_FASTMASK="${ATTN_FASTMASK:-0}"
  ATTN_FULLGRID="${ATTN_FULLGRID:-0}"
  ATTN_OCCTUNE="${ATTN_OCCTUNE:-0}"
else
  # ── GOLD: b64 full-causal Win-B(occ), C1-off — current knee 11,800 q/s ──
  # NB (2026-07-03/04, clean node): Plan 55 guarded dense-STU graph moved the clean-node knee
  # from q10,200 to q10,300. Plan 55 P2 offset reuse then made q10,400 PROD10min VALID
  # (10,395.39 completed q/s; p99 69.87ms). The NVE LinearUVM odd-set cache geometry fix
  # lets the item_id GPU cache move to 16GB and makes q10,500 PROD10min VALID (10,495.02
  # completed q/s; p99 68.97ms). Static candidate metadata plus optimized embedding lookup
  # then makes q10,600 PROD10min VALID (10,594.65 completed q/s; p99 67.66ms). Plan 57 local
  # small-table lookup plus 32GB odd-set NVE cache then makes q10,900 PROD10min VALID
  # (10,894.82 completed q/s; p99 61.50ms), q11,000 PROD10min VALID
  # (10,994.71 completed q/s; p99 64.11ms), q11,100 PROD10min VALID
  # (11,095.71 completed q/s; p99 68.53ms; clean-node p99 61.70ms),
  # Plan 59 q11,400 PROD10min VALID (11,395.30 completed q/s;
  # p50 52.2ms / p99 62.3ms / p99.9 72.4ms), Plan 61 P1 q11,500 PROD10min VALID
  # (11,495.85 completed q/s; p50 52.17ms / p99 63.35ms / p99.9 77.31ms),
  # and Plan 61 P3 lnaddfold q11,500 PROD10min VALID (11,495.73 completed q/s;
  # p50 52.18ms / p99 62.17ms / p99.9 71.73ms). Plan 62 disabling per-layer
  # sort-by-length makes q11,600 PROD10min VALID (11,595.72 completed q/s;
  # p50 51.62ms / p99 63.68ms / p99.9 75.41ms). Plan 62 output-LN fast-inference
  # path makes q11,700 PROD10min VALID (11,694.83 completed q/s; p50 52.20ms /
  # p99 69.98ms / p99.9 221.21ms). Plan 63 vectorized real-output response
  # buffer makes q11,800 PROD10min VALID (11,794.84 completed q/s; p50 52.13ms /
  # p99 64.92ms / p99.9 98.75ms).
  FUSE_EPILOGUE="${FUSE_EPILOGUE:-0}"
  BATCH="${BATCH:-64}"
  INFLIGHT="${INFLIGHT:-128}"
  CONF="${CONF:-user_mi355x8_nve_b64_qps12200_PROD10min.conf}"
  MAX_ATTN_LEN="${MAX_ATTN_LEN:-0}"
  LASTLAYER_TARGETS_ONLY="${LASTLAYER_TARGETS_ONLY:-1}"
  ATTN_FASTMASK="${ATTN_FASTMASK:-1}"
  ATTN_FULLGRID="${ATTN_FULLGRID:-1}"
  ATTN_OCCTUNE="${ATTN_OCCTUNE:-1}"
fi
MODE="${MODE:-performance}"
# MLPerf scenario. Default Server (perf cert). AccuracyOnly is conventionally run
# Offline (LoadGen issues the full QSL once); the governing C1 accuracy cert used
# Offline, so set SCENARIO=Offline for accuracy re-certs to match it.
SCENARIO="${SCENARIO:-Server}"
DATASET_PERCENTAGE="${DATASET_PERCENTAGE:-1}"
CKPT_MODE="${CKPT_MODE:-1}"
# Buffer ops: required at runtime on the OLD container (built with the retired patch); a fresh
# patch-dropped build has them on by default, so =1 is a harmless no-op there.
BUFFER_OPS="${BUFFER_OPS:-1}"
BF16_GATHER="${BF16_GATHER:-1}"
# Residual fold (OUTPROJ out+x via hipBLASLt beta*C): off | pin | heur. Default PIN: accuracy
# re-cert PASSED (GAUC 0.78628734 >= fp8 baseline 0.78624080, ~100% of fp16 ref) and it lifts the
# b64 knee 9,700 -> 9,800 (2026-06-14, MI355X). `heur` is NOT submission-safe (stock hipBLASLt
# heuristic mis-picks at the prod token count n~131072 -> net-negative; build-dependent) — probing
# only. `pin` is robust (pinned solution 454444 + safe fallback to the bf16 add). Set RESID=off to
# revert to the pre-fold separate-add path.
RESID="${RESID:-pin}"
case "$RESID" in off|pin|heur) ;; *) echo "[err] RESID must be off|pin|heur (got '$RESID')" >&2; exit 64;; esac
# delta V/K fp8-out fold: cast the UVQK V/K GEMM output straight to e4m3 in the GEMM
# epilogue so delta_hstu_mha sees k/v already fp8 — deletes the standalone bf16->e4m3
# `float8_copy` over all rows (~2% of worker GPU). Default ON: at the b64 9900 bar it cut
# p99 79.3 -> 74.8ms (-4.5ms) at iso-throughput (9895 q/s), and accuracy re-cert PASSED
# (GAUC 0.78628749 = 100.000% of fp16 ref, >= pin 0.78628734 >= fp8 baseline 0.78624080;
# 2026-06-14 MI355X). Not bit-exact (one fp32->e4m3 round vs two). Set DELTA_VK=0 to revert.
DELTA_VK="${DELTA_VK:-1}"
case "$DELTA_VK" in 0|1) ;; *) echo "[err] DELTA_VK must be 0|1 (got '$DELTA_VK')" >&2; exit 64;; esac
# Main-layer SiLU(u) epilogue fold (DLRM_HSTU_FUSE_SILU_MAINONLY): defers the standalone
# F.silu(u) on the MAIN (non-last) layers into the output LN/concat epilogue (silu_u=True),
# deleting the standalone SiLU launch (~2% worker GPU). Unlike the global D2 FUSE_EPILOGUE
# it does NOT touch the delta/last-layer path, so it COMPOSES with LASTLAYER_TARGETS_ONLY
# (D2 cannot). Reuses the certified SILU_U kernel epilogue (fast_dividef SiLU, GAUC-cleared
# in the windowed C1-on cert).
#   DEFAULT 1 (promoted 2026-06-16): lifts the b64 knee 10,000 -> 10,100 q/s (q10100 fold
#   VALID p99 73.9ms vs GOLD INVALID p99 99.7ms; both INVALID @ q10200). Iso-load @ q10000:
#   p99 -4.3ms (-5.7%), p99.9 -37.7ms. GAUC 0.78628750 >= baseline 0.78624080 (PASS). Set
#   SILU_MAIN=0 to revert to the pre-fold standalone-SiLU GOLD path.
#   *** CLIFF CAVEAT / RE-BENCHMARK MARKER ***: the saturation cliff just past the knee is
#   chaotic for this arm (a single fold run hit p99 195ms @q10100 and 942ms @q10200 while
#   p50 stayed ~58ms = backlog blow-ups, not steady state). The controlled same-session A/B
#   is the trustworthy read. Because the fold trades GPU time for a sharper tail near
#   saturation, it CAN backfire / interact with other latency levers: RE-BENCHMARK this
#   fold (A/B both arms) whenever another optimization lands, before trusting the knee.
SILU_MAIN="${SILU_MAIN:-1}"
case "$SILU_MAIN" in 0|1) ;; *) echo "[err] SILU_MAIN must be 0|1 (got '$SILU_MAIN')" >&2; exit 64;; esac
LN_ADD="${LN_ADD:-${DLRM_HSTU_FUSE_PREPROCESSOR_LN_ADD:-1}}"
case "$LN_ADD" in 0|1) ;; *) echo "[err] LN_ADD must be 0|1 (got '$LN_ADD')" >&2; exit 64;; esac
SORT_BY_LENGTH="${SORT_BY_LENGTH:-${DLRM_HSTU_SORT_BY_LENGTH:-0}}"
case "$SORT_BY_LENGTH" in 0|1) ;; *) echo "[err] SORT_BY_LENGTH must be 0|1 (got '$SORT_BY_LENGTH')" >&2; exit 64;; esac
OUTLN_FAST="${OUTLN_FAST:-${DLRM_HSTU_OUTPUT_LN_FAST_INFER:-1}}"
case "$OUTLN_FAST" in 0|1) ;; *) echo "[err] OUTLN_FAST must be 0|1 (got '$OUTLN_FAST')" >&2; exit 64;; esac
SKIP_BF16_NOOP_CAST="${SKIP_BF16_NOOP_CAST:-${DLRM_SKIP_BF16_NOOP_CAST:-1}}"
case "$SKIP_BF16_NOOP_CAST" in 0|1) ;; *) echo "[err] SKIP_BF16_NOOP_CAST must be 0|1 (got '$SKIP_BF16_NOOP_CAST')" >&2; exit 64;; esac
VECTORIZE_RESPONSE_BUFFER="${VECTORIZE_RESPONSE_BUFFER:-${DLRM_VECTORIZE_RESPONSE_BUFFER:-1}}"
case "$VECTORIZE_RESPONSE_BUFFER" in 0|1) ;; *) echo "[err] VECTORIZE_RESPONSE_BUFFER must be 0|1 (got '$VECTORIZE_RESPONSE_BUFFER')" >&2; exit 64;; esac
REUSE_PINNED_OUTPUT="${REUSE_PINNED_OUTPUT:-${DLRM_REUSE_PINNED_OUTPUT:-1}}"
case "$REUSE_PINNED_OUTPUT" in 0|1) ;; *) echo "[err] REUSE_PINNED_OUTPUT must be 0|1 (got '$REUSE_PINNED_OUTPUT')" >&2; exit 64;; esac
REUSE_LOADGEN_BUFFERS="${REUSE_LOADGEN_BUFFERS:-${DLRM_REUSE_LOADGEN_BUFFERS:-1}}"
case "$REUSE_LOADGEN_BUFFERS" in 0|1) ;; *) echo "[err] REUSE_LOADGEN_BUFFERS must be 0|1 (got '$REUSE_LOADGEN_BUFFERS')" >&2; exit 64;; esac
RESPONSE_BUFFER_RING_SIZE="${RESPONSE_BUFFER_RING_SIZE:-${DLRM_RESPONSE_BUFFER_RING_SIZE:-1024}}"
case "$RESPONSE_BUFFER_RING_SIZE" in ''|*[!0-9]*) echo "[err] RESPONSE_BUFFER_RING_SIZE must be a non-negative integer (got '$RESPONSE_BUFFER_RING_SIZE')" >&2; exit 64;; esac
# Gate-poly (DLRM_HSTU_GATE_POLY): exp-free SiLU gate. GATE_POLY_DEG=5 is the q12,200 GOLD default;
# GATE_POLY_DEG=9 reproduces the prior degree-9 polynomial gate. Both replace the transcendental
# fast_expf/fast_dividef SiLU with an odd polynomial in u=x², removing the v_exp/v_rcp hazard stalls.
#   DEFAULT 1/5 — load-bearing at the knee. Degree-5 passed production Offline GAUC
#   (0.7862875110), q12,200 PROD10min, and TEST08; exact-gate/poly-off remains available only for
#   lower-qps archaeology and is expected to fall off the near-knee cliff.
GATE_POLY="${GATE_POLY:-1}"
case "$GATE_POLY" in 0|1) ;; *) echo "[err] GATE_POLY must be 0|1 (got '$GATE_POLY')" >&2; exit 64;; esac
GATE_POLY_DEG="${GATE_POLY_DEG:-${DLRM_HSTU_GATE_POLY_DEG:-5}}"
case "$GATE_POLY_DEG" in 5|9) ;; *) echo "[err] GATE_POLY_DEG must be 5|9 (got '$GATE_POLY_DEG')" >&2; exit 64;; esac
# LinearUVM item_id GPU cache. Default 40GB after the Plan 64 c40p6 promotion (was 32GB after the
# odd-num_sets geometry fix + Plan 57 re-test). The larger cache pairs with the P6 runtime knobs
# (below) as the "c40p6" combo: a same-window q11,800 PROD10min A/B on a clean reserved node
# (2026-07-07) had c40p6 p99 66.89ms / p99.9 91.91ms vs 32GB/no-P6 71.28ms / 117.67ms (both VALID,
# same completed q/s ~11,795) — a −4.4ms p99 / −26ms p99.9 tail tightening at the same figure of
# record (q11,800; q11,900 stays INVALID). Bit-exact (residency only). Override with
# DLRM_NVE_GPU_CACHE_GB=... for sensitivity probes; +8GB/GPU is graph-off-affordable (Plan 60 §2).
NVE_GPU_CACHE_GB="${DLRM_NVE_GPU_CACHE_GB:-40}"
NVE_INT8_GATHER="${DLRM_NVE_INT8_GATHER:-0}"
OPTIMIZED_EMBED_LOOKUP="${OPTIMIZED_EMBED_LOOKUP:-${DLRM_OPTIMIZED_EMBED_LOOKUP:-1}}"
OPTIMIZED_EMBED_COMPARE="${OPTIMIZED_EMBED_COMPARE:-${DLRM_OPTIMIZED_EMBED_COMPARE:-1}}"
OPTIMIZED_EMBED_COMPARE_LIMIT="${OPTIMIZED_EMBED_COMPARE_LIMIT:-${DLRM_OPTIMIZED_EMBED_COMPARE_LIMIT:-1}}"
LOCAL_SMALL_TABLE_LOOKUP="${LOCAL_SMALL_TABLE_LOOKUP:-${DLRM_LOCAL_SMALL_TABLE_LOOKUP:-1}}"
case "${NVE_INT8_GATHER,,}" in 0|1|item_id|true|yes|false|no) ;; *) echo "[err] DLRM_NVE_INT8_GATHER must be 0|1|item_id|true|yes|false|no (got '$NVE_INT8_GATHER')" >&2; exit 64;; esac
case "$OPTIMIZED_EMBED_LOOKUP" in 0|1) ;; *) echo "[err] OPTIMIZED_EMBED_LOOKUP must be 0|1 (got '$OPTIMIZED_EMBED_LOOKUP')" >&2; exit 64;; esac
case "$OPTIMIZED_EMBED_COMPARE" in 0|1) ;; *) echo "[err] OPTIMIZED_EMBED_COMPARE must be 0|1 (got '$OPTIMIZED_EMBED_COMPARE')" >&2; exit 64;; esac
case "$LOCAL_SMALL_TABLE_LOOKUP" in 0|1) ;; *) echo "[err] LOCAL_SMALL_TABLE_LOOKUP must be 0|1 (got '$LOCAL_SMALL_TABLE_LOOKUP')" >&2; exit 64;; esac
# Static candidate metadata is a Server perf shortcut for the 2048-candidate inference path.
# AccuracyOnly uses the training/eval max-candidate shape (32), so run_accuracy.sh disables it.
HSTU_UNIFORM_TARGETS_METADATA="${HSTU_UNIFORM_TARGETS_METADATA:-${DLRM_HSTU_UNIFORM_TARGETS_METADATA:-1}}"
case "$HSTU_UNIFORM_TARGETS_METADATA" in 0|1) ;; *) echo "[err] HSTU_UNIFORM_TARGETS_METADATA must be 0|1 (got '$HSTU_UNIFORM_TARGETS_METADATA')" >&2; exit 64;; esac
# Plan 60: graph capture is parked and GOLD defaults to eager dense-STU. The graph
# runner remains opt-in for reproducing Plan 55/58/60 probes, but q11,400 PROD10min
# graph-off is VALID and leaves enough HBM headroom for NVE extension-cache work.
STU_GRAPH="${STU_GRAPH:-0}"
case "$STU_GRAPH" in 0|1) ;; *) echo "[err] STU_GRAPH must be 0|1 (got '$STU_GRAPH')" >&2; exit 64;; esac
# Plan 55 guarded dense-STU hipGraph runner. Default was ON after q10,300 PROD sustained VALID;
# Plan 60 turns it OFF by default after graph-off q11,400 PROD stayed valid and saved HBM.
# The 557056-row bucket used to trip hipBLASLt capture on ROCm (HIPBLAS_STATUS_INTERNAL_ERROR),
# so it is capped at the largest trusted bucket (540672) with eager fallback above it.
# Plan 58 Item A (BANKED 2026-07-05) root-caused that failure to VRAM *fragmentation* (each
# bucket captured into its own private graph mempool) and PROVED a shared graph mempool
# (STU_GRAPH_SHARED_POOL=1) makes 557056 capture TRUSTED, bit-exact, under the loaded worker.
# BUT the same-session q11,100 PROD10min A/B showed capturing the large bucket does NOT help
# the tail — it slightly REGRESSED it (p99 61.7->65.2, p99.9 66.3->87.4ms vs the cap-540672
# private-pool baseline), because (a) the dominant large band is 589824 (36*16384), not
# 557056, and is un-coverable (613 distinct max_seq_len + greedy warmup ordering exhaust the
# 16-bucket budget on small batches), and (b) the tail bursts are O(L^2)-compute-bound, not
# launch-jitter, so graph capture can't shrink them. So the cap stays 540672 and the shared
# pool defaults OFF (kept as an opt-in flag for future work). See plan 58 §6.
#   STU_GRAPH_MAX_ROWS: cap for graph capture; buckets with cap_rows above it stay eager.
#     540672 = defensive floor (2nd-largest bucket); raising it did not pay (see above).
STU_GRAPH_MAX_ROWS="${STU_GRAPH_MAX_ROWS:-540672}"
STU_GRAPH_MAX_BUCKETS="${STU_GRAPH_MAX_BUCKETS:-16}"
STU_GRAPH_L_GRAN="${STU_GRAPH_L_GRAN:-16384}"
STU_GRAPH_N_GRAN="${STU_GRAPH_N_GRAN:-1}"
STU_GRAPH_CAPTURE_ERROR_MODE="${STU_GRAPH_CAPTURE_ERROR_MODE:-thread_local}"
STU_GRAPH_DISABLE_ON_FAIL="${STU_GRAPH_DISABLE_ON_FAIL:-1}"
MEMORY_TRACE="${MEMORY_TRACE:-0}"
case "$MEMORY_TRACE" in 0|1) ;; *) echo "[err] MEMORY_TRACE must be 0|1 (got '$MEMORY_TRACE')" >&2; exit 64;; esac
MEMORY_TRACE_SNAPSHOT="${MEMORY_TRACE_SNAPSHOT:-0}"
case "$MEMORY_TRACE_SNAPSHOT" in 0|1) ;; *) echo "[err] MEMORY_TRACE_SNAPSHOT must be 0|1 (got '$MEMORY_TRACE_SNAPSHOT')" >&2; exit 64;; esac
MEMORY_TRACE_INTERVAL="${MEMORY_TRACE_INTERVAL:-2}"
MEMORY_EMPTY_CACHE_AFTER_DIRECT_WARMUP="${MEMORY_EMPTY_CACHE_AFTER_DIRECT_WARMUP:-0}"
case "$MEMORY_EMPTY_CACHE_AFTER_DIRECT_WARMUP" in 0|1) ;; *) echo "[err] MEMORY_EMPTY_CACHE_AFTER_DIRECT_WARMUP must be 0|1 (got '$MEMORY_EMPTY_CACHE_AFTER_DIRECT_WARMUP')" >&2; exit 64;; esac
# Plan 64 §19 (2026-07-07): warm-up count. The direct-warmup loop (inference_server.warmup) rotates
# *real* dataset slices through backend.predict, warming the NVE cache + JIT shape coverage; it MUST
# stay count-based (per-step collective Barrier under sharded sparse -> a wall-clock loop would
# desync ranks and gloo-abort). We tested WARMUP_STEPS=900 (+41s of cache warming) to try to fix the
# 90s PROF probe's tail spikes — it did NOT: direct measurement showed backend batches are uniformly
# ~40ms (0 batches >80ms, early bins as clean as late even at =10), so the p99 tail is queue-wait at
# rho~1 (drain ~11,905 q/s ~= arrival), NOT cold cache (batch time is attention-bound, not
# lookup-bound). So the big warm-up bought nothing on latency. Default dialed back to a modest 60
# (covers fp8 JIT shapes + light cache) — set WARMUP_STEPS=900 for full cache warmth on a cert if
# desired. MLPerf permits unlimited untimed warm-up. Revert-to-original with WARMUP_STEPS=10.
WARMUP_STEPS="${WARMUP_STEPS:-60}"
case "$WARMUP_STEPS" in ''|*[!0-9]*) echo "[err] WARMUP_STEPS must be a non-negative integer (got '$WARMUP_STEPS')" >&2; exit 64;; esac
BATCHING_WARMUP_STEPS="${BATCHING_WARMUP_STEPS:-15}"
case "$BATCHING_WARMUP_STEPS" in ''|*[!0-9]*) echo "[err] BATCHING_WARMUP_STEPS must be a non-negative integer (got '$BATCHING_WARMUP_STEPS')" >&2; exit 64;; esac
STU_GRAPH_STATS="${STU_GRAPH_STATS:-0}"
case "$STU_GRAPH_STATS" in 0|1) ;; *) echo "[err] STU_GRAPH_STATS must be 0|1 (got '$STU_GRAPH_STATS')" >&2; exit 64;; esac
STU_GRAPH_STATS_TOPK="${STU_GRAPH_STATS_TOPK:-24}"
STU_GRAPH_DEFER_CAPTURE="${STU_GRAPH_DEFER_CAPTURE:-0}"
case "$STU_GRAPH_DEFER_CAPTURE" in 0|1) ;; *) echo "[err] STU_GRAPH_DEFER_CAPTURE must be 0|1 (got '$STU_GRAPH_DEFER_CAPTURE')" >&2; exit 64;; esac
STU_GRAPH_DEFER_CAPTURE_STEPS="${STU_GRAPH_DEFER_CAPTURE_STEPS:-128}"
STU_GRAPH_FREEZE_AFTER_WARMUP="${STU_GRAPH_FREEZE_AFTER_WARMUP:-0}"
case "$STU_GRAPH_FREEZE_AFTER_WARMUP" in 0|1) ;; *) echo "[err] STU_GRAPH_FREEZE_AFTER_WARMUP must be 0|1 (got '$STU_GRAPH_FREEZE_AFTER_WARMUP')" >&2; exit 64;; esac
# Memory tracing / graph stats should include the existing STU-graph verbose logs unless explicitly disabled.
if [ "$MEMORY_TRACE" = "1" ] || [ "$STU_GRAPH_STATS" = "1" ]; then
  STU_GRAPH_VERBOSE="${STU_GRAPH_VERBOSE:-1}"
else
  STU_GRAPH_VERBOSE="${STU_GRAPH_VERBOSE:-0}"
fi
# Plan 58 A3: shared graph mempool across STU buckets (sizes the capture reservation to the
# largest bucket's transient peak, not the sum of <=16 private pools). Bit-exact
# (trust-after-replay still gates). DEFAULT OFF — Item A banked (raising the cap to use it
# regressed the tail; see the STU_GRAPH_MAX_ROWS note above). Opt-in for future capture work.
STU_GRAPH_SHARED_POOL="${STU_GRAPH_SHARED_POOL:-0}"
case "$STU_GRAPH_SHARED_POOL" in 0|1) ;; *) echo "[err] STU_GRAPH_SHARED_POOL must be 0|1 (got '$STU_GRAPH_SHARED_POOL')" >&2; exit 64;; esac
EXTRA_ENV="${EXTRA_ENV:-}"
if [ -z "${HSTU_TRITON_FULL_AUTOTUNE+x}" ]; then
  if [ "$MODE" = "accuracy" ]; then
    HSTU_TRITON_FULL_AUTOTUNE=0
  else
    HSTU_TRITON_FULL_AUTOTUNE=1
  fi
fi
case "$HSTU_TRITON_FULL_AUTOTUNE" in 0|1) ;; *) echo "[err] HSTU_TRITON_FULL_AUTOTUNE must be 0|1 (got '$HSTU_TRITON_FULL_AUTOTUNE')" >&2; exit 64;; esac
# Plan 58 A2: expandable-segments allocator. Lets the caching allocator grow/relocate
# segments so fragmented free VRAM becomes usable for the large one-time 557056 capture
# reservation. Env-only, default OFF (it shifts steady-state allocator behavior, so it needs
# its own q-knee A/B before promotion). EXPANDABLE_SEGMENTS=1 opts in for the capture probe.
EXPANDABLE_SEGMENTS="${EXPANDABLE_SEGMENTS:-0}"
case "$EXPANDABLE_SEGMENTS" in 0|1) ;; *) echo "[err] EXPANDABLE_SEGMENTS must be 0|1 (got '$EXPANDABLE_SEGMENTS')" >&2; exit 64;; esac
if [ "$EXPANDABLE_SEGMENTS" = "1" ]; then
  EXTRA_ENV="$EXTRA_ENV -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
fi
# Fused MXFP4 K-pack (Plan 36/37): emit the packed e2m1 + e8m0 K *inside* the fp8 UVQK GEMM
# epilogue (patched-hipBLASLt Tensile MXScaleE), deleting the standalone Triton K-pack pass.
# Requires the patched fused-pack library (lib_merged grafts the MXScaleE TN solution onto the
# stock library) via HIPBLASLT_TENSILE_LIBPATH, plus the mxfp4_ext op via DLRM_MXFP4_EXT_PATH.
#   *** CERTIFIED NET-NEGATIVE (2026-06-27, chi2872) — DO NOT SHIP. ***
#   e2e A/B at the b64 q10,200 GOLD knee: fp4-fused = INVALID (p99 387 ms) vs fp8 = VALID
#   (p99 78 ms). The fp8+lib_merged control is VALID (p99 76.6 ms), so the regression is the
#   fp4 path itself, not the library swap. Kernel is bit-exact (scales exact, 0 non-tie diffs)
#   and the run is stable (no wedge), but fp4 QK attention is VALU-bound (Plan 40) so a free
#   pack does not flip it net-positive. This flag exists ONLY to reproduce that closed cert.
FP4_FUSED="${FP4_FUSED:-0}"
case "$FP4_FUSED" in 0|1) ;; *) echo "[err] FP4_FUSED must be 0|1 (got '$FP4_FUSED')" >&2; exit 64;; esac
if [ "$FP4_FUSED" = "1" ]; then
  FP4_LIBPATH="${FP4_LIBPATH:-/work/route_a_probe/lib_merged/library}"
  MXFP4_EXT_PATH="${MXFP4_EXT_PATH:-/work/route_a_probe}"
  EXTRA_ENV="$EXTRA_ENV -e DLRM_HSTU_FP4_QK_FUSED=1 -e DLRM_MXFP4_EXT_PATH=$MXFP4_EXT_PATH -e HIPBLASLT_TENSILE_LIBPATH=$FP4_LIBPATH"
fi
# Plan 64 P6 — ROCm runtime launch/sync-latency knobs (bit-exact; the "P6" half of the c40p6 combo):
#   HSA_ENABLE_INTERRUPT=0  -> detect GPU completion by POLLING instead of an interrupt, cutting the
#                              per-batch wakeup latency of the load-bearing full-device sync.
#   HIP_FORCE_DEV_KERNARG=1 -> device-memory kernargs -> lower per-launch host latency.
#   AMD_DIRECT_DISPATCH=1   -> dispatch kernels from the calling thread -> lower launch latency.
# These do NOT change sync semantics (unlike the Plan 63 P2 defer arm that regressed) — only the
# wait/launch *implementation* gets cheaper. Promoted with the 40GB cache: the pair beat 32GB/no-P6
# by −4.4ms p99 / −26ms p99.9 in a same-window q11,800 PROD10min A/B (both VALID, 2026-07-07 clean
# node). P6 alone is neutral (it needs the cache-40 headroom); the validated unit is the pair.
# Default ON; set P6=0 to revert to interrupt-driven wait + stock dispatch.
P6="${P6:-1}"
case "$P6" in 0|1) ;; *) echo "[err] P6 must be 0|1 (got '$P6')" >&2; exit 64;; esac
if [ "$P6" = "1" ]; then
  EXTRA_ENV="$EXTRA_ENV -e HSA_ENABLE_INTERRUPT=0 -e HIP_FORCE_DEV_KERNARG=1 -e AMD_DIRECT_DISPATCH=1"
fi
PROFILE="${PROFILE:-0}"
TIMING="${TIMING:-0}"
case "$TIMING" in 0|1) ;; *) echo "[err] TIMING must be 0|1 (got '$TIMING')" >&2; exit 64;; esac
WAIT="${WAIT:-3600}"
RUN_RETRIES="${RUN_RETRIES:-1}"
RETRY_DRAIN_SECONDS="${RETRY_DRAIN_SECONDS:-20}"
TAG="${TAG:-$(echo "$CONF" | sed 's/^user_mi355x8_nve_//; s/\.conf$//')_$([ "$WINDOW" = 1 ] && echo win || echo gold)_fuse${FUSE_EPILOGUE}$([ "$SILU_MAIN" = 1 ] && echo "_silumain")$([ "$LN_ADD" = 1 ] && echo "_lnadd")$([ "$SORT_BY_LENGTH" = 0 ] && echo "_nosort")$([ "$OUTLN_FAST" = 1 ] && echo "_outlnfast")$([ "$SKIP_BF16_NOOP_CAST" = 1 ] && echo "_skipbf16noop")$([ "$VECTORIZE_RESPONSE_BUFFER" = 1 ] && echo "_vecresp")$([ "$REUSE_PINNED_OUTPUT" = 1 ] && echo "_reusepinout")$([ "$REUSE_LOADGEN_BUFFERS" = 1 ] && echo "_lgring${RESPONSE_BUFFER_RING_SIZE}")$([ "$P6" = 1 ] && echo "_p6")$([ "$GATE_POLY" = 1 ] && [ "$GATE_POLY_DEG" != 9 ] && echo "_gate${GATE_POLY_DEG}")_c${NVE_GPU_CACHE_GB}$([ "$RESID" != off ] && echo "_resid$RESID")$([ "$FP4_FUSED" = 1 ] && echo "_fp4fused")}"

# ── Pre-launch amdgpu/KFD floor guard ─────────────────────────────────────────
# NVE multi-GPU import relies on dmabuf / DRM-PRIME KFD paths that are driver-sensitive.
# The lowest known-good stack is amdgpu 6.16.6 (chi2810); newer 6.16.13 hosts are
# also expected to work. This is a host KMD requirement, not a container-userspace
# requirement.
MIN_AMDGPU_VERSION="${MIN_AMDGPU_VERSION:-6.16.6}"
if [ "${SKIP_AMDGPU_CHECK:-0}" != "1" ]; then
  if ! command -v modinfo >/dev/null 2>&1; then
    echo "[err] amdgpu guard: modinfo is required to validate the host driver; set SKIP_AMDGPU_CHECK=1 only for intentional probing." >&2
    exit 72
  fi
  if ! amdgpu_version="$(modinfo amdgpu 2>/dev/null | awk '/^version:/ {version=$2} END {if (version) print version}')"; then
    amdgpu_version=""
  fi
  if [ -z "$amdgpu_version" ]; then
    echo "[err] amdgpu guard: could not determine host amdgpu module version; set SKIP_AMDGPU_CHECK=1 only for intentional probing." >&2
    exit 72
  fi
  if ! MIN_AMDGPU_VERSION="$MIN_AMDGPU_VERSION" AMDGPU_VERSION="$amdgpu_version" python3 - <<'PY'
import os
import re
import sys


def parse(version: str):
    nums = [int(x) for x in re.findall(r"\d+", version)[:3]]
    return tuple(nums + [0] * (3 - len(nums)))


current = parse(os.environ["AMDGPU_VERSION"])
minimum = parse(os.environ["MIN_AMDGPU_VERSION"])
sys.exit(0 if current >= minimum else 1)
PY
  then
    echo "[err] amdgpu guard: host amdgpu ${amdgpu_version} is below required ${MIN_AMDGPU_VERSION} for NVE." >&2
    echo "[err] Upgrade the host KMD/driver, or set SKIP_AMDGPU_CHECK=1 only for non-cert debugging." >&2
    exit 72
  fi
fi

# ── Pre-launch CPU topology/governor guard ───────────────────────────────────
# This benchmark needs the full host CPU scheduling surface to keep LoadGen,
# batching, ZMQ, dataset collation, and result handling from starving the GPUs.
# A post-reboot node with SMT disabled showed only CPUs 0-127 online and turned
# q10,500/q10,600 into pathological queueing failures while GPU predict stayed
# healthy. Refuse that state up front.
MIN_ONLINE_CPUS="${MIN_ONLINE_CPUS:-256}"
REQUIRE_SMT="${REQUIRE_SMT:-1}"
REQUIRE_CPU_GOVERNOR="${REQUIRE_CPU_GOVERNOR:-performance}"
if [ "${SKIP_CPU_CHECK:-0}" != "1" ]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[err] CPU guard: python3 is required for topology validation; set SKIP_CPU_CHECK=1 to bypass." >&2
    exit 71
  fi

  ask_cpu_repair() {
    local prompt="$1"
    local reply=""
    if [ "${CPU_GUARD_ASSUME_YES:-0}" = "1" ]; then
      return 0
    fi
    if [ -r /dev/tty ]; then
      printf "%s [y/N] " "$prompt" > /dev/tty
      read -r reply < /dev/tty || reply=""
    else
      echo "[err] CPU guard: cannot prompt for repair without a TTY; re-run interactively or set CPU_GUARD_ASSUME_YES=1." >&2
      return 1
    fi
    case "$reply" in
      y|Y|yes|YES|Yes) return 0 ;;
      *) return 1 ;;
    esac
  }

  run_cpu_guard_check() {
    MIN_ONLINE_CPUS="$MIN_ONLINE_CPUS" REQUIRE_SMT="$REQUIRE_SMT" REQUIRE_CPU_GOVERNOR="$REQUIRE_CPU_GOVERNOR" python3 - <<'PY'
import os
import sys
from pathlib import Path

cpu_root = Path("/sys/devices/system/cpu")
min_online = int(os.environ.get("MIN_ONLINE_CPUS", "256"))
require_smt = os.environ.get("REQUIRE_SMT", "1") == "1"
required_governor = os.environ.get("REQUIRE_CPU_GOVERNOR", "performance").strip()


def parse_cpu_list(spec):
    cpus = []
    for part in spec.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


def read_text(path):
    return path.read_text().strip()


errors = []
online_spec = read_text(cpu_root / "online")
offline_spec = read_text(cpu_root / "offline") if (cpu_root / "offline").exists() else ""
online_cpus = parse_cpu_list(online_spec)

if len(online_cpus) < min_online:
    errors.append(
        f"only {len(online_cpus)} CPUs online ({online_spec}); need >= {min_online}"
    )

smt_control = "<missing>"
smt_active = "<missing>"
smt_control_path = cpu_root / "smt" / "control"
smt_active_path = cpu_root / "smt" / "active"
if smt_control_path.exists():
    smt_control = read_text(smt_control_path)
if smt_active_path.exists():
    smt_active = read_text(smt_active_path)
if require_smt and (smt_control != "on" or smt_active != "1"):
    errors.append(f"SMT is not active (control={smt_control}, active={smt_active})")

bad_governors = []
missing_governors = 0
if required_governor:
    for cpu in online_cpus:
        gov_path = cpu_root / f"cpu{cpu}" / "cpufreq" / "scaling_governor"
        try:
            governor = read_text(gov_path)
        except OSError:
            missing_governors += 1
            continue
        if governor != required_governor:
            bad_governors.append(f"cpu{cpu}={governor}")
    if missing_governors:
        errors.append(f"{missing_governors} online CPUs have unreadable cpufreq governors")
    if bad_governors:
        sample = ", ".join(bad_governors[:8])
        more = "" if len(bad_governors) <= 8 else f", ... (+{len(bad_governors) - 8})"
        errors.append(f"governor mismatch; expected {required_governor}: {sample}{more}")

print(f"  online={online_spec} ({len(online_cpus)} CPUs)")
print(f"  offline={offline_spec or '<none>'}")
print(f"  smt_control={smt_control} smt_active={smt_active}")
print(f"  required_governor={required_governor or '<skipped>'}")
if errors:
    for err in errors:
        print(f"  ERROR: {err}")
    sys.exit(1)
PY
  }

  enable_smt_now() {
    python3 - <<'PY'
from pathlib import Path

path = Path("/sys/devices/system/cpu/smt/control")
path.write_text("on")
print("  SMT control set to on")
PY
  }

  set_cpu_governor_now() {
    REQUIRE_CPU_GOVERNOR="$REQUIRE_CPU_GOVERNOR" python3 - <<'PY'
import os
from pathlib import Path

required = os.environ.get("REQUIRE_CPU_GOVERNOR", "performance").strip()
cpu_root = Path("/sys/devices/system/cpu")
failures = []

for path in sorted(
    cpu_root.glob("cpu[0-9]*/cpufreq/scaling_governor"),
    key=lambda p: int(p.parts[-3][3:]),
):
    try:
        path.write_text(required)
    except OSError as exc:
        failures.append(f"{path}: {exc}")

if failures:
    print(f"  ERROR: failed to set {len(failures)} CPU governors")
    for failure in failures[:8]:
        print(f"  ERROR: {failure}")
    raise SystemExit(1)
print(f"  Set CPU governors to {required}")
PY
  }

  if cpu_guard_out="$(run_cpu_guard_check)"; then
    : # CPU topology/governors satisfy the run requirements
  else
    echo "[warn] CPU guard: node CPU requirements are not satisfied." >&2
    echo "$cpu_guard_out" >&2
    if echo "$cpu_guard_out" | grep -q "SMT is not active"; then
      if ask_cpu_repair "Enable SMT now?"; then
        if ! enable_smt_now; then
          echo "[err] CPU guard: failed to enable SMT; refusing to launch." >&2
          exit 71
        fi
      else
        echo "[err] CPU guard: SMT repair declined; refusing to launch." >&2
        exit 71
      fi
    fi
    if echo "$cpu_guard_out" | grep -Eq "governor mismatch|unreadable cpufreq governors"; then
      if ask_cpu_repair "Set all online CPU governors to ${REQUIRE_CPU_GOVERNOR}?"; then
        if ! set_cpu_governor_now; then
          echo "[err] CPU guard: failed to set CPU governors; refusing to launch." >&2
          exit 71
        fi
      else
        echo "[err] CPU guard: governor repair declined; refusing to launch." >&2
        exit 71
      fi
    fi

    if cpu_guard_out="$(run_cpu_guard_check)"; then
      : # repaired successfully
    else
      echo "[err] CPU guard: node CPU requirements are still not satisfied after repair attempt." >&2
      echo "$cpu_guard_out" >&2
      echo "[err] Expected SMT on, >=${MIN_ONLINE_CPUS} online CPUs, and governor=${REQUIRE_CPU_GOVERNOR}." >&2
      echo "[err] Set SKIP_CPU_CHECK=1 only for intentional non-cert probing." >&2
      exit 71
    fi
  fi
fi

# REPO = this dlrmv3-rocm checkout (derived from the script's own location); host-side
# artifacts land under $REPO/artifacts. Override REPO to redirect the output dir.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STAMP=$(date -u +%Y%m%dT%H%M%S)
OUT="$REPO/artifacts/gold_${TAG}_${STAMP}"
mkdir -p "$OUT"

PROF_ENV=""
if [ "$PROFILE" = "1" ]; then
  PROF_ENV="-e DLRM_TORCH_PROFILER=1 -e DLRM_TORCH_PROFILER_SKIP=${PROF_SKIP:-120} -e DLRM_TORCH_PROFILER_N=${PROF_N:-20} -e DLRM_TORCH_PROFILER_DIR=$OUT/torchprof"
  mkdir -p "$OUT/torchprof"
fi
TIMING_ENV="-e DLRM_TIMING=$TIMING"
if [ "$TIMING" = "1" ]; then
  TIMING_ENV="$TIMING_ENV -e DLRM_TIMING_LOG=$OUT/timing.jsonl"
fi

# ── Pre-launch GPU clock-determinism guard (Plan 64 P8) ──────────────────────
# Server GOLD REQUIRES pinned SCLK. Under load the GPU DVFS ripples ~2075-2400 MHz;
# the dips stretch the O(L^2) attention burst tail past the 80ms p99 bar — q11,900 was INVALID
# (p99 100ms) on auto clocks but VALID (p99 70.27ms) with determinism (same c40p6 stack; Plan 64
# §17 update 5). Pin SCLK for minimal Server variation (analogous to the CPU governor=performance
# guard). It PERSISTS until `rocm-smi --resetperfdeterminism` (record it in the submission system
# description). Offline has no latency bar, so it defaults to resetting determinism and allowing
# auto clocks to avoid an unnecessary SCLK cap; set CLOCK_DETERMINISM=1 to force pinned clocks for
# Offline A/B.
if [ -z "${CLOCK_DETERMINISM+x}" ]; then
  if [ "$SCENARIO" = "Offline" ]; then
    CLOCK_DETERMINISM=0
  else
    CLOCK_DETERMINISM=1
  fi
fi
CLOCK_SCLK="${CLOCK_SCLK:-2400}"
if [ "$CLOCK_DETERMINISM" = "1" ] && command -v rocm-smi >/dev/null 2>&1; then
  if timeout 60 rocm-smi --setperfdeterminism "$CLOCK_SCLK" >/dev/null 2>&1; then
    echo "[launch] GPU clock determinism ON (SCLK<=${CLOCK_SCLK} MHz) — required for Server GOLD"
  else
    echo "[warn] failed to set perfdeterminism ${CLOCK_SCLK} MHz (privilege?); the q11,900 tail may regress —" >&2
    echo "[warn]   set CLOCK_DETERMINISM=0 to silence, or run a lower CONF (e.g. qps11800/qps9600)." >&2
  fi
elif [ "$CLOCK_DETERMINISM" = "0" ] && command -v rocm-smi >/dev/null 2>&1; then
  if timeout 60 rocm-smi --resetperfdeterminism >/dev/null 2>&1; then
    echo "[launch] GPU clock determinism OFF (auto clocks; default for Offline)"
  else
    echo "[warn] failed to reset perfdeterminism (privilege?); GPU clocks may remain capped from a prior Server run." >&2
  fi
fi

# ── Pre-launch VRAM guard ─────────────────────────────────────────────────────
# Refuse to launch unless every GPU has >= MIN_FREE_GB free. A busy node — most
# often a prior run's detached `run_benchmark` workers that didn't exit (they linger holding
# ~50-150 GB/GPU, sometimes D-state) — otherwise turns a launch into a ZMQ-starvation / GPU
# contention collapse (p50 in the tens of seconds, INVALID) or an OOM crash, not a clean run.
# Default is 210 GB for the bf16/GOLD path. For the MI350P true-int8 memory-fit path, use a
# config-aware lower floor: measured b16/i32/cache1 peak is ~110 GB/GPU, so require ~120 GB
# plus small conservative increments for larger cache/batch/inflight settings.
# Override with SKIP_VRAM_CHECK=1, or tune MIN_FREE_GB. Soft no-op if rocm-smi is unavailable.
if [ -z "${MIN_FREE_GB+x}" ]; then
  case "${NVE_INT8_GATHER,,}" in
    1|item_id|true|yes)
      MIN_FREE_GB="$(awk -v cache="$NVE_GPU_CACHE_GB" -v batch="$BATCH" -v inflight="$INFLIGHT" '
        BEGIN {
          v = 119 + cache
          if (batch > 16) v += (batch - 16) * 0.35
          if (inflight > 32) v += (inflight - 32) * 0.05
          if (v < 120) v = 120
          printf "%.0f", v
        }')"
      ;;
    *)
      MIN_FREE_GB=210
      ;;
  esac
fi
if [ "${SKIP_VRAM_CHECK:-0}" != "1" ] && command -v rocm-smi >/dev/null 2>&1; then
  if guard_out="$(timeout 30 rocm-smi --showmeminfo vram 2>/dev/null | awk -v min="$MIN_FREE_GB" '
        /VRAM Total Memory \(B\)/      { split($1,g,/[][]/); gpu=g[2]; tot=$NF }
        /VRAM Total Used Memory \(B\)/ { free=(tot-$NF)/1e9; printf "  GPU %s: %.0f GB free\n", gpu, free; if (free < min+0) bad=1 }
        END { exit bad+0 }')"; then
    : # all GPUs have >= MIN_FREE_GB free
  else
    echo "[err] VRAM guard: a GPU has < ${MIN_FREE_GB} GB free — refusing to launch (busy node / leftover workers?)." >&2
    echo "$guard_out" >&2
    echo "[err] Free the GPUs (kill stale run_benchmark workers; reboot if they are D-state), or set SKIP_VRAM_CHECK=1 to bypass." >&2
    exit 70
  fi
fi

echo "[launch] run_gold regime=$([ "$WINDOW" = 1 ] && echo WINDOWED-C1on || echo GOLD-fullcausal) max_attn_len=$MAX_ATTN_LEN fuse_epilogue=$FUSE_EPILOGUE silu_main=$SILU_MAIN ln_add=$LN_ADD sort_by_length=$SORT_BY_LENGTH outln_fast=$OUTLN_FAST skip_bf16_noop=$SKIP_BF16_NOOP_CAST vector_response=$VECTORIZE_RESPONSE_BUFFER reuse_pinned_output=$REUSE_PINNED_OUTPUT reuse_loadgen_buffers=$REUSE_LOADGEN_BUFFERS ring=$RESPONSE_BUFFER_RING_SIZE p6=$P6 gate_poly=$GATE_POLY gate_poly_deg=$GATE_POLY_DEG hip_visible=$HIP_VISIBLE_DEVICES hip_full_visibility=$HIP_FULL_VISIBILITY stu_graph=$STU_GRAPH stu_graph_max_rows=$STU_GRAPH_MAX_ROWS stu_graph_shared_pool=$STU_GRAPH_SHARED_POOL expandable_segments=$EXPANDABLE_SEGMENTS fp4_fused=$FP4_FUSED resid=$RESID delta_vk=$DELTA_VK nve_cache_gb=$NVE_GPU_CACHE_GB nve_int8=$NVE_INT8_GATHER min_free_gb=$MIN_FREE_GB local_small=$LOCAL_SMALL_TABLE_LOOKUP opt_embed=$OPTIMIZED_EMBED_LOOKUP uniform_targets=$HSTU_UNIFORM_TARGETS_METADATA warmup=$WARMUP_STEPS/$BATCHING_WARMUP_STEPS timing=$TIMING batch=$BATCH inflight=$INFLIGHT mode=$MODE scenario=$SCENARIO conf=$CONF -> $OUT"

VRAM_SAMPLER_PID=""
if [ "$MEMORY_TRACE" = "1" ] && command -v rocm-smi >/dev/null 2>&1; then
  VRAM_LOG="$OUT/vram.csv"
  echo "ts,gpu,total_B,used_B" > "$VRAM_LOG"
  (
    while :; do
      ts="$(date +%s.%N)"
      timeout 10 rocm-smi --showmeminfo vram 2>/dev/null | awk -v ts="$ts" '
        /VRAM Total Memory \(B\)/      { split($1,g,/[][]/); gpu=g[2]; total=$NF }
        /VRAM Total Used Memory \(B\)/ { print ts "," gpu "," total "," $NF; fflush() }
      ' >> "$VRAM_LOG"
      sleep "$MEMORY_TRACE_INTERVAL"
    done
  ) &
  VRAM_SAMPLER_PID="$!"
  echo "[launch] MEMORY_TRACE=1 writing host VRAM samples to $VRAM_LOG every ${MEMORY_TRACE_INTERVAL}s"
fi

cleanup_vram_sampler() {
  if [ -n "${VRAM_SAMPLER_PID:-}" ]; then
    kill "$VRAM_SAMPLER_PID" >/dev/null 2>&1 || true
    wait "$VRAM_SAMPLER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_vram_sampler EXIT

launch_benchmark() {
docker exec -d \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e DLRM_HIP_FULL_VISIBILITY="$HIP_FULL_VISIBILITY" \
  -e DLRM_ROCM_NVE=1 -e DLRM_USE_MPI_LOOKUP=1 -e DLRM_NVE_PARALLEL_CKPT_LOAD="$CKPT_MODE" \
  -e DLRM_NVE_GPU_CACHE_GB="$NVE_GPU_CACHE_GB" -e DLRM_NVE_INT8_GATHER="$NVE_INT8_GATHER" -e DLRM_SKIP_TORCH_BOOTSTRAP=1 \
  -e DLRM_NVE_BF16_GATHER="$BF16_GATHER" \
  -e DLRM_ZMQ_TRACE=0 \
  -e DLRM_ZMQ_MAX_INFLIGHT="$INFLIGHT" -e DLRM_HSTU_TRITON_FULL_AUTOTUNE="$HSTU_TRITON_FULL_AUTOTUNE" -e DLRM_CLAMP_OOB_IDS=1 \
  -e DLRM_HSTU_FP8_GEMM=1 -e DLRM_HSTU_FP8_ATTN=1 -e DLRM_HSTU_FP8_GEMM_STATIC_ASCALE=1 \
  -e DLRM_HSTU_FP8_FUSE_LN=1 -e DLRM_HSTU_FP8_FUSE_OUTLN=1 -e DLRM_HSTU_FP8_FUSE_QKV=1 \
  -e DLRM_HSTU_FP8_RESID="$RESID" \
  -e DLRM_HSTU_FP8_DELTA_VK_FP8OUT="$DELTA_VK" \
  -e DLRM_HSTU_MAX_ATTN_LEN="$MAX_ATTN_LEN" \
  -e DLRM_HSTU_LASTLAYER_TARGETS_ONLY="$LASTLAYER_TARGETS_ONLY" \
  -e DLRM_HSTU_ATTN_FASTMASK="$ATTN_FASTMASK" -e DLRM_HSTU_ATTN_FULLGRID="$ATTN_FULLGRID" \
  -e DLRM_HSTU_ATTN_OCCTUNE="$ATTN_OCCTUNE" -e AMDGCN_USE_BUFFER_OPS_GFX950="$BUFFER_OPS" \
  -e DLRM_HSTU_FUSE_EPILOGUE="$FUSE_EPILOGUE" \
  -e DLRM_HSTU_FUSE_SILU_MAINONLY="$SILU_MAIN" \
  -e DLRM_HSTU_FUSE_PREPROCESSOR_LN_ADD="$LN_ADD" \
  -e DLRM_HSTU_SORT_BY_LENGTH="$SORT_BY_LENGTH" \
  -e DLRM_HSTU_OUTPUT_LN_FAST_INFER="$OUTLN_FAST" \
  -e DLRM_SKIP_BF16_NOOP_CAST="$SKIP_BF16_NOOP_CAST" \
  -e DLRM_VECTORIZE_RESPONSE_BUFFER="$VECTORIZE_RESPONSE_BUFFER" \
  -e DLRM_REUSE_PINNED_OUTPUT="$REUSE_PINNED_OUTPUT" \
  -e DLRM_REUSE_LOADGEN_BUFFERS="$REUSE_LOADGEN_BUFFERS" \
  -e DLRM_RESPONSE_BUFFER_RING_SIZE="$RESPONSE_BUFFER_RING_SIZE" \
  -e DLRM_HSTU_GATE_POLY="$GATE_POLY" \
  -e DLRM_HSTU_GATE_POLY_DEG="$GATE_POLY_DEG" \
  -e DLRM_HSTU_STU_GRAPH="$STU_GRAPH" \
  -e DLRM_HSTU_UNIFORM_TARGETS_METADATA="$HSTU_UNIFORM_TARGETS_METADATA" \
  -e DLRM_OPTIMIZED_EMBED_LOOKUP="$OPTIMIZED_EMBED_LOOKUP" \
  -e DLRM_OPTIMIZED_EMBED_COMPARE="$OPTIMIZED_EMBED_COMPARE" \
  -e DLRM_OPTIMIZED_EMBED_COMPARE_LIMIT="$OPTIMIZED_EMBED_COMPARE_LIMIT" \
  -e DLRM_LOCAL_SMALL_TABLE_LOOKUP="$LOCAL_SMALL_TABLE_LOOKUP" \
  -e DLRM_HSTU_STU_GRAPH_VERBOSE="$STU_GRAPH_VERBOSE" \
  -e DLRM_HSTU_STU_GRAPH_L_GRAN="$STU_GRAPH_L_GRAN" \
  -e DLRM_HSTU_STU_GRAPH_N_GRAN="$STU_GRAPH_N_GRAN" \
  -e DLRM_HSTU_STU_GRAPH_MAX_BUCKETS="$STU_GRAPH_MAX_BUCKETS" \
  -e DLRM_HSTU_STU_GRAPH_DISABLE_ON_FAIL="$STU_GRAPH_DISABLE_ON_FAIL" \
  -e DLRM_HSTU_STU_GRAPH_CAPTURE_ERROR_MODE="$STU_GRAPH_CAPTURE_ERROR_MODE" \
  -e DLRM_HSTU_STU_GRAPH_MAX_ROWS="$STU_GRAPH_MAX_ROWS" \
  -e DLRM_HSTU_STU_GRAPH_SHARED_POOL="$STU_GRAPH_SHARED_POOL" \
  -e DLRM_HSTU_STU_GRAPH_STATS="$STU_GRAPH_STATS" \
  -e DLRM_HSTU_STU_GRAPH_STATS_TOPK="$STU_GRAPH_STATS_TOPK" \
  -e DLRM_HSTU_STU_GRAPH_DEFER_CAPTURE="$STU_GRAPH_DEFER_CAPTURE" \
  -e DLRM_HSTU_STU_GRAPH_DEFER_CAPTURE_STEPS="$STU_GRAPH_DEFER_CAPTURE_STEPS" \
  -e DLRM_HSTU_STU_GRAPH_FREEZE_AFTER_WARMUP="$STU_GRAPH_FREEZE_AFTER_WARMUP" \
  -e DLRM_MEMORY_TRACE="$MEMORY_TRACE" \
  -e DLRM_MEMORY_TRACE_SNAPSHOT="$MEMORY_TRACE_SNAPSHOT" \
  -e DLRM_MEMORY_EMPTY_CACHE_AFTER_DIRECT_WARMUP="$MEMORY_EMPTY_CACHE_AFTER_DIRECT_WARMUP" \
  $PROF_ENV $TIMING_ENV $EXTRA_ENV \
  -e NUM_WORKERS=8 -e BATCH_SIZE="$BATCH" -e WARMUP_STEPS="$WARMUP_STEPS" \
  -e DLRM_BATCHING_WARMUP_STEPS="$BATCHING_WARMUP_STEPS" \
  -e DATASET_PERCENTAGE="$DATASET_PERCENTAGE" \
  -e DATASET_PATH=/work/dlrmv3_preprocessed_full \
  -e CHECKPOINT_PATH=/work/dlrmv3_trained_checkpoint/dlrm-v3-checkpoint \
  -e USER_CONF="$CONF" -e DLRM_LAZY_PREPROCESSED=1 \
  -e PYTHONPATH=/work/pynve-rocm/python -e PYTHONUNBUFFERED=1 \
  -w /work/dlrm-v3-harness-rocm/benchmarks \
  "$CONTAINER" \
  bash -lc "exec bash MI355_run_performance_harness.sh SCENARIO=$SCENARIO MODE=$MODE OUTPUT_DIR=$OUT > $OUT/run.log 2>&1"
}

cleanup_benchmark_processes() {
  docker exec "$CONTAINER" bash -lc '
    for pat in run_benchmark.py MI355_run_performance_harness.sh mpirun; do
      pgrep -f "$pat" | xargs -r kill -TERM 2>/dev/null || true
    done
    sleep 5
    for pat in run_benchmark.py MI355_run_performance_harness.sh mpirun; do
      pgrep -f "$pat" | xargs -r kill -KILL 2>/dev/null || true
    done
  ' >/dev/null 2>&1 || true
}

is_transient_nve_init_failure() {
  grep -Eq 'MPIMemBlock|validate_and_fence|RuntimeError: invalid argument|EBUSY' "$OUT/run.log" 2>/dev/null
}

# Wait on the benchmark PROCESS finishing rather than a fixed timer, so we never return while
# the run is still on the GPUs. WAIT is only a safety cap for a stuck/hung run.
#   rc=0  completed (perf summary written, or accuracy/process-exit with a summary present)
#   rc=1  process exited but no summary (crash) — see run.log
#   rc=2  cap hit while still running — run LEFT in place, completion NOT signalled
rc=1
attempt=1
max_attempts=$((RUN_RETRIES + 1))
STARTUP_GRACE="${STARTUP_GRACE:-900}"
while (( attempt <= max_attempts )); do
  if (( attempt > 1 )); then
    prev=$((attempt - 1))
    [ -f "$OUT/run.log" ] && mv "$OUT/run.log" "$OUT/run.attempt${prev}.log"
    echo "[retry] relaunching after transient NVE init failure (attempt ${attempt}/${max_attempts})"
  fi

  launch_benchmark
  echo "[launch] started (detached in $CONTAINER, attempt ${attempt}/${max_attempts}); waiting on completion, cap ${WAIT}s ..."

  rc=0
  start=$SECONDS
  started=0
  transient_startup_failure=0
  # The harness loads the dense+sparse checkpoint and does MPI setup BEFORE run_benchmark.py
  # appears, so we wait for the process to first show up (within STARTUP_GRACE) and only treat
  # its disappearance as completion AFTER we've seen it — otherwise the first liveness check
  # races the spawn and falsely reports "exited".
  while :; do
    if grep -q "Result is :" "$OUT/mlperf_log_summary.txt" 2>/dev/null; then
      echo "[done] MLPerf summary written after ~$((SECONDS - start))s"; break
    fi
    # accuracy runs don't emit "Result is :"; the summary appears at the end of the pass
    if [ "$MODE" = "accuracy" ] && grep -q "No errors encountered during test" "$OUT/mlperf_log_summary.txt" 2>/dev/null; then
      echo "[done:accuracy] after ~$((SECONDS - start))s"; break
    fi
    if is_transient_nve_init_failure; then
      echo "[warn] transient NVE/MPIMemBlock startup failure detected; cleaning detached MPI ranks." >&2
      cleanup_benchmark_processes
      transient_startup_failure=1
      rc=1
      break
    fi
    if docker exec "$CONTAINER" pgrep -f run_benchmark.py >/dev/null 2>&1; then
      started=1
    elif [ "$started" = 1 ]; then
      # was running, now gone -> finished (or crashed)
      sleep 5
      if [ -s "$OUT/mlperf_log_summary.txt" ]; then
        echo "[note] benchmark process exited; summary present (after ~$((SECONDS - start))s)"
      else
        echo "[warn] run_benchmark exited with no summary; see $OUT/run.log"; rc=1
      fi
      break
    elif (( SECONDS - start >= STARTUP_GRACE )); then
      echo "[warn] run_benchmark.py never started within ${STARTUP_GRACE}s; see $OUT/run.log"; rc=1; break
    fi
    if (( SECONDS - start >= WAIT )); then
      echo "[warn] still running after ${WAIT}s cap — the run is LEFT in place ($CONTAINER:$OUT)."
      echo "[warn] NOT signalling completion (rc=2); re-check $OUT/mlperf_log_summary.txt or raise WAIT."
      rc=2; break
    fi
    sleep 10
  done

  if [ "$transient_startup_failure" = "1" ] && (( attempt < max_attempts )); then
    echo "[retry] waiting ${RETRY_DRAIN_SECONDS}s for KFD/VRAM drain before retry."
    sleep "$RETRY_DRAIN_SECONDS"
    attempt=$((attempt + 1))
    continue
  fi
  break
done

echo "===== SUMMARY ($TAG) ====="
sed -n '1,24p' "$OUT/mlperf_log_summary.txt" 2>/dev/null || echo "(no summary)"
echo "ARTIFACT=$OUT"
exit $rc
