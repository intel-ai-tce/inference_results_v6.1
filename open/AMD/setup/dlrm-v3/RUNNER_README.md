# dlrm-v3-rocm-runner

Minimal, self-contained runner that reproduces the **GOLD DLRM-v3 ROCm/gfx950 NVE
MLPerf-Inference Server result** for the closed submission — **full causal (C1-off), b64
Win-B + bf16 gather + guarded STU graph + 40GB NVE cache + optimized embedding glue
+ local small-table lookup + LN-add fold + sort-by-length off + output-LN fast-infer + vectorized real-output response + P6 runtime knobs + GPU clock determinism + P0/P2/P1 host hygiene with ring1024 + degree-5 production gate → VALID 12,200 issued q/s, 12,198.90 completed q/s, p99 58.32 ms with latest LoadGen 6.0.16** — on an MI355X (`gfx950:sramecc+`) host,
independent of the `jiaweichen-amd` working repo.

> **Windowed (1024 sliding-window) attention is NOT permitted in the closed submission.** The
> GOLD default therefore runs **full causal**. The legacy windowed C1-on numbers (10,595 q/s b48
> / ~11,994 q/s b64) are retained only as an explicit, non-submission opt-in (`WINDOW=1`).

It does four things: **stage data → clone the cert repos → build the container/stack →
launch the cert run.**

## Quickstart

```bash
# from this repo checkout, which must sit IN the workspace (its parent = $WORKSPACE == /work)
bash run.sh                                          # all stages: data → workspace → submission → run
```

Or per-stage (each script is independently runnable):

```bash
bash scripts/build/setup_data.sh         # stage/verify dataset + checkpoint
bash scripts/build/setup_workspace.sh    # clone the cert port repos + loadgen baseline
bash scripts/build/setup_submission.sh   # (re)create container, build fbgemm + pynve, verify sentinels  (~15 min)
bash scripts/run/run_gold.sh             # launch the GOLD b64 full-causal Server cert  (~22 min wall)
```

Release zips may extract to a package-specific directory such as
`dlrm-v3-rocm-q12200-gold-quickstart`. That is supported: `setup_submission.sh`
derives the runner directory name automatically. If you manually place or rename
the runner under a different workspace subdirectory, set `REPO_SUB=<that-dir>`.

### Vendored / Offline Source Setup

The zip includes source snapshots under:

- `vendor/dlrm-v3-harness-rocm`
- `vendor/dlrm-v3-gr-rocm`
- `vendor/pynve-rocm`

`setup_workspace.sh` uses those vendored snapshots automatically when they are
present, so users without `AMD-AGI/*` GitHub access can still populate the
workspace. The expected unpacked layout is:

```text
$WORKSPACE/
  dlrm-v3-rocm-q12200-gold-quickstart/   # or any runner dir name
    vendor/
      dlrm-v3-harness-rocm/
      dlrm-v3-gr-rocm/
      pynve-rocm/
```

Then run `bash scripts/build/setup_workspace.sh` from the runner directory. It
will copy the vendored private repos to `$WORKSPACE/dlrm-v3-harness-rocm`,
`$WORKSPACE/mlcommons-inference/recommendation/dlrm_v3`, and
`$WORKSPACE/pynve-rocm` if private clones are unavailable. To force online clones,
set `USE_VENDOR_SNAPSHOTS=0`.

For a fully air-gapped host, also pre-stage the public `mlcommons-inference`
sparse checkout and `FBGEMM` tree at the pins in `manifests/resources.lock.yaml`;
the zip vendors the private cert-port repos, not those larger public build trees.

After a fresh container rebuild, LoadGen rebuild, or Triton cache reset, treat the **first measured perf run as a
warm-up/throwaway** if it collapses: checkpoint/page-cache, Triton autotune, and NVE cache may not be steady yet
even after untimed warmup. Verify teardown + VRAM drain, then use the second run as the performance read.

Run a subset with `STAGES`:

```bash
STAGES=workspace,submission,run bash run.sh          # data already present
STAGES=run bash run.sh                                # just re-run the GOLD perf cert
STAGES=run WINDOW=1 bash run.sh                       # legacy windowed C1-on path (NOT submission-legal)
STAGES=accuracy bash run.sh                          # Offline AccuracyOnly cert + GAUC scoring
STAGES=submission,run,accuracy bash run.sh           # build, then perf + accuracy certs
```

### Accuracy cert

`scripts/run/run_accuracy.sh` reuses `run_gold.sh`'s env block with
`MODE=accuracy SCENARIO=Offline` for standalone GAUC scoring, then scores the resulting
`mlperf_log_accuracy.json` with `score_accuracy.py` (streams the ~17 GB log).
The MLPerf DLRM-v3 bar is **relative: GAUC ≥ 99.9% of the fp16 reference**; the certified
q12,200 figure of record uses the production degree-5 gate, which was Offline-recertified:
**GAUC 0.7862875110, PASS**. Re-score an
existing run with `SCORE_ONLY=<artifact-dir> bash scripts/run/run_accuracy.sh`.

For compliance, `scripts/run/_test08_chain.sh` is the source-of-truth TEST08 flow. It runs
an Offline AccuracyOnly reference and an audited Server PerformanceOnly run with the shared
determinism fixes enabled, then calls the official MLCommons verifier.

### Building a submission

The closed-division submission scaffold is **self-contained under `submission/`**. It tracks
runner-owned staging folders that mirror AMD's published MLPerf layout
(`documentation/`, `results/`, `setup/`, `src/`, `systems/`, `tools/`). When ready, generate a
separate final `closed/AMD/...` tree and run the official MLCommons submission tools against
that output. See [`submission/README.md`](submission/README.md).

## Layout

```
run.sh                                  one-shot orchestrator (data → workspace → submission → run)
scripts/build/
  setup_data.sh                         stage/verify dataset + checkpoint (pull only if a *_SRC is given)
  setup_workspace.sh                    clone the 3 cert repos + mlcommons loadgen baseline (sparse), at PINNED commits
  setup_submission.sh                   (re)create container, build stack, write confs, VERIFY cert sentinels
  setup_stack_rocm723.sh                fbgemm gfx950:sramecc+ build + deps + loadgen (Triton UNPATCHED: buffer ops ON)
  build_fbgemm_gfx950_sramecc.sh        fbgemm-gpu from source (applies patches/fbgemm_rocm7.patch)
  patch_triton_compiler.sh              FALLBACK ONLY (no longer called): blanket-disables gfx950 buffer ops; superseded by GR *_multirow routing
  build_pynve_rocm_full.sh              build the certified AMD-AGI/pynve-rocm port in place (HIP)
scripts/run/
  run_gold.sh                           the GOLD Server-cert launcher (b64 full-causal Win-B, qps12200 PROD10min knee + degree-5 gate + GPU clock determinism; WINDOW=1 = legacy windowed C1-on)
  run_accuracy.sh                       Offline AccuracyOnly cert (reuses run_gold's stack) + GAUC scoring
  score_accuracy.py                     streaming GAUC/accuracy/NE scorer for mlperf_log_accuracy.json
submission/                             MLPerf submission scaffold (`code/` + `results/`) for q12,200 assembly
patches/
  fbgemm_rocm7.patch                    the ONLY applied patch (fbgemm build); the port repos are cert source
docs/
  GOLD_PROMOTION.md                     how to check/commit/push a new GOLD (nested-GR gotcha) + which docs to update
  full_causal_run.md                    how to run with C1 (1024 window) OFF — full-causal config + certified numbers
results/
  fullcausal_c1off/                     C1-off characterization + tuning sweep + 600s certs (README + TUNING_PLAN)
                                        + PROFILE_attention_breakdown.md (AMD TraceLens per-kernel profile)
```

## What gets pulled (3 PRIVATE `AMD-AGI/*` repos need git/gh auth)

`setup_workspace.sh` clones, at pinned commits:

| Tree | Repo | Visibility | Pin | Notes |
|---|---|---|---|---|
| harness | `AMD-AGI/dlrm-v3-harness-rocm` | private | `5088d7c` (main) | cloned as `dlrm-v3-harness-rocm/`; launchers cd into its `benchmarks/`. Carries Plan 55/57/59 cache/glue controls, Plan 61 P1 harness sync cleanup, Plan 62 sort-by-length/output path defaults, Plan 63 vectorized real-output response, Plan 64 q11,900/q12,000/q11,970 configs, q12,200 GOLD configs, and the P0/P1 host-output reuse plumbing (Triton cache epoch, worker pinned-output reuse, LoadGen response-buffer ring). |
| GR | `AMD-AGI/dlrm-v3-gr-rocm` | private | `7eb52e9` (main) | the certified `recommendation/dlrm_v3`; cloned INTO the mlcommons baseline at that path. Carries the full-causal C1-off stack incl. Plan 61 P3 preprocessor LN-add fold and the Plan 62 output-LN fast-inference path (GOLD defaults), the q12,200 degree-5 gate selector, default-off Plan 58 STU-graph stats, Plan 59 target-action-emb probe, and the local `datasets` package marker required by the harness import path. |
| pynve | `AMD-AGI/pynve-rocm` | private | `d34a5fe` (main) | built in place by `build_pynve_rocm_full.sh`; carries Plan 55 cache metrics, admission knobs, and odd-set geometry. |
| loadgen | `mlcommons/inference` | public | `393d8ef` | sparse checkout of `loadgen/` only (pip-installed at runtime; LoadGen 6.0.16) |
| FBGEMM | `pytorch/fbgemm` | public | `5beb3e6e` | built from source for `gfx950:sramecc+` |

Only the three `AMD-AGI/*` repos are private and need git/gh auth; `mlcommons/inference` and
`pytorch/fbgemm` are public.

Override any pin with `HARNESS_REPO_REF` / `GR_REPO_REF` / `PYNVE_REPO_REF` (empty = repo default branch).

> The certified ports are **repo-sourced**, not patch-reconstructed. The per-plan patch
> history that produced them lives in those repos; `setup_submission.sh` only **verifies**
> the cert sentinels (harness `DLRM_ROCM_NVE`/`DLRM_CLAMP_OOB_IDS`; GR Plan-12 MPI-lookup +
> Plan-10.1.b reorder fallback + `model_family.py`; pynve Plan-18 own-device grant). The only
> patch this runner applies is `patches/fbgemm_rocm7.patch` during the fbgemm build.

> **Submission note — minor OOV patch (harness):** for the MLPerf submission, a minor
> correctness patch makes out-of-vocabulary embedding ids resolve to a **zero embedding**
> (instead of aliasing onto a valid row) during accuracy — `AMD-AGI/dlrm-v3-harness-rocm@4d382ee`
> (branch `fix/nve-oov-zero-fill`, on top of cert `ec130e3`), vendored in `submission/`. The
> harness pin here is **deliberately not bumped** (stays at `ec130e3`): this runner reproduces
> the certified figures with the cert commit.

## Data (fleet-local / out-of-band — no public URL)

The dataset (**preprocessed** MLPerf DLRM-v3, ~140 GB) and the sharded DCP checkpoint
(~964 GB) must resolve inside the workspace at:

- `$WORKSPACE/dlrmv3_preprocessed_full/`
- `$WORKSPACE/dlrmv3_trained_checkpoint/dlrm-v3-checkpoint/` (`sparse/*.distcp` + `non_sparse.ckpt`)

`setup_data.sh` **pulls** them only if you point it at a source, else it just verifies:

```bash
DATASET_SRC=<local-dir | host:path | rclone-remote:path> \
CHECKPOINT_SRC=<...> bash scripts/build/setup_data.sh
```

If they already live elsewhere on the host, either copy them in (above) or symlink and set
`MOUNT_ROOT` to the lowest common ancestor of the workspace and the data (the build's
preflight prints the exact value to use).

## Prerequisites

- MI355X / `gfx950:sramecc+` host with `/dev/kfd` + `/dev/dri`, docker, and the base image
  `rocm/atom:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom20260511` pullable.
- git/gh auth for the three private `AMD-AGI/*` repos.

The container image is large (~47 GB). The dataset and checkpoint are much
larger: about 140 GB and 964 GB respectively. In practice, reserve at least
~1.2 TB for inputs plus additional space for Docker layers, FBGEMM/pynve builds,
artifacts, and temporary packaging output.

The q12,200 GOLD number is certified on the MI355X host class below. On smaller
or virtual-function hosts, use the package as a functional/non-cert smoke unless
you reproduce the certified host surface. In particular, `run_gold.sh`'s CPU guard
expects >=256 online CPUs, SMT enabled, and `performance` governors; hosts with
fewer CPUs, no SMT, or no cpufreq governors should run non-cert probes with:

```bash
SKIP_CPU_CHECK=1 CLOCK_DETERMINISM=0 bash scripts/run/run_gold.sh
```

`CLOCK_DETERMINISM=0` is also appropriate on VF hosts where `rocm-smi
--setperfdeterminism` is not permitted.

## Validated system configuration (certified reference)

The figures of record below were certified on the exact stack in the table. The **container is the unit
of reproducibility**, but the **host kernel + amdgpu/KFD driver also matter** for the multi-GPU NVE path:
`MPIMemBlock` imports each GPU's shard over dmabuf / DRM-PRIME, and that kernel path is version-sensitive.

| Layer | Certified value (`chi2835`) |
|---|---|
| GPUs | 8× AMD Instinct MI355X (`gfx950:sramecc+`), ~288 GB HBM each |
| Host OS | Ubuntu 24.04.3 LTS (Noble) |
| Host kernel | `6.8.0-107-generic` |
| amdgpu (KMD) | DKMS `6.16.13-2278356.24.04` (module `6.16.13`) |
| Host ROCm | `7.2.0` |
| Container image | `rocm/atom:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom20260511` |
| Container ROCm / HIP | `7.2.3` / `7.2.53211` |
| PyTorch | `2.10.0+rocm7.2.3.git1a270074` |
| Triton | `3.6.0` (unpatched; gfx950 buffer ops ON via GR `*_multirow` jagged routing) |
| torchrec / tensordict | `1.4.0` / `0.12.4` |
| rocBLAS / hipBLASLt | `5.2.0` / `1.2.2` |

> **Host floor (important).** Run on a host with a known-good MI355X KMD stack. The launcher
> enforces `amdgpu >= 6.16.6` by default (`chi2810` is known-good at 6.16.6; 6.16.13 hosts are
> also expected to work). Older Ubuntu 22.04 / kernel `5.15` / host ROCm `7.1.1` stacks have
> been observed to crash at NVE backend init (`MPIMemBlock` → `validate_and_fence -16 / EBUSY`)
> with the identical 7.2.3 container. The container userspace alone is not sufficient.

## Results (figures of record)

The closed submission runs **full causal** (windowed attention not permitted), so the **GOLD**
figure of record is the b64 full-causal point — what `run_gold.sh` produces by default:

| Cert | Scenario | Batch | Attention | Figure | Result |
|---|---|---|---|---|---|
| **Perf (GOLD)** | Server | 64 | full causal | **12,200 issued q/s, 12,198.90 completed q/s, p50 50.04 / p99 58.32 ms** (clean-node 600s PROD with **latest LoadGen 6.0.16 + degree-5 gate + GPU clock determinism + P0/P2/P1 host hygiene, ring1024**; p99.9 69.14 ms — noisy tail diagnostic, the Server bar is p99) | **VALID** |
| **Compliance TEST08** | Offline ref + Server audit | 64 | full causal | Offline ref `349,823` entries + Server audit `4,017` sampled entries; audited Server **VALID @ 12,198.91/s, p99 57.36 ms**; official verifier `num_matched=4017`, `num_unmatched=0`, `num_ne_mismatch=0`, tolerance `0.10%` | **PASS** |

The GOLD recipe = b64, full causal (C1-off last-layer target-only lever), **Win-B(occ)** fp8
attention (`FASTMASK`+`FULLGRID`+`OCCTUNE`), `inflight=128`, **bf16 gather**, OUTPROJ resid-fold
(`RESID=pin`), delta V/K fp8-out fold (`DELTA_VK=1`), main-layer SiLU fold (`SILU_MAIN=1`),
**preprocessor LN-add fold** (`LN_ADD=1`, Plan 61 P3), **sort-by-length off** (`SORT_BY_LENGTH=0`,
Plan 62 — a bit-exact scheduling change), **output-LN fast-inference path** (`OUTLN_FAST=1`, Plan 62 —
fused LN+gate, GAUC re-cert PASS), **production degree-5 SiLU gate** (`GATE_POLY_DEG=5`, GAUC + TEST08 PASS),
**vectorized real-output response buffer** (`VECTORIZE_RESPONSE_BUFFER=1`, Plan 63),
**P6 ROCm runtime latency knobs** (`P6=1`: `HSA_ENABLE_INTERRUPT=0` / `HIP_FORCE_DEV_KERNARG=1` /
`AMD_DIRECT_DISPATCH=1`, Plan 64 "c40p6" — bit-exact, pairs with the 40GB cache),
**P0/P2/P1 host hygiene** (`REUSE_PINNED_OUTPUT=1`, `TIMING=0`, `DLRM_REUSE_LOADGEN_BUFFERS=1` with
`DLRM_RESPONSE_BUFFER_RING_SIZE=1024`: worker-side pinned output-buffer reuse, timing_stats disabled in the
cert hot path, and LoadGen response-buffer ring reuse; bit-exact),
graph-off eager dense-STU (`STU_GRAPH=0`, Plan 60),
40GB LinearUVM item_id GPU cache with odd-set geometry (32→40GB, Plan 64 c40p6), local bf16 lookup for the fully resident
`user_id` / `item_category_id` tables, static candidate metadata, optimized embedding lookup with a one-batch compare guard, buffer
ops on. `run_gold.sh` bakes all of this in as the zero-arg default. This is **+55%** over the prior
b40 ~7,500 q/s full-causal point — the faster Win-B attention kernel is what makes b64 full-causal
feasible (the old kernel forced b40 and a ~7,500 q/s knee). The knee progression that got here:
bf16-gather 9,700 → `RESID=pin` 9,800 → `DELTA_VK` 10,000 → `SILU_MAIN` 10,100 → clean-node
re-measure 10,200 → guarded STU graph 10,300 → offset reuse 10,400 → NVE odd-set 16GB cache
10,500 → optimized embedding glue 10,600 → local small-table lookup + NVE 32GB cache 10,900
→ q11,000 → q11,100 → q11,400 → q11,500 → q11,600 → q11,700 → q11,800 → q11,900. Plan 61 promoted **q11,500** via P1 (harness
sync / `inference_mode` cleanup) and P3 (preprocessor LN-add fold, p99 62.17 ms); Plan 62 then took the
knee to **q11,600** by disabling per-batch sort-by-length (bit-exact, p99 63.68 ms) and to **q11,700
PROD10min VALID** via the output-LN fast-inference path — 11,694.83 completed q/s, **p50 52.20 / p99
69.98 ms** (p99.9 221 ms is a noisy tail diagnostic; the Server bar is p99). The output-LN path is a
fused (non-bit-exact) LN+gate kernel, Offline-recertified **GAUC 0.7858985604 (PASS)**. Plan 63 then took
the real-output path to **q11,800 PROD10min VALID** with response-buffer vectorization — 11,794.84 completed
q/s, **p50 52.13 / p99 64.92 ms** (p99.9 98.75 ms). Plan 64 then promoted the **"c40p6" tail-robustness combo**
(40GB item cache + P6 runtime knobs): a same-window q11,800 PROD10min A/B on a clean reserved node had c40p6
**p99 66.89 / p99.9 91.91 ms** vs the 32GB/no-P6 default **p99 71.28 / p99.9 117.67 ms** (both VALID, ~11,795 q/s)
— a −4.4 ms p99 / −26 ms p99.9 tightening (q11,800 knee at that step). **Plan 64 P8 then cleared q11,900**:
GPU clock determinism (`rocm-smi --setperfdeterminism 2400`, now set by `run_gold.sh`'s clock guard) pins the
2075–2400 MHz SCLK ripple that was stretching the O(L²) attention burst tail — q11,900 PROD10min went from
INVALID p99 100 (auto clocks) to **VALID p99 70.27 / p99.9 88.60 ms** (11,894 q/s), making q11,900 the
new figure of record. Plan 64 P0/P2 then tightened the same q11,900 GOLD point to **p99 58.58 / p99.9
75.66 ms** with `REUSE_PINNED_OUTPUT=1 TIMING=0` (11,894.24 q/s, PROD10min VALID). Plan 64 P1 then moved
the legacy LoadGen figure to q12,000 through safe LoadGen response-buffer reuse, but latest
LoadGen 6.0.16 put q12,000 just over the Server tail bar before the degree-5 gate (p99 82.61 ms
on chi2761). Plan 4 ring1024 then certified **q11,970 PROD10min VALID @ 11,968.67 completed q/s,
p50 51.18 / p99 61.91 / p99.9 84.99 ms**. The production degree-5 SiLU gate then moved the GOLD
knee to **q12,200 PROD10min VALID @ 12,198.90 completed q/s, p50 50.04 / p99 58.32 /
p99.9 69.14 ms**. It was separately GAUC-scored (`0.7862875110`, PASS) and TEST08-verified with
the shared FP8 scale store, fixed attention normalization length, and flush-trim fix:
`num_matched=4017`, `num_unmatched=0`, `num_ne_mismatch=0`; the audited Server leg was also
VALID @ **12,198.91 completed q/s, p99 57.36 ms**. Requires a clean/un-contended node + determinism.
NB q11,800/q11,900/q11,970/q12,000/q12,200
**PROF90s** is a single-window coin-flip (O(L²) tail-burst variance) — rank levers on **PROD10min**, not 90s.
Set `CONF=user_mi355x8_nve_b64_qps9600_PROD10min.conf` for the
conservative tail-margin point (~9,595 q/s, p99 ~70 ms, more headroom below the cliff).
An AMD TraceLens per-kernel profile shows **attention (`_hstu_attn_fwd`) is the dominant cost
(~55% of GPU)** under full causal — see
[`results/fullcausal_c1off/PROFILE_attention_breakdown.md`](results/fullcausal_c1off/PROFILE_attention_breakdown.md)
and the C1-off runbook [`docs/full_causal_run.md`](docs/full_causal_run.md).

<details>
<summary><b>Legacy windowed C1-on figures (1024 sliding window) — NOT submission-legal (reference only)</b></summary>

Reachable via `WINDOW=1`. These run with the 1024 sliding window **on**, which is not permitted
in the closed submission:

| Cert | Scenario | Batch | Figure | Result |
|---|---|---|---|---|
| Perf | Server | 64 | 11,993.9 q/s, p99 58.95 ms | VALID |
| Perf | Server | 48 | 10,595.2 q/s, p99 51.07 ms | VALID |
| Perf | Offline | 64 | 12,586.1 q/s | VALID |
| Perf | Offline | 48 | 9,841.8 q/s | VALID |
| Accuracy | Offline | 48 | GAUC 0.78624 (99.9941% of fp16 ref) | PASS |

```bash
# reference only — NOT submission-legal (windowed attention)
WINDOW=1 BATCH=64 CONF=user_mi355x8_nve_b64_qps12000_PROD10min.conf STAGES=run bash run.sh
```
</details>

> **Triton buffer ops re-enabled (2026-06-07).** The build used to blanket-disable AMD buffer-op
> lowering on gfx950 (`patch_triton_compiler.sh`) to dodge a `CanonicalizePointers` crash in the
> basic `_concat_2D_jagged` kernel — which also turned the optimization off on the dominant
> `_hstu_attn_fwd`. The proper fix routes 2D-jagged concat/split to their mask-based `*_multirow`
> variants on HIP (in GR `triton_jagged_tensors.py`), which compile cleanly with buffer ops **on**,
> so the patch is dropped and buffer ops stay enabled for every kernel. Bit-exact (GAUC neutral).
> Re-cert is VALID with **lower p99 / higher qps**: C1-on b48 p99 **51.07 → 45.9 ms** (~10%, same
> 10,595 q/s); C1-off b40 knee **7,400 → 7,600 q/s** (+2.7%; 7,500 recommended at p99 58.1 ms). To
> run on the current (pre-fix) container add `-e AMDGCN_USE_BUFFER_OPS_GFX950=1`; fresh builds have
> it on by default. See the *Follow-up* section of the profile doc above.

## Success criterion

`run_gold.sh` writes `artifacts/gold_*/mlperf_log_summary.txt` with **`Result is : VALID`**
and **Completed samples per second ≈ 12,199** (p99 ≈ 58 ms on a clean node with latest
LoadGen 6.0.16, degree-5 gate, GPU clock determinism, and P0/P2/P1 ring1024; target 80 ms).
`scripts/run/_test08_chain.sh` should finish with **`TEST PASS`** from the official verifier
(`num_ne_mismatch=0`, `num_unmatched=0`). Performance validity of the audited Server leg still
depends on clean-node headroom at this near-cliff q12,200 point.
