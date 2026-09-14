# FLOP utilization (MFU) vs GPU utilization — GOLD DLRM-v3, MI355X (2026-06-27)

What fraction of the GPU's compute power the GOLD run actually uses, and why that is very different from
the ~99–100% "GPU utilization" rocm-smi reports. Probe: `scripts/profile/plan_flop_util.py` (attention,
measured) + analytic whole-model FLOPs from the confirmed model config + the certified 10,200 q/s.

## TL;DR

| metric | value | what it measures |
|---|---:|---|
| GPU-busy (rocm-smi / GRBM_GUI_ACTIVE) | **~99–100%** | TIME the GPU is doing *anything* |
| attention executed-matrix MFU | ~54% | matrix engine's actual fp8 FLOP-rate during attention |
| attention **useful** MFU | **17.6%** | productive matmul (executed minus causal/mask/padding waste) |
| **whole-model useful MFU @ 10,200 q/s** | **~9.5%** | productive fp8 matmul ÷ (8×peak), end-to-end |

**~100% GPU-busy but only ~9.5% of fp8 FLOP peak is productive work.** Expected for a recommendation
model: it is memory/latency/comms-bound, not matmul-bound.

## Hardware peak (MI355X, CDNA4, per GPU)
OCP-FP8 e4m3 / MXFP8 **dense = 5.0 PFLOP/s** (sparsity 10.1, N/A here); bf16 matrix 2.5 PF; 256 CUs,
1024 matrix cores, 2.4 GHz. 8-GPU system fp8 dense peak = 40 PFLOP/s.

## The three different "utilization" numbers (the common confusion)
1. **GPU-busy % (time)** — is an instruction issuing this cycle? ~100% here; says nothing about *what* or
   *how efficiently*.
2. **Executed-matrix MFU (throughput)** — achieved matrix FLOP/s ÷ peak. Attention ≈ **54%**: the matrix
   engine is fed fairly densely; VALU between MFMAs caps it below 100% ("VALU-bound" ceiling).
3. **Useful MFU (productive throughput)** — of executed matrix FLOPs, only the *unmasked* ones count.
   Attention executes ~3.1× the minimal MFMA (causal triangle ~2× + contextual first-block full scan +
   tile padding) ⇒ useful = 54% / 3.1 ≈ **17.6%**.

Relationship:
```
useful FLOP/s = PEAK × (GPU-busy%) × (matrix-issue efficiency) × (1 / waste factor)
  0.88 PF     =  5.0  ×   ~1.00     ×        ~0.54              ×   1/3.1      (attention kernel)
```

## Whole-model MFU @ the 10,200 q/s knee
**Model config** (`dlrm-v3-harness-rocm/.../tools/model_configs.py`, "production"): 5 HSTU layers,
`hstu_transducer_embedding_dim` (d_model) = 512, `hstu_num_heads`=4, `hstu_attn_qk_dim`=128,
`hstu_attn_linear_dim`=128 ⇒ per layer the UVQK projection is 512→2048 and the output projection 512→512.

**Useful matmul FLOPs per sample** (realistic batch Z=16, H=4, hist~6k, targets=2048, ctx=1, L≈126k tok):
- attention: 4 full layers + 1 targets-only last layer ≈ **4.3e12 / batch** (one full layer measured =
  9.76e11 useful, 1.11 ms, 17.6% MFU)
- UVQK + output GEMMs, 5 layers: `2·L·512·(2048+512)·5` ≈ **1.65e12 / batch**
- total ≈ 5.95e12 / 16 = **3.7e11 useful matmul FLOPs/sample**

```
3.7e11 FLOP/sample × 10,200 samples/s = 3.8 PFLOP/s (system, useful)
3.8 PF ÷ (8 GPU × 5.0 PF) = ~9.5% whole-model useful MFU
```
Contribution: **attention ≈ 6.9%**, **GEMMs ≈ 2.6%**; everything else (NVE embedding gather, cross-GPU
ZMQ comms, layernorm, elementwise, reduce) ≈ **0 matmul FLOPs** but consumes ~25–30% of wall-time → pure
MFU drag. (q/s is the true end-to-end rate, so this ~9.5% already folds in all of that.)

## Why it's low (and why that's fine)
1. **Attention is FLOP-heavy but low-MFU** (gate VALU + sync between sparse MFMAs; ~3× causal/mask waste).
2. **GEMMs are efficient but a small FLOP share** (~25% of matmul FLOPs).
3. **~25–30% of wall-time is non-matmul** (memory-bound embedding gather, comms, LN/elementwise) — the
   structural reason a *recommendation* model is low-MFU vs a dense LLM.

Implication: the figure of record is set by the **saturation knee (q/s)**, not by raw FLOPs — the matrix
engine is never the bottleneck. This is the same conclusion the attention-floor work reached (Plans 44–49):
the kernel is VALU/latency/comms-bound (`MemUnitStalled≈0`), so matmul-throughput levers don't move it;
the real step change is the hardware ask (async shared-memory-operand MMA). See
`../../../cursor/jiaweichen-amd/agent/docker/dlrmv3-rocm/ATTENTION_FLOOR_PROBLEM_GENERIC.md`.

## Caveats
Analytical (grounded in confirmed config dims + measured attention + certified q/s); ignores the small
preprocessor/dense MLP (<~5% of FLOPs) and approximates the targets-only last layer. Exact per-kernel
measured split would need `run_gold.sh PROFILE=1` (~30 min) — the ~9.5% is solid to a point or two.
