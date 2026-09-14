# Calibration — Naeem Khoshnevis, Individual Submitter — MLPerf Inference v6.1

## llama3.1-8b (Offline, Server) — FP8, calibration performed

Quantized with **NVIDIA ModelOpt** post-training quantization to **FP8 (W8A8)** with an **FP8 KV
cache**, then built into a TensorRT-LLM 1.0.0 engine. Both submitted scenarios serve the same
engine.

### Calibration data

Source of truth is the MLPerf-official calibration id list,
`calibration/CNNDailyMail/calibration-list.txt` (1,000 lines, **998 unique ids** — the official
file contains two duplicate ids, `ae05bddb7e816fd0e14e95cc525e06caf9392918` and
`b05f9fa99ca30d7ce2611a6deb139f2274d1ad3b`). No data outside that list was used.

Verified against our materialized set: **998 unique ids, 0 off-list, 0 missing, order preserved.**

**Disclosure — input preprocessing differs from the reference calibration script.** The reference
`language/llama3.1-8b/prepare-calibration.py` wraps each article in the benchmark's instruction
template ("Summarize the following news article in 128 tokens...") and reads the `train` split
only. Our builder (`src/llama3.1-8b/build_official_calib.py`) instead resolves each official id
against `abisee/cnn_dailymail` 3.0.0, searching `validation, test, train` in that order, and
writes the **raw article text** with no instruction template. The id set is identical to the
official list; the surrounding prompt text is not. This is stated explicitly because PTQ
activation statistics are sensitive to it, and because the endpoints harness *does* apply the
reference instruction template at inference time.

### Method

- ModelOpt static **per-tensor** scales for weights and activations, computed from forward passes
  over the calibration set. `--qformat fp8 --kv_cache_dtype fp8 --dtype bfloat16`.
- **Scope:** all Linear layers, including `lm_head`. ModelOpt emitted
  `UserWarning: Enable lm_head quantization` during the run; `lm_head` was **not** excluded.
- **Subset:** `--calib_size 512`. The materialized `train.jsonl` is written in official-list
  order, so ModelOpt consumed a deterministic 512-row prefix. Note that `train.jsonl` has one row
  per line of the official list, not per unique id: it is 1,000 rows carrying 998 unique ids, and
  the two duplicated ids fall at 0-based rows 146/318 and 208/346 (lines 147/319 and 209/347 as `grep -n` counts them), both inside the 512-row window. ModelOpt
  therefore saw **510 unique documents**, not 512. Using a subset of the official calibration set is
  expressly permitted.
- No retraining, no fine-tuning, no architecture change (`retraining: "no"` in all
  `measurements.json`).

### Toolchain

| Component | Version |
|---|---|
| Container | `trtllm-release-1.0.0.sif` (TensorRT-LLM 1.0.0) |
| Quantizer | NVIDIA ModelOpt, as bundled in that container |
| Quantize entrypoint | `/app/tensorrt_llm/examples/quantization/quantize.py` |
| Engine build | `trtllm-build` (`--gemm_plugin auto --max_seq_len 4096 --max_batch_size 512 --max_num_tokens 8192 --use_paged_context_fmha enable --multiple_profiles enable`) |
| Base weights | `meta-llama/Llama-3.1-8B-Instruct`, bf16 safetensors |
| CUDA / driver | 12.9 / 575.57.08 |

**Unverified:** the local copy of the base weights carries no git or HF-snapshot metadata, so we
cannot attest the checkpoint revision hash against the `be673f32...` revision pinned in the
reference README.

### Reproduction

1. `src/llama3.1-8b/build_official_calib.py` — materializes `train.jsonl` from the official id
   list. **This must be run first**; the pipeline exits with `CALIB_MISSING` (rc 5) without it.

   The script reads `/cal/calibration-list.txt` and writes `/work/mlperf_l31_calib/train.jsonl`,
   so both paths must be bound, and `/work` must be the same `$WORK` the pipeline later uses:

   ```
   singularity exec --cleanenv \
       --bind $BASE/mlperf_inference/calibration/CNNDailyMail:/cal:ro \
       --bind $SCRATCH/trtllm-work:/work \
       --bind $BASE/.../src/llama3.1-8b:/src:ro \
       $SCRATCH/containers/trtllm-release-1.0.0.sif \
       python /src/build_official_calib.py
   ```

   **This step needs network access.** It calls `load_dataset("abisee/cnn_dailymail", "3.0.0")`,
   whereas the benchmark stages run with `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`. Neither
   sbatch invokes this script or binds `/cal`; it is a prerequisite run separately.
2. `src/llama3.1-8b/l31_8b_fp8_pipeline.sbatch` — stage 1 quantizes, stage 2 builds, stage 3
   serves and runs the Offline benchmark. Stages 1 and 2 are skipped when their artifacts already
   exist, so a from-scratch reproduction must start with an empty `$WORK`.
3. `src/llama3.1-8b/l31_8b_server_acc.sbatch` — Server, against the cached engine.

Benchmark configurations are shipped under `src/llama3.1-8b/config/staged/`; see
`src/llama3.1-8b/config/README.md` for the staging step.
