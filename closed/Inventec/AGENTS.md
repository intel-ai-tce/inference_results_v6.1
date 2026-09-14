# AGENTS.md

This file provides general guidance to AI Agents when working with this repository. 

## Repository Overview

NVIDIA's closed division submission for the MLPerf Inference Benchmark (v6.0). Implements optimized inference harnesses for multiple benchmarks using various Nvidia Internal or Open Source frameworks. Only files in `closed/NVIDIA/` are released publicly; everything else is internal.

**Primary working directory (run every CLI command under this):** `closed/NVIDIA/` (mounted as `/work` inside Docker containers).

**Source code location**: `src/nv_mlpinf`

**Package usage guidelines:** [closed/NVIDIA/docs/src/README.md](closed/NVIDIA/docs/src/README.md)

**Package Developing guidelines:** [closed/NVIDIA/docs/src/DEVELOPMENT_GUIDE.md](closed/NVIDIA/docs/src/DEVELOPMENT_GUIDE.md)

## Benchmark Support Matrix

For per-benchmark, per-system support matrices, see **[Docker support](configs/DOCKER_SUPPORT.md)** and **[SLURM support](configs/SLURM_SUPPORT.md)**.

## Configuration Folder

Benchmark configurations are organized in `closed/NVIDIA/configs/` by model, system, and scenario. 

For detailed config authorship guidelines (naming conventions, YAML formats, topology schemas), see **[closed/NVIDIA/docs/configs/EDITING.md](closed/NVIDIA/docs/configs/EDITING.md)**.

## Run Benchmark Step 1: Container Setup Instructions

While the codebase supports launching various benchmarks, each benchmark has a different way of setting up the benchmarking environments (software stack, benchmarking components, environments).

Generally, for a benchmark, it can support two types of container environments:

- **Single-node Docker**: For benchmarking on a single node with interactive or detached mode
- **Multi-node Enroot (SLURM)**: For benchmarking across multiple nodes

**For detailed environment setup instructions per benchmark**, see **[closed/NVIDIA/docs/ENV_SETUP.md](closed/NVIDIA/docs/ENV_SETUP.md)**.

## Run Benchmark Step 2: Launch Instructions

Benchmarks can be launched in two modes:

1. **Single-node Docker** — Interactive session on a single node (e.g., B200x8, B300x8)
2. **Multi-node SLURM** — Scaleout across multiple nodes using Enroot (e.g., GB200x72), benchmarks are launched on cluster login node.

For complete launching instructions, command examples, and parameter details, refer to the individual benchmark READMEs:

- [DeepSeek-R1 README](closed/NVIDIA/src/nv_mlpinf/benchmarks/deepseek_r1/README.md)
- [DLRMv3 README](closed/NVIDIA/src/nv_mlpinf/benchmarks/dlrm_v3/README.md)
- [GPT-OSS-120B README](closed/NVIDIA/src/nv_mlpinf/benchmarks/gpt_oss_120b/README.md)
- [Llama2-70B README](closed/NVIDIA/src/nv_mlpinf/benchmarks/llama2_70b/README.md)
- [Qwen3 VL (Q3VL) README](closed/NVIDIA/src/nv_mlpinf/benchmarks/q3vl/vllm/README.md)
- [Whisper README](closed/NVIDIA/src/nv_mlpinf/benchmarks/whisper/README.md)
- [Wan2.2-T2V-A14B (wan22) README](closed/NVIDIA/src/nv_mlpinf/benchmarks/wan22_a14b/README.md)

## MLPerf Submission Instructions

Once result logs are ready, use [docs/SUBMISSION.md](docs/SUBMISSION.md) for post-result-collection submission.

## Supplemental Information:

### Makefile Structure

Modular includes in `closed/NVIDIA/`:

- `Makefile.const` — System detection (GPU arch, CUDA/TRT versions, OS, Python version)
- `Makefile.docker` — Container management and prebuild targets
- `Makefile.data` — Dataset download/preprocessing
- `Makefile.tests` — Test infrastructure (unit_tests, e2e_tests, regression_presubmit)
- `Makefile.submission` — Results staging, compliance, submission pipeline

### MLPERF_SCRATCH_PATH Directory

This directory will hold all necessary files in order to run each benchmarks, it contains original dataset, preprocessed dataset, model checkpoints, the way to download these files are recorded in per benchmark README files. By default, the code will always look for files using this environment variable

```
$MLPERF_SCRATCH_PATH/
├── data/                   # Raw datasets for each benchmarks
├── models/                 # Pre-trained checkpoints
└── preprocessed_data/      # Processed inference inputs
```

Default: 

on compute lab:`/home/mlperf_inference_storage`,
on slurm clusters (ptyche, lyris, bia, prenyx...): `/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone/`. 
To override: `export MLPERF_SCRATCH_PATH=/path/to/scratch` or edit nv_mlpinf_paths.yml under src folder

### Runtime Paths Configuration

Runtime paths are resolved via a 3-tier fallback: **environment variable → YAML config → hardcoded default**.

- **YAML config:** `~/.config/nv_mlpinf/paths.yml` (created from bundled template on first run)
- **Override config location:** `NV_MLPINF_PATHS_CONFIG=/path/to/paths.yml`
- **Per-invocation override:** Environment variables (e.g., `BUILD_DIR=/tmp nv-mlpinf ...`)
- **Debug paths:** `nv-mlpinf show_paths` prints all resolved paths with their sources

### 3rd Party Submodules (`closed/NVIDIA/3rdparty/`)

- `trtllm/` — TensorRT-LLM (LLM inference engine)
- `mlc-inference/` — MLCommons inference repo (LoadGen)
- `mitten/` — NVIDIA configuration framework (`nvmitten`)
- `endpoints/` — MLCommons `inference-endpoint` client (wan22-a14b video-gen load generator; submodule removed, use the pre-built container image instead — see wan22 README)

## Rules and Restrictions

### DO:

- **Always log benchmark outputs to disk**: When running `nv-mlpinf run_harness` using Docker command line, serialize stdout logs to closed/NVIDIA with explicit, descriptive naming
- **Monitor results from log files**: When checking harness progress, always read from the saved log files
- when you run `nv-mlpinf run_llm_server`, you can put it on background, and monitor the task's output logs
- when you run `nv-mlpinf run_llm_server`, you need to add LOG_DIR=/build/logs/<descriptive_name> so that the log is dumped to that location
- when you need to run benchmarks based on user instructions, read the AGENTS.md and when you are not sure where the instructions are, go to docs folder for relevant resources

### DON'T:

- **Never modify files under `MLPERF_SCRATCH_PATH`**: This directory contains datasets and model files that should remain unchanged. Any modifications could corrupt benchmark data or invalidate results.
