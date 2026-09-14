# MAKE Targets and RUN_ARGS Documentation
Our Makefile includes many targets and options. Below are some of the more relevant and commonly used ones:

## Make Targets

### Outside the Container

- `make prebuild BENCHMARK=<benchmark>`: Builds and launches a docker container for a specific benchmark.
    - Valid BENCHMARK values: `whisper`, `llama` (or `llama2-70b`, `llama3.1-8b`, etc.), `deepseek` (or `deepseek-r1`), `wan22-a14b`
    - Optional: `ENV=dev` (default) or `ENV=release` to include TRTLLM in the container build
- `make stage_results`: Updates the results/ directory with the logs in `build/logs`. Run during the submission process.
- `make check_submission`: Runs the official submission checker on the current state of the repo. Run during the submission process.

### Inside the Container

- `make download_dataset`: Downloads datasets.
- `make preprocess_data`: Preprocesses the downloaded datasets.
- `make build`: Runs the following steps:
    - `make link_dirs`: Adds symlinks in `build/` to the relevant directories from `$MLPERF_SCRATCH_PATH`
    - `make clone_loadgen`: Clone the official MLPerf inference GitHub repo.
    - `make build_plugins`: Builds TensorRT plugins.
    - `make build_loadgen`: Builds LoadGen source codes.
    - `make build_harness`: Builds the harnesses.
- `make build_trt_llm`: Builds TensorRT-LLM (required if `ENV=dev` was used in prebuild).
- `make clean`: Cleans up all the build directories. Please note that you will need to exit the docker container and run `make prebuild` again after the cleaning.
- `make clean_shallow`: Cleans up only the files needed to make a clean build. This includes any binaries and files that require compilation.

## Running Benchmarks

There are two main pathways to running benchmarks:

### Path 1: TensorRT Engines (Traditional)

Used for benchmarks that build TRT engines:

```bash
# Generate TensorRT engines
make generate_engines RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"

# Run the harness
make run_harness RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"
```

### Path 2: TRTLLM Endpoint (LLM Benchmarks)

Used for DeepSeek-R1, Llama3.1-8B, GPT-OSS-120B, and other LLM benchmarks with TRTLLM serve:

```bash
# Start the LLM server
make run_llm_server RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO> --core_type=trtllm_endpoint"

# In a separate terminal or after server is ready, run the harness
make run_harness RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO> --core_type=trtllm_endpoint"
```

### Available Benchmarks

- `deepseek-r1`: DeepSeek-R1 (uses TRTLLM endpoint)
- `gpt-oss-120b`: GPT-OSS-120B (uses TRTLLM endpoint)
- `llama2-70b`: Llama2-70B
- `llama3.1-8b`: Llama3.1-8B (uses TRTLLM endpoint)
- `llama3.1-405b`: Llama3.1-405B
- `whisper`: Whisper
- `rgat`: R-GAT
- `dlrm-v3`: DLRMv3 (Generative Recommender)
- `wan22-a14b`: WAN-2.2-T2V-A14B (Text-to-Video)
- `qwen3-vl-235b-a22b`: Qwen3-VL-235B-A22B (Vision-Language)

### Available Scenarios

- `Offline`: Datacenter offline scenario
- `Server`: Datacenter server scenario
- `Interactive`: Datacenter interactive scenario (for LLMs)
- `SingleStream`: Edge single stream scenario
- `MultiStream`: Edge multi stream scenario

## RUN_ARGS Flags

- `--benchmarks=comma,separated,list,of,benchmark,names`
- `--scenarios=comma,separated,list,of,scenario,names`
- `--config_ver=comma,separated,list,of,config,versions`
- `--test_mode=[PerformanceOnly,AccuracyOnly]`: Specifies which LoadGen mode to run with.
- `--core_type=trtllm_endpoint`: Use TRTLLM serve endpoint for LLM benchmarks.
- `--test_run`: Reduces minimum runtime from 10 minutes to 1 minute for development/testing.
- `--force_calibration`: Forces recalculation of calibration cache.
- `--log_dir=path/to/logs`: Specifies where to save logs.
- `--verbose`: Prints out verbose logs.
- `--verbose_glog=1`: Enable detailed TRTLLM iteration logging.

## Config Versions

There are several config versions which can be passed into the `--config_ver` flag:

1. `default`: Default config with low accuracy target. Supported for all benchmarks.
2. `high_accuracy`: Runs the benchmark for the 99.9% of FP32 accuracy target. Supported only for Llama2-70B.

## Scaleout Commands (Multi-Node)

For multi-node runs on GB200/GB300 NVL72 systems, see `scaleout/REPRODUCE.md` for detailed instructions and reproduction commands.

## Benchmark-Specific Instructions

For detailed setup, data preparation, and execution instructions for each benchmark, see the respective README files:

| Benchmark | README Location |
|-----------|-----------------|
| DeepSeek-R1 | `code/deepseek-r1/tensorrt/README.md` |
| GPT-OSS-120B | `code/gpt-oss-120b/tensorrt/README.md` |
| Llama2-70B | `code/llama2-70b/tensorrt/README.md` |
| Llama3.1-8B | `code/llama3_1-8b/tensorrt/README.md` |
| Llama3.1-405B | `code/llama3_1-405b/tensorrt/README.md` |
| Whisper | `code/whisper/tensorrt/README.md` |
| R-GAT | `code/rgat/pytorch/README.md` |
| DLRMv3 | `code/dlrm-v3/README.md` |
| WAN-2.2-T2V-A14B | `code/wan22-a14b/tensorrt/README.md` |
| Qwen3-VL-235B-A22B | `code/qwen3-vl-235b-a22b/vllm/README.md` |

## Compliance Tests

```bash
# Run all compliance tests for a benchmark
make run_audit_harness RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"

# Run individual compliance tests
make run_audit_test01 RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"
make run_audit_test04 RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"
make run_audit_test05 RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"
make run_audit_test07 RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"
make run_audit_test09 RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>"
```
