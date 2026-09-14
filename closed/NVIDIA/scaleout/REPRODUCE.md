# MLPerf Inference Scaleout - nv-sflow Reproduction Commands

All commands below should be run from `closed/NVIDIA/` on the SLURM login node.

The commands use `sflow batch --submit` with three inputs: the benchmark/scenario nv-sflow config, the colocated `slurm_env_sflow.yaml`, and the matching template from `scaleout/sflow/templates/`.

Replace `ACCT=<your-slurm-account>` and the `--partition` value if your cluster uses different account or partition names. The container image is pulled from `variables.CONTAINER_IMAGE` in each `slurm_env_sflow.yaml` unless you override it with `--set CONTAINER_IMAGE=...`.

---

## DeepSeek-R1

### GB200x72 - Offline

```bash
ACCT=<your-slurm-account>
RUN=deepseek_r1_gb200x72_offline_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/deepseek_r1/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/deepseek_config_sflow.yaml \
  -f configs/deepseek_r1/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=gb200 \
  --set SLURM_TIME=04:00:00 \
  --nodes=18 \
  --partition=gb200 \
  --account=$ACCT \
  --time=04:00:00 \
  --job-name=deepseek_r1_gb200x72_offline \
  -o build/sbatch_scripts_sflow/$RUN.sh \
  --submit
```

### GB200x72 - Server

```bash
ACCT=<your-slurm-account>
RUN=deepseek_r1_gb200x72_server_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/deepseek_r1/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Server/deepseek_config_sflow.yaml \
  -f configs/deepseek_r1/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Server/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=gb200 \
  --set SLURM_TIME=04:00:00 \
  --nodes=18 \
  --partition=gb200 \
  --account=$ACCT \
  --time=04:00:00 \
  --job-name=deepseek_r1_gb200x72_server \
  -o build/sbatch_scripts_sflow/$RUN.sh \
  --submit
```

### GB300x72 - Offline

```bash
ACCT=<your-slurm-account>
RUN=deepseek_r1_gb300x72_offline_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/deepseek_r1/GB300-NVL72_GB300-288GB_aarch64x72/TRTLLM/Offline/deepseek_config_sflow.yaml \
  -f configs/deepseek_r1/GB300-NVL72_GB300-288GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=gb300 \
  --set SLURM_TIME=04:00:00 \
  --nodes=18 \
  --partition=gb300 \
  --account=$ACCT \
  --time=04:00:00 \
  --job-name=deepseek_r1_gb300x72_offline \
  -o build/sbatch_scripts_sflow/$RUN.sh \
  --submit
```

### GB300x72 - Server

```bash
ACCT=<your-slurm-account>
RUN=deepseek_r1_gb300x72_server_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/deepseek_r1/GB300-NVL72_GB300-288GB_aarch64x72/TRTLLM/Server/deepseek_config_sflow.yaml \
  -f configs/deepseek_r1/GB300-NVL72_GB300-288GB_aarch64x72/TRTLLM/Server/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=gb300 \
  --set SLURM_TIME=04:00:00 \
  --nodes=18 \
  --partition=gb300 \
  --account=$ACCT \
  --time=04:00:00 \
  --job-name=deepseek_r1_gb300x72_server \
  -o build/sbatch_scripts_sflow/$RUN.sh \
  --submit
```

---

## GPT-OSS-120B

### GB200x72 - Offline

```bash
ACCT=<your-slurm-account>
RUN=gpt_oss_120b_gb200x72_offline_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=gb200 \
  --set SLURM_TIME=04:00:00 \
  --nodes=18 \
  --partition=gb200 \
  --account=$ACCT \
  --time=04:00:00 \
  --job-name=gpt_oss_120b_gb200x72_offline \
  -o build/sbatch_scripts_sflow/$RUN.sh \
  --submit
```

### GB200x72 - Server

```bash
ACCT=<your-slurm-account>
RUN=gpt_oss_120b_gb200x72_server_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Server/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Server/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=gb200 \
  --set SLURM_TIME=04:00:00 \
  --nodes=18 \
  --partition=gb200 \
  --account=$ACCT \
  --time=04:00:00 \
  --job-name=gpt_oss_120b_gb200x72_server \
  -o build/sbatch_scripts_sflow/$RUN.sh \
  --submit
```

### GB200x72 - Interactive (Disaggregated)

```bash
ACCT=<your-slurm-account>
RUN=gpt_oss_120b_gb200x72_interactive_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Interactive/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Interactive/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_disagg_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=gb200 \
  --set SLURM_TIME=04:00:00 \
  --nodes=18 \
  --partition=gb200 \
  --account=$ACCT \
  --time=04:00:00 \
  --job-name=gpt_oss_120b_gb200x72_interactive \
  -o build/sbatch_scripts_sflow/$RUN.sh \
  --submit
```

### GB300x72 - Offline

```bash
ACCT=<your-slurm-account>
RUN=gpt_oss_120b_gb300x72_offline_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/gpt_oss_120b/GB300-NVL72_GB300-288GB_aarch64x72/TRTLLM/Offline/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/GB300-NVL72_GB300-288GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=gb300 \
  --set SLURM_TIME=04:00:00 \
  --nodes=18 \
  --partition=gb300 \
  --account=$ACCT \
  --time=04:00:00 \
  --job-name=gpt_oss_120b_gb300x72_offline \
  -o build/sbatch_scripts_sflow/$RUN.sh \
  --submit
```

### GB300x72 - Server

```bash
ACCT=<your-slurm-account>
RUN=gpt_oss_120b_gb300x72_server_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/gpt_oss_120b/GB300-NVL72_GB300-288GB_aarch64x72/TRTLLM/Server/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/GB300-NVL72_GB300-288GB_aarch64x72/TRTLLM/Server/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=gb300 \
  --set SLURM_TIME=04:00:00 \
  --nodes=18 \
  --partition=gb300 \
  --account=$ACCT \
  --time=04:00:00 \
  --job-name=gpt_oss_120b_gb300x72_server \
  -o build/sbatch_scripts_sflow/$RUN.sh \
  --submit
```

### GB300x72 - Interactive (Disaggregated)

```bash
ACCT=<your-slurm-account>
RUN=gpt_oss_120b_gb300x72_interactive_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/gpt_oss_120b/GB300-NVL72_GB300-288GB_aarch64x72/TRTLLM/Interactive/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/GB300-NVL72_GB300-288GB_aarch64x72/TRTLLM/Interactive/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_disagg_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=gb300 \
  --set SLURM_TIME=04:00:00 \
  --nodes=18 \
  --partition=gb300 \
  --account=$ACCT \
  --time=04:00:00 \
  --job-name=gpt_oss_120b_gb300x72_interactive \
  -o build/sbatch_scripts_sflow/$RUN.sh \
  --submit
```

---

## Additional Options

### Accuracy Testing

Add `--set TEST_MODE=AccuracyOnly` to any command to run the accuracy dataset/mode for that benchmark:

```bash
--set TEST_MODE=AccuracyOnly \
```

### Extra Harness Arguments

Pass additional harness flags through `HARNESS_EXTRA_ARGS`. For example:

```bash
--set HARNESS_EXTRA_ARGS="--offline_expected_qps=<qps>" \
```

### Interactive Debugging

For interactive debugging, use the same `-f` files with `sflow run --tui` instead of `sflow batch --submit`:

```bash
sflow run \
  -f <benchmark_config_sflow.yaml> \
  -f <slurm_env_sflow.yaml> \
  -f <template.yaml> \
  --set WORK_DIR=$PWD \
  --tui
```

### Compliance Testing

The current `trtllm_*_loadgen.yaml` templates call `nv-mlpinf run_harness`. They do not provide a generic `run_audit_harness`/compliance workflow yet. Add or use an audit-capable nv-sflow template before replacing a compliance run with `sflow`.

---

## Reference

For more details on nv-sflow orchestration, see:

- [nv-sflow Documentation](sflow/README.md) - Detailed nv-sflow scaleout documentation
- [../docs/ENV_SETUP.md](../docs/ENV_SETUP.md) - Environment setup guide
- [../configs/SLURM_SUPPORT.md](../configs/SLURM_SUPPORT.md) - SLURM config support matrix
