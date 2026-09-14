# SLURM Support Matrix

For running benchmarks from a login node when the selected SLURM partition contains the target machines. See the [multi-node SLURM environment setup](../docs/ENV_SETUP.md#multi-node-slurm-environment-setup) for setup instructions.

Per-benchmark support for SLURM configs. For Qwen3-VL, single-node and multi-node SLURM entries are merged into the `SLURM` column; the system name identifies the node setup.

## deepseek_r1 (10 configs)

| System                               | Scenario    | SLURM      |
| ------------------------------------ | ----------- | ---------- |
| `B200-SXM-180GBx8`                   | Offline     | ✅          |
| `B200-SXM-180GBx8`                   | Server      | ✅          |
| `B300-SXM-270GBx8`                   | Offline     | ✅          |
| `B300-SXM-270GBx8`                   | Server      | ✅          |
| `GB200-NVL72_GB200-186GB_aarch64x72` | Interactive | ✅ (disagg) |
| `GB200-NVL72_GB200-186GB_aarch64x72` | Offline     | ✅          |
| `GB200-NVL72_GB200-186GB_aarch64x72` | Server      | ✅          |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Interactive | ✅ (disagg) |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Offline     | ✅          |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Server      | ✅          |


## gpt_oss_120b (14 configs)

| System                               | Scenario    | SLURM      |
| ------------------------------------ | ----------- | ---------- |
| `B200-SXM-180GBx8`                   | Offline     | ✅          |
| `B200-SXM-180GBx8`                   | Server      | ✅          |
| `B300-SXM-270GBx8`                   | Offline     | ✅          |
| `B300-SXM-270GBx8`                   | Server      | ✅          |
| `GB200-NVL72_GB200-186GB_aarch64x4`  | Offline     | ✅          |
| `GB200-NVL72_GB200-186GB_aarch64x4`  | Server      | ✅          |
| `GB300-NVL72_GB300-288GB_aarch64x4`  | Offline     | ✅          |
| `GB300-NVL72_GB300-288GB_aarch64x4`  | Server      | ✅          |
| `GB200-NVL72_GB200-186GB_aarch64x72` | Interactive | ✅ (disagg) |
| `GB200-NVL72_GB200-186GB_aarch64x72` | Offline     | ✅          |
| `GB200-NVL72_GB200-186GB_aarch64x72` | Server      | ✅          |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Interactive | ✅ (disagg) |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Offline     | ✅          |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Server      | ✅          |


## llama2_70b (14 configs)

| System                               | Scenario    | SLURM   |
| ------------------------------------ | ----------- | ------- |
| `B200-SXM-180GBx8`                   | Offline     | ✅       |
| `B200-SXM-180GBx8`                   | Server      | ✅       |
| `B300-SXM-270GBx8`                   | Offline     | ✅       |
| `B300-SXM-270GBx8`                   | Server      | ✅       |
| `GB200-NVL72_GB200-186GB_aarch64x4`  | Offline     | ✅       |
| `GB200-NVL72_GB200-186GB_aarch64x4`  | Server      | ✅       |
| `GB300-NVL72_GB300-288GB_aarch64x4`  | Offline     | ✅       |
| `GB300-NVL72_GB300-288GB_aarch64x4`  | Server      | ✅       |
| `GB200-NVL72_GB200-186GB_aarch64x72` | Interactive | pending |
| `GB200-NVL72_GB200-186GB_aarch64x72` | Offline     | pending |
| `GB200-NVL72_GB200-186GB_aarch64x72` | Server      | pending |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Interactive | pending |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Offline     | pending |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Server      | pending |


## qwen3_vl_235b_a22b (12 configs)

| System                               | Scenario    | SLURM       |
|--------------------------------------| ----------- |-------------|
| `B300-SXM-270GBx8`                   | Interactive | ✅           |
| `B300-SXM-270GBx8`                   | Offline     | ✅           |
| `B300-SXM-270GBx8`                   | Server      | ✅           |
| `GB200-NVL72_GB200-186GB_aarch64x4`  | Interactive | ✅           |
| `GB200-NVL72_GB200-186GB_aarch64x4`  | Offline     | ✅           |
| `GB200-NVL72_GB200-186GB_aarch64x4`  | Server      | ✅           |
| `GB300-NVL72_GB300-288GB_aarch64x4`  | Interactive | ✅           |
| `GB300-NVL72_GB300-288GB_aarch64x4`  | Offline     | ✅           |
| `GB300-NVL72_GB300-288GB_aarch64x4`  | Server      | ✅           |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Interactive | ✅ (disagg)  |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Offline     | ✅           |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Server      | ✅           |


## wan22_a14b (12 configs)

| System                               | Scenario     | SLURM   |
| ------------------------------------ | ------------ | ------- |
| `B200-SXM-180GBx8`                   | Offline      | ✅       |
| `B200-SXM-180GBx8`                   | SingleStream | ✅       |
| `B300-SXM-270GBx8`                   | Offline      | ✅       |
| `B300-SXM-270GBx8`                   | SingleStream | ✅       |
| `GB200-NVL72_GB200-186GB_aarch64x4`  | Offline      | ✅       |
| `GB200-NVL72_GB200-186GB_aarch64x4`  | SingleStream | ✅       |
| `GB200-NVL72_GB200-186GB_aarch64x72` | Offline      | ✅       |
| `GB200-NVL72_GB200-186GB_aarch64x72` | SingleStream | ✅       |
| `GB300-NVL72_GB300-288GB_aarch64x4`  | Offline      | ✅       |
| `GB300-NVL72_GB300-288GB_aarch64x4`  | SingleStream | ✅       |
| `GB300-NVL72_GB300-288GB_aarch64x72` | Offline      | ✅       |
| `GB300-NVL72_GB300-288GB_aarch64x72` | SingleStream | ✅       |
