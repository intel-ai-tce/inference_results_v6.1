# Docker Support Matrix

For running benchmarks on a single-node server using Docker. See the [single-node Docker environment setup](../docs/ENV_SETUP.md#single-node-docker-environment-setup) for setup instructions.

## deepseek_r1 (4 configs)

| System                              | Scenario | Single-node (Docker) |
| ----------------------------------- | -------- | -------------------- |
| `B200-SXM-180GBx8`                  | Offline  | ✅                    |
| `B200-SXM-180GBx8`                  | Server   | ✅                    |
| `B300-SXM-270GBx8`                  | Offline  | ✅                    |
| `B300-SXM-270GBx8`                  | Server   | ✅                    |

## gpt_oss_120b (8 configs)

| System                              | Scenario | Single-node (Docker) |
| ----------------------------------- | -------- | -------------------- |
| `B200-SXM-180GBx8`                  | Offline  | ✅                    |
| `B200-SXM-180GBx8`                  | Server   | ✅                    |
| `B300-SXM-270GBx8`                  | Offline  | ✅                    |
| `B300-SXM-270GBx8`                  | Server   | ✅                    |
| `GB200-NVL72_GB200-186GB_aarch64x4` | Offline  | ✅                    |
| `GB200-NVL72_GB200-186GB_aarch64x4` | Server   | ✅                    |
| `GB300-NVL72_GB300-288GB_aarch64x4` | Offline  | ✅                    |
| `GB300-NVL72_GB300-288GB_aarch64x4` | Server   | ✅                    |

## llama2_70b (8 configs)

| System                              | Scenario | Single-node (Docker) |
| ----------------------------------- | -------- | -------------------- |
| `B200-SXM-180GBx8`                  | Offline  | ✅                    |
| `B200-SXM-180GBx8`                  | Server   | ✅                    |
| `B300-SXM-270GBx8`                  | Offline  | ✅                    |
| `B300-SXM-270GBx8`                  | Server   | ✅                    |
| `GB200-NVL72_GB200-186GB_aarch64x4` | Offline  | ✅                    |
| `GB200-NVL72_GB200-186GB_aarch64x4` | Server   | ✅                    |
| `GB300-NVL72_GB300-288GB_aarch64x4` | Offline  | ✅                    |
| `GB300-NVL72_GB300-288GB_aarch64x4` | Server   | ✅                    |

## qwen3_vl_235b_a22b (0 configs)

No single-node Docker configs are currently listed for this benchmark. Run this benchmark from a SLURM cluster login node and refer to [SLURM_SUPPORT.md](SLURM_SUPPORT.md#qwen3_vl_235b_a22b-12-configs). Note: You can use slurm and still get a single node benchmark number. 

## wan22_a14b (0 configs)

No single-node Docker configs are currently listed for this benchmark. Run this benchmark from a SLURM cluster login node and refer to [SLURM_SUPPORT.md](SLURM_SUPPORT.md#wan22_a14b-12-configs). Note: You can use slurm and still get a single node benchmark number.
