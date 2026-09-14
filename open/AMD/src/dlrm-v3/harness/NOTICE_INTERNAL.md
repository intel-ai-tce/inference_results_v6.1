# AMD INTERNAL — WORK IN PROGRESS — DO NOT DISTRIBUTE

**Status:** AMD-internal, work-in-progress. **Not for release.**

This repository is the **certified ROCm/gfx950 port of the NVIDIA DLRM-v3 inference
harness** (`closed/NVIDIA/code/dlrm-v3` from `mlcommons/inference_results_v6.0`),
maintained for **internal AMD evaluation only**. It is **not** an official, supported,
or released product.

## What this is

- A complete, version-controlled copy of the **certified** harness working tree (the one
  that produced the figure of record), replacing the previous vendored snapshot
  (`vendor/harness-src/`) and the per-plan `nvidia-harness--*` patch reconstruction.
- The harness port is a *living working tree* (8 modified + ~84 new files); the per-plan
  patches **cannot** faithfully rebuild it from the upstream baseline (6/14 fail to
  forward-apply), so this repo is the **source of truth**.

## Provenance

- Upstream: `mlcommons/inference_results_v6.0` subtree `closed/NVIDIA/code/dlrm-v3`,
  baseline pin `4d3916a`.
- Seeded from the certified snapshot `dlrm-v3-cert-4d3916a-20260601T222944`
  (sha256 `834be9495c3a14378e94da1bb8927fb58428d21a25d7d6919419916f6eb4c0be`).
- NVE-cert sentinels present: `DLRM_ROCM_NVE`, `DLRM_NVE_PARALLEL_CKPT_LOAD`,
  `DLRM_CLAMP_OOB_IDS`.

## Do NOT

- Publish this repository or share it/its artifacts outside AMD.
- Treat it as production-ready.

## Licensing

Derivative work of MLCommons `inference_results_v6.0` (`closed/NVIDIA/code/dlrm-v3`), which
is published under **Apache-2.0**. That upstream license governs the upstream code.

**Attribution status (restored):**

- The Apache-2.0 `LICENSE` is now present in this tree, and a repository-level `NOTICE`
  records upstream attribution (MLCommons / NVIDIA) and AMD's modifications. (A previous
  version of this notice incorrectly stated that LICENSE-bearing upstream files were
  "retained in-tree"; they were not — they have now been restored.)
- **Residual item:** the source files carry essentially no per-file copyright/SPDX headers
  (exceptions: 2 files with a Meta Platforms header — `benchmarks/accuracy.py`,
  `inference_harness/backends/model/STU_custom.py`). Restoring the originals' per-file
  upstream headers and adding per-file AMD "modified" notices (Apache-2.0 §4(b)) remains a
  manual OSPO task; the repository-level `LICENSE` + `NOTICE` cover attribution in the interim.

The **internal/WIP/do-not-distribute** restriction is an AMD-internal handling policy for
this port snapshot during development, not a change to the upstream license. See the
`LICENSE` and `NOTICE` files in this directory.
