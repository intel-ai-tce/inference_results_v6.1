# AMD INTERNAL — WORK IN PROGRESS — DO NOT DISTRIBUTE

**Status:** AMD-internal, work-in-progress. **Not for release.**

This repository is the **certified ROCm/gfx950 port of the DLRM-v3 / HSTU generative-
recommenders tree** (`recommendation/dlrm_v3` from `mlcommons/inference`), maintained for
**internal AMD evaluation only**. It is **not** an official, supported, or released product.

## What this is

- A complete, version-controlled copy of the **certified** GR working tree (the HSTU model
  + our ROCm port), replacing the previous vendored snapshot (`vendor/gr-src/`).
- It is consumed by cloning it into the `mlcommons/inference` baseline at
  `recommendation/dlrm_v3` (the baseline still supplies `loadgen/`); the harness imports it
  via `PYTHONPATH=.../recommendation/dlrm_v3`.

## Why a repo (the patches don't reproduce the cert)

The `mlcommons-inference--*` patches forward-apply cleanly but **miss cert-path code** that
only lived in the working tree:
- **`sparse_routing.py` — Plan-12 MPI-lookup swap** (`_use_mpi_lookup_env`,
  `set_route_lookup_comm`, `Alltoallv`): load-bearing — `run_plan24.sh` sets
  `DLRM_USE_MPI_LOOKUP=1` and the harness Phase-12.3 calls into it.
- **`generative_recommenders/ops/rocm_compat.py` — Plan-10.1.b** CPU fallbacks for
  `reorder_batched_ad_lengths/_indices` (they SIGSEGV on HIP fbgemm).

## Reconciliation note (now fixed in this repo)

The original cursor working tree had a **spurious uncommitted deletion** of
`model_family.py` (which defines `HSTUModelFamily`, still imported via the re-export shim).
The snapshot restored it from HEAD; here it is **committed properly**, so the tree is
internally consistent.

## Provenance

- Upstream: `mlcommons/inference` subtree `recommendation/dlrm_v3`, baseline pin `33c8ee9`.
- Seeded from the certified snapshot `dlrm_v3-cert-33c8ee9-20260601T224822`
  (sha256 `fe47c72c3357978de8e1091ed8fddab9c5effc8a3d2f0e2b873c6e9eb3ec70b3`).

## Do NOT

- Publish this repository or share it/its artifacts outside AMD.
- Treat it as production-ready.

## Licensing

Derivative work of `mlcommons/inference` (`recommendation/dlrm_v3`) and Meta's
`generative_recommenders` library, both published under **Apache-2.0**. Those upstream
licenses govern the upstream code.

**Attribution status (restored):**

- The Apache-2.0 `LICENSE` is now present in this tree, and a repository-level `NOTICE`
  records upstream attribution (MLCommons `inference` + Meta `generative_recommenders`) and
  AMD's modifications.
- This tree bundles Meta's `generative_recommenders/` in-tree; 47 files carry
  `Copyright (c) Meta Platforms, Inc. and affiliates.` (retained — good).
- **Residual item:** AMD's ROCm/gfx950 modifications are not yet individually marked
  (Apache-2.0 §4(b)); the repository-level `LICENSE` + `NOTICE` cover attribution in the
  interim. Per-file AMD "modified" annotation remains a manual OSPO task.
- Build-time Python deps (pinned in `requirements.txt`, not vendored): `torch`,
  `fbgemm_gpu`, `torchrec`, `gin_config`, `pandas`, `tensorboard` — each under its own
  (BSD-3 / Apache-2.0) upstream license.

The **internal/WIP/do-not-distribute** restriction is an AMD-internal handling policy for
this port during development, not a change to the upstream license. See the `LICENSE` and
`NOTICE` files in this directory.
