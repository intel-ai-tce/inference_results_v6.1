# AMD INTERNAL — WORK IN PROGRESS — DO NOT DISTRIBUTE

**Status:** AMD-internal, work-in-progress. **Not for release.**

This repository is a ROCm/gfx950 port of the NVIDIA Embedding Cache (`nv-embedding-cache`).
It is maintained for **internal AMD evaluation only** and is **not** an official, supported,
or released product.

## Do NOT

- Publish this repository (public GitHub, package registries, etc.).
- Share it, its tarballs/snapshots, or its build artifacts outside AMD.
- Treat it as production-ready — the port is incomplete and under active development
  (see, e.g., the warp-width caveat for `NVEmbeddingBag` pooling backward in
  `docs/training_on_rocm.md`).

## Notes

- Upstream `nv-embedding-cache` (https://github.com/NVIDIA/nv-embedding-cache) is licensed
  Apache-2.0 (see `LICENSE`, retained); that license governs the upstream code. Source files
  retain NVIDIA Apache-2.0 SPDX headers (212 files: 187 dated 2026, 25 dated 2024, 1 dated
  2023). This **internal/WIP/do-not-distribute** restriction is an AMD-internal handling
  policy for *this* port snapshot during development, not a change to the upstream license.
- A repository-level `NOTICE` records upstream attribution (NVIDIA `nv-embedding-cache`,
  bundled NVIDIA cuEmbed) and AMD's modifications.
- **Bundled third-party actually present here:** `third_party/cuembed/` — NVIDIA cuEmbed,
  Apache-2.0 (`third_party/cuembed/LICENSE`). The other `third_party/*` directories are git
  **submodules** (`.gitmodules`), fetched at build time (and omitted from `git archive`
  snapshots). Their licenses (googletest BSD-3, json MIT, argparse MIT, pybind11 BSD-3,
  dlpack Apache-2.0, abseil-cpp Apache-2.0, parallel-hashmap Apache-2.0, rocksdb dual
  GPLv2/Apache-2.0 → use Apache-2.0, hiredis BSD-3, redis++ Apache-2.0) are recorded in the
  `NOTICE` file in this directory.
- **Residual item:** AMD-modified and AMD-new files still carry `Copyright (c) 2026 NVIDIA`
  SPDX headers, with no per-file AMD copyright or "modified by AMD" notice (Apache-2.0 §4(b)).
  The repository-level `NOTICE` records AMD's modifications in the interim; per-file
  annotation remains a manual OSPO task.
- Clear this notice (and the banner in `README.md`) only when the work is explicitly approved
  for release.

**Author / maintainer:** Jiawei Chen
