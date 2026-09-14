# Path Conventions For Final Submission

The runner checkout path must not leak into the final MLPerf submission. AMD's
v6.0 submissions avoid this by treating the generated submission tree as a stable
runtime root inside the container. Representative AMD result READMEs invoke code
through paths like:

```text
/lab-mlperf-inference/code/...
/lab-mlperf-inference/results/...
```

and scripts such as `src/run_harness.sh` derive their own location with:

```bash
CODE_DIR=$(dirname -- $0)
SCRIPTS_DIR=${CODE_DIR}/scripts
```

For q12,200 DLRM-v3, use the same idea:

- runner-owned staging lives under `submission/` in this repo;
- final generation creates a separate `closed/AMD/...` tree;
- container/runtime docs should refer to a stable in-container root, not to the
  original runner checkout path;
- setup/run wrappers copied from the runner should compute paths from their own
  location or accept explicit env overrides (`WORKSPACE_HOST`, `REPO_SUB`,
  `MOUNT_ROOT`, `OUTPUT_DIR`).

Do not hard-code paths such as `/mnt/shared/...`, `/mnt/vast/...`, or a specific
extracted quickstart directory name in final submission-facing scripts.
