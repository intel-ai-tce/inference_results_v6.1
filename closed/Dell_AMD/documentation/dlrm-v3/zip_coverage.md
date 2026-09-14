# q12,200 Quickstart Zip Coverage

The `submission-6.1` branch stages the contents needed to turn the q12,200
quickstart zip into an AMD-style MLPerf submission payload.

## Covered From The Zip

- `vendor/dlrm-v3-harness-rocm` -> `submission/src/dlrm-v3/harness`
- `vendor/dlrm-v3-gr-rocm` -> `submission/src/dlrm-v3/gr`
- `vendor/pynve-rocm` -> `submission/src/dlrm-v3/pynve-rocm`
- `scripts/build/*`, `scripts/run/run_gold.sh`, `scripts/run/run_accuracy.sh`,
  `scripts/run/_test08_chain.sh`, `scripts/run/score_accuracy.py` ->
  `submission/setup/dlrm-v3/`
- `scripts/package/create_minimal_zip.sh` ->
  `submission/tools/dlrm-v3/`
- q12,200 proof and full-causal markdown ->
  `submission/documentation/dlrm-v3/`
- q12,200 LoadGen configs ->
  `submission/src/dlrm-v3/harness/benchmarks/`
- system JSON and result directory scaffold ->
  `submission/systems/` and `submission/results/`

## Intentionally Not Copied Into Submission Staging

The quickstart zip also contains runner development/profiling helpers that are
useful for engineering but should not become submission-facing setup:

- exploratory A/B scripts such as `ab_*`, `sweep_*`, `knee_sweep.sh`,
  `winb_batch_sweep.sh`;
- old debug chains such as `_eosfix_chain.sh`, `_eostrace_chain.sh`,
  `_roof_probe_chain.sh`, `_profile*_chain.sh`;
- local-path-only GR test/bench helpers under `fp8tuned_ext/`.

Those were deliberately pruned from `submission/setup` or `submission/src` to
avoid local path assumptions and non-submission debug surface.

## Path Handling

Final submission generation should normalize runtime paths to a stable container
root, following the AMD v6.0 pattern of `/lab-mlperf-inference/code` and
`/lab-mlperf-inference/results`, rather than preserving the original runner or
quickstart extraction path.
