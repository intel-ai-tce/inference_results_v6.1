# Submission-checker notes for the e2e-rag-db workload

Status as of 2026-07-29. **Bottom line: the submission passes cleanly —
`SUMMARY: submission looks OK`, `errors=0`, no skip flags.**

Build a submission tree and check it with:

```bash
# From inside container mk-e2e-rag-cpu, at /workspace/code:
HOST_OS="Ubuntu 24.04.4 LTS" HOST_STORAGE="879G" \
    bash tools/make_submission_db.sh
```

`HOST_OS` is written to `operating_system` verbatim, so keep it byte-identical
across rebuilds if you want reproducible trees.

The script is idempotent: it wipes and rebuilds `submission/` each run, then
truncates the accuracy log and invokes the checker. Overridable via env:
`DIVISION`, `SUBMITTER`, `SYSTEM_ID`, `BENCHMARK`, `SCENARIO`, `VERSION`,
`PERF_SRC`, `ACC_SRC`, `HOST_OS`, `HOST_STORAGE`, `HOST_STORAGE_TYPE`, `LENIENT`.

`LENIENT=1` adds `--skip-dataset-size-check`. It is **no longer needed** and is
kept only as an escape hatch for older checkers.

Current result:

| field | value |
|---|---|
| Result | `22.6141` Samples/s |
| Accuracy | `E2E_ACCURACY: 98.1952` |
| MlperfModel / Scenario | `e2e-rag-db` / `Offline` |
| Division / Availability | `closed` / `available` |
| errors | `0` |

## Naming

The workload is **`e2e-rag-db`** and the QnA workload is **`e2e-rag-qna`**,
matching the LoadGen `user.conf` section names.

These were renamed upstream in mlcommons/inference `c22d843b` (PR #2645); before
that they were `e2e_vectorDB` and `e2e`. Passing an old name is not a clean
failure — `get_required()` returns `None` and the checker crashes inside
`lower_list()` with `TypeError: 'NoneType' object is not iterable`. If you see
that, the vendored checker and the `BENCHMARK` name disagree.

The checker entrypoint is
`third_party/mlperf-inference/tools/submission/submission_checker/main.py`
(invoked as `python3 -m submission_checker.main` from
`third_party/mlperf-inference/tools/submission/`). Note there is *no* doubled
`submission/submission/` path.

## Layout the checker expects (v6.1)

```
closed/Intel/
├── src/e2e-rag-db/README.md                # SRC_PATH is "src" for v6.0+, "code" only up to v5.1
├── systems/1-node-2S-Xeon6787P.json        # SYSTEM_DESC_REQUIRED_FIELDS
├── documentation/README.md
└── results/1-node-2S-Xeon6787P/e2e-rag-db/Offline/
    ├── measurements.json                   # SYSTEM_IMP_REQUIRED_FILES keys
    ├── user.conf, README.md                # REQUIRED_MEASURE_FILES
    ├── accuracy/                           # REQUIRED_ACC_FILES
    │   ├── accuracy.txt                    # needs "Accuracy: <v>" + "hash=<sha256>"
    │   ├── mlperf_log_accuracy.json        # must be <= 10 KiB (truncated)
    │   ├── mlperf_log_detail.txt
    │   └── mlperf_log_summary.txt
    └── performance/run_1/                  # REQUIRED_PERF_FILES
        ├── mlperf_log_detail.txt
        └── mlperf_log_summary.txt
```

Source artifacts: `output_datasetup/` (performance run) and
`output_datasetup_accuracy/` (accuracy run), plus the repo's `user.conf`.

## The one real bug on our side

**`accuracy.txt` reported a raw fraction where the checker wants a percentage.**

```
ERROR ... accuracy not met: expected=98.000000, found=0.9819523809523809
```

`MODEL_CONFIG['v6.1']['accuracy-target']['e2e-rag-db']` is
`("E2E_ACCURACY", 98)` — a percentage. `ACC_PATTERN['E2E_ACCURACY']`
(`Accuracy:\s*([\d\.]+)`) matched fine, so the number parsed; it was simply 100×
too small and failed the comparison.

`evaluation/accuracy_eval_ingestion.py` wrote the manifest's
`retrieval_accuracy` verbatim, while `evaluation/accuracy_eval_qna.py` had always
scaled its metric by 100. Fixed by scaling the same way, which is what makes the
first line read `Accuracy: 98.1952`.

Worth knowing when re-testing: `make_submission_db.sh` copies `accuracy.txt`
**verbatim** from `${ACC_SRC}`. Editing the eval script changes nothing until the
eval is re-run — check the file's mtime if a fix appears to have no effect.

## Structural gotchas that cost real debugging time

- **`SRC_PATH` changed name.** It is `{division}/{submitter}/code` up to v5.1 but
  `{division}/{submitter}/src` for v6.0+. `MeasurementsCheck.directory_exist_check`
  looks for exactly that, so a `code/` dir silently fails on v6.1.
- **`accelerators_per_node` must be the *string* `"0"`.** It is in
  `SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS` and the test is
  `not self.system_json[k]`, so a CPU-only system with numeric `0` fails as
  "requires a meaningful response but is empty".
- **The accuracy log must be truncated by the official tool.**
  `tools/submission/truncate_accuracy_log.py` both shrinks
  `mlperf_log_accuracy.json` to under `MAX_ACCURACY_LOG_SIZE` (10 KiB; ours was
  133 KB) *and* appends the `hash=<sha256>` line to `accuracy.txt` that
  `AccuracyCheck.accuracy_result_check` requires. `make_submission_db.sh` runs
  it automatically and keeps the untruncated original in
  `submission_accuracy_backup/` (gitignored).
- **`--csv` defaults to a bare relative `summary.csv`.** The script `cd`s into
  the root-owned `third_party/` checkout, so the default write fails with
  `PermissionError`. We pass `--csv "${REPO_ROOT}/submission_summary.csv"`
  explicitly (gitignored).

## Previously-blocking checker constants — all fixed upstream

Four errors blocked the checker before `third_party/mlperf-inference` was pulled
to `c22d843b`. Every one was an upstream constant, and all are now resolved. Kept
here as a record, and because an older vendored checkout will reproduce them.

| Was | Now (verified at `c22d843b`) |
|---|---|
| `accuracy-target: ("E2E_ACCURACY", "")` → `TypeError: '>=' not supported between 'float' and 'str'` | `("E2E_ACCURACY", 98)` — `constants.py:167` |
| `dataset-size: 824` (the QnA query count) vs our 2515-page corpus | `2515` — `constants.py:208`, `accuracy-sample-count` at `:242` |
| `min-queries: {"Offline": 824}`, unsatisfiable because LoadGen hardcodes `min_query_count = 1` for Offline | `{"Offline": 1}` — `constants.py:303` |
| `TEST_DURATION_MS = 600000` enforced with no exemption, but the DB run takes ~111 s over a fixed corpus | explicit exemption for both e2e-rag workloads — `performance_check.py:370` |

The min-queries diagnosis is worth remembering: LoadGen *unconditionally* sets
`min_query_count = 1` for Offline (`loadgen/test_settings_internal.cc:158`) after
folding the requested count into `samples_per_query`, and the check reads
`effective_min_query_count`. So an Offline floor could never be met by any
submitter. The intended mechanism is `OFFLINE_MIN_SPQ_SINCE_V4`, checked against
`samples_per_query`, where `e2e-rag-db: 2515` is satisfied by our run.

## Judgment call: `user.conf` is copied verbatim

An earlier version of the script rewrote `min_duration` to `600000` in the
*submitted* copy to force the duration check to pass. That was removed: it would
contradict `effective_min_duration_ms: 0` in the `mlperf_log_detail.txt` sitting
in the same directory, making the submission internally inconsistent with its own
logs. The upstream exemption is the correct fix, and it now exists.

The general rule: never edit `user.conf` or result logs to satisfy a check. If a
config value is wrong, re-run.

## System description caveats

- **Container-vs-host detection.** The script runs inside `mk-e2e-rag-cpu`, so
  `lscpu`/`free` see through to the host correctly, but `/etc/os-release` and
  `df /` describe the *container*. One build recorded `Ubuntu 22.04.5 LTS`
  (container) instead of `Ubuntu 24.04.4 LTS` (host). The script now warns when
  `/.dockerenv` exists and `HOST_OS` is unset; pass `HOST_OS` and `HOST_STORAGE`
  explicitly. An earlier version also shelled out to `docker exec` for framework
  versions, which silently fell back to hardcoded defaults from inside the
  container — now read from the running interpreter directly.

- **Two placeholders still need real data.** `dmidecode` is unavailable in this
  environment, so `host_processor_caches` is `"N/A"` and
  `host_memory_configuration` is a bare `"DDR5"`. Replace both before any real
  submission.

Current values (verified in the built tree):

| field | value |
|---|---|
| host_processor_model_name | `Intel(R) Xeon(R) 6787P` |
| host_processors_per_node / core_count | 2 / 86 (344 CPUs, 2 threads/core, 4 NUMA nodes) |
| host_memory_capacity | 503GB |
| host_storage_capacity / type | 879G / NVMe SSD |
| operating_system | whatever `HOST_OS` is set to — `Ubuntu 24.04.4 LTS` in the current tree |
| framework | PyTorch 2.11.0+cpu, FAISS 1.14.3 |
| other_software_stack | transformers 4.57.6, langchain-community, bm25s |
| accelerators_per_node | `"0"` (string — see gotchas) |

## Accuracy metric

The reported accuracy is the DB manifest probe-query retrieval accuracy: the
mean top-K document-URL set overlap against the reference database over 50 fixed
probe queries. Gate: >= 0.95 (reported as >= 95 after percentage scaling).

The manifest lives at `assets/db_manifest_intel_xpu.json.gz` and is a
**`v2-behavioral`** manifest (`probe_top_k: 10`), so it must be verified with
`ingestion/db_manifest_v2.py` — v1's `verify` rejects it on the version field.
`evaluation/accuracy_eval_ingestion.py` calls
`ingestion.db_manifest_v2.verify_manifest()`.

The accuracy run additionally verifies file-processing success rate, the database
MD5 reported by the SUT, vector/docstore consistency, index dimension, and the
FAISS index parameters.

## Known artifact oddities (not blocking)

`output_datasetup_accuracy/mlperf_log_summary.txt` is essentially empty — it
contains only "No warnings encountered during test." with no MLPerf Results
Summary block. The checker accepts it (it parses `mlperf_log_detail.txt` for
accuracy runs, not the summary), but it is worth understanding before submitting.
The perf-mode summary is complete and `VALID`.

Related: the `--pipelined` path in `main_ingestion.py` calls `os._exit(0)` before
the "Results saved to" print, so a pipelined run reports no output paths.

## Next steps

1. **Fill in `host_processor_caches` and `host_memory_configuration`** from a
   host with `dmidecode` available.
2. **Build the QnA (`e2e-rag-qna`) submission** the same way. Its accuracy target
   is `33.95` (`constants.py:166`) and its dataset size is the 824 FRAMES
   queries. It is also the only e2e-rag workload in `models_TEST09`, so it needs
   the TEST09 compliance run (`scripts/run_compliance_test09.sh`).
3. **Investigate the empty accuracy-mode summary log.**
