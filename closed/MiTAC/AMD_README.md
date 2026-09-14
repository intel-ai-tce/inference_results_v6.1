# AMD MLPerf Inference v6.1

This package covers four benchmarks. Three of them (`llama2-70b-99`, `gpt-oss-120b`,
`llama3.1-8b`) use the same tool (`submission.py`) and the same workflow, each with its
own setup. `wan-2.2-t2v-a14b` (text-to-video) has its **own** self-contained harness (its
own Docker image, `wan-harness` CLI, and packaging scripts); it is run and packaged
separately, and its finished tree can **optionally** be folded into the combined zip by
`submission.py combine`. Full instructions in [`WAN_Readme.md`](WAN_Readme.md).

| Benchmark          | `--model` value | Scenarios                    | Setup guide                                                                                              |
|--------------------|-----------------|------------------------------|----------------------------------------------------------------------------------------------------------|
| `llama2-70b-99`    | `llama2-70b-99` | Offline, Server, Interactive | [`setup.md#llama2-70b-99`](setup.md#llama2-70b-99)                                                        |
| `gpt-oss-120b`     | `gpt-oss-120b`  | Offline, Server              | [`setup.md#gpt-oss-120b`](setup.md#gpt-oss-120b)                                                          |
| `llama3.1-8b`      | `llama3_1-8b`   | Offline, Server, Interactive | [`setup.md#llama3.1-8b`](setup.md#llama31-8b)                                                             |
| `wan-2.2-t2v-a14b` | (own CLI)       | Offline, SingleStream        | [`WAN_Readme.md`](WAN_Readme.md) (separate harness and workflow)                                         |

> **MI350X vs MI355X:** the `<gpu>` placeholder below is `mi355x` or `mi350x` - pick the
> one matching your hardware for both the `*_<gpu>.yaml` model-conf and `user_<gpu>.conf`
> user-conf, and set `GPU_NAME` to match (e.g. `mi350x`). All three benchmarks ship both
> `mi355x` and `mi350x` configs with per-GPU tuned values.

---

# Setup

For setup instructions, follow the per-benchmark guide for each model you plan to run:

- **llama2-70b-99** -> [`setup.md#llama2-70b-99`](setup.md#llama2-70b-99)
- **gpt-oss-120b** -> [`setup.md#gpt-oss-120b`](setup.md#gpt-oss-120b)
- **llama3.1-8b** -> [`setup.md#llama3.1-8b`](setup.md#llama31-8b)
  (separate Docker image and its own subtree)
- **wan-2.2-t2v-a14b** -> [`WAN_Readme.md`](WAN_Readme.md)
  (separate harness, Docker image, and workflow)

Each guide walks through the dataset download, model download/quantization, and how to
build and start that benchmark's container. Everything in the sections below runs
**inside the container**, from the `submission/` folder,
and applies to the three `submission.py` benchmarks only.

---

# Running Experiments

All commands run **inside the container**, from `/lab-mlperf-inference/submission` -
the **same path in every benchmark's container**, so once you're inside you don't need
to worry about host directories. Each container bind-mounts its own subtree there, so
`../code/<config-dir>/` always resolves to that benchmark's configs. The only host-side
difference is which folder you **start the container from** during setup: the repo root
for `llama2-70b-99` / `gpt-oss-120b`, and the nested `code_llama3.1-8b/mlperf-inference/`
subtree for `llama3.1-8b` (its setup guide does this for you).

## 1. Generate performance data

To generate performance data for a `<benchmark>`, run once per scenario it supports.
Substitute the placeholders below (see the `#` comments for valid values):

```bash
# <benchmark>  :  llama2-70b-99  |  gpt-oss-120b  |  llama3_1-8b   (the --model value)
# <config-dir> :  the config folder under ../code/ - same as <benchmark>, EXCEPT
#                 llama3_1-8b, whose folder is ../code/llama3.1-8b/ (dot, not underscore):
#                   llama2-70b-99 -> ../code/llama2-70b-99/
#                   gpt-oss-120b  -> ../code/gpt-oss-120b/
#                   llama3_1-8b   -> ../code/llama3.1-8b/
# <scenario>   :  Offline  |  Server  |  Interactive
#   - gpt-oss-120b supports Offline and Server only
# <gpu>        :  mi355x  |  mi350x   (must match your hardware; set GPU_NAME to match)
#   - config filenames use the lowercase scenario + gpu, e.g. offline_mi355x.yaml / user_mi350x.conf
#   - llama3_1-8b runs from the nested submission/ folder (ships both mi355x and mi350x configs)
python3 submission.py --model <benchmark> experiment --scenario <scenario> \
    --model-conf ../code/<config-dir>/<scenario>_<gpu>.yaml \
    --user-conf ../code/<config-dir>/user_<gpu>.conf
```

To force a specific hand-run experiment to be treated as "the best":

```bash
python3 submission.py --model <benchmark> update_best --scenario <scenario>
```

## 2. Check status

Read-only check of what exists so far (PERF/ACC/COMP per scenario). Run it any time -
it's most useful after accuracy and compliance, as a final check before packaging.

```bash
# <benchmark> :  llama2-70b-99  |  gpt-oss-120b  |  llama3_1-8b
python3 submission.py --model <benchmark> status
```

## 3. Prepare accuracy

Run once per scenario you ran:

```bash
# <benchmark> :  llama2-70b-99  |  gpt-oss-120b  |  llama3_1-8b
# <scenario>  :  Offline  |  Server  |  Interactive
python3 submission.py --model <benchmark> prepare --scenario <scenario> accuracy
```

## 4. Prepare compliance

Run for **every scenario** you ran. Each benchmark has different compliance tests:

- **llama2-70b-99** - one test (TEST06), no `--test-version` needed:

```bash
python3 submission.py --model llama2-70b-99 prepare --scenario Offline compliance
python3 submission.py --model llama2-70b-99 prepare --scenario Server compliance
python3 submission.py --model llama2-70b-99 prepare --scenario Interactive compliance
```

- **gpt-oss-120b** - two tests (TEST07 + TEST09), so `--test-version` is **required** for each scenario:

```bash
python3 submission.py --model gpt-oss-120b prepare --scenario Offline --test-version TEST07 compliance
python3 submission.py --model gpt-oss-120b prepare --scenario Offline --test-version TEST09 compliance
python3 submission.py --model gpt-oss-120b prepare --scenario Server  --test-version TEST07 compliance
python3 submission.py --model gpt-oss-120b prepare --scenario Server  --test-version TEST09 compliance
```

- **llama3.1-8b** - one test (TEST06), in the nested container:

```bash
python3 submission.py --model llama3_1-8b prepare compliance --scenario Offline
python3 submission.py --model llama3_1-8b prepare compliance --scenario Server
python3 submission.py --model llama3_1-8b prepare compliance --scenario Interactive
```

> Tip: to redo a result that already exists, add `--force` right after `prepare`, e.g.
> `python3 submission.py --model llama2-70b-99 prepare --force --scenario Offline accuracy`

## 5. Package the submission

Fill in your system specs in `<GPU_NAME>_system.json` (`mi355x_system.json` /
`mi350x_system.json`) first - it must match your hardware (see
[`README_system_json.md`](submission/README_system_json.md)). The `submission/`
folder ships only `dummy_system.json` - **create your `submission/mi355x_system.json`
(or `submission/mi350x_system.json`) by copying `dummy_system.json` as a reference**
and editing the values to match your machine. Set these environment
variables (the same values in every container, so all three packages match):

| Variable    | Example      | What it is                                       |
|-------------|--------------|--------------------------------------------------|
| `GPU_COUNT` | `8`          | Number of GPUs                                   |
| `GPU_NAME`  | `mi355x`     | GPU name (lowercase; must match the system JSON) |
| `CPU_COUNT` | `2`          | Number of CPUs                                   |
| `CPU_NAME`  | `EPYC-9575F` | CPU model (find it with `lscpu \| grep name`)    |
| `COMPANY`   | `AMD`        | Your company name                                |

```bash
export GPU_COUNT=8 GPU_NAME="mi355x" CPU_COUNT=2 CPU_NAME="EPYC-9575F" COMPANY="AMD"
```

> Run this `export` once in **each** shell before packaging - the llama2-70b-99 container,
> the gpt-oss-120b container, the nested llama3.1-8b container, and again before `combine`
> if you opened a new shell. A fresh shell won't have these set.

**a. Package each benchmark.** Each command runs from
`/lab-mlperf-inference/submission`. These are **three separate containers** (each model has
its own image), but they all bind-mount the same host `submission/` folder, so every
package lands side by side there.

In the **llama2-70b-99** container:

```bash
python3 submission.py --model llama2-70b-99 package
```

In the **gpt-oss-120b** container:

```bash
python3 submission.py --model gpt-oss-120b package
```

In the **nested llama3.1-8b** container (adds `MLPERF_INFERENCE_DIR`):

```bash
MLPERF_INFERENCE_DIR=/app/mlperf_inference python3 submission.py --model llama3_1-8b package
```

**b. Combine into one submission.** Only after **all three** benchmarks in step 5a are
packaged. Run this from either the **llama2-70b-99** or **gpt-oss-120b** container (both
mount the shared `submission/` folder, and via the `$HOME`->`/workdir` mount `combine` also
finds the packaged llama3.1-8b subtree). With the same env vars set, from
`/lab-mlperf-inference/submission` run:

```bash
python3 submission.py combine
```

This merges the per-benchmark packages, runs the submission_checker once over the
combined tree, and writes **one** `submission/inference_results_6.1.zip` (all three
benchmarks) plus `submission/summary.csv`.

> **Including wan-2.2-t2v-a14b (optional):** if a WAN submission tree is present,
> `combine` folds it into the same zip (set `WAN_SUBMISSION_DIR`, or leave it under
> `code_wan/submissions/`). See [`WAN_Readme.md`](WAN_Readme.md) for how to build and
> combine WAN.

---

## Quick reference

| Benchmark          | `--model` value | Scenarios                    | Compliance test(s) | Package output                                             |
|--------------------|-----------------|------------------------------|--------------------|------------------------------------------------------------|
| `llama2-70b-99`    | `llama2-70b-99` | Offline, Server, Interactive | TEST06             | `submission/` (`.zip`)                                     |
| `gpt-oss-120b`     | `gpt-oss-120b`  | Offline, Server              | TEST07 + TEST09    | `submission/` (`.zip`)                                     |
| `llama3.1-8b`      | `llama3_1-8b`   | Offline, Server, Interactive | TEST06             | `code_llama3.1-8b/mlperf-inference/submission/` (`.tar.gz`)|
| **all three**      | `combine`       | -                            | -                  | `submission/inference_results_6.1.zip` (one combined zip)  |
| `wan-2.2-t2v-a14b` | `wan-harness` CLI (separate) | Offline, SingleStream | TEST04    | `code_wan/submissions/...` (own tree); optionally folded into the `combine` zip |
