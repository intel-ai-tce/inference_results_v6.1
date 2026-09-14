#!/usr/bin/env bash
# Build the image once, then run the SAME (scenario, mode) test in N fresh
# containers -- one replicate per container -- to sample run-to-run variability.
#
# In accuracy mode each replicate is, in turn, generated, scored with VBench
# (run_vbench.sh), and then pruned -- so you see PASS/FAIL as it happens and the
# large MLPerf accuracy logs never pile up. Pruning keeps only ONE
# mlperf_log_accuracy.json on disk at a time: the highest vbench_score among the
# runs that PASS so far; every other log (failed VBench, or a lower-scoring pass)
# is deleted immediately. Pruning deletes by default; preview with
# --dry-run-prune, or keep every log with --no-prune.
#
# Runs on the HOST (it drives ./launch.sh); do NOT run it inside the container.
# Because ./launch.sh uses `docker run --rm`, every step gets a brand-new
# container that is torn down when the inner command returns, so
# "create container -> run -> exit" happens once per step by construction.
# VBench scoring and pruning run in-container too (accuracy logs are root-owned
# on the host, so they must be removed from inside the container).
#
# Each replicate writes to  <output-dir>/run_<n>  (n = 1..RUNS by default). The
# path is interpreted inside the container (relative to /workspace/wan-harness);
# keep it under runs/ -- or any repo-relative path -- so the bind mounts persist
# it back to the host.
#
# Usage:
#   ./scripts/replicate_scenario.sh --scenario SCENARIO --output-dir DIR --runs N \
#       [--backend wan22] [--mode accuracy] [--start-index 1] \
#       [--vbench-nproc 1] [--no-vbench] [--no-prune] [--dry-run-prune] \
#       [--skip-build] [--fail-fast] [-- extra args forwarded to run_scenario.sh]
#
# Examples:
#   ./scripts/replicate_scenario.sh --scenario Offline \
#       --output-dir runs/wan22/offline_replicates --runs 5
#
#   # Score + preview the prune without deleting anything:
#   ./scripts/replicate_scenario.sh --scenario Offline \
#       --output-dir runs/wan22/offline_replicates --runs 5 --dry-run-prune
#
#   # Continue an interrupted sweep without rebuilding, starting at run_6:
#   ./scripts/replicate_scenario.sh --scenario SingleStream \
#       --output-dir runs/wan22/ss_replicates --runs 5 \
#       --start-index 6 --skip-build
#
# Note: ./launch.sh runs `docker run --rm -it`, so invoke this from an
# interactive terminal (a TTY). For background/CI use, drop -it in launch.sh.

set -uo pipefail   # NOT -e: per-replicate failures are tracked explicitly

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="wan22"
SCENARIO=""
MODE="accuracy"
OUTPUT_DIR=""
RUNS=""
START_INDEX=1
VBENCH_NPROC=1
NO_VBENCH=0
NO_PRUNE=0
DRY_RUN_PRUNE=0
SKIP_BUILD=0
FAIL_FAST=0
EXTRA_ARGS=()

die() { echo "[replicate] error: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)       BACKEND="$2"; shift 2 ;;
        --scenario)      SCENARIO="$2"; shift 2 ;;
        --mode)          MODE="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --runs|--count)  RUNS="$2"; shift 2 ;;
        --start-index)   START_INDEX="$2"; shift 2 ;;
        --vbench-nproc)  VBENCH_NPROC="$2"; shift 2 ;;
        --no-vbench)     NO_VBENCH=1; shift ;;
        --no-prune)      NO_PRUNE=1; shift ;;
        --dry-run-prune) DRY_RUN_PRUNE=1; shift ;;
        --skip-build)    SKIP_BUILD=1; shift ;;
        --fail-fast)     FAIL_FAST=1; shift ;;
        --)              shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '1,/^set -uo/p' "$0" | head -50
            exit 0 ;;
        *)               die "unknown argument: $1 (use -- to forward extra args)" ;;
    esac
done

[[ -n "${SCENARIO}"   ]] || die "--scenario is required"
[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"
[[ -n "${RUNS}"       ]] || die "--runs is required"
[[ "${RUNS}"         =~ ^[0-9]+$ ]] || die "--runs must be a positive integer (got '${RUNS}')"
[[ "${START_INDEX}"  =~ ^[0-9]+$ ]] || die "--start-index must be a non-negative integer"
[[ "${VBENCH_NPROC}" =~ ^[0-9]+$ ]] || die "--vbench-nproc must be a positive integer"
(( RUNS >= 1 )) || die "--runs must be >= 1"

LAUNCH="${REPO_ROOT}/launch.sh"
[[ -x "${LAUNCH}" ]] || die "cannot find executable ${LAUNCH}"

END_INDEX=$(( START_INDEX + RUNS - 1 ))

# VBench scoring + pruning only make sense for accuracy runs.
DO_VBENCH=0
if [[ "${NO_VBENCH}" -eq 0 && "${MODE}" == "accuracy" ]]; then
    DO_VBENCH=1
fi

echo "======================================================================"
echo "[replicate] backend=${BACKEND} scenario=${SCENARIO} mode=${MODE}"
echo "[replicate] output base=${OUTPUT_DIR}   run_${START_INDEX}..run_${END_INDEX}  (${RUNS} replicate(s))"
if [[ "${DO_VBENCH}" -eq 1 ]]; then
    if [[ "${NO_PRUNE}" -eq 1 ]]; then
        prune_desc="score only (--no-prune)"
    elif [[ "${DRY_RUN_PRUNE}" -eq 1 ]]; then
        prune_desc="score + prune PREVIEW (--dry-run-prune)"
    else
        prune_desc="score + prune (delete redundant/failed accuracy logs)"
    fi
    echo "[replicate] vbench: ${prune_desc}   (nproc-per-node=${VBENCH_NPROC})"
elif [[ "${NO_VBENCH}" -eq 1 ]]; then
    echo "[replicate] vbench: disabled (--no-vbench)"
else
    echo "[replicate] vbench: skipped (only runs in accuracy mode; mode=${MODE})"
fi
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[replicate] extra args -> run_scenario.sh: ${EXTRA_ARGS[*]}"
echo "======================================================================"

# --- 1. Build the image once -----------------------------------------------
if [[ "${SKIP_BUILD}" -eq 1 ]]; then
    echo "[replicate] --skip-build: reusing existing image"
else
    echo "[replicate] building image: ./launch.sh --build"
    "${LAUNCH}" --build || die "image build failed"
fi

# --- Prune helper (in-container python) ------------------------------------
# Reads each run's vbench/vbench_summary.json in run_<start>..run_<end>, keeps
# the mlperf_log_accuracy.json of the highest-scoring PASSING run, and deletes
# the rest (failed VBench, or redundant passing runs). Runs in-container so the
# root-owned logs can actually be removed. Called after EVERY replicate with
# end=<current run>, so at most one accuracy log is ever on disk at a time.
read -r -d '' PRUNE_PY <<'PY' || true
import json, os, sys

base = sys.argv[1]
start, end = int(sys.argv[2]), int(sys.argv[3])
dry = sys.argv[4] == "1"


def human(nbytes):
    x = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{x:.1f} {unit}"
        x /= 1024.0


runs = []
for n in range(start, end + 1):
    rundir = os.path.join(base, f"run_{n}")
    if not os.path.isdir(rundir):
        continue
    acc = os.path.join(rundir, "mlperf_log_accuracy.json")
    summ = os.path.join(rundir, "vbench", "vbench_summary.json")
    passed, score, scored = False, None, False
    if os.path.isfile(summ):
        try:
            with open(summ, encoding="utf-8") as fh:
                d = json.load(fh)
            passed = bool(d.get("pass_99", False))
            score = d.get("vbench_score")
            scored = True
        except (OSError, ValueError) as exc:
            print(f"[prune] WARN: cannot read {summ}: {exc}")
    acc_exists = os.path.isfile(acc)
    runs.append(
        dict(
            n=n,
            acc=acc,
            acc_exists=acc_exists,
            acc_bytes=os.path.getsize(acc) if acc_exists else 0,
            passed=passed,
            score=score,
            scored=scored,
        )
    )

if not runs:
    print("[prune] no run directories found; nothing to prune.")
    sys.exit(0)

passing = [r for r in runs if r["passed"] and r["acc_exists"] and r["score"] is not None]
best = max(passing, key=lambda r: r["score"], default=None)

print("[prune] per-run VBench status:")
for r in runs:
    tag = "PASS" if r["passed"] else ("FAIL" if r["scored"] else "UNSCORED")
    score_s = f"{r['score']:.4f}" if r["score"] is not None else "n/a"
    keep = best is not None and r["n"] == best["n"]
    print(
        f"[prune]   run_{r['n']:<3d} {tag:<8s} score={score_s:>9s} "
        f"acc_log={'yes' if r['acc_exists'] else 'no ':>3s}"
        + ("   <-- KEEP" if keep else "")
    )

if best is None:
    print("[prune] WARNING: no run passed VBench -- no accuracy log will be retained.")
else:
    print(
        f"[prune] retaining run_{best['n']} accuracy log "
        f"(highest passing vbench_score {best['score']:.4f})."
    )

freed, deleted = 0, []
for r in runs:
    if best is not None and r["n"] == best["n"]:
        continue
    if not r["acc_exists"]:
        continue
    freed += r["acc_bytes"]
    deleted.append(r["n"])
    if dry:
        print(f"[prune] would delete {r['acc']} ({human(r['acc_bytes'])})")
    else:
        try:
            os.remove(r["acc"])
            print(f"[prune] deleted {r['acc']} ({human(r['acc_bytes'])})")
        except OSError as exc:
            print(f"[prune] ERROR deleting {r['acc']}: {exc}")

verb = "would free" if dry else "freed"
print(
    f"[prune] {verb} {human(freed)} across {len(deleted)} accuracy log(s); "
    f"retained {0 if best is None else 1}."
)
PY

prune_logs() {
    # $1 = highest run index to consider (prune over run_START..run_$1).
    local last="$1"
    if [[ "${DRY_RUN_PRUNE}" -eq 1 ]]; then
        echo "[replicate] (${last}) pruning run_${START_INDEX}..run_${last} (DRY RUN -- nothing deleted)"
    else
        echo "[replicate] (${last}) pruning run_${START_INDEX}..run_${last} (keeping best passing log)"
    fi
    "${LAUNCH}" python3 -c "${PRUNE_PY}" \
        "${OUTPUT_DIR}" "${START_INDEX}" "${last}" "${DRY_RUN_PRUNE}" \
        || echo "[replicate] WARN: prune step exited non-zero" >&2
}

# --- Run each replicate in turn: generate -> VBench score -> prune ---------
declare -a GEN_OK=()
declare -a GEN_FAILED=()
declare -a VBENCH_OK=()
declare -a VBENCH_FAILED=()

for (( n = START_INDEX; n <= END_INDEX; n++ )); do
    run_dir="${OUTPUT_DIR}/run_${n}"
    echo ""
    echo "----------------------------------------------------------------------"
    echo "[replicate] replicate ${n}/${END_INDEX} -> ${run_dir}"
    echo "----------------------------------------------------------------------"

    # (1) generate -- a single ./launch.sh call = create container, run, remove.
    echo "[replicate] (${n}) generating (run_scenario.sh)"
    if "${LAUNCH}" ./scripts/run_scenario.sh \
            --backend "${BACKEND}" \
            --scenario "${SCENARIO}" \
            --mode "${MODE}" \
            --output-dir "${run_dir}" \
            "${EXTRA_ARGS[@]}"; then
        echo "[replicate] (${n}) generation PASS"
        GEN_OK+=("${n}")
    else
        rc=$?
        echo "[replicate] (${n}) generation FAIL (exit ${rc})" >&2
        GEN_FAILED+=("${n}")
        if [[ "${FAIL_FAST}" -eq 1 ]]; then
            die "--fail-fast: aborting after replicate ${n} (generation)"
        fi
        # Nothing worth scoring; still prune so a partial log does not linger.
        [[ "${DO_VBENCH}" -eq 1 && "${NO_PRUNE}" -eq 0 ]] && prune_logs "${n}"
        continue
    fi

    # (2) score with VBench, then (3) prune immediately -- accuracy mode only.
    if [[ "${DO_VBENCH}" -eq 1 ]]; then
        echo "[replicate] (${n}) scoring (run_vbench.sh, nproc-per-node=${VBENCH_NPROC})"
        if "${LAUNCH}" ./scripts/run_vbench.sh \
                --backend "${BACKEND}" \
                --scenario "${SCENARIO}" \
                --accuracy-dir "${run_dir}" \
                --nproc-per-node "${VBENCH_NPROC}"; then
            echo "[replicate] (${n}) VBench scoring done"
            VBENCH_OK+=("${n}")
        else
            rc=$?
            echo "[replicate] (${n}) VBench scoring FAILED (exit ${rc})" >&2
            VBENCH_FAILED+=("${n}")
            if [[ "${FAIL_FAST}" -eq 1 ]]; then
                # Prune before bailing so we do not leave logs piled up.
                [[ "${NO_PRUNE}" -eq 0 ]] && prune_logs "${n}"
                die "--fail-fast: aborting after replicate ${n} (VBench scoring)"
            fi
        fi

        [[ "${NO_PRUNE}" -eq 0 ]] && prune_logs "${n}"
    fi
done

# --- Summary ---------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[replicate] generation: ${#GEN_OK[@]} ok, ${#GEN_FAILED[@]} failed (of ${RUNS})"
[[ ${#GEN_OK[@]}     -gt 0 ]] && echo "[replicate]   generated: ${GEN_OK[*]/#/run_}"
[[ ${#GEN_FAILED[@]} -gt 0 ]] && echo "[replicate]   failed:    ${GEN_FAILED[*]/#/run_}"
if [[ "${DO_VBENCH}" -eq 1 ]]; then
    echo "[replicate] vbench scoring: ${#VBENCH_OK[@]} ok, ${#VBENCH_FAILED[@]} failed"
    [[ ${#VBENCH_FAILED[@]} -gt 0 ]] && echo "[replicate]   scoring failed: ${VBENCH_FAILED[*]/#/run_}"
fi
echo "======================================================================"

[[ ${#GEN_FAILED[@]} -eq 0 && ${#VBENCH_FAILED[@]} -eq 0 ]] || exit 1
