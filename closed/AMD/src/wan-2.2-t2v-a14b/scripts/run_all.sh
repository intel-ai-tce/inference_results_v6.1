#!/usr/bin/env bash
# Run all four (scenario, mode) combinations needed for a submission, then
# TEST04 compliance and VBench scoring for each scenario (TEST04 is the
# only compliance test required for wan-2.2-t2v-a14b per
# mlcommons/inference/compliance/README.md; accuracy is scored via VBench).
#
# Intended to be invoked inside the wan-harness container. On the host,
# start a shell with ``./launch.sh``, then run:
#
#     ./scripts/run_all.sh --backend wan22
#
# For automation, ``./launch.sh ./scripts/run_all.sh --backend wan22`` from
# the host is equivalent (fresh container, same bind mounts).
#
# Each invocation creates a fresh timestamped + git-SHA-stamped experiment
# directory under ``runs/${backend}/`` so multiple runs co-exist without
# clobbering each other. The matrix runs to completion even if one combo
# fails; a final pass/fail summary is printed and the script exits
# non-zero iff any combo failed.
#
# Default layout:
#   runs/${BACKEND}/${TIMESTAMP}_g${SHORT_SHA}[-dirty][__${NAME}]/
#   ├── MANIFEST.json                        (git SHA, host, args, etc.)
#   ├── Offline/performance/run_1/{harness_metadata.json, mlperf_log_*}
#   ├── Offline/performance/run.log
#   ├── Offline/accuracy/{run.log, accuracy.txt, artefacts/, vbench/}
#   ├── Offline/accuracy/run_vbench.log
#   ├── Offline/TEST04/{mlperf_log_*, ...}   (compliance run)
#   ├── Offline/compliance/TEST04/           (verification artefacts)
#   ├── SingleStream/performance/run_1/{...}
#   ├── SingleStream/accuracy/{...}
#   ├── SingleStream/TEST04/{...}
#   └── SingleStream/compliance/TEST04/
#
# Plus a ``runs/${BACKEND}/latest`` symlink refreshed to point at the
# newest experiment directory (only when --root is left at its default).
#
# Usage:
#   ./scripts/run_all.sh [--backend mock|wan22] [--name LABEL]
#                        [--root PATH] [--skip-compliance] [--skip-vbench]
#                        [--dry-run-plan]
#                        [-- extra args forwarded to wan-harness]
#
# Environment overrides (consumed by run_scenario.sh / run_vbench.sh):
#   BACKEND_CONFIG=<path>       per-scenario YAML override
#   NPROC_PER_NODE=<int>        torchrun --nproc-per-node (default: 8)
#   VBENCH_NPROC_PER_NODE=<int>  VBench torchrun size (default: 1)

set -uo pipefail   # NOT -e: we track per-combo failures explicitly

BACKEND="mock"
NAME=""
ROOT=""
DRY_RUN_PLAN=0
SKIP_COMPLIANCE=0
SKIP_VBENCH=0
EXTRA_ARGS=()
ROOT_EXPLICIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)          BACKEND="$2"; shift 2 ;;
        --name)             NAME="$2"; shift 2 ;;
        --root)             ROOT="$2"; ROOT_EXPLICIT=1; shift 2 ;;
        --skip-compliance)  SKIP_COMPLIANCE=1; shift ;;
        --skip-vbench)      SKIP_VBENCH=1; shift ;;
        --dry-run-plan)     DRY_RUN_PLAN=1; shift ;;
        --)                 shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '1,/^set -uo/p' "$0" | head -40
            exit 0 ;;
        *)                EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------
# Git introspection. ``-c safe.directory=*`` keeps git happy when the
# repo is owned by the host user but the container shell is root
# (common when ./launch.sh bind-mounts a host-owned repo).
# ---------------------------------------------------------------------
git_safe() { git -c safe.directory='*' "$@" 2>/dev/null; }

SHORT_SHA="$(git_safe rev-parse --short=7 HEAD || echo nogit)"
DIRTY=""
if [[ "${SHORT_SHA}" != "nogit" ]]; then
    if ! git_safe diff --quiet --ignore-submodules HEAD; then
        DIRTY="-dirty"
    fi
fi

TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

if [[ -z "${ROOT}" ]]; then
    SUFFIX=""
    [[ -n "${NAME}" ]] && SUFFIX="__${NAME}"
    ROOT="runs/${BACKEND}/${TIMESTAMP}_g${SHORT_SHA}${DIRTY}${SUFFIX}"
fi

# ---------------------------------------------------------------------
# Dry-run-plan: print the resolved commands and exit. Useful before
# kicking off a multi-hour wan22 matrix.
# ---------------------------------------------------------------------
if [[ ${DRY_RUN_PLAN} -eq 1 ]]; then
    echo "[run_all] DRY-RUN PLAN (no commands executed)"
    echo "[run_all]   backend     = ${BACKEND}"
    echo "[run_all]   root        = ${ROOT}"
    echo "[run_all]   sha         = ${SHORT_SHA}${DIRTY}"
    echo "[run_all]   name        = ${NAME:-(none)}"
    echo "[run_all]   extra_args  = ${EXTRA_ARGS[*]:-(none)}"
    echo "[run_all]   compliance  = $([[ ${SKIP_COMPLIANCE} -eq 1 ]] && echo skipped || echo TEST04 per scenario)"
    if [[ ${SKIP_VBENCH} -eq 1 ]]; then
        echo "[run_all]   vbench      = skipped (--skip-vbench)"
    elif [[ "${BACKEND}" == "mock" ]]; then
        echo "[run_all]   vbench      = skipped (mock backend has no .mp4 artefacts)"
    else
        echo "[run_all]   vbench      = per scenario after accuracy"
    fi
    echo "[run_all]   env BACKEND_CONFIG = ${BACKEND_CONFIG:-(unset)}"
    echo "[run_all]   env NPROC_PER_NODE = ${NPROC_PER_NODE:-(unset)}"
    echo "[run_all]   env VBENCH_NPROC_PER_NODE = ${VBENCH_NPROC_PER_NODE:-(unset)}"
    echo "[run_all]"
    echo "[run_all]   would run:"
    for SCENARIO in Offline SingleStream; do
        for MODE in performance accuracy; do
            if [[ "${MODE}" == "performance" ]]; then
                OUT="${ROOT}/${SCENARIO}/performance/run_1"
            else
                OUT="${ROOT}/${SCENARIO}/${MODE}"
            fi
            echo "     ./scripts/run_scenario.sh --backend ${BACKEND} --scenario ${SCENARIO} --mode ${MODE} --output-dir ${OUT} -- ${EXTRA_ARGS[*]:-}"
        done
        if [[ ${SKIP_COMPLIANCE} -eq 0 ]]; then
            echo "     ./scripts/verify_compliance.sh --backend ${BACKEND} --scenario ${SCENARIO} --scenario-dir ${ROOT}/${SCENARIO}"
        fi
        if [[ ${SKIP_VBENCH} -eq 0 && "${BACKEND}" != "mock" ]]; then
            echo "     ./scripts/run_vbench.sh --backend ${BACKEND} --scenario ${SCENARIO} --accuracy-dir ${ROOT}/${SCENARIO}/accuracy"
        fi
    done
    exit 0
fi

mkdir -p "${ROOT}"

# ---------------------------------------------------------------------
# Experiment manifest. Captures everything needed to recreate / trace
# the run: git state, host, resolved CLI, env overrides, start time.
# Written via python3 to avoid hand-escaping JSON. We pass dynamic
# values through environment variables (immune to shell-quoting
# pitfalls in commit subjects, branch names, etc.) and a literal
# heredoc so nothing in the Python source is shell-interpolated.
# ---------------------------------------------------------------------
RA_DIRTY_FLAG="false"
[[ -n "${DIRTY}" ]] && RA_DIRTY_FLAG="true"

RA_EXTRA_ARGS_JSON="$(
    python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
)"

export RA_STARTED_AT="${TIMESTAMP}"
export RA_BACKEND="${BACKEND}"
export RA_NAME="${NAME}"
export RA_ROOT="${ROOT}"
export RA_HOST="$(hostname)"
export RA_DIRTY_FLAG
export RA_EXTRA_ARGS_JSON
export RA_SKIP_COMPLIANCE="${SKIP_COMPLIANCE}"
export RA_SKIP_VBENCH="${SKIP_VBENCH}"

python3 - >"${ROOT}/MANIFEST.json" <<'PYEOF'
import json, os, subprocess

def git(*args):
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", *args],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

manifest = {
    "started_at_utc": os.environ["RA_STARTED_AT"],
    "backend": os.environ["RA_BACKEND"],
    "name": os.environ.get("RA_NAME") or None,
    "root": os.environ["RA_ROOT"],
    "git": {
        "sha": git("rev-parse", "HEAD"),
        "short_sha": git("rev-parse", "--short=7", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": os.environ["RA_DIRTY_FLAG"] == "true",
        "commit_subject": git("log", "-1", "--format=%s"),
    },
    "host": os.environ["RA_HOST"],
    "env": {
        "BACKEND_CONFIG": os.environ.get("BACKEND_CONFIG"),
        "NPROC_PER_NODE": os.environ.get("NPROC_PER_NODE"),
        "VBENCH_NPROC_PER_NODE": os.environ.get("VBENCH_NPROC_PER_NODE"),
    },
    "extra_args": json.loads(os.environ["RA_EXTRA_ARGS_JSON"]),
    "scenarios": ["Offline", "SingleStream"],
    "modes": ["performance", "accuracy"],
    "compliance": os.environ.get("RA_SKIP_COMPLIANCE") != "1",
    "compliance_tests": ["TEST04"],
    "vbench": (
        os.environ.get("RA_SKIP_VBENCH") != "1"
        and os.environ.get("RA_BACKEND") != "mock"
    ),
    "vbench_skip_reason": (
        "mock backend" if os.environ.get("RA_BACKEND") == "mock"
        else ("--skip-vbench" if os.environ.get("RA_SKIP_VBENCH") == "1" else None)
    ),
}
print(json.dumps(manifest, indent=2, sort_keys=True))
PYEOF
echo "[run_all] wrote ${ROOT}/MANIFEST.json"

# ---------------------------------------------------------------------
# Run the matrix. Track per-combo exit codes; the matrix runs to
# completion even when one (scenario, mode) fails.
# ---------------------------------------------------------------------
declare -A RESULTS=()
declare -A LATENCIES=()
START_EPOCH="$(date +%s)"

for SCENARIO in Offline SingleStream; do
    for MODE in performance accuracy; do
        if [[ "${MODE}" == "performance" ]]; then
            OUT="${ROOT}/${SCENARIO}/performance/run_1"
            LOG="${ROOT}/${SCENARIO}/performance/run.log"
        else
            OUT="${ROOT}/${SCENARIO}/${MODE}"
            LOG="${OUT}/run.log"
        fi
        mkdir -p "${OUT}"
        echo
        echo "=============================================================="
        echo "[run_all] backend=${BACKEND} scenario=${SCENARIO} mode=${MODE}"
        echo "[run_all] output_dir=${OUT}"
        echo "[run_all] log=${LOG}"
        echo "=============================================================="
        T0="$(date +%s)"
        ./scripts/run_scenario.sh \
            --backend "${BACKEND}" \
            --scenario "${SCENARIO}" \
            --mode "${MODE}" \
            --output-dir "${OUT}" \
            -- "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee "${LOG}"
        # PIPESTATUS[0] = run_scenario.sh exit; tee will normally be 0.
        RC=${PIPESTATUS[0]}
        ELAPSED=$(( $(date +%s) - T0 ))
        RESULTS["${SCENARIO}/${MODE}"]=${RC}
        LATENCIES["${SCENARIO}/${MODE}"]=${ELAPSED}
        if [[ ${RC} -ne 0 ]]; then
            echo "[run_all] FAILED ${SCENARIO}/${MODE} (rc=${RC}, ${ELAPSED}s); continuing"
        fi
    done

    # TEST04 compliance for this scenario (requires a successful performance run).
    if [[ ${SKIP_COMPLIANCE} -eq 0 ]]; then
        PERF_KEY="${SCENARIO}/performance"
        if [[ ${RESULTS[${PERF_KEY}]} -ne 0 ]]; then
            echo
            echo "[run_all] skipping TEST04 for ${SCENARIO} (performance run failed)"
            RESULTS["${SCENARIO}/TEST04"]=1
            LATENCIES["${SCENARIO}/TEST04"]=0
        else
            SCENARIO_DIR="${ROOT}/${SCENARIO}"
            LOG="${SCENARIO_DIR}/TEST04/run.log"
            mkdir -p "${SCENARIO_DIR}/TEST04"
            echo
            echo "=============================================================="
            echo "[run_all] compliance TEST04 backend=${BACKEND} scenario=${SCENARIO}"
            echo "[run_all] scenario_dir=${SCENARIO_DIR}"
            echo "[run_all] log=${LOG}"
            echo "=============================================================="
            T0="$(date +%s)"
            ./scripts/verify_compliance.sh \
                --backend "${BACKEND}" \
                --scenario "${SCENARIO}" \
                --scenario-dir "${SCENARIO_DIR}" \
                -- "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee "${LOG}"
            RC=${PIPESTATUS[0]}
            ELAPSED=$(( $(date +%s) - T0 ))
            RESULTS["${SCENARIO}/TEST04"]=${RC}
            LATENCIES["${SCENARIO}/TEST04"]=${ELAPSED}
            if [[ ${RC} -ne 0 ]]; then
                echo "[run_all] FAILED ${SCENARIO}/TEST04 (rc=${RC}, ${ELAPSED}s); continuing"
            fi
        fi
    fi

    # VBench scoring for this scenario (requires a successful accuracy run).
    if [[ ${SKIP_VBENCH} -eq 0 && "${BACKEND}" != "mock" ]]; then
        ACC_KEY="${SCENARIO}/accuracy"
        if [[ ${RESULTS[${ACC_KEY}]} -ne 0 ]]; then
            echo
            echo "[run_all] skipping VBench for ${SCENARIO} (accuracy run failed)"
            RESULTS["${SCENARIO}/VBench"]=1
            LATENCIES["${SCENARIO}/VBench"]=0
        else
            ACCURACY_DIR="${ROOT}/${SCENARIO}/accuracy"
            LOG="${ACCURACY_DIR}/run_vbench.log"
            echo
            echo "=============================================================="
            echo "[run_all] VBench backend=${BACKEND} scenario=${SCENARIO}"
            echo "[run_all] accuracy_dir=${ACCURACY_DIR}"
            echo "[run_all] log=${LOG}"
            echo "=============================================================="
            T0="$(date +%s)"
            ./scripts/run_vbench.sh \
                --backend "${BACKEND}" \
                --scenario "${SCENARIO}" \
                --accuracy-dir "${ACCURACY_DIR}" \
                2>&1 | tee "${LOG}"
            RC=${PIPESTATUS[0]}
            ELAPSED=$(( $(date +%s) - T0 ))
            RESULTS["${SCENARIO}/VBench"]=${RC}
            LATENCIES["${SCENARIO}/VBench"]=${ELAPSED}
            if [[ ${RC} -ne 0 ]]; then
                echo "[run_all] FAILED ${SCENARIO}/VBench (rc=${RC}, ${ELAPSED}s); continuing"
            fi
        fi
    fi
done

TOTAL_ELAPSED=$(( $(date +%s) - START_EPOCH ))

# ---------------------------------------------------------------------
# Summary. Print in deterministic order (Offline first, performance
# before accuracy) regardless of associative-array iteration order.
# ---------------------------------------------------------------------
echo
echo "=============================================================="
echo "[run_all] summary"
echo "  root  : ${ROOT}"
echo "  total : ${TOTAL_ELAPSED}s"
echo "=============================================================="
FAILED=0
for SCENARIO in Offline SingleStream; do
    for MODE in performance accuracy; do
        KEY="${SCENARIO}/${MODE}"
        RC=${RESULTS[${KEY}]}
        SECS=${LATENCIES[${KEY}]}
        if [[ ${RC} -eq 0 ]]; then
            printf "  %-30s OK     (%4ds)\n" "${KEY}" "${SECS}"
        else
            printf "  %-30s FAIL   (%4ds, rc=%d)\n" "${KEY}" "${SECS}" "${RC}"
            FAILED=1
        fi
    done
    if [[ ${SKIP_COMPLIANCE} -eq 0 ]]; then
        KEY="${SCENARIO}/TEST04"
        RC=${RESULTS[${KEY}]:-1}
        SECS=${LATENCIES[${KEY}]:-0}
        if [[ ${RC} -eq 0 ]]; then
            printf "  %-30s OK     (%4ds)\n" "${KEY}" "${SECS}"
        else
            printf "  %-30s FAIL   (%4ds, rc=%d)\n" "${KEY}" "${SECS}" "${RC}"
            FAILED=1
        fi
    fi
    if [[ ${SKIP_VBENCH} -eq 0 && "${BACKEND}" != "mock" ]]; then
        KEY="${SCENARIO}/VBench"
        RC=${RESULTS[${KEY}]:-1}
        SECS=${LATENCIES[${KEY}]:-0}
        if [[ ${RC} -eq 0 ]]; then
            printf "  %-30s OK     (%4ds)\n" "${KEY}" "${SECS}"
        else
            printf "  %-30s FAIL   (%4ds, rc=%d)\n" "${KEY}" "${SECS}" "${RC}"
            FAILED=1
        fi
    fi
done

# ---------------------------------------------------------------------
# Latest symlink. Only when --root was left at its default, so an
# explicit override doesn't drop a stray symlink in some unrelated dir.
# Refreshed unconditionally (pass or fail) so "go look at the latest
# experiment" always works during exploratory work.
# ---------------------------------------------------------------------
if [[ ${ROOT_EXPLICIT} -eq 0 ]]; then
    LATEST_DIR="$(dirname "${ROOT}")"
    LATEST_LINK="${LATEST_DIR}/latest"
    TARGET="$(basename "${ROOT}")"
    ln -sfn "${TARGET}" "${LATEST_LINK}"
    echo "[run_all] symlink ${LATEST_LINK} -> ${TARGET}"
fi

POST_STEPS=()
[[ ${SKIP_COMPLIANCE} -eq 0 ]] && POST_STEPS+=("TEST04")
if [[ ${SKIP_VBENCH} -eq 0 && "${BACKEND}" != "mock" ]]; then
    POST_STEPS+=("VBench")
fi
if [[ ${#POST_STEPS[@]} -gt 0 ]]; then
    echo "[run_all] matrix + ${POST_STEPS[*]} finished under ${ROOT}/"
else
    echo "[run_all] all four combinations finished under ${ROOT}/ (post-steps skipped)"
fi
exit ${FAILED}
