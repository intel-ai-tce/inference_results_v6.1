#!/bin/bash
# Run an MLPerf inference benchmark.
#
# Backends (only two active; legacy *_zmq and HTTP variants kept under
# src/sut/*_zmq.py and src/sut/pd_http_legacy.py for reference only):
#   standalone  - N independent engines per node, full prefill+decode
#   pd          - N prefill engines + M decode engines, KV over NIXL
#
# Workers must already be running before run.sh is invoked. Start them
# with ./start_server.sh --role <prefill|decode|standalone> ...
#
# Usage:
#   ./run.sh <model> <scenario> <mode> [--backend <backend>] [--hardware <hw>] [-- extra_overrides...]
#
# Examples:
#   ./run.sh gptoss_120b server performance --backend standalone --hardware mi350x
#   ./run.sh gptoss_120b server performance --backend pd --hardware mi350x
#   ./run.sh llama2_70b  offline performance --backend pd

set -eu

cleanup() { jobs -p | xargs -r kill -9 2>/dev/null; exit 130; }
trap cleanup INT TERM

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-${SCRIPT_DIR}/config/deployment.env}"
if [ ! -f "${DEPLOYMENT_ENV}" ]; then
    echo "ERROR: deployment environment file not found: ${DEPLOYMENT_ENV}" >&2
    exit 1
fi
set -a
source "${DEPLOYMENT_ENV}"
set +a
: "${WORK_DIR:?Set WORK_DIR in config/deployment.env}"
: "${MODEL_ROOT:?Set MODEL_ROOT in config/deployment.env}"
: "${DATA_ROOT:?Set DATA_ROOT in config/deployment.env}"
if [ -n "${MLPERF_LOADGEN_PYTHONPATH:-}" ]; then
    export PYTHONPATH="${MLPERF_LOADGEN_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
elif [ -d "${SCRIPT_DIR}/artifacts/loadgen" ]; then
    export PYTHONPATH="${SCRIPT_DIR}/artifacts/loadgen${PYTHONPATH:+:${PYTHONPATH}}"
fi

# --- Logging ---
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
if [ -z "${_LOGGING_WRAPPED:-}" ]; then
    LOGFILE="${LOG_DIR}/run_${1:-unknown}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to ${LOGFILE}"
    export _LOGGING_WRAPPED=1
    exec > >(tee -a "${LOGFILE}") 2>&1
fi

if [ $# -lt 3 ]; then
    cat <<EOF
Usage: $0 <model> <scenario> <mode> [options] [-- extra_overrides...]

Arguments:
  model      Config name under config/model/ (e.g. llama2_70b, gptoss_120b)
  scenario   offline | server | interactive
  mode       performance | accuracy

Options:
  --backend   standalone | pd            (default: from model config)
  --hardware  h100 | h200 | mi350x       (applies per-hardware overrides)
  --profile   named complete driver profile (for example dp)

Workers must already be running. Start them with:
  ./start_server.sh --role <prefill|decode|standalone> --hardware <hw> --model <name>
EOF
    exit 1
fi

MODEL="$1"; SCENARIO="$2"; MODE="$3"
shift 3

BACKEND_OVERRIDE=""
HARDWARE_OVERRIDE=""
PROFILE_OVERRIDE=""
while [ $# -gt 0 ]; do
    case "${1:-}" in
        --backend)  BACKEND_OVERRIDE="$2";  shift 2 ;;
        --hardware) HARDWARE_OVERRIDE="$2"; shift 2 ;;
        --profile)  PROFILE_OVERRIDE="$2";  shift 2 ;;
        --)         shift; break ;;
        *)          break ;;
    esac
done

MODEL_CONFIG="${SCRIPT_DIR}/config/model"
MODEL_FILE="${MODEL_CONFIG}/${MODEL}.yaml"

if [ ! -f "${MODEL_FILE}" ]; then
    echo "ERROR: Model config not found: ${MODEL_FILE}"
    echo "Available models:"
    ls "${MODEL_CONFIG}"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml
    exit 1
fi

OUTPUT_DIR="${MLPERF_OUTPUT_DIR:-${SCRIPT_DIR}/results/${MODEL}/${SCENARIO}/${MODE}}"

# --- Determine backend ---
if [ -n "${BACKEND_OVERRIDE}" ]; then
    BACKEND="${BACKEND_OVERRIDE}"
else
    BACKEND="$(python3 -c "import yaml; print(yaml.safe_load(open('${MODEL_FILE}')).get('backend', 'pd'))")"
fi
if [ -z "${PROFILE_OVERRIDE}" ] && [ "${BACKEND}" = "pd" ]; then
    PROFILE_OVERRIDE="$(python3 - "${MODEL_FILE}" <<'PYTHON'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
profiles = config.get("profiles") or {}
print("pd" if isinstance(profiles, dict) and "pd" in profiles else "")
PYTHON
)"
fi

# --- Servers parsing helper ---
source "${SCRIPT_DIR}/scripts/lib/servers.sh"

print_banner_common() {
    echo "=========================================="
    echo " Model:    ${MODEL}"
    echo " Scenario: ${SCENARIO}"
    echo " Mode:     ${MODE}"
    echo " Backend:  ${BACKEND}"
}

case "${BACKEND}" in
    standalone)
        parse_servers "${MODEL_FILE}" standalone STANDALONE_ADDR STANDALONE_N "${HARDWARE_OVERRIDE}" "${SCENARIO}" "${PROFILE_OVERRIDE}"
        if [ -z "${STANDALONE_ADDR}" ]; then
            echo "ERROR: servers.standalone is empty in ${MODEL_FILE}" >&2
            echo "       Add N endpoints (one per engine) and re-run." >&2
            exit 1
        fi
        echo "=========================================="
        echo " AMD-style Multi-Engine Standalone Benchmark"
        print_banner_common
        echo " Workers:  ${STANDALONE_N} engines @ ${STANDALONE_ADDR}"
        echo " Output:   ${OUTPUT_DIR}"
        echo "=========================================="
        echo "NOTE: Workers must already be running. Start them with:"
        echo "        ./start_server.sh --role standalone \\"
        echo "            --hardware ${HARDWARE_OVERRIDE:-<hw>} --model ${MODEL}"
        echo "=========================================="
        ;;

    pd)
        parse_servers "${MODEL_FILE}" prefill PREFILL_ADDR PREFILL_N "${HARDWARE_OVERRIDE}" "${SCENARIO}" "${PROFILE_OVERRIDE}"
        parse_servers "${MODEL_FILE}" decode  DECODE_ADDR  DECODE_N "${HARDWARE_OVERRIDE}" "${SCENARIO}" "${PROFILE_OVERRIDE}"
        if [ -z "${PREFILL_ADDR}" ] || [ -z "${DECODE_ADDR}" ]; then
            echo "ERROR: servers.prefill and servers.decode must be set in ${MODEL_FILE}" >&2
            exit 1
        fi
        echo "=========================================="
        echo " PD Disaggregated Inference Benchmark"
        print_banner_common
        echo " Prefill:  ${PREFILL_N} engines @ ${PREFILL_ADDR}"
        echo " Decode:   ${DECODE_N} engines @ ${DECODE_ADDR}"
        echo " Output:   ${OUTPUT_DIR}"
        echo "=========================================="
        echo "NOTE: Workers must already be running. Start them with:"
        echo "        ./start_server.sh --role prefill --hardware <hw> --model ${MODEL}   (on prefill node)"
        echo "        ./start_server.sh --role decode  --hardware <hw> --model ${MODEL}   (on decode node)"
        echo "=========================================="
        ;;

    pd_zmq)
        echo "Using retained single-engine PD ZMQ relay backend."
        ;;

    *)
        echo "ERROR: backend '${BACKEND}' is deprecated." >&2
        echo "       Use --backend standalone or --backend pd." >&2
        echo "       Legacy backends (pd_zmq, standalone_zmq, http pd) live under" >&2
        echo "       src/sut/*_zmq.py / src/sut/pd_http_legacy.py for reference only." >&2
        exit 1
        ;;
esac

# --- Launch the harness ---
HARNESS_ARGS=(
    --config-path "${MODEL_CONFIG}"
    --config-name "${MODEL}"
    "scenario=${SCENARIO}"
    "test_mode=${MODE}"
    "harness_config.output_log_dir=${OUTPUT_DIR}"
    "backend=${BACKEND}"
)
[ -n "${HARDWARE_OVERRIDE}" ] && HARNESS_ARGS+=("hardware=${HARDWARE_OVERRIDE}")
[ -n "${PROFILE_OVERRIDE}" ] && HARNESS_ARGS+=("profile=${PROFILE_OVERRIDE}")

# --- Select the LoadGen user.conf by submission system ---
case "${BACKEND}" in
    pd|pd_zmq) SYS_CONF="mlperf_pd_h200_mi350x.conf" ;;
    standalone)
        case "${HARDWARE_OVERRIDE}" in
            mi350x) SYS_CONF="mlperf_mi350x.conf" ;;
            *)      SYS_CONF="mlperf_h200.conf" ;;
        esac ;;
    *) SYS_CONF="" ;;
esac
if [ -n "${SYS_CONF}" ] && [ -f "${SCRIPT_DIR}/mlperf/${SYS_CONF}" ]; then
    HARNESS_ARGS+=("harness_config.user_conf_path=${SCRIPT_DIR}/mlperf/${SYS_CONF}")
fi

MODE_SAMPLING_ARGS=()
MODEL_BASENAME="${MODEL##*/}"
if [[ "${MODEL_BASENAME}" == gptoss_120b* ]]; then
    MODE_SAMPLING_VALUES="$(
        python3 -c '
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)
profile_name = sys.argv[3]
scenario = sys.argv[4].lower()
if profile_name:
    config = (config.get("profiles") or {}).get(profile_name, {})
    if isinstance(config.get(scenario), dict):
        config = config[scenario]
profile = config.get("sampling_mode_overrides", {}).get(sys.argv[2])
if profile is not None:
    print("{} {}".format(profile["min_tokens"], profile["max_tokens"]))
' "${MODEL_FILE}" "${MODE}" "${PROFILE_OVERRIDE}" "${SCENARIO}"
    )"
    read -r MODE_MIN_TOKENS MODE_MAX_TOKENS <<< "${MODE_SAMPLING_VALUES}"
    if ! [[ "${MODE_MIN_TOKENS}" =~ ^[0-9]+$ ]] || ! [[ "${MODE_MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: missing valid sampling_mode_overrides.${MODE} in ${MODEL_FILE}" >&2
        exit 1
    fi
    MODE_SAMPLING_ARGS=(
        "vllm_sampling_config.min_tokens=${MODE_MIN_TOKENS}"
        "vllm_sampling_config.max_tokens=${MODE_MAX_TOKENS}"
    )
    echo "GPT-OSS sampling: enforcing ${MODE} max_tokens=${MODE_MAX_TOKENS}; caller token limits are overridden."
fi

"${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/src/main.py" "${HARNESS_ARGS[@]}" "$@" "${MODE_SAMPLING_ARGS[@]}" &
wait $!
