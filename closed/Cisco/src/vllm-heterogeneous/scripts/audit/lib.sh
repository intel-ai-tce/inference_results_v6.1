#!/usr/bin/env bash

set -o pipefail

audit_die() {
    printf 'AUDIT_ERROR: %s\n' "$*" >&2
    exit 2
}

audit_require_file() {
    [[ -s "$1" ]] || audit_die "required file is missing or empty: $1"
}

audit_valid_summary() {
    grep -Fq 'Result is : VALID' "$1"
}

audit_validate_scenario() {
    case "$1" in
        offline|server|interactive) ;;
        *) audit_die "scenario must be offline, server, or interactive" ;;
    esac
}

audit_mlperf_scenario() {
    case "$1" in
        offline) printf 'Offline' ;;
        server) printf 'Server' ;;
        interactive) printf 'Interactive' ;;
    esac
}

audit_validate_name() {
    [[ "$1" =~ ^[A-Za-z0-9._/-]+$ && "$1" != *..* ]] || audit_die "unsafe model name: $1"
}

audit_validate_tag() {
    [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || audit_die "tag may contain only letters, digits, dot, underscore, and dash"
}

audit_load_deployment() {
    AUDIT_PACKAGE="$1"
    local deployment="${DEPLOYMENT_ENV:-${AUDIT_PACKAGE}/config/deployment.env}"
    [[ -f "$deployment" ]] || audit_die "deployment environment file not found: $deployment"
    set -a
    source "$deployment"
    set +a
    [[ -n "${WORK_DIR:-}" ]] || audit_die 'set WORK_DIR in config/deployment.env'
    [[ -n "${MODEL_ROOT:-}" ]] || audit_die 'set MODEL_ROOT in config/deployment.env'
    [[ -n "${DATA_ROOT:-}" ]] || audit_die 'set DATA_ROOT in config/deployment.env'
    [[ -n "${MLPERF_INFERENCE_DIR:-}" ]] || audit_die 'set MLPERF_INFERENCE_DIR in config/deployment.env'
    AUDIT_COMPLIANCE_DIR="${MLPERF_INFERENCE_DIR}/compliance"
    [[ -d "$AUDIT_COMPLIANCE_DIR" ]] || audit_die "MLPerf compliance directory not found: $AUDIT_COMPLIANCE_DIR"
}

audit_setup_root() {
    local test="$1" model="$2" scenario="$3" tag="$4"
    AUDIT_ROOT="${WORK_DIR}/results/audits/${test}/${model//\//_}/${scenario}/${tag}"
    [[ ! -e "$AUDIT_ROOT" ]] || audit_die "refusing to overwrite existing audit capture: $AUDIT_ROOT"
    AUDIT_RAW="$AUDIT_ROOT/raw"
    AUDIT_EVIDENCE="$AUDIT_ROOT/evidence"
    AUDIT_VERIFY="$AUDIT_ROOT/verification"
    mkdir -p "$AUDIT_RAW" "$AUDIT_EVIDENCE/accuracy" \
        "$AUDIT_EVIDENCE/performance/run_1" "$AUDIT_VERIFY" "$AUDIT_ROOT/config"
    local config="${AUDIT_PACKAGE}/config/model/${model}.yaml"
    audit_require_file "$config"
    install -m 0644 "$config" "$AUDIT_ROOT/config/$(basename "$config")"
    AUDIT_CONFIG="$config"
}

audit_capture_required_logs() {
    local raw="$1"
    audit_require_file "$raw/mlperf_log_accuracy.json"
    audit_require_file "$raw/mlperf_log_detail.txt"
    audit_require_file "$raw/mlperf_log_summary.txt"
    install -m 0644 "$raw/mlperf_log_accuracy.json" "$AUDIT_EVIDENCE/accuracy/mlperf_log_accuracy.json"
    install -m 0644 "$raw/mlperf_log_detail.txt" "$AUDIT_EVIDENCE/performance/run_1/mlperf_log_detail.txt"
    install -m 0644 "$raw/mlperf_log_summary.txt" "$AUDIT_EVIDENCE/performance/run_1/mlperf_log_summary.txt"
}

audit_find_config() {
    local test="$1" canonical_model="$2"
    local model_config="${AUDIT_COMPLIANCE_DIR}/${test}/${canonical_model}/audit.config"
    local generic_config="${AUDIT_COMPLIANCE_DIR}/${test}/audit.config"
    if [[ -f "$model_config" ]]; then
        printf '%s' "$model_config"
    elif [[ -f "$generic_config" ]]; then
        printf '%s' "$generic_config"
    else
        audit_die "official ${test} audit.config not found beneath ${AUDIT_COMPLIANCE_DIR}"
    fi
}

audit_check_gpt_sampling() {
    local config="$1" profile="${2:-}"
    python3 - "$config" "$profile" <<'PYTHON'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
profile = sys.argv[2]
profiles = config.get('profiles') or {}
candidates = []
if profile and profiles.get(profile) is not None:
    candidates.append(profiles[profile])
candidates.append(config)
candidates.extend(profiles.values())
selected = None
for candidate in candidates:
    overrides = candidate.get('sampling_mode_overrides') or {}
    performance = overrides.get('performance') or candidate.get('vllm_sampling_config') or {}
    harness = candidate.get('harness_config') or {}
    if performance and harness:
        selected = (performance, harness)
        break
if selected is None:
    raise SystemExit('no selected GPT-OSS profile defines sampling and harness settings')
performance, harness = selected
if int(performance.get('min_tokens', -1)) != 1:
    raise SystemExit('performance min_tokens must be 1 for the GPT-OSS audit')
if int(performance.get('max_tokens', -1)) != 10000:
    raise SystemExit('performance max_tokens must be 10000 for the GPT-OSS audit')
if harness.get('strip_output_special_tokens', None) is not False:
    raise SystemExit('strip_output_special_tokens must be false for the GPT-OSS audit')

def decode_limits(value):
    if hasattr(value, 'items'):
        for key, child in value.items():
            if key == 'decode_max_tokens':
                yield child
            yield from decode_limits(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from decode_limits(child)

limits = list(decode_limits(config))
if limits and any(int(limit) != 10000 for limit in limits):
    raise SystemExit('decode_max_tokens must be 10000 when configured for the GPT-OSS audit')
PYTHON
}

audit_write_manifest() {
    local test="$1" model="$2" scenario="$3" backend="$4" hardware="$5" profile="$6" qps="$7"
    {
        printf 'test=%s\nmodel=%s\nscenario=%s\nbackend=%s\nhardware=%s\nprofile=%s\ntarget_qps=%s\n' \
            "$test" "$model" "$scenario" "$backend" "$hardware" "$profile" "$qps"
        printf 'config=%s\nconfig_sha256=%s\n' "$AUDIT_CONFIG" "$(sha256sum "$AUDIT_CONFIG" | awk '{print $1}')"
    } > "$AUDIT_ROOT/manifest.txt"
}

audit_stage_config() {
    local config="$1"
    install -m 0644 "$config" "$AUDIT_ROOT/audit.config"
}

audit_run_harness() {
    local model="$1" scenario="$2" backend="$3" hardware="$4" profile="$5" qps="$6"
    AUDIT_WORKSPACE="$(mktemp -d "${TMPDIR:-/tmp}/mlperf-audit.XXXXXX")"
    install -m 0644 "$AUDIT_ROOT/audit.config" "$AUDIT_WORKSPACE/audit.config"
    local args=("${AUDIT_PACKAGE}/run.sh" "$model" "$scenario" performance --backend "$backend")
    [[ -n "$hardware" ]] && args+=(--hardware "$hardware")
    [[ -n "$profile" ]] && args+=(--profile "$profile")
    args+=(-- "harness_config.target_qps=$qps")
    shift 6
    args+=("$@")
    if (
        cd "$AUDIT_WORKSPACE"
        MLPERF_OUTPUT_DIR="$AUDIT_RAW" "${args[@]}"
    ) >"$AUDIT_ROOT/driver.log" 2>&1; then
        AUDIT_DRIVER_RC=0
    else
        AUDIT_DRIVER_RC=$?
    fi
    rm -f -- "$AUDIT_WORKSPACE/audit.config"
    rmdir -- "$AUDIT_WORKSPACE" 2>/dev/null || true
    audit_capture_required_logs "$AUDIT_RAW"
    grep -Fq 'Found Audit Config file (audit.config).' "$AUDIT_RAW/mlperf_log_detail.txt" || \
        audit_die 'LoadGen did not acknowledge the staged audit.config'
    if grep -Eq 'Multiple conf files are used|error_invalid_config' "$AUDIT_RAW/mlperf_log_detail.txt"; then
        audit_die 'LoadGen reported an invalid or multiple audit configuration'
    fi
}
