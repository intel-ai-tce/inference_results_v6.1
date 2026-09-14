#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

usage() {
    cat <<'USAGE'
Usage:
  scripts/audit/run_test09.sh --model gptoss_120b --scenario <offline|server|interactive> \
    --target-qps <qps> --tag <tag> [--backend standalone|pd] [--hardware <hardware>] [--profile <profile>]

Runs the official GPT-OSS TEST09 performance audit from a temporary working
directory and records LoadGen logs plus output-length verifier evidence below
results/audits/. Dataset and sampling limits remain those in the selected model
configuration; this script supplies no performance-tuning overrides.
USAGE
}

model='' scenario='' target_qps='' tag='' backend=standalone hardware='' profile=''
while (($#)); do
    case "$1" in
        --model) model="${2:?--model requires a value}"; shift 2 ;;
        --scenario) scenario="${2:?--scenario requires a value}"; shift 2 ;;
        --target-qps) target_qps="${2:?--target-qps requires a value}"; shift 2 ;;
        --tag) tag="${2:?--tag requires a value}"; shift 2 ;;
        --backend) backend="${2:?--backend requires a value}"; shift 2 ;;
        --hardware) hardware="${2:?--hardware requires a value}"; shift 2 ;;
        --profile) profile="${2:?--profile requires a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) audit_die "unknown option: $1" ;;
    esac
done
[[ -n "$model" && -n "$scenario" && -n "$target_qps" && -n "$tag" ]] || { usage >&2; exit 2; }
audit_validate_name "$model"
audit_validate_scenario "$scenario"
audit_validate_tag "$tag"
[[ "$backend" == standalone || "$backend" == pd ]] || audit_die 'backend must be standalone or pd'
[[ "$target_qps" =~ ^[0-9]+([.][0-9]+)?$ ]] || audit_die 'target QPS must be numeric'
audit_load_deployment "$(cd "$SCRIPT_DIR/../.." && pwd)"
audit_setup_root TEST09 "$model" "$scenario" "$tag"
audit_check_gpt_sampling "$AUDIT_CONFIG" "$profile"
audit_config="$(audit_find_config TEST09 gpt-oss-120b)"
audit_stage_config "$audit_config"
audit_write_manifest TEST09 "$model" "$scenario" "$backend" "$hardware" "$profile" "$target_qps"
printf 'audit_config_sha256=%s\n' "$(sha256sum "$audit_config" | awk '{print $1}')" >> "$AUDIT_ROOT/manifest.txt"

audit_run_harness "$model" "$scenario" "$backend" "$hardware" "$profile" "$target_qps"

if (
    cd "$AUDIT_VERIFY"
    python3 "$AUDIT_PACKAGE/scripts/eval/verify_gptoss_compliance.py" \
        --test TEST09 --logs "$AUDIT_RAW" --out "$AUDIT_VERIFY" \
        --model "$model" --canonical-model gpt-oss-120b --compliance-dir "$AUDIT_COMPLIANCE_DIR"
) >"$AUDIT_ROOT/verify_test09.log" 2>&1; then
    verify_rc=0
else
    verify_rc=$?
fi
verify_log="$(find "$AUDIT_VERIFY" -type f -name verify_output_len.txt -print -quit)"
audit_require_file "$verify_log"
install -m 0644 "$verify_log" "$AUDIT_EVIDENCE/verify_output_len.txt"
if (( AUDIT_DRIVER_RC == 0 && verify_rc == 0 )) && audit_valid_summary "$AUDIT_RAW/mlperf_log_summary.txt" && grep -Fq 'TEST PASS' "$verify_log"; then
    verdict=PASS
else
    verdict=FAIL
fi
printf 'driver_exit_code=%s\nverifier_exit_code=%s\nverdict=%s\n' "$AUDIT_DRIVER_RC" "$verify_rc" "$verdict" >> "$AUDIT_ROOT/manifest.txt"
printf 'TEST09_RESULT: %s\nTEST09_EVIDENCE: %s\n' "$verdict" "$AUDIT_EVIDENCE"
[[ "$verdict" == PASS ]]
