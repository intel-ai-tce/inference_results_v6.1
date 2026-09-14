#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

usage() {
    cat <<'USAGE'
Usage:
  scripts/audit/run_test06.sh --model <config> --scenario <offline|server|interactive> \
    --accuracy-dir <AccuracyOnly-result-dir> --performance-dir <performance-result-dir> --tag <tag>

Captures matching existing AccuracyOnly and performance logs, runs the official
TEST06 verifier, and writes a self-contained audit capture below results/audits/.
It does not start a service or modify a benchmark configuration.
USAGE
}

model='' scenario='' accuracy_dir='' performance_dir='' tag=''
while (($#)); do
    case "$1" in
        --model) model="${2:?--model requires a value}"; shift 2 ;;
        --scenario) scenario="${2:?--scenario requires a value}"; shift 2 ;;
        --accuracy-dir) accuracy_dir="${2:?--accuracy-dir requires a value}"; shift 2 ;;
        --performance-dir) performance_dir="${2:?--performance-dir requires a value}"; shift 2 ;;
        --tag) tag="${2:?--tag requires a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) audit_die "unknown option: $1" ;;
    esac
done
[[ -n "$model" && -n "$scenario" && -n "$accuracy_dir" && -n "$performance_dir" && -n "$tag" ]] || { usage >&2; exit 2; }
audit_validate_name "$model"
audit_validate_scenario "$scenario"
audit_validate_tag "$tag"
audit_load_deployment "$(cd "$SCRIPT_DIR/../.." && pwd)"
audit_setup_root TEST06 "$model" "$scenario" "$tag"

for source in \
    "$accuracy_dir/mlperf_log_accuracy.json" \
    "$accuracy_dir/mlperf_log_detail.txt" \
    "$accuracy_dir/mlperf_log_summary.txt" \
    "$performance_dir/mlperf_log_detail.txt" \
    "$performance_dir/mlperf_log_summary.txt"; do
    audit_require_file "$source"
done
audit_valid_summary "$accuracy_dir/mlperf_log_summary.txt" || audit_die 'AccuracyOnly source summary is not VALID'
audit_valid_summary "$performance_dir/mlperf_log_summary.txt" || audit_die 'performance source summary is not VALID'

install -m 0644 "$accuracy_dir/mlperf_log_accuracy.json" "$AUDIT_RAW/mlperf_log_accuracy.json"
install -m 0644 "$accuracy_dir/mlperf_log_detail.txt" "$AUDIT_RAW/mlperf_log_detail.txt"
install -m 0644 "$accuracy_dir/mlperf_log_summary.txt" "$AUDIT_RAW/mlperf_log_summary.txt"
mkdir -p "$AUDIT_ROOT/performance/run_1"
install -m 0644 "$performance_dir/mlperf_log_detail.txt" "$AUDIT_ROOT/performance/run_1/mlperf_log_detail.txt"
install -m 0644 "$performance_dir/mlperf_log_summary.txt" "$AUDIT_ROOT/performance/run_1/mlperf_log_summary.txt"
install -m 0644 "$AUDIT_RAW/mlperf_log_accuracy.json" "$AUDIT_EVIDENCE/accuracy/mlperf_log_accuracy.json"
install -m 0644 "$AUDIT_ROOT/performance/run_1/mlperf_log_detail.txt" "$AUDIT_EVIDENCE/performance/run_1/mlperf_log_detail.txt"
install -m 0644 "$AUDIT_ROOT/performance/run_1/mlperf_log_summary.txt" "$AUDIT_EVIDENCE/performance/run_1/mlperf_log_summary.txt"
audit_write_manifest TEST06 "$model" "$scenario" existing existing '' ''

official="${AUDIT_COMPLIANCE_DIR}/TEST06/run_verification.py"
audit_require_file "$official"
mlperf_scenario="$(audit_mlperf_scenario "$scenario")"
if python3 "$official" -c "$AUDIT_RAW" -o "$AUDIT_VERIFY" -s "$mlperf_scenario" >"$AUDIT_ROOT/verify_test06.log" 2>&1; then
    verify_rc=0
else
    verify_rc=$?
fi
verify_log="$(find "$AUDIT_VERIFY" -type f -name verify_accuracy.txt -print -quit)"
audit_require_file "$verify_log"
install -m 0644 "$verify_log" "$AUDIT_EVIDENCE/verify_accuracy.txt"
if (( verify_rc == 0 )) && grep -Fq 'TEST06 verification complete' "$verify_log" && ! grep -Fq 'check pass: False' "$verify_log"; then
    verdict=PASS
else
    verdict=FAIL
fi
printf 'verifier_exit_code=%s\nverdict=%s\n' "$verify_rc" "$verdict" >> "$AUDIT_ROOT/manifest.txt"
printf 'TEST06_RESULT: %s\nTEST06_EVIDENCE: %s\n' "$verdict" "$AUDIT_EVIDENCE"
[[ "$verdict" == PASS ]]
