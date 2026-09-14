#!/usr/bin/env bash
# Path-sanitized launcher for GPT-OSS-120B x64 TRTLLM-serve IFB Offline.

#SBATCH --job-name=gptoss_x64_offline_ifb
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mem=0
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --time=06:00:00

set -euo pipefail

: "${MLPERF_CISCO_ROOT:?Set to the closed/Cisco checkout}"
: "${MLPERF_SFLOW_BIN:?Set to the sflow executable}"
: "${MLPERF_CONTAINER_IMAGE:?Set to the pinned MLPerf container image}"
: "${MLPERF_DATA_DIR:?Set to the MLPerf data/model root}"
: "${MLPERF_SLURM_ACCOUNT:?Set to the target Slurm account}"
: "${MLPERF_SLURM_PARTITION:?Set to the target Slurm partition}"
: "${MLPERF_OUTPUT_DIR:?Set to a writable benchmark output directory}"
: "${MLPERF_SERVER_HOSTS:?Set one comma-separated host per IFB replica}"

collected_container_sha256=5112cac5674c92ac1e6e7db69d0fc9e8882bd5211d1d97636229facf153f30dc
collected_oci_digest=sha256:54c5995ded61b0b5640c641ca79248b14422d0720ef8b293aba8f5fab2e5243d
if [[ -f "${MLPERF_CONTAINER_IMAGE}" ]]; then
  actual_container_sha256=$(sha256sum "${MLPERF_CONTAINER_IMAGE}" | awk '{print $1}')
  [[ "${actual_container_sha256}" == "${collected_container_sha256}" ]] || {
    echo "Container SHA256 mismatch: ${actual_container_sha256}" >&2
    exit 2
  }
elif [[ "${MLPERF_CONTAINER_IMAGE}" != *"@${collected_oci_digest}"* ]]; then
  echo "Container must be the collected squashfs or exact OCI digest ${collected_oci_digest}" >&2
  exit 2
fi

config="${MLPERF_CISCO_ROOT}/configs/gpt_oss_120b/Cisco_UCS_B300x64_G200_4x2_RO/TRTLLM/Offline"
template="${MLPERF_CISCO_ROOT}/src/nv_mlpinf/scaleout/templates/trtllm_ifb_portable.yaml"
output_dir="${MLPERF_OUTPUT_DIR}"
workspace="${output_dir}/${SLURM_JOB_ID:-manual}-workspace"
test_mode="${MLPERF_TEST_MODE:-PerformanceOnly}"

case "${test_mode}" in
  PerformanceOnly)
    harness_mode=PerformanceOnly
    harness_extra_args='--offline_expected_qps=635 --accuracy_target=0.99'
    ;;
  AccuracyOnly)
    harness_mode=AccuracyOnly
    harness_extra_args='--offline_expected_qps=635 --accuracy_target=0.99'
    ;;
  TEST07)
    harness_mode=PerformanceOnly
    harness_extra_args='--offline_expected_qps=635 --accuracy_target=0.99 --audit_test=TEST07'
    ;;
  TEST09)
    harness_mode=PerformanceOnly
    harness_extra_args='--offline_expected_qps=635 --accuracy_target=0.99 --audit_test=TEST09'
    ;;
  *)
    echo "Unsupported MLPERF_TEST_MODE=${test_mode}" >&2
    exit 2
    ;;
esac

mkdir -p "${output_dir}" "${workspace}"
cd "${MLPERF_CISCO_ROOT}"

"${MLPERF_SFLOW_BIN}" run \
  --file "${template}" \
  --file "${config}/gptoss_ifb_config_sflow.yaml" \
  --file "${MLPERF_CISCO_ROOT}/src/nv_mlpinf/scaleout/templates/slurm_env_sflow.example.yaml" \
  --set WORK_DIR="${MLPERF_CISCO_ROOT}" \
  --set SCRATCH_DIR="${MLPERF_DATA_DIR}" \
  --set CONTAINER_IMAGE="${MLPERF_CONTAINER_IMAGE}" \
  --set SLURM_ACCOUNT="${MLPERF_SLURM_ACCOUNT}" \
  --set SLURM_PARTITION="${MLPERF_SLURM_PARTITION}" \
  --set "SERVER_HOSTS=${MLPERF_SERVER_HOSTS}" \
  --set "HTTP_INTERFACE=${MLPERF_HTTP_INTERFACE:-}" \
  --set "BOOTSTRAP_INTERFACE=${MLPERF_BOOTSTRAP_INTERFACE:-}" \
  --set DP_MULTIPLICITY=64 \
  --set GPUS_PER_DP_RANK=1 \
  --set TOTAL_GPUS=64 \
  --set SYSTEM_NAME=B300-SXM-270GBx64 \
  --set TEST_MODE="${harness_mode}" \
  --set LOADGEN_MODE=full \
  --set "HARNESS_EXTRA_ARGS=${harness_extra_args}" \
  --workspace-dir "${workspace}" \
  --output-dir "${output_dir}"
