import os

from nv_mlpinf.common.mlcommons.compliance_overrides import TEST06_COMPLIANCE_OVERRIDES
from nv_mlpinf.common.mlcommons.mlperf_loadgen_defaults import (
    MIN_DURATION_MS,
    OFFLINE_MIN_SAMPLE_COUNT,
)

import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.common.paths as paths

# Llama2-70B Offline IFB, B300 x64 (64 DP ranks x TP1, 8 nodes).
harness_config = {
    llm_fields.llm_gen_config_path: 'src/nv_mlpinf/benchmarks/llama2_70b/generation_config.json',
    harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / 'llama2-70b/',
    loadgen_fields.min_duration: MIN_DURATION_MS,
    loadgen_fields.offline_expected_qps: 3600,
    # mlperf.conf: llama2-70b.Offline.min_query_count
    loadgen_fields.min_query_count: OFFLINE_MIN_SAMPLE_COUNT["llama2-70b"],
    harness_fields.max_concurrency: 34560,
    # Selected harness-side HTTP fan-out.
    harness_fields.workers_per_core: 4,
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',

    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: False,
}

# --------------------------------------------------------------------------- #
# Dev / iteration loadgen profile (NOT a submission run).
# Activated by MLPINF_LOADGEN_MODE=dev (via the sflow LOADGEN_MODE variable).
# Offline is throughput-bound, so the dev knob is a short run; override
# offline_expected_qps (NOT server_target_qps) via MLPINF_DEV_QPS for sizing.
# --------------------------------------------------------------------------- #
if os.environ.get("MLPINF_LOADGEN_MODE", "full").lower() == "dev":
    _dev_qps = float(os.environ.get("MLPINF_DEV_QPS", "3600"))
    _dev_min_duration_ms = int(os.environ.get("MLPINF_DEV_MIN_DURATION_MS", "180000"))
    _dev_sample_count = int(os.environ.get("MLPINF_DEV_SAMPLE_COUNT", "2048"))
    harness_config = {
        **harness_config,
        loadgen_fields.offline_expected_qps: _dev_qps,
        loadgen_fields.min_duration: _dev_min_duration_ms,
        loadgen_fields.min_query_count: 1,
        loadgen_fields.performance_sample_count_override: _dev_sample_count,
    }

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): harness_config,
}

WORKLOAD = C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP)
ACCURACY_OVERRIDES = {
    WORKLOAD: {
        loadgen_fields.min_query_count: OFFLINE_MIN_SAMPLE_COUNT["llama2-70b"],
    },
}

COMPLIANCE_OVERRIDES = TEST06_COMPLIANCE_OVERRIDES

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
    },
}
