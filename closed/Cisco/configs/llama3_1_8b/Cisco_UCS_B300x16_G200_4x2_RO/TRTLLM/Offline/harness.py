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

# Llama3.1-8B Offline IFB, B300 x16.
harness_config = {
    llm_fields.llm_gen_config_path: "src/nv_mlpinf/benchmarks/llama3_1_8b/generation_config.json",
    harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / "llama3.1-8b/",
    loadgen_fields.min_duration: MIN_DURATION_MS,
    # Selected full-run offered load.
    loadgen_fields.offline_expected_qps: 2800,
    # mlperf.conf: llama3_1-8b.Offline.min_query_count
    loadgen_fields.min_query_count: OFFLINE_MIN_SAMPLE_COUNT["llama3_1-8b"],
    # Selected concurrency for sustained throughput.
    harness_fields.max_concurrency: 8192,
    # Selected worker count.
    harness_fields.workers_per_core: 16,
    llm_fields.traffic_distribution_policy: "isl_load_balancing",
    model_fields.precision: "fp4",
    model_fields.input_dtype: "int32",
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: False,
}


_workers_override = os.environ.get("MLPINF_WORKERS_PER_CORE")
if _workers_override:
    harness_config = {
        **harness_config,
        harness_fields.workers_per_core: int(_workers_override),
    }

if os.environ.get("MLPINF_LOADGEN_MODE", "full").lower() == "dev":
    _dev_qps = float(os.environ.get("MLPINF_DEV_QPS", "16"))
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
        loadgen_fields.min_query_count: OFFLINE_MIN_SAMPLE_COUNT["llama3_1-8b"],
    },
}

COMPLIANCE_OVERRIDES = TEST06_COMPLIANCE_OVERRIDES

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
    },
}
