import os

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

# DeepSeek-R1 Offline IFB, B300 x16 (2x DEP8). QPS scaled from NV x8 ref (19).
harness_config = {
    llm_fields.llm_gen_config_path: "src/nv_mlpinf/benchmarks/deepseek_r1/generation_config.json",
    harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / "deepseek-r1/",
    loadgen_fields.min_duration: MIN_DURATION_MS,
    loadgen_fields.offline_expected_qps: 38,
    # mlperf.conf: deepseek-r1.Offline.min_query_count
    loadgen_fields.min_query_count: OFFLINE_MIN_SAMPLE_COUNT["deepseek-r1"],
    llm_fields.warmup_iterations: 0,
    llm_fields.use_token_latencies: True,
    harness_fields.max_concurrency: 10240,
    model_fields.precision: "fp4",
    model_fields.input_dtype: "int32",
}


if os.environ.get("MLPINF_LOADGEN_MODE", "full").lower() == "dev":
    _dev_qps = float(os.environ.get("MLPINF_DEV_QPS", "38"))
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
        loadgen_fields.min_query_count: OFFLINE_MIN_SAMPLE_COUNT["deepseek-r1"],
    },
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
    },
}
