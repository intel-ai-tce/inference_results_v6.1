import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields

import os
import nv_mlpinf.common.paths as paths
os.environ['TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER'] = '1'
os.environ['MLPINF_HTTP_USE_COMPLETIONS'] = '1'
os.environ['TLLM_PROFILE_START_STOP'] = '12000-12100'

# Base config (PerformanceOnly)
harness_config = {
    llm_fields.llm_gen_config_path: 'src/nv_mlpinf/benchmarks/gpt_oss_120b/generation_config_performance.json',
    harness_fields.tensor_path: paths.DATA_DIR / 'gpt-oss/v4/perf',

    loadgen_fields.min_duration: 1800000,
    loadgen_fields.min_query_count: 25584,
    llm_fields.warmup_iterations: 0,
    llm_fields.use_token_latencies: True,

    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32', 
    loadgen_fields.offline_expected_qps: 53,

    harness_fields.enable_metrics: False,
}


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): harness_config,
}

# Accuracy-specific overrides (deep merged when test_mode=AccuracyOnly)
ACCURACY_OVERRIDES = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        llm_fields.llm_gen_config_path: 'src/nv_mlpinf/benchmarks/gpt_oss_120b/generation_config_accuracy.json',
        harness_fields.tensor_path: paths.DATA_DIR / 'gpt-oss/v4/acc',
        loadgen_fields.min_query_count: 4395,
    },
}

# Compliance test overrides (keyed by AuditTest)
COMPLIANCE_OVERRIDES = {
    C.AuditTest.TEST07: {
        C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
            llm_fields.llm_gen_config_path: 'src/nv_mlpinf/benchmarks/gpt_oss_120b/generation_config_performance.json',
            harness_fields.tensor_path: paths.DATA_DIR / 'gpt-oss/v4/compliance/test07',
            loadgen_fields.performance_sample_count: 990,
            loadgen_fields.performance_sample_count_override: 990,
            loadgen_fields.accuracy_sample_count_override: 990,
            loadgen_fields.min_query_count: 990,
        },
    },
    C.AuditTest.TEST09: {
        C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
            loadgen_fields.min_query_count: 6396,
            loadgen_fields.min_duration: 0,
        },
    },
}
