import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.common.paths as paths
import os

os.environ["MLPINF_HTTP_USE_COMPLETIONS"] = "1"

base = {
    llm_fields.llm_gen_config_path: paths.PROJECT_BASE_DIR / 'src/nv_mlpinf/benchmarks/gpt_oss_120b/generation_config_performance.json',
    harness_fields.tensor_path: paths.DATA_DIR / 'gpt-oss/v4/perf',
    loadgen_fields.min_duration: 600000 * 2,
    loadgen_fields.min_query_count: 51168,
    loadgen_fields.server_target_qps: 63.0,
    llm_fields.warmup_iterations: 0,
    llm_fields.use_token_latencies: True,
    harness_fields.max_concurrency: 512 + 1024,
    llm_fields.server_instance_size: 1,
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    harness_fields.enable_metrics: False,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base,
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": base,
    },
}

ACCURACY_OVERRIDES = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        llm_fields.llm_gen_config_path: paths.PROJECT_BASE_DIR / 'src/nv_mlpinf/benchmarks/gpt_oss_120b/generation_config_accuracy.json',
        harness_fields.tensor_path: paths.DATA_DIR / 'gpt-oss/v4/acc',
        loadgen_fields.min_query_count: 4395,
    },
}

# Compliance test overrides (keyed by AuditTest)
COMPLIANCE_OVERRIDES = {
    C.AuditTest.TEST07: {
        C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
            llm_fields.llm_gen_config_path: paths.PROJECT_BASE_DIR / 'src/nv_mlpinf/benchmarks/gpt_oss_120b/generation_config_performance.json',
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
