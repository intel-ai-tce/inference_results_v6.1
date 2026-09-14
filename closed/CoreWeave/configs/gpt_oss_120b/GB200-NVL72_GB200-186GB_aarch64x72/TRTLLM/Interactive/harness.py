import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields

import os
import nv_mlpinf.common.paths as paths
os.environ['MLPINF_HTTP_USE_COMPLETIONS'] = '1'


harness_config = {
    llm_fields.llm_gen_config_path: paths.PROJECT_BASE_DIR / 'src/nv_mlpinf/benchmarks/gpt_oss_120b/generation_config_performance.json',
    harness_fields.tensor_path: paths.DATA_DIR / 'gpt-oss/v4/perf',
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 6396,
    loadgen_fields.server_target_qps: 480,
    llm_fields.warmup_iterations: 5,
    llm_fields.use_token_latencies: True,
    harness_fields.max_concurrency: 256 + 1024,
    llm_fields.server_instance_size: 1,
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    harness_fields.enable_metrics: False,
}

# Dynamo disaggregated endpoint config
# 24 CTX TP1 + 12 GEN TP4EP4 = 24 + 48 = 72 GPUs
# Worker TRT-LLM configs:
# - configs/GB200-NVL72_GB200-186GB_aarch64x1/Interactive/disagg_prefill/gpt-oss-120b.yml (CTX, TP1)
# - configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_decode/gpt-oss-120b.yml (GEN, TP4EP4)
dynamo_cluster = {
    **harness_config,
    llm_fields.dynamo_cluster: {
        'num_prefill_workers': 24,
        'num_decode_workers': 12,
        'num_frontends': 24,
        'gpus_per_node': 4,
        'frontend': {
            'router_mode': 'kv',
            'kv_overlap_weight': 0,
            'distribute_frontends': True,
        },
        'prefill': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x1',
            'trtllm_yml_override': '/work/configs/GB200-NVL72_GB200-186GB_aarch64x1/Interactive/disagg_prefill/gpt-oss-120b.yml',
            'env_vars': 'configs/GB200-NVL72_GB200-186GB_aarch64x1/Interactive/disagg_prefill/gptoss_env.yaml',
        },
        'decode': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x4',
            'trtllm_yml_override': '/work/configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_decode/gpt-oss-120b.yml',
            'env_vars': 'configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_decode/gptoss_env.yaml',
        },
    },
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): harness_config,
}

# Accuracy-specific overrides (deep merged when test_mode=AccuracyOnly)
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

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
        "dynamo_cluster": dynamo_cluster,
    },
}
