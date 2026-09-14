import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.common.paths as paths

harness_config = {
    llm_fields.llm_gen_config_path: paths.PROJECT_BASE_DIR / 'src/nv_mlpinf/benchmarks/llama3_1_8b/generation_config.json',
    harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / 'llama3.1-8b/',
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 13368,
    loadgen_fields.offline_expected_qps: 420,
    llm_fields.warmup_iterations: 0,
    llm_fields.use_token_latencies: True,
    harness_fields.max_concurrency: 1280, #3456, #4320, #2304, #1280,

    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: False,
    llm_fields.server_instance_size: 1,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): harness_config,
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
    },
}
