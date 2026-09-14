import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.common.paths as paths

harness_config = {
    llm_fields.llm_gen_config_path: paths.PROJECT_BASE_DIR / 'src/nv_mlpinf/benchmarks/llama2_70b/generation_config.json',
    harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / 'llama2-70b/',
    loadgen_fields.min_duration: 1200000,
    loadgen_fields.server_target_qps: 327.6,
    harness_fields.max_concurrency: 4320,
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',

    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: False,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): harness_config,
}
