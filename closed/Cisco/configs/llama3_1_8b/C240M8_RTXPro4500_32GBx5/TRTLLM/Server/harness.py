import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.llmlib.fields as llm_fields


harness_config = {
    llm_fields.llm_gen_config_path: 'src/nv_mlpinf/benchmarks/llama3_1_8b/generation_config.json',
    harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / 'llama3.1-8b/',
    loadgen_fields.min_duration: 600000,

    # Required Server target for this four-GPU system.
    loadgen_fields.server_target_qps: 88.5,

    # Keep harness concurrency aligned with the qualified endpoint batch capacity.
    harness_fields.max_concurrency: 2272,
    harness_fields.workers_per_core: 2,
    llm_fields.server_instance_size: 1,
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    llm_fields.readiness_timeout: 1800,

    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: False,
}


EXPORTS = {
    C.WorkloadSetting(
        C.HarnessType.Custom,
        C.AccuracyTarget(0.99),
        C.PowerSetting.MaxP,
    ): harness_config,
}
