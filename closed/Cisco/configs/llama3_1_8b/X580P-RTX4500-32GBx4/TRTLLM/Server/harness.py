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

    loadgen_fields.server_target_qps: 71.9,
    # System-wide cap: 192 in-flight requests per one-GPU endpoint.
    harness_fields.max_concurrency: 768,
    # TP=1/PP=1 in the server YAML makes this one endpoint per GPU (four total).
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
