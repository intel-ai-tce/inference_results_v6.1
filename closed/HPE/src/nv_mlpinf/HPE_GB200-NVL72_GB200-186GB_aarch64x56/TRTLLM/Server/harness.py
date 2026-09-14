import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.llmlib.fields as llm_fields


base = {
    llm_fields.llm_gen_config_path: "src/nv_mlpinf/benchmarks/deepseek_r1/generation_config.json",
    harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / "deepseek-r1/",
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 26328 * 8,
    llm_fields.warmup_iterations: 0,
    llm_fields.use_token_latencies: True,
    llm_fields.traffic_distribution_policy: "isl_load_balancing",
    harness_fields.max_concurrency: 5120,
    model_fields.precision: "fp4",
    model_fields.input_dtype: "int32",
    loadgen_fields.server_target_qps: 73.5,
    harness_fields.enable_metrics: False,
}


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base,
}
