import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.llmlib.fields as llm_fields


harness_config = {
    llm_fields.llm_gen_config_path:
        'src/nv_mlpinf/benchmarks/llama3_1_8b/generation_config.json',

    harness_fields.tensor_path:
        paths.PREPROCESSED_DATA_DIR / 'llama3.1-8b/',

    # Interactive reference configuration uses a 20-minute run.
    loadgen_fields.min_duration: 1200000,

    # Conservative starting point for 5x RTX PRO 4500.
    # Tune only after checking TTFT <= 500 ms and TPOT <= 30 ms.
    loadgen_fields.server_target_qps: 65.6,

    # This field is per endpoint. With four TP1 endpoints, the system-wide
    # upper bound is 4 * 128 = 512 concurrent requests.
    harness_fields.max_concurrency: 64,

    # TP1 means one independent TRT-LLM endpoint per GPU.
    llm_fields.server_instance_size: 1,

    # Distribute queries according to current in-flight sequence load.
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',

    # Allow sufficient time for four server processes to initialize and
    # compile/capture the configured CUDA graphs.
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
