import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.common.paths as paths
import os

os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1'
os.environ['TRTLLM_WORKER_DISABLE_GC'] = '1'
os.environ['TRTLLM_ENABLE_PDL'] = '1'

harness_config = {
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'src/nv_mlpinf/benchmarks/llama2_70b/generation_config.json',
    loadgen_fields.min_duration: 600000,
    loadgen_fields.server_target_qps: 108,
    model_fields.precision: 'fp4',
    harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / 'llama2-70b/',
    llm_fields.harness_use_hf_tokenizer: False,
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    llm_fields.use_token_latencies: True,
    harness_fields.enable_metrics: False,
    llm_fields.server_instance_size: 1,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): harness_config,
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): harness_config,
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
    },
}
