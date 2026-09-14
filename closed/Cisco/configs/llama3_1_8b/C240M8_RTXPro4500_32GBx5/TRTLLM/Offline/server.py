import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.llmlib.fields as llm_fields


ifb_config = {
    llm_fields.trtllm_yml_override: paths.PROJECT_BASE_DIR / (
        'configs/llama3_1_8b/C240M8-RTX4500-32GBx5/'
        'TRTLLM/Offline/trtllm-serve-ifb-1gpu.yaml'
    ),
    llm_fields.env_yml_override: paths.PROJECT_BASE_DIR / (
        'configs/llama3_1_8b/C240M8-RTX4500-32GBx5/'
        'TRTLLM/Offline/trtllm-serve-ifb-1gpu-env.yaml'
    ),
    model_fields.precision: 'fp4',
}


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): ifb_config,
}
