import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.llmlib.fields as llm_fields

# Used in single-node benchmark with docker environment only

ifb_config = {
    llm_fields.trtllm_yml_override: paths.PROJECT_BASE_DIR / 'configs/deepseek_r1/B300-SXM-270GBx8/TRTLLM/Server/trtllm-serve-ifb-dep8.yaml',
    llm_fields.env_yml_override: paths.PROJECT_BASE_DIR / 'configs/deepseek_r1/B300-SXM-270GBx8/TRTLLM/Server/trtllm-serve-ifb-dep8-env.yaml',
    model_fields.precision: 'fp4',
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): ifb_config,
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": ifb_config,
    },
}
