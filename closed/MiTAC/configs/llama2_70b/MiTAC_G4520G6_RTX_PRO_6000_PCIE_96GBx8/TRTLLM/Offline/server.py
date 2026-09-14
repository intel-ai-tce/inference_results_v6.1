import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.llmlib.fields as llm_fields

# Used in single-node benchmark with docker environment only

ifb_config = {
    llm_fields.trtllm_yml_override: paths.PROJECT_BASE_DIR / 'configs/llama2_70b/MiTAC_G4520G6_RTX_PRO_6000_PCIE_96GBx8/TRTLLM/Offline/trtllm-serve-ifb-8gpu.yaml',
    llm_fields.env_yml_override: paths.PROJECT_BASE_DIR / 'configs/llama2_70b/MiTAC_G4520G6_RTX_PRO_6000_PCIE_96GBx8/TRTLLM/Offline/trtllm-serve-ifb-8gpu-env.yaml',
    llm_fields.server_instance_size: 1,
    model_fields.precision: 'fp4',
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): ifb_config,
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): ifb_config,
}
