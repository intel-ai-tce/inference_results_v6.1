import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.llmlib.fields as llm_fields


CONFIG_ROOT = (
    paths.PROJECT_BASE_DIR
    / "configs/gpt_oss_120b/Cisco_UCS_B300x16_G200_4x2_RO/TRTLLM/Server"
)
ifb_config = {
    llm_fields.trtllm_yml_override: CONFIG_ROOT / "trtllm-serve-ifb-offline-baseline.yaml",
    llm_fields.env_yml_override: CONFIG_ROOT
    / "trtllm-serve-ifb-offline-baseline-env.yaml",
    model_fields.precision: "fp4",
}
WORKLOAD = C.WorkloadSetting(
    C.HarnessType.Custom,
    C.AccuracyTarget(0.99),
    C.PowerSetting.MaxP,
)
EXPORTS = {WORKLOAD: ifb_config}
ATOMIC_EXPORTS = {WORKLOAD: {"default": ifb_config}}
