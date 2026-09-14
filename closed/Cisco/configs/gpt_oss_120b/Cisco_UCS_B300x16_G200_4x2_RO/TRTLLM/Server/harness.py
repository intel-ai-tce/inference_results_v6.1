import os

import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.llmlib.fields as llm_fields


os.environ["MLPINF_HTTP_USE_COMPLETIONS"] = "1"
harness_config = {
    llm_fields.llm_gen_config_path: (
        paths.PROJECT_BASE_DIR
        / "configs/gpt_oss_120b/Cisco_UCS_B300x16_G200_4x2_RO/TRTLLM/Server/"
        "generation_config_performance.json"
    ),
    harness_fields.tensor_path: paths.DATA_DIR / "gpt-oss/v4/perf",
    llm_fields.use_token_latencies: True,
    model_fields.precision: "fp4",
    model_fields.input_dtype: "int32",
    llm_fields.server_instance_size: 1,
    harness_fields.workers_per_core: 8,
    loadgen_fields.server_target_qps: 170,
    loadgen_fields.performance_sample_count_override: 6396,
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 25584,
}

WORKLOAD = C.WorkloadSetting(
    C.HarnessType.Custom,
    C.AccuracyTarget(0.99),
    C.PowerSetting.MaxP,
)
EXPORTS = {WORKLOAD: harness_config}
COMPLIANCE_OVERRIDES = {
    C.AuditTest.TEST07: {
        WORKLOAD: {
            llm_fields.llm_gen_config_path: (
                paths.PROJECT_BASE_DIR
                / "configs/gpt_oss_120b/Cisco_UCS_B300x16_G200_4x2_RO/TRTLLM/Server/"
                "generation_config_performance.json"
            ),
            harness_fields.tensor_path: paths.DATA_DIR
            / "gpt-oss/v4/compliance/test07",
            loadgen_fields.min_query_count: 990,
        },
    },
    C.AuditTest.TEST09: {
        WORKLOAD: {
            loadgen_fields.min_query_count: 6396,
            loadgen_fields.min_duration: 0,
        },
    },
}
ATOMIC_EXPORTS = {WORKLOAD: {"default": harness_config}}
