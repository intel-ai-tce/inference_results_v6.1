import os

from nv_mlpinf.common.mlcommons.mlperf_loadgen_defaults import (
    GPT_OSS_ACCURACY_SAMPLE_COUNT,
    MIN_DURATION_MS,
    SERVER_MIN_QUERY_COUNT,
)

import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.common.paths as paths

# GPT-OSS Server IFB, B300 x64: 64 independent TP1/EP1 endpoints.
# QPS 350 is the selected full-run target.
os.environ["MLPINF_HTTP_USE_COMPLETIONS"] = "1"
harness_config = {
    llm_fields.llm_gen_config_path: "src/nv_mlpinf/benchmarks/gpt_oss_120b/generation_config_performance.json",
    harness_fields.tensor_path: paths.DATA_DIR / "gpt-oss/v4/perf",
    loadgen_fields.min_duration: MIN_DURATION_MS,
    loadgen_fields.server_target_qps: 350,
    loadgen_fields.min_query_count: SERVER_MIN_QUERY_COUNT,
    llm_fields.use_token_latencies: True,
    model_fields.precision: "fp4",
    model_fields.input_dtype: "int32",
    llm_fields.server_instance_size: 1,
    harness_fields.workers_per_core: 8,
}


_loadgen_mode = os.environ.get("MLPINF_LOADGEN_MODE", "full").lower()
if _loadgen_mode == "dev":
    _dev_qps = float(os.environ.get("MLPINF_DEV_QPS", "350"))
    _dev_min_duration_ms = int(os.environ.get("MLPINF_DEV_MIN_DURATION_MS", "180000"))
    _dev_sample_count = int(os.environ.get("MLPINF_DEV_SAMPLE_COUNT", "6396"))
    harness_config = {
        **harness_config,
        loadgen_fields.server_target_qps: _dev_qps,
        loadgen_fields.min_duration: _dev_min_duration_ms,
        loadgen_fields.min_query_count: 1,
        loadgen_fields.performance_sample_count_override: _dev_sample_count,
    }
elif _loadgen_mode == "full" and "MLPINF_FULL_QPS" in os.environ:
    _full_qps = float(os.environ["MLPINF_FULL_QPS"])
    if _full_qps <= 0:
        raise ValueError("MLPINF_FULL_QPS must be positive")
    harness_config = {
        **harness_config,
        loadgen_fields.server_target_qps: _full_qps,
    }
elif _loadgen_mode not in {"full", "dev"}:
    raise ValueError(f"Unsupported MLPINF_LOADGEN_MODE: {_loadgen_mode}")


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): harness_config,
}

ACCURACY_OVERRIDES = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        llm_fields.llm_gen_config_path: "src/nv_mlpinf/benchmarks/gpt_oss_120b/generation_config_accuracy.json",
        harness_fields.tensor_path: paths.DATA_DIR / "gpt-oss/v4/acc",
        loadgen_fields.min_query_count: GPT_OSS_ACCURACY_SAMPLE_COUNT,
    },
}

COMPLIANCE_OVERRIDES = {
    C.AuditTest.TEST07: {
        C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
            llm_fields.llm_gen_config_path: "src/nv_mlpinf/benchmarks/gpt_oss_120b/generation_config_performance.json",
            harness_fields.tensor_path: paths.DATA_DIR / "gpt-oss/v4/compliance/test07",
            loadgen_fields.min_query_count: 990,
        },
    },
    C.AuditTest.TEST09: {
        C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
            loadgen_fields.min_query_count: SERVER_MIN_QUERY_COUNT,
            loadgen_fields.min_duration: 0,
        },
    },
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
    },
}
