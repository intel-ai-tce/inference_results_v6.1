import os

from nv_mlpinf.common.mlcommons.mlperf_loadgen_defaults import (
    GPT_OSS_ACCURACY_SAMPLE_COUNT,
    MIN_DURATION_MS,
    OFFLINE_MIN_SAMPLE_COUNT,
)

import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.common.paths as paths

# gpt-oss-120b Offline, B300 x16 IFB (16 TP1 replicas, one per GPU).
# The harness uses the completions HTTP path.
os.environ["MLPINF_HTTP_USE_COMPLETIONS"] = "1"

harness_config = {
    llm_fields.llm_gen_config_path: "src/nv_mlpinf/benchmarks/gpt_oss_120b/generation_config_performance.json",
    harness_fields.tensor_path: paths.DATA_DIR / "gpt-oss/v4/perf",
    loadgen_fields.min_duration: MIN_DURATION_MS,
    # mlperf.conf: gpt-oss-120b.*.performance_sample_count_override (no Offline.min_query_count entry)
    loadgen_fields.min_query_count: OFFLINE_MIN_SAMPLE_COUNT["gpt-oss-120b"],
    loadgen_fields.offline_expected_qps: 162,
    llm_fields.use_token_latencies: True,
    model_fields.precision: "fp4",
    model_fields.input_dtype: "int32",
    llm_fields.server_instance_size: 1,
    harness_fields.workers_per_core: 16,
}

# --------------------------------------------------------------------------- #
# Dev / iteration loadgen profile (NOT a submission run).
# Activated by MLPINF_LOADGEN_MODE=dev (wired via the sflow LOADGEN_MODE variable).
# Short duration-bound run for plumbing + throughput probing; env-overridable:
#     --set LOADGEN_MODE=dev  (then optionally) MLPINF_DEV_QPS=200 MLPINF_DEV_MIN_DURATION_MS=180000
# The full config above stays submission-valid and untouched.
# --------------------------------------------------------------------------- #
if os.environ.get("MLPINF_LOADGEN_MODE", "full").lower() == "dev":
    _dev_qps = float(os.environ.get("MLPINF_DEV_QPS", "162"))
    _dev_min_duration_ms = int(os.environ.get("MLPINF_DEV_MIN_DURATION_MS", "180000"))
    _dev_sample_count = int(os.environ.get("MLPINF_DEV_SAMPLE_COUNT", "2048"))
    harness_config = {
        **harness_config,
        loadgen_fields.offline_expected_qps: _dev_qps,
        loadgen_fields.min_duration: _dev_min_duration_ms,
        loadgen_fields.min_query_count: 1,  # duration-bound, not query-count-bound
        loadgen_fields.performance_sample_count_override: _dev_sample_count,
    }

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
            loadgen_fields.performance_sample_count: 990,
            loadgen_fields.performance_sample_count_override: 990,
            loadgen_fields.accuracy_sample_count_override: 990,
            loadgen_fields.min_query_count: 990,
        },
    },
    C.AuditTest.TEST09: {
        C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
            loadgen_fields.min_query_count: OFFLINE_MIN_SAMPLE_COUNT["gpt-oss-120b"],
            loadgen_fields.min_duration: 0,
        },
    },
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
    },
}
