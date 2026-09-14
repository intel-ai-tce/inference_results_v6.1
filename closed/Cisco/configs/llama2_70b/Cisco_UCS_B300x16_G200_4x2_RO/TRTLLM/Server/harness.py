import os

from nv_mlpinf.common.mlcommons.compliance_overrides import TEST06_COMPLIANCE_OVERRIDES
from nv_mlpinf.common.mlcommons.mlperf_loadgen_defaults import (
    MIN_DURATION_MS,
    OFFLINE_MIN_SAMPLE_COUNT,
    SERVER_MIN_QUERY_COUNT,
)

import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.common.paths as paths

# Llama2-70B Server, B300 x16 PD-disagg (6 CTX TP1 + 10 GEN TP1).
# Server TTFT/TPOT latency constraints are enforced by LoadGen.
harness_config = {
    llm_fields.llm_gen_config_path: 'src/nv_mlpinf/benchmarks/llama2_70b/generation_config.json',
    harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / 'llama2-70b/',
    # mlperf.conf: *.Server.min_duration
    loadgen_fields.min_duration: MIN_DURATION_MS,
    # submission_checker/constants.py v6.1 min_queries[<benchmark>]["Server"]
    loadgen_fields.min_query_count: SERVER_MIN_QUERY_COUNT,
    loadgen_fields.server_target_qps: 650,
    harness_fields.max_concurrency: 8640,
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',

    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: False,
}

# --------------------------------------------------------------------------- #
# Dev / iteration loadgen profile (NOT a submission run).
#
# Activated by MLPINF_LOADGEN_MODE=dev (wired via the sflow `LOADGEN_MODE`
# variable -> the harness task exports it). Produces a short run for fast
# plumbing + capacity probing, and flows straight into the auto-generated
# user.conf via the same loadgen fields. The default (full) config above stays
# submission-valid and untouched.
#
# QPS / duration / sample-count are additionally overridable via env so capacity
# sweeps need no code edits, e.g.:
#     --set LOADGEN_MODE=dev   (then optionally)
#     MLPINF_DEV_QPS=450 MLPINF_DEV_MIN_DURATION_MS=180000
# --------------------------------------------------------------------------- #
if os.environ.get("MLPINF_LOADGEN_MODE", "full").lower() == "dev":
    _dev_qps = float(os.environ.get("MLPINF_DEV_QPS", "400"))
    _dev_min_duration_ms = int(os.environ.get("MLPINF_DEV_MIN_DURATION_MS", "180000"))
    _dev_sample_count = int(os.environ.get("MLPINF_DEV_SAMPLE_COUNT", "2048"))
    harness_config = {
        **harness_config,
        loadgen_fields.server_target_qps: _dev_qps,
        loadgen_fields.min_duration: _dev_min_duration_ms,
        loadgen_fields.min_query_count: 1,  # duration-bound, not query-count-bound
        loadgen_fields.performance_sample_count_override: _dev_sample_count,
    }

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): harness_config,
}

WORKLOAD = C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP)
ACCURACY_OVERRIDES = {
    WORKLOAD: {
        loadgen_fields.min_query_count: OFFLINE_MIN_SAMPLE_COUNT["llama2-70b"],
    },
}

COMPLIANCE_OVERRIDES = TEST06_COMPLIANCE_OVERRIDES

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
    },
}
