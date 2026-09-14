"""Shared MLPerf compliance LoadGen overrides for Cisco harness configs.

Mirrors the MLCommons compliance audit.config files so audit runs stay short
(100 samples for TEST06) even when LoadGen cannot read audit.config from the
process working directory.

Used by all x16 and x64 Offline/Server cells for llama2, llama3.1, and
deepseek-r1 (TEST06). gpt-oss keeps TEST07/TEST09 overrides in its harness.py.
"""

from __future__ import annotations

import nv_mlpinf.common.constants as C
import nv_mlpinf.fields.loadgen as loadgen_fields

_SUBMISSION_WORKLOAD = C.WorkloadSetting(
    C.HarnessType.Custom,
    C.AccuracyTarget(0.99),
    C.PowerSetting.MaxP,
)

# MLPerf compliance/TEST06/audit.config equivalents.
TEST06_LOADGEN_OVERRIDES = {
    loadgen_fields.min_query_count: 100,
    loadgen_fields.min_duration: 0,
    loadgen_fields.performance_sample_count_override: 100,
    loadgen_fields.accuracy_sample_count_override: 100,
}

TEST06_COMPLIANCE_OVERRIDES = {
    C.AuditTest.TEST06: {
        _SUBMISSION_WORKLOAD: dict(TEST06_LOADGEN_OVERRIDES),
    },
}
