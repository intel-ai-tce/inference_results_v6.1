"""LoadGen fields derived from MLCommons mlperf-inference (source of truth).

Values are taken from:
  closed/Cisco/3rdparty/mlc-inference/mlperf.conf
  closed/Cisco/3rdparty/mlc-inference/tools/submission/submission_checker/constants.py

Re-read those files when bumping the mlc-inference submodule for a new MLPerf round.
"""

from __future__ import annotations

# mlperf.conf: *.Server.min_duration / *.Offline.min_duration
MIN_DURATION_MS = 600_000

# mlperf.conf: <model>.Offline.min_query_count (accuracy / perf-sample pool for most models)
OFFLINE_MIN_SAMPLE_COUNT = {
    "llama2-70b": 24_576,
    "llama3_1-8b": 13_368,
    "deepseek-r1": 4_388,
    # gpt-oss-120b has no Offline.min_query_count entry; use performance_sample_count_override
    "gpt-oss-120b": 6_396,
}

# Full Offline coalesced-query size for PerformanceOnly (NV submission reference).
# DeepSeek: mlperf.conf Offline.min_query_count (4388) is the perf pool only;
# the submission FULL query is REQUIRED_OFFLINE_QUERIES = 105312 (24 × 4388).
OFFLINE_REQUIRED_QUERY_COUNT = {
    "deepseek-r1": 105_312,
}

# submission_checker/constants.py v6.1 min_queries[<benchmark>]["Server"]
SERVER_MIN_QUERY_COUNT = 270_336

# mlperf.conf: gpt-oss-120b.*.accuracy_sample_count_override
GPT_OSS_ACCURACY_SAMPLE_COUNT = 4_395
