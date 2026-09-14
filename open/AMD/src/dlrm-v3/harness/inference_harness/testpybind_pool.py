"""Python fallback for TestPybind.QuerySamplesCompletePool (ROCm bring-up)."""
from __future__ import annotations

from typing import List

import mlperf_loadgen as lg


class QuerySamplesCompletePool:
    def __init__(self, num_threads: int = 10, test_mode: bool = False) -> None:
        self.num_threads = num_threads
        self.test_mode = test_mode

    def enqueue_batch(
        self, query_ids: List[int], base_ptr: int, bytes_per_query: int
    ) -> None:
        for i, query_id in enumerate(query_ids):
            addr = base_ptr + i * bytes_per_query
            lg.QuerySamplesComplete(
                [lg.QuerySampleResponse(query_id, addr, bytes_per_query)]
            )
