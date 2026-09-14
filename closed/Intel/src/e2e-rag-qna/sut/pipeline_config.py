# Copyright 2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# =============================================================================

"""Hardcoded knobs for the pipelined ingestion SUT."""


class PipelineConfig:
    num_parse_workers = 32
    embed_concurrency = 64
    embed_batch_size = 32
    embed_queue_passages = 16
    index_batch_size = 512
    queue1_maxsize = 2000  # parse -> embed
    queue2_maxsize = 2000  # embed -> index

    def summary(self) -> str:
        return (
            f"{self.num_parse_workers} parse + 1 embed(HTTP, concurrency="
            f"{self.embed_concurrency}, batch={self.embed_batch_size}) + 1 index"
        )
