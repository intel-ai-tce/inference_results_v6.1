# Copyright 2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# =============================================================================

"""Stage 1: parse + chunk HTML into passage batches."""

import logging
import os
import queue as pyqueue
import time
from multiprocessing import Queue

from ingestion.read_docs import HTMLExtractor
from ingestion.text_splitter import split_into_fixed_passages
from common.utils import load_url_mapping, get_base_filename
from sut.pipeline_config import PipelineConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Stage1Parse")


def stage1_parse_chunk_worker(
    worker_id: int,
    config: PipelineConfig,
    input_queue: Queue,
    output_queue: Queue,
    timing_queue: Queue,
    chunk_size: int,
    chunk_overlap: int,
    text_boundary: str,
    ready_queue: Queue,
    done_queue: Queue,
):
    html_extractor = HTMLExtractor(
        preserve_tables=True, preserve_lists=True, text_boundary=text_boundary,
    )
    _ = split_into_fixed_passages(
        "warmup " * 256, fixed_length=chunk_size, overlap=chunk_overlap,
    )

    log.info(f"[Stage1-{worker_id}] started")
    ready_queue.put(('stage1', worker_id))

    pass_batch = []

    # Cache the frozen-corpus URL mapping (filename -> canonical Wikipedia URL)
    # per source directory. Passages must record original_url so the DB manifest
    # gate can compare canonical URLs across systems; without it they only carry
    # the filename-form source and cross-system verification fails.
    url_mapping_cache = {}

    def _url_mapping_for(file_path):
        doc_dir = os.path.dirname(file_path)
        if doc_dir not in url_mapping_cache:
            url_mapping_cache[doc_dir] = load_url_mapping(doc_dir)
        return url_mapping_cache[doc_dir]

    def _flush():
        nonlocal pass_batch
        if pass_batch:
            output_queue.put(pass_batch)
            pass_batch = []

    while True:
        try:
            doc_info = input_queue.get(timeout=0.5)
        except pyqueue.Empty:
            continue

        if doc_info is None:
            break

        sample_id, query_id, file_path, file_name = doc_info
        stage_start = time.time()

        try:
            parse_start = time.time()
            text = html_extractor.extract_text(file_path)
            parse_time = time.time() - parse_start
            if not text or len(text.strip()) == 0:
                text = f"Document: {file_name}"

            chunk_start = time.time()
            passages = split_into_fixed_passages(
                text, fixed_length=chunk_size, overlap=chunk_overlap)
            chunk_time = time.time() - chunk_start
            if not passages:
                passages = [text]

            original_url = _url_mapping_for(file_path).get(
                get_base_filename(file_name), "")
            total = len(passages)
            for idx, ptext in enumerate(passages):
                pass_batch.append({
                    'kind': 'passage',
                    'doc_id': sample_id,
                    'query_id': query_id,
                    'file_name': file_name,
                    'passage_idx': idx,
                    'total_passages': total,
                    'text': ptext,
                    'metadata': {'source': file_name, 'passage_id': idx,
                                 'original_url': original_url},
                })
                if len(pass_batch) >= config.embed_queue_passages:
                    _flush()

            if timing_queue is not None:
                timing_queue.put(('stage1', 'parse_html', parse_time,
                                  {'file_name': file_name, 'text_len': len(text)}))
                timing_queue.put(('stage1', 'chunk_text', chunk_time,
                                  {'file_name': file_name, 'num_passages': total}))
                timing_queue.put(('stage1', 'total_per_doc',
                                  time.time() - stage_start,
                                  {'file_name': file_name, 'num_passages': total}))

        except Exception as e:
            log.error(f"[Stage1-{worker_id}] Error on {file_name}: {e}")
            _flush()
            output_queue.put([{
                'kind': 'doc_header', 'doc_id': sample_id, 'query_id': query_id,
                'file_name': file_name, 'total_passages': 0, 'error': str(e),
            }])

    _flush()
    done_queue.put(('stage1_done', worker_id))
    log.info(f"[Stage1-{worker_id}] shutting down")
