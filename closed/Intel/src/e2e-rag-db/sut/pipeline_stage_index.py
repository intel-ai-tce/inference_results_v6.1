# Copyright 2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# =============================================================================

"""Stage 3: Buffer per-doc passages, then batch-add to FAISS + save DB."""

import hashlib
import logging
import os
import queue as pyqueue
import time
from multiprocessing import Queue

from engine import VectorDB
from sut.pipeline_config import PipelineConfig

_MD5_CHUNK = 1 << 20

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Stage3Index")


def _add_passages_to_index(passages, rag_db: VectorDB, timing_queue: Queue):
    if not passages:
        return
    start = time.time()
    texts, embeddings, metadatas = (list(c) for c in zip(*passages))
    rag_db._vector_store.add_embeddings(
        text_embeddings=list(zip(texts, embeddings)),
        metadatas=metadatas,
    )
    if timing_queue is not None:
        timing_queue.put(('stage3', 'faiss_add_batch', time.time() - start,
                          {'num_passages': len(passages)}))


def stage3_index_worker(
    config: PipelineConfig,
    input_queue: Queue,
    timing_queue: Queue,
    results_queue: Queue,
    total_documents: int,
    last_sample_id: int,
    database: str,
    embedding_model: str,
    reranker_model: str,
    vector_index_method: str,
    ready_queue: Queue,
):
    """VectorDB is built inside this process — its live object holds
    unpicklable weakrefs so 'spawn' can't ship it from the parent."""
    log.info("[Stage3] Index worker started")

    rag_db = VectorDB(
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        device='cpu',
        database=database,
        num_embedding_devices=1,
        benchmark=False,
        vector_index_method=vector_index_method,
    )
    log.info("[Stage3] Vector database initialized")
    ready_queue.put(('stage3', os.getpid()))

    # HTTP embed may complete batches out of order; buffer per doc_id and
    # only index a document once every passage_idx has arrived.
    doc_buffers = {}
    completed_docs = 0
    pending_passages = []
    pending_responses = []            # (query_id, response_bytes) awaiting flush
    last_doc_query_id = None          # deferred until MD5 is ready

    def _flush_pending(force=False):
        nonlocal pending_passages, pending_responses
        if pending_passages and (force or len(pending_passages) >= config.index_batch_size):
            _add_passages_to_index(pending_passages, rag_db, timing_queue)
            pending_passages = []
        for query_id, response_bytes in pending_responses:
            results_queue.put((query_id, response_bytes))
        pending_responses = []

    def _finalize_doc(buf):
        nonlocal completed_docs, last_doc_query_id
        items = sorted(buf['items'], key=lambda it: it['passage_idx'])
        for it in items:
            pending_passages.append((it['text'], it['embedding'], it['metadata']))
        success = 1 if items else 0
        if buf['doc_id'] == last_sample_id:
            # Held back — must carry MD5 in its response, sent after DB save.
            last_doc_query_id = buf['query_id']
        else:
            pending_responses.append((buf['query_id'], bytes([success])))
        completed_docs += 1
        _flush_pending()

    def _consume(batch):
        for it in batch:
            did = it['doc_id']
            if it.get('kind') == 'doc_header':
                _finalize_doc({
                    'doc_id': did, 'file_name': it['file_name'],
                    'query_id': it['query_id'], 'items': [], 'total': 0,
                })
                continue
            buf = doc_buffers.get(did)
            if buf is None:
                buf = doc_buffers[did] = {
                    'doc_id': did, 'file_name': it['file_name'],
                    'query_id': it['query_id'],
                    'total': it['total_passages'], 'items': [],
                }
            buf['items'].append(it)
            if len(buf['items']) >= buf['total']:
                _finalize_doc(buf)
                del doc_buffers[did]

    while True:
        try:
            batch = input_queue.get(timeout=0.5)
        except pyqueue.Empty:
            continue

        if batch is None:
            break

        _consume(batch)

    for did, buf in list(doc_buffers.items()):
        log.warning(f"[Stage3] Doc {did} ({buf['file_name']}) incomplete: "
                    f"{len(buf['items'])}/{buf['total']} passages")
        _finalize_doc(buf)
    _flush_pending(force=True)
    log.info(f"[Stage3] Indexed {completed_docs}/{total_documents} docs")

    save_start = time.time()
    db_path = f"{database}.db"
    rag_db.serialize(db_path)
    if timing_queue is not None:
        timing_queue.put(('stage3', 'save_database', time.time() - save_start,
                          {'db_path': db_path}))

    md5_start = time.time()
    # usedforsecurity=False: file-identity checksum, not a security digest
    # (silences bandit B324).
    h = hashlib.md5(usedforsecurity=False)
    with open(db_path, 'rb') as f:
        for chunk in iter(lambda: f.read(_MD5_CHUNK), b""):
            h.update(chunk)
    db_md5 = h.hexdigest()
    if timing_queue is not None:
        timing_queue.put(('stage3', 'compute_md5', time.time() - md5_start,
                          {'md5': db_md5}))
    log.info(f"[Stage3] Database MD5: {db_md5}")

    if last_doc_query_id is not None:
        results_queue.put((last_doc_query_id, db_md5.encode('utf-8')))
    results_queue.put(None)

    log.info("[Stage3] Index worker shutting down")
