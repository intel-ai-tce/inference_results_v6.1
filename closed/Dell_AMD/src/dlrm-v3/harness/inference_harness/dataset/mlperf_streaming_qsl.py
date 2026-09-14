"""
MLPerf streaming Query Sample Library (QSL) for DLRM inference.

Implements dataset loading, batching, and query sampling for MLPerf LoadGen
benchmarking with support for streaming data, GPU-accelerated batching,
and multi-process data loading for high throughput inference.
"""

from __future__ import annotations

# NOTE: Samples import is deferred to after @torch.jit.script functions
# to avoid "could not get source code" error during JIT compilation
from generative_recommenders.dlrm_v3.datasets.dataset import Dataset
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from generative_recommenders.dlrm_v3.datasets.utils import (
    json_loads,
    maybe_truncate_seq,
)
from dataclasses import dataclass
import copy

from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from generative_recommenders.dlrm_v3.datasets.dataset import Samples
import math
import os
import csv
import sys
import pandas as pd
import torch
from inference_harness.dataset.util.timer import timer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import nvtx

# Import worker functions from isolated module (MPI-safe for spawn context)
from inference_harness.dataset.data_loader_worker import (
    load_samples_worker,
    split_combined_kjt,
    ProcessLineConfig,
    LoadItemConfig,
)

from torchrec import KeyedJaggedTensor
import logging
# Increase CSV field size limit to handle large fields
csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)


# Plan 21 §0.4 (OOB id guard): the preprocessed dataset's latest timestamp(s)
# contain item ids up to exactly num_embeddings (e.g. item_id == 1_000_000_000
# for the 1e9-row table), i.e. one past the last valid row. The trained
# checkpoint is also exactly 1e9 rows, so this id is genuinely out-of-vocab and
# must be treated as OOV rather than dereferenced. The NVE LinearUVM lookup does
# not bounds-check, so id == num_embeddings produces an out-of-bounds GPU write
# during cache-fill -> "Memory access fault ... RW: write", crashing any run
# that reaches the offending data (~93% into a single pass, ts_99). Clamping the
# lookup features to [0, num_embeddings - 1] before the H2D transfer removes the
# fault. Gated by DLRM_CLAMP_OOB_IDS (default off) so the change is reversible
# and does not touch the in-bounds CUDA/baseline paths.
#
# NOTE: min-clamp (id -> num_embeddings - 1) is correct for a *performance* run
# (LoadGen does not score outputs). For an *accuracy* run the OOV id should map
# to a dedicated zero row instead; revisit once the perf ceiling is certified.
_CLAMP_OOB_IDS: bool = os.environ.get("DLRM_CLAMP_OOB_IDS", "0") == "1"
_FEATURE_CLAMP_MAX: Dict[str, int] = {
    "item_id": 1_000_000_000 - 1,
    "item_candidate_id": 1_000_000_000 - 1,
    "item_category_id": 128 - 1,
    "item_candidate_category_id": 128 - 1,
    "user_id": 10_000_000 - 1,
}


def _clamp_lookup_features(values: torch.Tensor, keys: List[str], lengths: torch.Tensor) -> None:
    """Clamp each embedding-lookup feature's value-segment to its table bounds.

    ``values`` is the flattened KJT value buffer (concatenated per key, in
    ``keys`` order); ``lengths`` gives the jagged length of each key for this
    sample. Clamps in place; keys without a configured bound are left untouched.
    """
    offset = 0
    for key, length in zip(keys, lengths.tolist()):
        end = offset + int(length)
        cap = _FEATURE_CLAMP_MAX.get(key)
        if cap is not None and end > offset:
            seg = values[offset:end]
            seg.clamp_(max=cap)
        offset = end


class _LazyTimestampSamples:
    """mmap-backed samples; build KJTs on first access (low peak RAM)."""

    __slots__ = (
        "_uih_keys",
        "_candidates_keys",
        "_sample_ids",
        "_id_to_idx",
        "_uih_lengths",
        "_uih_values",
        "_uih_offsets",
        "_cand_lengths",
        "_cand_values",
        "_cand_offsets",
        "_cache",
        "_cache_max",
    )

    def __init__(
        self,
        ts_dir: str,
        uih_keys: List[str],
        candidates_keys: List[str],
        cache_max: int = 64,
    ) -> None:
        import numpy as np

        self._uih_keys = uih_keys
        self._candidates_keys = candidates_keys
        self._sample_ids = np.load(os.path.join(ts_dir, "sample_ids.npy"))
        self._id_to_idx = {int(sid): i for i, sid in enumerate(self._sample_ids)}
        mmap = "r"
        self._uih_lengths = np.load(os.path.join(ts_dir, "uih_lengths.npy"), mmap_mode=mmap)
        self._uih_values = np.load(os.path.join(ts_dir, "uih_values.npy"), mmap_mode=mmap)
        self._uih_offsets = np.load(os.path.join(ts_dir, "uih_offsets.npy"), mmap_mode=mmap)
        self._cand_lengths = np.load(os.path.join(ts_dir, "cand_lengths.npy"), mmap_mode=mmap)
        self._cand_values = np.load(os.path.join(ts_dir, "cand_values.npy"), mmap_mode=mmap)
        self._cand_offsets = np.load(os.path.join(ts_dir, "cand_offsets.npy"), mmap_mode=mmap)
        self._cache: Dict[int, Tuple[KeyedJaggedTensor, KeyedJaggedTensor]] = {}
        self._cache_max = cache_max

    def get(self, sample_id: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        if sample_id in self._cache:
            return self._cache[sample_id]
        i = self._id_to_idx[sample_id]
        uih_lens = torch.from_numpy(self._uih_lengths[i].copy())
        uih_vals = torch.from_numpy(
            self._uih_values[self._uih_offsets[i] : self._uih_offsets[i + 1]].copy()
        )
        cand_lens = torch.from_numpy(self._cand_lengths[i].copy())
        cand_vals = torch.from_numpy(
            self._cand_values[self._cand_offsets[i] : self._cand_offsets[i + 1]].copy()
        )
        if _CLAMP_OOB_IDS:
            _clamp_lookup_features(uih_vals, self._uih_keys, uih_lens)
            _clamp_lookup_features(cand_vals, self._candidates_keys, cand_lens)
        pair = (
            KeyedJaggedTensor(keys=self._uih_keys, lengths=uih_lens, values=uih_vals),
            KeyedJaggedTensor(
                keys=self._candidates_keys, lengths=cand_lens, values=cand_vals
            ),
        )
        if len(self._cache) >= self._cache_max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[sample_id] = pair
        return pair


def collate_fn(
    samples: List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]],
    device: torch.device,
    batching_on_gpu: bool = False,
    buffer_pair: "Optional[KJTBatchBufferPair]" = None,
) -> Samples:
    """
    Collate function for batching KJT samples.

    Args:
        samples: List of (uih_kjt, candidates_kjt) tuples
        device: Target CUDA device
        batching_on_gpu: If True, use ``kjt_batched_func_cuda_upgrade``
            with pre-allocated pinned-CPU + GPU buffers (faster, but each
            call corrupts its source pinned region until the H2D completes —
            so multiple in-flight collates need distinct buffers).
        buffer_pair: Plan 10 §10.1 — per-batch buffer pair from a rotating
            ``KJTBatchBufferPool``. If supplied, used instead of the NV-ref
            module-level ``_uih_buffer`` / ``_candidates_buffer`` globals.
            The pool form is required when ``DLRM_LISTENER_QUEUE_DEPTH >= 2``
            because the listener thread starts batch N+1's CPU writes
            into the pinned region while batch N's H2D may still be
            queued on ``data_stream``. NV's reference does not pipeline
            the listener, so its globals are safe; ours does, so we pool.
    """
    (
        uih_features_kjt_list,
        candidates_features_kjt_list,
    ) = list(zip(*samples))

    if batching_on_gpu:
        if buffer_pair is not None:
            uih_buf = buffer_pair.uih
            cand_buf = buffer_pair.candidates
        else:
            uih_buf = _uih_buffer
            cand_buf = _candidates_buffer
        if uih_buf is not None and cand_buf is not None:
            feature = kjt_batched_func_cuda_upgrade(uih_features_kjt_list, device, uih_buf)
            candidates_feature = kjt_batched_func_cuda_upgrade(candidates_features_kjt_list, device, cand_buf)
        else:
            feature = kjt_batch_func(uih_features_kjt_list)
            candidates_feature = kjt_batch_func(candidates_features_kjt_list)
    else:
        feature = kjt_batch_func(uih_features_kjt_list)
        candidates_feature = kjt_batch_func(candidates_features_kjt_list)
    return Samples(
        uih_features_kjt=feature,
        candidates_features_kjt=candidates_feature,
    )


def kjt_batch_func(
    kjt_list: List[KeyedJaggedTensor],
) -> KeyedJaggedTensor:
    bs_list = [kjt.stride() for kjt in kjt_list]
    bs = sum(bs_list)
    batched_length = torch.cat([kjt.lengths() for kjt in kjt_list], dim=0)
    batched_indices = torch.cat([kjt.values() for kjt in kjt_list], dim=0)
    bs_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(bs_list)
    ).int()
    batched_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(batched_length)
    reorder_length = torch.ops.fbgemm.reorder_batched_ad_lengths(
        batched_length, bs_offset, bs
    )
    reorder_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(reorder_length)
    reorder_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
        batched_offset, batched_indices, reorder_offsets, bs_offset, bs
    )
    out = KeyedJaggedTensor(
        keys=kjt_list[0].keys(),
        lengths=reorder_length.long().pin_memory(),
        values=reorder_indices.long().pin_memory(),
    )
    return out


if not getattr(torch.version, "hip", None):
    kjt_batch_func = torch.jit.script(kjt_batch_func)

# Import Samples AFTER kjt_batch_func definition (TorchScript on CUDA only)
# (JIT tries to get source for all classes in scope, but Samples source isn't accessible)
from generative_recommenders.dlrm_v3.datasets.dataset import Samples  # noqa: E402


def kjt_batch_func_cuda(
    kjt_list: List[KeyedJaggedTensor],
    device: torch.device,
) -> KeyedJaggedTensor:
    bs_list = [1] * len(kjt_list)
    bs = 32

    with nvtx.annotate(f"gather indices on GPU", color="green"):
        cat_indices = [kjt.values() for kjt in kjt_list]

    with nvtx.annotate(f"concat indices tensor on CPU", color="green"):
        batched_indices = torch.cat(cat_indices, dim=0)

    with nvtx.annotate(f"send indices tensor to GPU", color="green"):
        batched_indices = batched_indices.to(device, non_blocking=True)

    with nvtx.annotate(f"gather and concat lengths on cpu", color="green"):
        cat_lengths = [kjt.lengths() for kjt in kjt_list]
        batched_length = torch.cat(cat_lengths, dim=0)
        batched_length = batched_length.to(device, non_blocking=True)

    with nvtx.annotate(f"concat jagged on GPU", color="green"):
        # bs_offset must be on CUDA for GPU kernel
        bs_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.tensor(bs_list, device=device, dtype=torch.int32)
        )
        batched_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(batched_length)
        reorder_length = torch.ops.fbgemm.reorder_batched_ad_lengths(
            batched_length, bs_offset, bs
        )
        reorder_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(reorder_length)
        reorder_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            batched_offset, batched_indices, reorder_offsets, bs_offset, bs
        )
        out = KeyedJaggedTensor(
            keys=kjt_list[0].keys(),
            lengths=reorder_length.long(),
            values=reorder_indices.long(),
        )
    return out


@dataclass
class KJTBatchBuffer:
    """Pre-allocated buffers for kjt_batched_func_cuda_upgrade to avoid repeated allocations."""
    indices_buffer_cpu: torch.Tensor  # Pinned CPU buffer for indices
    lengths_buffer_cpu: torch.Tensor  # Pinned CPU buffer for lengths
    indices_buffer_gpu: torch.Tensor  # GPU buffer for indices
    lengths_buffer_gpu: torch.Tensor  # GPU buffer for lengths

    @staticmethod
    def create(max_indices_size: int, max_lengths_size: int, device: torch.device) -> "KJTBatchBuffer":
        """Create a new buffer with the specified maximum sizes."""
        return KJTBatchBuffer(
            indices_buffer_cpu=torch.empty(max_indices_size, dtype=torch.long, pin_memory=True),
            lengths_buffer_cpu=torch.empty(max_lengths_size, dtype=torch.long, pin_memory=True),
            indices_buffer_gpu=torch.empty(max_indices_size, dtype=torch.long, device=device),
            lengths_buffer_gpu=torch.empty(max_lengths_size, dtype=torch.long, device=device),
        )


@dataclass
class KJTBatchBufferPair:
    """A uih + candidates buffer pair consumed atomically by one ``collate_fn`` call.

    Plan 10 §10.1: ``collate_fn`` calls ``kjt_batched_func_cuda_upgrade``
    once for ``uih`` and once for ``candidates`` in sequence; both writes
    + their async H2Ds belong to the same logical batch.  Bundling them
    keeps the rotation index in lockstep so we never split a batch
    across pool slots.
    """
    uih: KJTBatchBuffer
    candidates: KJTBatchBuffer


_LIVE_POOLS: "List[weakref.ReferenceType]" = []


def _atexit_drop_pools() -> None:
    """Plan 10 §10.1.b — drop all live ``KJTBatchBufferPool`` references
    BEFORE MPI Finalize / CUDA teardown. The 2-buffer-pair pool at b=6
    holds ~96 MB pinned + ~96 MB GPU per dataset instance; if its
    tensor destructors run during Python interpreter shutdown they
    contend with mpi4py's atexit Finalize and openmpi's vader BTL,
    causing a benign-but-ugly SIGSEGV in ``ompi_coll_base_sendrecv_actual``
    after the LG summary has already been written.  Registering this
    handler from inside ``apply_rocm_env`` ordering would be too late —
    the import of ``mlperf_streaming_qsl`` happens after mpi4py is
    already imported, so any atexit we register here runs BEFORE
    mpi4py's Finalize hook (LIFO ordering).
    """
    import atexit  # local import; cheap if already loaded
    for ref in _LIVE_POOLS:
        pool = ref()
        if pool is not None:
            try:
                pool._buffers.clear()  # release pinned + GPU tensors
            except Exception:
                pass
    _LIVE_POOLS.clear()


import atexit as _atexit
import weakref
_atexit.register(_atexit_drop_pools)


class KJTBatchBufferPool:
    """Round-robin pool of ``KJTBatchBufferPair``s for pipe-safe collation.

    Plan 10 §10.1 deviation from NV reference: the NV harness keeps a
    single module-level ``_uih_buffer`` / ``_candidates_buffer`` pair and
    reuses it for every collate.  That is correct ONLY when the listener
    thread cannot start collate(N+1) before collate(N)'s H2D has
    consumed the pinned source region — i.e. listener pipelining depth
    1.  Our harness defaults to ``DLRM_LISTENER_QUEUE_DEPTH=2`` (Plan 05),
    so the listener can start CPU writes for batch N+1 while batch N's
    H2D is still queued on ``data_stream``.  Sharing one buffer between
    those two operations would let CPU overwrites race against the
    in-flight DMA, silently corrupting batch N's KJT.

    The pool size MUST be >= the listener pipeline depth so that
    consecutive collates never alias.  Default is 2 (ping-pong, matching
    ``DLRM_LISTENER_QUEUE_DEPTH=2``); bump via
    ``DLRM_BATCHING_ON_GPU_POOL_SIZE`` if you also bump the listener
    depth.  At b=64 each pair is ~96 MB pinned + ~96 MB GPU, so a
    size-2 pool costs ~384 MB total per rank — well within MI355's
    288 GB HBM headroom (Plan 09 §9.1).
    """

    def __init__(
        self,
        size: int,
        max_indices_size: int,
        max_lengths_size: int,
        device: torch.device,
    ) -> None:
        assert size >= 1, f"pool size must be >= 1, got {size}"
        self._buffers: List[KJTBatchBufferPair] = [
            KJTBatchBufferPair(
                uih=KJTBatchBuffer.create(max_indices_size, max_lengths_size, device),
                candidates=KJTBatchBuffer.create(max_indices_size, max_lengths_size, device),
            )
            for _ in range(size)
        ]
        self._idx = 0
        self._size = size
        torch.cuda.synchronize(device)
        logger.info(
            f"[KJTBatchBufferPool] allocated size={size} "
            f"max_indices={max_indices_size} max_lengths={max_lengths_size} "
            f"device={device}"
        )
        _LIVE_POOLS.append(weakref.ref(self))

    def next(self) -> KJTBatchBufferPair:
        pair = self._buffers[self._idx]
        self._idx = (self._idx + 1) % self._size
        return pair

    @property
    def size(self) -> int:
        return self._size


# Global pre-allocated buffers for collate_fn (NV-reference path; kept for
# back-compat — DLRMv3StreamingMLPerfDataset uses a per-instance pool
# instead, see Plan 10 §10.1).
_uih_buffer: Optional[KJTBatchBuffer] = None
_candidates_buffer: Optional[KJTBatchBuffer] = None


def init_collate_buffers(
    max_indices_size: int,
    max_lengths_size: int,
    device: torch.device,
) -> None:
    """
    Initialize global pre-allocated buffers for collate_fn.
    Must be called before using collate_fn with buffer reuse.

    Args:
        max_indices_size: Maximum number of indices (e.g., batch_size * max_seq_len * num_features)
        max_lengths_size: Maximum number of lengths (e.g., batch_size * num_features)
        device: Target CUDA device
    """
    global _uih_buffer, _candidates_buffer
    _uih_buffer = KJTBatchBuffer.create(max_indices_size, max_lengths_size, device)
    _candidates_buffer = KJTBatchBuffer.create(max_indices_size, max_lengths_size, device)
    # Warmup: touch the buffers to ensure memory is allocated
    torch.cuda.synchronize(device)


def kjt_batched_func_cuda_upgrade(
    kjt_list: List[KeyedJaggedTensor],
    device: torch.device,
    buffer: KJTBatchBuffer,
) -> KeyedJaggedTensor:
    """
    Optimized version of kjt_batch_func_cuda that reuses pre-allocated buffers.

    Args:
        kjt_list: List of KeyedJaggedTensors to batch together
        device: Target CUDA device
        buffer: Pre-allocated buffer (must be large enough, no dynamic expansion)

    Returns:
        Batched KeyedJaggedTensor

    Raises:
        AssertionError: If buffer is too small for the input data
    """
    bs_list = [1] * len(kjt_list)
    bs = len(kjt_list)

    with nvtx.annotate("gather indices and lengths", color="green"):
        cat_indices = [kjt.values() for kjt in kjt_list]
        cat_lengths = [kjt.lengths() for kjt in kjt_list]
    # Calculate total sizes needed
    total_indices_size = sum(t.numel() for t in cat_indices)
    total_lengths_size = sum(t.numel() for t in cat_lengths)    # Assert buffer is large enough (no dynamic expansion)
    assert buffer.indices_buffer_cpu.numel() >= total_indices_size, \
        f"Buffer too small for indices: need {total_indices_size}, have {buffer.indices_buffer_cpu.numel()}"
    assert buffer.lengths_buffer_cpu.numel() >= total_lengths_size, \
        f"Buffer too small for lengths: need {total_lengths_size}, have {buffer.lengths_buffer_cpu.numel()}"

    with nvtx.annotate("copy indices to buffer on CPU", color="green"):
        # Copy indices into the pre-allocated CPU buffer
        offset = 0
        for t in cat_indices:
            size = t.numel()
            buffer.indices_buffer_cpu[offset:offset + size].copy_(t)
            offset += size
        batched_indices_cpu = buffer.indices_buffer_cpu[:total_indices_size]

    with nvtx.annotate("copy lengths to buffer on CPU", color="green"):
        # Copy lengths into the pre-allocated CPU buffer
        offset = 0
        for t in cat_lengths:
            size = t.numel()
            buffer.lengths_buffer_cpu[offset:offset + size].copy_(t)
            offset += size
        batched_length_cpu = buffer.lengths_buffer_cpu[:total_lengths_size]

    with nvtx.annotate("send buffers to GPU", color="green"):
        # Copy from pinned CPU buffer to GPU buffer (async)
        buffer.indices_buffer_gpu[:total_indices_size].copy_(batched_indices_cpu, non_blocking=True)
        buffer.lengths_buffer_gpu[:total_lengths_size].copy_(batched_length_cpu, non_blocking=True)
        batched_indices = buffer.indices_buffer_gpu[:total_indices_size]
        batched_length = buffer.lengths_buffer_gpu[:total_lengths_size]

    with nvtx.annotate("concat jagged on GPU", color="green"):
        # bs_offset must be on CUDA for GPU kernel; ROCm cumsum patch in
        # generative_recommenders/ops/rocm_compat.py routes through
        # torch.cumsum which upcasts int32->int64, so cast back to int32
        # for fbgemm reorder_batched_ad_lengths (Plan 10 §10.1.b smoke
        # surfaced this — kernel rejects Long batch_offsets).  On CUDA
        # fbgemm preserves int32 so the cast is a no-op.
        bs_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.tensor(bs_list, device=device, dtype=torch.int32)
        ).int()
        batched_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(batched_length)
        reorder_length = torch.ops.fbgemm.reorder_batched_ad_lengths(
            batched_length, bs_offset, bs
        )
        reorder_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(reorder_length)
        reorder_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            batched_offset, batched_indices, reorder_offsets, bs_offset, bs
        )
        out = KeyedJaggedTensor(
            keys=kjt_list[0].keys(),
            lengths=reorder_length.long(),
            values=reorder_indices.long(),
        )
    return out


class DLRMv3StreamingMLPerfDataset(Dataset):
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        ratings_file_prefix: str,
        is_inference: bool,
        train_ts: int,
        total_ts: int,
        num_files: int,
        num_users: int,
        num_items: int,
        num_categories: int,
        verbose: int = 0,
        device: torch.device = None,
        batching_on_gpu: bool = False,
        max_buffer_indices: int = 1500000,
        max_buffer_lengths: int = 256,
    ) -> None:
        self._max_num_candidates: int = hstu_config.max_num_candidates
        self._max_num_candidates_inference: int = hstu_config.max_num_candidates_inference
        self._max_seq_len: int = hstu_config.max_seq_len
        self._uih_keys: List[str] = hstu_config.hstu_uih_feature_names
        self._candidates_keys: List[str] = hstu_config.hstu_candidate_feature_names
        self._contextual_feature_to_max_length: Dict[str, int] = (
            hstu_config.contextual_feature_to_max_length
        )
        self._max_uih_len: int = (
            self._max_seq_len
            - self._max_num_candidates
            - (
                len(self._contextual_feature_to_max_length)
                if self._contextual_feature_to_max_length
                else 0
            )
        )
        self.num_files: int = num_files

        self.ratings_file_prefix = ratings_file_prefix
        self.file_to_offsets: Dict[int, List[int]] = {}
        with open(f"{self.ratings_file_prefix}/offset.csv", "r") as file:
            reader = csv.reader(file)
            size = 0
            for row in reader:
                assert len(row) == 1
                offset = json_loads(row[0])
                assert len(offset) == num_users // num_files
                self.file_to_offsets[size] = offset
                size += 1
        self.ts_requests_offsets: List[int] = []
        with open(f"{self.ratings_file_prefix}/requests_per_ts_offset.csv", "r") as file:
            reader = csv.reader(file)
            row = next(reader)
            assert len(row) == 1
            self.ts_requests_offsets = json_loads(row[0])
            assert len(self.ts_requests_offsets) == total_ts
        self.requests: List[int] = []

        self.ts_to_users_cumsum: Dict[int, List[int]] = {}
        with open(
            f"{self.ratings_file_prefix}/users_cumsum_per_ts.csv", "r"
        ) as cumsum_file:
            reader = csv.reader(cumsum_file)
            ts = 0
            for row in reader:
                assert len(row) == 1
                cumsum = json_loads(row[0])
                self.ts_to_users_cumsum[ts] = cumsum[0:self.num_files]
                ts += 1
        self.ts_to_requests: Dict[int, List[int]] = {}
        with open(f"{self.ratings_file_prefix}/requests_per_ts.csv", "r") as file:
            reader = csv.reader(file)
            ts = 0
            for row in reader:
                assert len(row) == 1
                requests = json_loads(row[0])
                self.ts_to_requests[ts] = requests
                logger.debug(f"ts {ts} requests len: {len(requests)}")
                ts += 1

        self.train_ts = train_ts
        self.total_ts = total_ts
        self.num_files = num_files
        self.ts: int = -1
        self.is_inference: bool = is_inference
        self.is_eval: bool = False
        self.users_per_file: int = num_users // num_files
        self.items_per_category: int = num_items // num_categories
        assert hstu_config.action_weights is not None
        self.action_weights: List[int] = hstu_config.action_weights
        self.items_in_memory: Dict[
            int, Dict[int, Tuple[KeyedJaggedTensor, KeyedJaggedTensor]]
        ] = {}

        self.verbose = verbose
        if self.verbose >= 1:
            logger.setLevel(logging.DEBUG)
        elif self.verbose == 0:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        self.device = device
        self.batching_on_gpu = batching_on_gpu
        self.max_buffer_indices = max_buffer_indices
        self.max_buffer_lengths = max_buffer_lengths
        self._buffer_pool: Optional[KJTBatchBufferPool] = None
        if self.batching_on_gpu:
            # Plan 10 §10.1 — replace NV's module-level globals with a
            # rotating per-instance pool so collate(N+1) cannot race
            # collate(N)'s in-flight H2D under DLRM_LISTENER_QUEUE_DEPTH=2.
            try:
                pool_size = int(os.environ.get("DLRM_BATCHING_ON_GPU_POOL_SIZE", "2"))
            except ValueError:
                pool_size = 2
            self._buffer_pool = KJTBatchBufferPool(
                size=max(1, pool_size),
                max_indices_size=self.max_buffer_indices,
                max_lengths_size=self.max_buffer_lengths,
                device=self.device,
            )

    def set_ts(self, ts: int) -> None:
        logger.info(f"Streaming dataset ts set to {ts}")
        if ts == self.ts:
            return
        self.ts = ts
        # at this ts, we have the requests id for this ts
        self.requests = self.ts_to_requests[self.ts]

    # @timer
    def load_query_samples(self, sample_list: List[int]) -> None:
        self.user_data_file = open(f"{self.ratings_file_prefix}/0.csv", "r")
        max_num_candidates = (
            self._max_num_candidates_inference
            if self.is_inference
            else self._max_num_candidates
        )
        for ind, idx in enumerate(sample_list):
            logger.debug(f"loading sample {ind} from ts {self.ts}, total samples: {len(sample_list)}")
            data = self.iloc(idx)
            sample = self.load_item(data, max_num_candidates)
            if self.ts not in self.items_in_memory:
                self.items_in_memory[self.ts] = {}
            self.items_in_memory[self.ts][idx] = sample

    @timer
    def load_query_samples_multi_processing(self, sample_list: List[int]) -> None:
        max_num_candidates = (
            self._max_num_candidates_inference
            if self.is_inference
            else self._max_num_candidates
        )

        process_line_config = ProcessLineConfig(
            ts=self.ts,
            train_ts=self.train_ts,
            total_ts=self.total_ts,
            is_eval=self.is_eval,
            is_inference=self.is_inference
        )

        li_config = LoadItemConfig(
            max_num_candidates=max_num_candidates,
            max_uih_len=self._max_uih_len,
            action_weights=self.action_weights,
            contextual_feature_to_max_length=self._contextual_feature_to_max_length,
            items_per_category=self.items_per_category,
            uih_keys=self._uih_keys,
            candidates_keys=self._candidates_keys,
        )
        THREADS_PER_PROCESS = 4  # Tune this if needed (try 1, 2, or 4)

        torch.set_num_threads(THREADS_PER_PROCESS)
        torch.multiprocessing.set_sharing_strategy('file_system')
        os.environ['OMP_NUM_THREADS'] = str(THREADS_PER_PROCESS)
        os.environ['MKL_NUM_THREADS'] = str(THREADS_PER_PROCESS)
        os.environ['OPENBLAS_NUM_THREADS'] = str(THREADS_PER_PROCESS)
        num_process = 8
        data_per_process = math.ceil(len(sample_list) / num_process)

        process_local_data = []
        for i in range(num_process):
            chunk_start = i * data_per_process
            chunk_end = chunk_start + data_per_process if chunk_start + data_per_process <= len(sample_list) else len(sample_list)
            process_local_data.append(sample_list[chunk_start:chunk_end])

        result_at_ts = {}
        tmp = []

        # Original code commented out for testing:
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_process, mp_context=ctx) as executor:
            futures = []
            ratings_file_path = f"{self.ratings_file_prefix}0.csv"
            for i in range(len(process_local_data)):
                chunk = process_local_data[i]
                logger.debug(f"[rank {i}] loading ({len(chunk)}/{len(sample_list)}) samples")
                future = executor.submit(load_samples_worker, (i, chunk,
                                                               self.file_to_offsets[0], self.requests, self.users_per_file, process_line_config, li_config, ratings_file_path))
                futures.append(future)

            # Wait for all processes to complete
            for i, future in enumerate(futures):
                logger.debug(f"[rank {i}] loading ({i}/{len(futures)}) samples")
                worker_results = future.result()
                # Split combined KJTs back into separate UIH and candidates KJTs
                for sample_idx, combined_kjt, num_uih_keys in worker_results:
                    uih_kjt, candidates_kjt = split_combined_kjt(combined_kjt, num_uih_keys)
                    tmp.append((sample_idx, (uih_kjt, candidates_kjt)))
                    del combined_kjt
                logger.debug(f"[rank {i}] loading ({i}/{len(futures)}) samples done")
        torch.cuda.set_device(self.device)
        for sample_idx, sample in tmp:
            result_at_ts[sample_idx] = sample
        logger.debug(f"ts {self.ts} result_at_ts len: {len(result_at_ts)}, total samples/ts: {len(sample_list)}")
        return result_at_ts

    def unload_query_samples(self, sample_list: List[int]) -> None:
        self.items_in_memory = {}

    def iloc(self, idx: int) -> pd.Series:
        cumsum: List[int] = self.ts_to_users_cumsum[self.ts]
        assert cumsum != []
        assert idx < cumsum[-1]
        file_idx: int = 0
        while cumsum[file_idx] <= idx:
            file_idx += 1
        assert file_idx == 0, "file_idx should be 0 for this mlperf streaming dataset"

        user_idx = self.requests[idx]
        idx = user_idx % self.users_per_file
        self.user_data_file.seek(self.file_to_offsets[file_idx][idx])
        line = self.user_data_file.readline()
        data = self._process_line(line=line, user_id=user_idx)
        return data

    def _process_line(self, line: str, user_id: int) -> pd.Series:
        reader = csv.reader([line])
        parsed_line = next(reader)
        # total ts + one more eval ts + one base ts so that uih won't be zero
        # for each ts, ordered as candidate_ids, candidate_ratings, uih_ids, uih_ratings
        assert len(parsed_line) == 4 * (self.total_ts + 2)
        uih_item_ids_list = []
        uih_ratings_list = []
        candidate_item_ids = ""
        candidate_ratings = ""
        if (not self.is_eval) and (not self.is_inference):
            assert self.ts < self.train_ts
            for i in range(self.ts + 1):
                if parsed_line[4 * i]:
                    uih_item_ids_list.append(parsed_line[2 + 4 * i])
                    uih_ratings_list.append(parsed_line[3 + 4 * i])
            candidate_item_ids = parsed_line[4 * (self.ts + 1)]
            candidate_ratings = parsed_line[1 + 4 * (self.ts + 1)]
        elif self.is_eval:
            for i in range(self.ts + 1):
                if parsed_line[4 * i]:
                    uih_item_ids_list.append(parsed_line[2 + 4 * i])
                    uih_ratings_list.append(parsed_line[3 + 4 * i])
            candidate_item_ids = parsed_line[4 * (self.ts + 1)]
            candidate_ratings = parsed_line[1 + 4 * (self.ts + 1)]
        else:
            assert self.is_inference is True
            assert self.ts >= self.train_ts
            for i in range(self.train_ts + 1):
                if parsed_line[4 * i]:
                    uih_item_ids_list.append(parsed_line[2 + 4 * i])
                    uih_ratings_list.append(parsed_line[3 + 4 * i])
            for i in range(self.train_ts + 2, self.ts + 2):
                if parsed_line[4 * i]:
                    uih_item_ids_list.append(parsed_line[2 + 4 * i])
                    uih_ratings_list.append(parsed_line[3 + 4 * i])
            candidate_item_ids = parsed_line[4 * (self.ts + 2)]
            candidate_ratings = parsed_line[1 + 4 * (self.ts + 2)]
        uih_item_ids = ",".join(uih_item_ids_list)
        uih_ratings = ",".join(uih_ratings_list)
        assert candidate_item_ids != "" and candidate_ratings != ""
        return pd.Series(
            data={
                "user_id": user_id,
                "uih_item_ids": uih_item_ids,
                "uih_ratings": uih_ratings,
                "candidate_item_ids": candidate_item_ids,
                "candidate_ratings": candidate_ratings,
            }
        )

    def load_item(
        self, data: pd.Series, max_num_candidates: int
    ) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        ids_uih = json_loads(data.uih_item_ids)
        ids_candidates = json_loads(data.candidate_item_ids)
        ratings_uih = json_loads(data.uih_ratings)
        ratings_candidates = json_loads(data.candidate_ratings)
        timestamps_uih = self.get_timestamp_uih(
            data=data,
            max_num_candidates=max_num_candidates,
            size=len(ids_uih),
        )
        assert len(ids_uih) == len(
            timestamps_uih
        ), "history len differs from timestamp len."
        assert len(ids_uih) == len(
            ratings_uih
        ), f"history len {len(ids_uih)} differs from ratings len {len(ratings_uih)}."
        assert (
            len(ids_candidates) == len(ratings_candidates)
        ), f"candidates len {len(ids_candidates)} differs from ratings len {len(ratings_candidates)}."

        ids_uih = maybe_truncate_seq(ids_uih, self._max_uih_len)
        ratings_uih = maybe_truncate_seq(ratings_uih, self._max_uih_len)
        timestamps_uih = maybe_truncate_seq(timestamps_uih, self._max_uih_len)
        ids_candidates = maybe_truncate_seq(ids_candidates, max_num_candidates)
        num_candidates = len(ids_candidates)
        ratings_candidates = maybe_truncate_seq(ratings_candidates, max_num_candidates)
        action_weights_uih = [
            self.action_weights[int(rating) - 1] for rating in ratings_uih
        ]
        action_weights_candidates = [
            int(rating >= 3.5) for rating in ratings_candidates
        ]

        uih_kjt_values: List[int] = []
        uih_kjt_lengths: List[int] = []
        for name, length in self._contextual_feature_to_max_length.items():
            uih_kjt_values.append(data[name])
            uih_kjt_lengths.append(length)

        uih_seq_len = len(ids_uih)
        dummy_watch_times_uih = [0 for _ in range(uih_seq_len)]
        item_category_ids = [id // self.items_per_category for id in ids_uih]

        uih_keys = copy.deepcopy(self._uih_keys)
        extend_uih_kjt_values: List[int] = (
            ids_uih
            + ratings_uih
            + timestamps_uih
            + action_weights_uih
            + dummy_watch_times_uih
            + item_category_ids
        )
        uih_kjt_values.extend(extend_uih_kjt_values)
        uih_kjt_lengths.extend(
            [
                uih_seq_len
                for _ in range(
                    len(uih_keys) - len(self._contextual_feature_to_max_length)
                )
            ]
        )

        dummy_query_time = 0 if timestamps_uih == [] else max(timestamps_uih)
        uih_kjt_values.append(dummy_query_time)
        uih_kjt_lengths.append(1)
        uih_features_kjt: KeyedJaggedTensor = KeyedJaggedTensor(
            keys=uih_keys + ["dummy_query_time"],
            lengths=torch.tensor(uih_kjt_lengths).long(),
            values=torch.tensor(uih_kjt_values).long(),
        )

        candidates_keys = copy.deepcopy(self._candidates_keys)
        item_candidate_category_ids = [
            id // self.items_per_category for id in ids_candidates
        ]
        candidates_kjt_values = (
            ids_candidates
            + ratings_candidates
            + [dummy_query_time] * num_candidates  # item_query_time
            + action_weights_candidates
            + [1] * num_candidates  # item_dummy_watchtime
            + item_candidate_category_ids
        )
        candidates_kjt_lengths = num_candidates * torch.ones(len(candidates_keys))

        candidates_features_kjt: KeyedJaggedTensor = KeyedJaggedTensor(
            keys=candidates_keys,
            lengths=candidates_kjt_lengths.detach().clone().long(),
            # values=torch.tensor(candidates_kjt_values, dtype=torch.int64),
            values=torch.tensor(candidates_kjt_values).long(),
        )

        # Find and cap values that exceed 1000000000
        # mask = candidates_features_kjt.values() >= 1000000000
        # if mask.any():
        #     indices = [i.item() for i in torch.nonzero(mask, as_tuple=False)]
        #     for i in indices:
        #         candidates_features_kjt.values()[i] = 999999999

        return uih_features_kjt, candidates_features_kjt

    def get_timestamp_uih(
        self, data: pd.Series, max_num_candidates: int, size: int
    ) -> List[int]:
        return [1] * size

    def get_sample_with_ts(
        self, id: int, ts: int
    ) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        store = self.items_in_memory[ts]
        if isinstance(store, _LazyTimestampSamples):
            return store.get(id)
        return store[id]

    def get_samples_with_ts(self, id_list: List[int], ts: int) -> Samples:
        self.use_buffer = False
        list_samples = [self.get_sample_with_ts(ix, ts) for ix in id_list]
        buffer_pair = self._buffer_pool.next() if self._buffer_pool is not None else None
        return collate_fn(
            list_samples,
            device=self.device,
            batching_on_gpu=self.batching_on_gpu,
            buffer_pair=buffer_pair,
        )

    def get_samples_with_ts_updated(self, ts_request_pairs: List[Tuple[int, int]]) -> Samples:
        self.use_buffer = False
        list_samples = [self.get_sample_with_ts(ix, ts) for ts, ix in ts_request_pairs]
        buffer_pair = self._buffer_pool.next() if self._buffer_pool is not None else None
        return collate_fn(
            list_samples,
            device=self.device,
            batching_on_gpu=self.batching_on_gpu,
            buffer_pair=buffer_pair,
        )

    # this is actually used in submission run, for faster load up time
    def serialize_ds(self, output_dir: str) -> None:
        """
        Serialize all loaded KJT data to numpy arrays for fast reloading.

        Creates a directory structure:
            output_dir/
                metadata.json          # keys, sample counts per ts
                ts_{ts}/
                    sample_ids.npy     # array of sample IDs in order
                    uih_lengths.npy    # (num_samples, num_uih_keys) - lengths per sample
                    uih_values.npy     # concatenated values
                    uih_offsets.npy    # offsets into uih_values for each sample
                    cand_lengths.npy   # (num_samples, num_cand_keys)
                    cand_values.npy    # concatenated values  
                    cand_offsets.npy   # offsets into cand_values for each sample
        """
        import numpy as np

        os.makedirs(output_dir, exist_ok=True)

        metadata = {
            'timestamps': [],
            'uih_keys': None,
            'candidates_keys': None,
        }

        for ts, samples_dict in self.items_in_memory.items():
            if not samples_dict:
                continue

            logger.info(f"Serializing timestamp {ts} with {len(samples_dict)} samples...")
            ts_dir = os.path.join(output_dir, f"ts_{ts}")
            os.makedirs(ts_dir, exist_ok=True)

            # Get sample IDs in sorted order for deterministic loading
            sample_ids = sorted(samples_dict.keys())

            # Initialize lists to collect data
            uih_lengths_list = []
            uih_values_list = []
            uih_offsets = [0]

            cand_lengths_list = []
            cand_values_list = []
            cand_offsets = [0]

            for sample_id in sample_ids:
                uih_kjt, cand_kjt = samples_dict[sample_id]

                # Extract keys from first sample (same for all)
                if metadata['uih_keys'] is None:
                    metadata['uih_keys'] = list(uih_kjt.keys())
                    metadata['candidates_keys'] = list(cand_kjt.keys())

                # UIH data
                uih_lens = uih_kjt.lengths().numpy()
                uih_vals = uih_kjt.values().numpy()
                uih_lengths_list.append(uih_lens)
                uih_values_list.append(uih_vals)
                uih_offsets.append(uih_offsets[-1] + len(uih_vals))

                # Candidates data
                cand_lens = cand_kjt.lengths().numpy()
                cand_vals = cand_kjt.values().numpy()
                cand_lengths_list.append(cand_lens)
                cand_values_list.append(cand_vals)
                cand_offsets.append(cand_offsets[-1] + len(cand_vals))
            # Stack and save - preserve original dtypes from tensors
            np.save(os.path.join(ts_dir, 'sample_ids.npy'), np.array(sample_ids, dtype=np.int32))
            np.save(os.path.join(ts_dir, 'uih_lengths.npy'), np.stack(uih_lengths_list))  # preserve original dtype
            np.save(os.path.join(ts_dir, 'uih_values.npy'), np.concatenate(uih_values_list))  # preserve original dtype
            np.save(os.path.join(ts_dir, 'uih_offsets.npy'), np.array(uih_offsets, dtype=np.int64))
            np.save(os.path.join(ts_dir, 'cand_lengths.npy'), np.stack(cand_lengths_list))  # preserve original dtype
            np.save(os.path.join(ts_dir, 'cand_values.npy'), np.concatenate(cand_values_list))  # preserve original dtype
            np.save(os.path.join(ts_dir, 'cand_offsets.npy'), np.array(cand_offsets, dtype=np.int64))

            metadata['timestamps'].append({
                'ts': ts,
                'num_samples': len(sample_ids),
            })

            logger.info(f"  Saved {len(sample_ids)} samples, "
                        f"uih_values: {uih_offsets[-1]}, cand_values: {cand_offsets[-1]}")

        # Save metadata
        import json
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Serialization complete. Saved to: {output_dir}")

    # this is actually used in submission run, for faster load up time
    def load_preprocessed_dataset(self, preprocessed_dir: str) -> None:
        """
        Load KJT data from preprocessed numpy arrays.

        This is much faster than loading from CSV because:
        1. No CSV parsing overhead
        2. numpy.load with mmap_mode='r' is very fast
        3. torch.from_numpy() is zero-copy
        4. All data is allocated in main process (NUMA-local)

        Args:
            preprocessed_dir: Directory containing serialized numpy arrays
        """
        import numpy as np
        import json

        metadata_path = os.path.join(preprocessed_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata.json found in {preprocessed_dir}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        uih_keys = metadata['uih_keys']
        candidates_keys = metadata['candidates_keys']
        lazy = os.environ.get("DLRM_LAZY_PREPROCESSED", "1") == "1"
        cache_max = int(os.environ.get("DLRM_LAZY_CACHE_SAMPLES", "64"))

        for ts_info in metadata['timestamps']:
            ts = ts_info['ts']
            num_samples = ts_info['num_samples']
            ts_dir = os.path.join(preprocessed_dir, f"ts_{ts}")

            if lazy:
                self.items_in_memory[ts] = _LazyTimestampSamples(
                    ts_dir, uih_keys, candidates_keys, cache_max=cache_max
                )
                logger.info(
                    f"Lazy-mapped {num_samples} samples for timestamp {ts} from {ts_dir}"
                )
                continue

            sample_ids = np.load(os.path.join(ts_dir, 'sample_ids.npy'))
            uih_lengths = np.load(os.path.join(ts_dir, 'uih_lengths.npy'))
            uih_values = np.load(os.path.join(ts_dir, 'uih_values.npy'))
            uih_offsets = np.load(os.path.join(ts_dir, 'uih_offsets.npy'))
            cand_lengths = np.load(os.path.join(ts_dir, 'cand_lengths.npy'))
            cand_values = np.load(os.path.join(ts_dir, 'cand_values.npy'))
            cand_offsets = np.load(os.path.join(ts_dir, 'cand_offsets.npy'))

            self.items_in_memory[ts] = {}
            for i, sample_id in enumerate(sample_ids):
                uih_lens = torch.from_numpy(uih_lengths[i].copy())
                uih_vals = torch.from_numpy(
                    uih_values[uih_offsets[i]:uih_offsets[i + 1]].copy()
                )
                cand_lens = torch.from_numpy(cand_lengths[i].copy())
                cand_vals = torch.from_numpy(
                    cand_values[cand_offsets[i]:cand_offsets[i + 1]].copy()
                )
                if _CLAMP_OOB_IDS:
                    _clamp_lookup_features(uih_vals, uih_keys, uih_lens)
                    _clamp_lookup_features(cand_vals, candidates_keys, cand_lens)
                uih_kjt = KeyedJaggedTensor(
                    keys=uih_keys, lengths=uih_lens, values=uih_vals
                )
                cand_kjt = KeyedJaggedTensor(
                    keys=candidates_keys, lengths=cand_lens, values=cand_vals
                )
                self.items_in_memory[ts][int(sample_id)] = (uih_kjt, cand_kjt)
            logger.info(f"Loaded {len(self.items_in_memory[ts])} samples for timestamp {ts}")
