"""Unit tests for the POSIX SHM result pool."""

from __future__ import annotations

import uuid
from multiprocessing import shared_memory

import pytest

from wan_harness.shm_pool import (
    ResultShmRef,
    ShmResultPool,
    SlotRelease,
    _release_attached_shm,
    compute_slot_layout,
    is_shm_available,
    segment_name,
)
from wan_harness.wire import ResultMeta


def test_compute_slot_layout_default_dims() -> None:
    layout = compute_slot_layout(height=720, width=1280, num_frames=81)
    assert layout.frames_cap == 720 * 1280 * 81 * 3
    assert layout.mp4_cap >= layout.frames_cap // 4
    assert layout.slot_bytes == layout.frames_cap + layout.mp4_cap


def test_is_shm_available_single_node() -> None:
    assert is_shm_available(world_size=8) is True


def test_result_shm_ref_and_slot_release_picklable() -> None:
    import pickle

    ref = ResultShmRef(
        meta=ResultMeta(sample_index=1, frame_count=2, height=8, width=8),
        src_rank=3,
        slot_id=0,
        frames_nbytes=100,
        mp4_nbytes=0,
    )
    ack = SlotRelease(rank=3, slot_id=0)
    assert pickle.loads(pickle.dumps(ref)) == ref
    assert pickle.loads(pickle.dumps(ack)) == ack


def test_shm_pool_write_read_roundtrip() -> None:
    pool_id = uuid.uuid4().hex[:12]
    layout = compute_slot_layout(height=8, width=8, num_frames=1, mp4_cap_mib=1)
    frames = bytes([1, 2, 3] * 20)
    mp4 = b"mp4-payload"

    own = shared_memory.SharedMemory(
        name=segment_name(pool_id, 1),
        create=True,
        size=layout.slot_bytes,
    )
    try:
        pool = ShmResultPool(
            pool_id=pool_id,
            rank=1,
            world_size=2,
            layout=layout,
            own_shm=own,
            foreign_shm={},
        )
        pool.write_result(frames_bytes=frames, mp4_bytes=mp4)
        ref = ResultShmRef(
            meta=ResultMeta(sample_index=0, frame_count=1, height=8, width=8),
            src_rank=1,
            slot_id=0,
            frames_nbytes=len(frames),
            mp4_nbytes=len(mp4),
        )
        back = pool.read_as_result(ref)
        assert back.frames_bytes == frames
        assert back.mp4_bytes == mp4
    finally:
        own.close()
        own.unlink()


def test_shm_pool_rejects_oversized_frames() -> None:
    pool_id = uuid.uuid4().hex[:12]
    layout = compute_slot_layout(height=8, width=8, num_frames=1, mp4_cap_mib=1)
    own = shared_memory.SharedMemory(
        name=segment_name(pool_id, 0),
        create=True,
        size=layout.slot_bytes,
    )
    try:
        pool = ShmResultPool(
            pool_id=pool_id,
            rank=0,
            world_size=1,
            layout=layout,
            own_shm=own,
            foreign_shm={},
        )
        with pytest.raises(ValueError, match="frames_bytes length"):
            pool.write_result(
                frames_bytes=b"x" * (layout.frames_cap + 1),
                mp4_bytes=None,
            )
    finally:
        own.close()
        own.unlink()


def test_foreign_attach_unregister_on_close() -> None:
    """Rank-0-style attach must not leave resource_tracker entries behind."""
    import warnings

    pool_id = uuid.uuid4().hex[:12]
    layout = compute_slot_layout(height=8, width=8, num_frames=1, mp4_cap_mib=1)
    creator = shared_memory.SharedMemory(
        name=segment_name(pool_id, 1),
        create=True,
        size=layout.slot_bytes,
    )
    try:
        attached = shared_memory.SharedMemory(
            name=segment_name(pool_id, 1), create=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)
            _release_attached_shm(attached)
        # Block still owned by creator; re-attach cleanly.
        again = shared_memory.SharedMemory(name=segment_name(pool_id, 1), create=False)
        _release_attached_shm(again)
    finally:
        creator.close()
        creator.unlink()


def test_rank0_reads_foreign_segment() -> None:
    pool_id = uuid.uuid4().hex[:12]
    layout = compute_slot_layout(height=8, width=8, num_frames=1, mp4_cap_mib=1)
    frames = b"\xab" * 48
    worker = shared_memory.SharedMemory(
        name=segment_name(pool_id, 1),
        create=True,
        size=layout.slot_bytes,
    )
    reader_own = shared_memory.SharedMemory(
        name=segment_name(pool_id, 0),
        create=True,
        size=layout.slot_bytes,
    )
    try:
        worker_pool = ShmResultPool(
            pool_id=pool_id,
            rank=1,
            world_size=2,
            layout=layout,
            own_shm=worker,
            foreign_shm={},
        )
        worker_pool.write_result(frames_bytes=frames, mp4_bytes=None)

        reader = ShmResultPool(
            pool_id=pool_id,
            rank=0,
            world_size=2,
            layout=layout,
            own_shm=reader_own,
            foreign_shm={},
        )
        ref = ResultShmRef(
            meta=ResultMeta(sample_index=5, frame_count=1, height=8, width=8),
            src_rank=1,
            slot_id=0,
            frames_nbytes=len(frames),
            mp4_nbytes=0,
        )
        got_frames, got_mp4 = reader.read_result(ref)
        assert got_frames == frames
        assert got_mp4 is None
        reader.close_foreign_attachments()
    finally:
        worker.close()
        worker.unlink()
        reader_own.close()
        reader_own.unlink()
