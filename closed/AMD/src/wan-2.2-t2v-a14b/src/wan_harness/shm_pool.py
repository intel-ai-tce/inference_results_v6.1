"""POSIX shared-memory pool for Result bulk payloads (single-node DP).

Workers write frame/MP4 bytes into a fixed-layout slot in a per-rank
``multiprocessing.shared_memory`` segment; rank 0 reads by descriptor.
Gloo carries only small control messages (metadata + slot handoff).
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch.distributed import ProcessGroup

    from .wire import Result, ResultMeta

_log = logging.getLogger(__name__)

__all__ = [
    "ShmSlotLayout",
    "ShmResultPool",
    "ResultShmRef",
    "SlotRelease",
    "compute_slot_layout",
    "init_shm_result_pool",
    "is_shm_available",
    "segment_name",
]

_MIB = 1024 * 1024
_NAME_PREFIX = "wan_harness"


@dataclass(frozen=True)
class ShmSlotLayout:
    """Fixed byte layout for one result slot."""

    frames_cap: int
    mp4_cap: int

    @property
    def slot_bytes(self) -> int:
        return int(self.frames_cap) + int(self.mp4_cap)


@dataclass(frozen=True)
class ResultShmRef:
    """Descriptor sent on the Gloo control plane when bulk data lives in SHM."""

    meta: "ResultMeta"
    src_rank: int
    slot_id: int
    frames_nbytes: int
    mp4_nbytes: int


@dataclass(frozen=True)
class SlotRelease:
    """ACK from rank 0 telling a worker its SHM slot may be reused."""

    rank: int
    slot_id: int


def compute_slot_layout(
    *,
    height: int,
    width: int,
    num_frames: int,
    mp4_cap_mib: int = 16,
) -> ShmSlotLayout:
    """Derive slot capacity from harness frame dimensions."""
    frames_cap = int(height) * int(width) * int(num_frames) * 3
    mp4_cap = max(frames_cap // 4, int(mp4_cap_mib) * _MIB)
    return ShmSlotLayout(frames_cap=frames_cap, mp4_cap=mp4_cap)


def is_shm_available(*, world_size: int) -> bool:
    """Return True when all ranks are on this host (torchrun single-node)."""
    local = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    return local == int(world_size)


def segment_name(pool_id: str, rank: int) -> str:
    return f"{_NAME_PREFIX}_{pool_id}_r{int(rank)}"


def _release_attached_shm(shm: shared_memory.SharedMemory) -> None:
    """Close a foreign SHM attach without unlinking the creator's block.

    Python 3.12 registers *every* ``SharedMemory`` mapping with
    ``resource_tracker``, including ``create=False`` attaches.  Rank 0
    must unregister those names on close or process exit tries to
    ``shm_unlink`` segments it did not create (ENOENT + leak warnings).
    """
    from multiprocessing import resource_tracker

    tracker_name = getattr(shm, "_name", None) or shm.name
    shm.close()
    if tracker_name:
        try:
            resource_tracker.unregister(tracker_name, "shared_memory")
        except Exception:  # noqa: BLE001 – best-effort cleanup
            pass


class ShmResultPool:
    """Per-rank SHM segments with a single fixed slot (slot_id=0)."""

    def __init__(
        self,
        *,
        pool_id: str,
        rank: int,
        world_size: int,
        layout: ShmSlotLayout,
        own_shm: shared_memory.SharedMemory,
        foreign_shm: dict[int, shared_memory.SharedMemory],
    ) -> None:
        self.pool_id = pool_id
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.layout = layout
        self._own = own_shm
        self._foreign = foreign_shm
        self._creator = int(rank)

    @property
    def slot_bytes(self) -> int:
        return self.layout.slot_bytes

    def write_result(
        self,
        *,
        frames_bytes: bytes,
        mp4_bytes: bytes | None,
        slot_id: int = 0,
    ) -> None:
        if slot_id != 0:
            raise ValueError(f"only slot_id=0 is supported, got {slot_id!r}")
        mp4 = mp4_bytes or b""
        if len(frames_bytes) > self.layout.frames_cap:
            raise ValueError(
                f"frames_bytes length {len(frames_bytes)} exceeds cap "
                f"{self.layout.frames_cap}"
            )
        if len(mp4) > self.layout.mp4_cap:
            raise ValueError(
                f"mp4_bytes length {len(mp4)} exceeds cap {self.layout.mp4_cap}"
            )
        buf = self._own.buf
        buf[: len(frames_bytes)] = frames_bytes
        off = self.layout.frames_cap
        if mp4:
            buf[off : off + len(mp4)] = mp4

    def read_result(self, ref: ResultShmRef) -> tuple[bytes, bytes | None]:
        if ref.slot_id != 0:
            raise ValueError(f"only slot_id=0 is supported, got {ref.slot_id!r}")
        shm = self._segment_for_rank(ref.src_rank)
        buf = shm.buf
        frames = bytes(buf[: ref.frames_nbytes])
        mp4 = b""
        if ref.mp4_nbytes > 0:
            off = self.layout.frames_cap
            mp4 = bytes(buf[off : off + ref.mp4_nbytes])
        return frames, (mp4 if mp4 else None)

    def read_as_result(self, ref: ResultShmRef) -> "Result":
        from .wire import Result, meta_to_result

        frames, mp4 = self.read_result(ref)
        return meta_to_result(ref.meta, frames_bytes=frames, mp4_bytes=mp4)

    def close(self, *, unlink: bool = False) -> None:
        for shm in list(self._foreign.values()):
            _release_attached_shm(shm)
        self._foreign.clear()
        if unlink:
            try:
                self._own.unlink()
            except FileNotFoundError:
                pass
        self._own.close()

    def _segment_for_rank(self, src_rank: int) -> shared_memory.SharedMemory:
        r = int(src_rank)
        if r == self.rank:
            return self._own
        if r not in self._foreign:
            name = segment_name(self.pool_id, r)
            self._foreign[r] = shared_memory.SharedMemory(name=name, create=False)
        return self._foreign[r]

    def close_foreign_attachments(self) -> None:
        """Release rank-0 foreign SHM views (does not unlink worker blocks)."""
        for shm in list(self._foreign.values()):
            _release_attached_shm(shm)
        self._foreign.clear()


def init_shm_result_pool(
    *,
    rank: int,
    world_size: int,
    layout: ShmSlotLayout,
    group: "ProcessGroup | None" = None,
) -> ShmResultPool | None:
    """Collectively create/attach per-rank SHM segments. Returns None if unavailable."""
    if not is_shm_available(world_size=world_size):
        _log.warning(
            "SHM result transport unavailable (multi-node or LOCAL_WORLD_SIZE "
            "mismatch); falling back to Gloo bulk transfer"
        )
        return None

    import torch.distributed as dist  # noqa: WPS433

    pool_id_holder: list[str | None] = [None]
    if int(rank) == 0:
        pool_id_holder[0] = uuid.uuid4().hex[:12]
    dist.broadcast_object_list(pool_id_holder, src=0, group=group)
    pool_id = str(pool_id_holder[0])
    slot_bytes = layout.slot_bytes

    name = segment_name(pool_id, rank)
    own = shared_memory.SharedMemory(name=name, create=True, size=slot_bytes)

    dist.barrier(group=group)

    # Foreign segments are opened lazily on first read (rank 0 only).
    foreign: dict[int, shared_memory.SharedMemory] = {}

    _log.info(
        "ShmResultPool: rank=%d pool_id=%s slot_bytes=%.1f MiB",
        rank,
        pool_id,
        slot_bytes / _MIB,
    )
    return ShmResultPool(
        pool_id=pool_id,
        rank=rank,
        world_size=world_size,
        layout=layout,
        own_shm=own,
        foreign_shm=foreign,
    )
