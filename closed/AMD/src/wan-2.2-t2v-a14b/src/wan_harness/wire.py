"""Wire types and collective helpers used by the dispatchers.

This module is the seam between the dispatcher (``dispatcher.py``) and
``torch.distributed``. It carries two concerns:

1. The small Python dataclasses dispatched on the wire:

   * :class:`WorkUnit`  – one (prompt, qsl index) to process.
   * :class:`WorkBatch` – the Ulysses broadcast payload (a list of
     WorkUnits plus their requested generation args, so all ranks see
     the same input).
   * :class:`Result`    – a serialised :class:`GeneratedVideo`, gathered
     from a worker rank back to rank 0.
   * :class:`Shutdown`  – sentinel that tells workers to leave their
     loops.

2. Helpers wrapping ``torch.distributed``'s object-aware collective ops:

   * :func:`broadcast_object` – single-object broadcast on the world
     PG. Used by :class:`UlyssesDispatcher` to push one ``WorkUnit`` at
     a time.
   * :func:`broadcast_str_list` / :func:`gather_results_to_rank0` and
     the ``encode_/decode_wave_command*`` helpers – building blocks for
     the wave wire protocol used by :class:`WaveDispatcher`.

Per-wave / per-broadcast payloads are intentionally small (single-prompt
granularity) so any one collective is a quick op rather than a giant
tensor blob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch.distributed import ProcessGroup

    from .shm_pool import ShmResultPool

__all__ = [
    "WorkUnit",
    "WorkBatch",
    "Result",
    "Shutdown",
    "WorkerFailure",
    "ResultShmRef",
    "SlotRelease",
    "broadcast_object",
    "configure_result_transport",
    "use_shm_for_results",
    "WAVE_CMD_RUN",
    "WAVE_CMD_EXIT",
    "WAVE_INACTIVE_INDEX",
    "encode_wave_command_payload",
    "decode_wave_command_payload",
    "encode_wave_command",
    "decode_wave_command",
    "broadcast_str_list",
    "gather_results_to_rank0",
    "send_object_pt2pt",
    "recv_object_any_src",
    "recv_object_from_src",
    "release_worker_slot",
]


# ----------------------------------------------------------------------
# Wave dispatcher wire protocol.
#
# A "wave" is a collective unit of up to ``world_size`` samples that all
# ranks process in lockstep. Each wave is exactly:
#
#   1. ``dist.broadcast`` of a fixed-shape int64 command tensor.
#   2. ``dist.broadcast_object_list`` of the per-rank prompt strings.
#   3. ``backend.run_unit`` on every rank in parallel.
#   4. ``dist.gather_object`` of the per-rank :class:`Result` back to rank 0.
#
# Step 1's tensor layout is:
#     [cmd, n, idx_0, idx_1, ..., idx_{world_size-1}]
#
# where:
#     * ``cmd`` ∈ {WAVE_CMD_RUN, WAVE_CMD_EXIT}.
#     * ``n`` is the number of *active* slots (0 .. world_size).
#     * ``idx_i`` is the QSL sample index for rank ``i``, or
#       :data:`WAVE_INACTIVE_INDEX` when ``i >= n`` (i.e. a tail slot
#       that has no work in the current wave).
#
# We carry the tensor at fixed width (``2 + world_size`` int64 elements)
# so the recv side can pre-allocate without first sizing the message.
# Keeping the wire format encoded in plain ints (no Python pickling on
# the hot path) keeps the per-wave overhead in the microsecond range.
# ----------------------------------------------------------------------

WAVE_CMD_RUN = 1
"""Command code: rank 0 has a wave of work for the world to process."""

WAVE_CMD_EXIT = -1
"""Command code: rank 0 is tearing down; workers should exit
``run_worker_loop``."""

WAVE_INACTIVE_INDEX = -1
"""Sentinel placed in unused tail slots of a wave (the wave is shorter
than ``world_size``). Ranks observing this value must skip their local
``run_unit`` call but still participate in the gather (with ``None`` as
their contribution) so the collective stays balanced."""


# ----------------------------------------------------------------------
# Message types.
# ----------------------------------------------------------------------


@dataclass
class WorkUnit:
    """One prompt to generate. ``input_args`` holds the kwargs that
    :meth:`xfuser.xFuserModel._run_pipe` expects (height, width,
    num_inference_steps, guidance_scale, seed, ...). Rank 0 builds one
    per active slot in a wave; the Ulysses dispatcher broadcasts a
    single :class:`WorkUnit` at a time while the wave dispatcher builds
    them inline from the broadcast prompt + index pair.
    """

    sample_index: int
    prompt: str
    input_args: dict[str, Any]


@dataclass
class WorkBatch:
    """A list of :class:`WorkUnit`s broadcast from rank 0 to all ranks.

    The Ulysses dispatcher uses this to push 1 unit at a time; the API
    accepts a list so a future scenario can push more without changing
    the wire format.
    """

    units: list[WorkUnit]


@dataclass
class Result:
    """A worker's reply: one generated sample, serialised by reference
    (raw bytes) so pickle stays small."""

    sample_index: int
    frames_bytes: bytes
    frame_count: int
    height: int
    width: int
    mp4_bytes: bytes | None = None


@dataclass
class Shutdown:
    """Sentinel broadcast by rank 0 when the LoadGen test is finished."""


@dataclass
class WorkerFailure:
    """Sentinel returned by a worker rank when ``backend.run_unit`` raises.

    Both data-parallel dispatchers route worker exceptions through this
    type instead of letting them propagate inside the collective: the
    failing rank still has to participate in the wave gather (or the
    async result send) so the other ranks do not hang. Rank 0 detects
    the sentinel post-collective and raises a single ``RuntimeError``
    blaming the offending rank + sample, matching the harness'
    fail-fast policy.
    """

    rank: int
    sample_index: int
    error_repr: str


@dataclass
class ResultMeta:
    """Lightweight :class:`Result` header sent on the control plane.

    Frame and optional MP4 bytes are transferred separately as raw
    ``uint8`` tensors so we never pickle hundreds of MiB of pixel data.
    """

    sample_index: int
    frame_count: int
    height: int
    width: int


@dataclass
class ResultWireMeta:
    """Gather-scatter header for :class:`Result` in :func:`gather_results_to_rank0`."""

    meta: ResultMeta
    frames_nbytes: int
    mp4_nbytes: int


# Re-export SHM wire types (defined in shm_pool to avoid circular imports).
from .shm_pool import ResultShmRef, SlotRelease  # noqa: E402


# ----------------------------------------------------------------------
# Result transport configuration (Gloo bulk vs POSIX SHM data plane).
# ----------------------------------------------------------------------

_shm_pool: "ShmResultPool | None" = None
_use_shm: bool = False


def configure_result_transport(
    pool: "ShmResultPool | None", *, use_shm: bool
) -> None:
    """Set module-level SHM pool used by send/recv helpers."""
    global _shm_pool, _use_shm
    _shm_pool = pool
    _use_shm = bool(use_shm and pool is not None)


def use_shm_for_results() -> bool:
    return _use_shm and _shm_pool is not None


def reset_result_transport() -> None:
    """Clear transport state (tests only)."""
    configure_result_transport(None, use_shm=False)


def result_to_meta(result: Result) -> ResultMeta:
    return ResultMeta(
        sample_index=int(result.sample_index),
        frame_count=int(result.frame_count),
        height=int(result.height),
        width=int(result.width),
    )


def meta_to_result(
    meta: ResultMeta,
    *,
    frames_bytes: bytes,
    mp4_bytes: bytes | None,
) -> Result:
    return Result(
        sample_index=int(meta.sample_index),
        frames_bytes=frames_bytes,
        frame_count=int(meta.frame_count),
        height=int(meta.height),
        width=int(meta.width),
        mp4_bytes=mp4_bytes,
    )


# ----------------------------------------------------------------------
# Collective helpers.
# ----------------------------------------------------------------------


def _torch_modules():
    """Late-import torch so the wire module is importable without torch
    available – useful for unit tests that exercise the dataclasses only.
    """
    import torch
    import torch.distributed as dist

    return torch, dist


def broadcast_object(
    obj: Any | None,
    *,
    src: int,
    rank: int,
    group: "ProcessGroup | None" = None,
    device: "torch.device | str | None" = None,
) -> Any:
    """Broadcast a Python object from ``src`` to every rank in ``group``.

    Wrapper around ``torch.distributed.broadcast_object_list`` that fits
    the one-object-at-a-time idiom the dispatchers use. Non-``src`` ranks
    pass ``obj=None`` and receive the broadcast value as the return.
    """
    torch, dist = _torch_modules()
    del device  # broadcast_object_list infers device from the current PG

    payload: list[Any] = [obj] if rank == src else [None]
    dist.broadcast_object_list(payload, src=src, group=group)
    return payload[0]


def all_ranks(world_size: int) -> Sequence[int]:
    """Convenience: list of ranks in a single-node world."""
    return tuple(range(world_size))


# ----------------------------------------------------------------------
# Wave wire helpers.
# ----------------------------------------------------------------------


def encode_wave_command_payload(
    cmd: int, indices: Sequence[int], world_size: int
) -> list[int]:
    """Encode a wave command into the fixed-shape int payload.

    Returns a ``list[int]`` of length ``2 + world_size``:
    ``[cmd, n, idx_0, idx_1, ..., idx_{world_size-1}]``.
    Trailing slots beyond ``n = len(indices)`` are filled with
    :data:`WAVE_INACTIVE_INDEX`.

    Pure Python so the encode logic is testable without torch.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    n = len(indices)
    if n > world_size:
        raise ValueError(
            f"wave has {n} indices but world_size is {world_size}; the caller "
            f"is responsible for splitting overly long batches into waves."
        )
    payload: list[int] = [int(cmd), int(n)]
    payload.extend(int(i) for i in indices)
    payload.extend([WAVE_INACTIVE_INDEX] * (world_size - n))
    return payload


def decode_wave_command_payload(payload: Sequence[int]) -> tuple[int, list[int]]:
    """Decode the inverse of :func:`encode_wave_command_payload`.

    Returns ``(cmd, indices)`` where ``len(indices) == n`` (the value
    written by the encoder; padding slots are stripped). Pure Python.
    """
    if len(payload) < 2:
        raise ValueError(
            f"wave payload must hold at least [cmd, n]; got len={len(payload)}"
        )
    cmd = int(payload[0])
    n = int(payload[1])
    if n < 0:
        raise ValueError(f"wave payload has negative n={n}")
    if 2 + n > len(payload):
        raise ValueError(
            f"wave payload claims n={n} but only carries {len(payload) - 2} "
            f"slot ints"
        )
    indices = [int(payload[2 + i]) for i in range(n)]
    return cmd, indices


def encode_wave_command(
    cmd: int,
    indices: Sequence[int],
    world_size: int,
    device: "torch.device | str | None" = None,
) -> "torch.Tensor":
    """Tensor wrapper around :func:`encode_wave_command_payload`.

    Returned tensor is ``int64`` and lives on ``device`` (CPU when
    ``device is None``). Always has length ``2 + world_size`` so the
    recv side can pre-allocate.
    """
    torch, _ = _torch_modules()
    payload = encode_wave_command_payload(cmd, indices, world_size)
    return torch.tensor(payload, dtype=torch.long, device=device)


def decode_wave_command(tensor: "torch.Tensor") -> tuple[int, list[int]]:
    """Tensor wrapper around :func:`decode_wave_command_payload`."""
    return decode_wave_command_payload(tensor.tolist())


def broadcast_str_list(
    items: list[str] | None,
    *,
    src: int,
    rank: int,
    world_size: int,
    group: "ProcessGroup | None" = None,
) -> list[str]:
    """Broadcast a fixed-length list of strings from ``src`` to all ranks.

    The wave dispatcher uses this to push the per-rank prompts to the
    workers. ``items`` is required on rank ``src`` (must have length
    ``world_size``) and ignored on every other rank (pass ``None``).

    Note: ``broadcast_object_list`` requires *every* rank to provide a
    list of the same length – it broadcasts each slot separately and
    sizes its internal byte-length tensor by that length. Receivers
    therefore allocate ``[None] * world_size`` placeholders before the
    collective; an earlier version passed ``[None]`` here and crashed
    Gloo with "Received data size doesn't match expected size".
    """
    _, dist = _torch_modules()
    if rank == src:
        items_list = list(items) if items is not None else []
        if len(items_list) != world_size:
            raise ValueError(
                f"broadcast_str_list: source list must have length "
                f"world_size={world_size}, got {len(items_list)}"
            )
        payload: list[Any] = items_list
    else:
        payload = [None] * world_size
    dist.broadcast_object_list(payload, src=src, group=group)
    return [str(x) if x is not None else "" for x in payload]


def gather_results_to_rank0(
    local: "Result | WorkerFailure | None",
    *,
    world_size: int,
    rank: int,
    dst: int = 0,
    group: "ProcessGroup | None" = None,
    device: "torch.device | str | None" = None,
) -> list["Result | WorkerFailure | None"]:
    """Gather one payload from each rank to ``dst``.

    :class:`Result` values are gathered as :class:`ResultWireMeta` plus a
    separate raw-byte transfer so the collective never pickles full frame
    buffers.  :class:`WorkerFailure` and ``None`` still use
    ``gather_object`` directly.
    """
    _, dist = _torch_modules()

    local_wire: ResultWireMeta | WorkerFailure | None
    local_frames = b""
    local_mp4 = b""
    if isinstance(local, Result):
        frames_n = len(local.frames_bytes)
        mp4_n = len(local.mp4_bytes) if local.mp4_bytes is not None else 0
        if use_shm_for_results():
            assert _shm_pool is not None
            _shm_pool.write_result(
                frames_bytes=bytes(local.frames_bytes),
                mp4_bytes=local.mp4_bytes,
            )
            local_frames = b""
            local_mp4 = b""
        else:
            local_frames = bytes(local.frames_bytes)
            local_mp4 = (
                bytes(local.mp4_bytes) if local.mp4_bytes is not None else b""
            )
        local_wire = ResultWireMeta(
            meta=result_to_meta(local),
            frames_nbytes=frames_n,
            mp4_nbytes=mp4_n,
        )
    else:
        local_wire = local

    if rank == dst:
        wire_bucket: list[Any] = [None] * world_size
        dist.gather_object(local_wire, wire_bucket, dst=dst, group=group)
        out: list["Result | WorkerFailure | None"] = []
        for slot, cell in enumerate(wire_bucket):
            if cell is None:
                out.append(None)
            elif isinstance(cell, WorkerFailure):
                out.append(cell)
            elif isinstance(cell, ResultWireMeta):
                if use_shm_for_results():
                    assert _shm_pool is not None
                    ref = ResultShmRef(
                        meta=cell.meta,
                        src_rank=slot,
                        slot_id=0,
                        frames_nbytes=cell.frames_nbytes,
                        mp4_nbytes=cell.mp4_nbytes,
                    )
                    out.append(_shm_pool.read_as_result(ref))
                elif slot == rank:
                    frames = local_frames
                    mp4 = local_mp4
                    out.append(
                        meta_to_result(
                            cell.meta,
                            frames_bytes=frames,
                            mp4_bytes=mp4 if mp4 else None,
                        )
                    )
                else:
                    frames = _recv_payload_bytes(
                        cell.frames_nbytes,
                        src=slot,
                        group=group,
                        device=device,
                    )
                    mp4 = _recv_payload_bytes(
                        cell.mp4_nbytes,
                        src=slot,
                        group=group,
                        device=device,
                    )
                    out.append(
                        meta_to_result(
                            cell.meta,
                            frames_bytes=frames,
                            mp4_bytes=mp4 if mp4 else None,
                        )
                    )
            else:
                raise RuntimeError(
                    f"gather_results_to_rank0: unexpected payload "
                    f"{type(cell).__name__} from rank {slot}"
                )
        return out

    dist.gather_object(local_wire, None, dst=dst, group=group)
    if isinstance(local, Result) and not use_shm_for_results():
        _send_payload_bytes(local_frames, dst=dst, group=group, device=device)
        _send_payload_bytes(local_mp4, dst=dst, group=group, device=device)
    return []


# ----------------------------------------------------------------------
# Point-to-point object exchange (used by the async DP dispatcher).
#
# Small control messages (``WorkUnit``, ``Shutdown``, ``WorkerFailure``)
# travel as compact pickles.  :class:`Result` is split: metadata goes as a
# tiny pickle and ``frames_bytes`` / ``mp4_bytes`` follow as raw ``uint8``
# tensors without ``list()`` / ``tolist()`` conversion.
#
# Wire layout (all messages start with a 4×int64 header on ``group``):
#
#     [wire_kind, meta_nbytes, bulk_nbytes, aux_nbytes]
#
# * ``wire_kind == 0`` (pickle): ``meta`` holds the full pickled object;
#   ``bulk`` and ``aux`` are zero.
# * ``wire_kind == 1`` (result): ``meta`` is a pickled :class:`ResultMeta`;
#   ``bulk`` is ``len(frames_bytes)``; ``aux`` is ``len(mp4_bytes)`` (0
#   when absent).  ``bulk`` and (if non-zero) ``aux`` byte payloads follow
#   as separate ``uint8`` tensor sends pinned to the same source rank.
# ----------------------------------------------------------------------

WIRE_KIND_PICKLE = 0
WIRE_KIND_RESULT = 1
WIRE_KIND_RESULT_SHM = 2


def _resolve_device(device: "torch.device | str | None" = None) -> "torch.device":
    torch, _ = _torch_modules()
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _pickle_blob(obj: Any) -> bytes:
    import pickle  # noqa: WPS433

    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _unpickle_blob(blob: bytes) -> Any:
    import pickle  # noqa: WPS433

    return pickle.loads(blob)


def _bytes_to_uint8_tensor(
    blob: bytes, *, device: "torch.device | str | None" = None
) -> "torch.Tensor":
    """Copy ``blob`` into a contiguous ``uint8`` tensor suitable for ``dist.send``."""
    torch, _ = _torch_modules()
    dev = _resolve_device(device)
    if not blob:
        return torch.empty(0, dtype=torch.uint8, device=dev)
    # ``torch.frombuffer`` refuses immutable ``bytes`` (PyTorch warns and the
    # view is technically writable-through).  Copy into a ``bytearray`` first,
    # then ``clone`` so the send buffer is owned by the returned tensor and
    # outlives the temporary wrapper.
    writable = bytearray(blob)
    return torch.frombuffer(writable, dtype=torch.uint8).clone().to(dev)


def _uint8_tensor_to_bytes(buf: "torch.Tensor") -> bytes:
    return buf.detach().cpu().numpy().tobytes()


def _send_header_and_meta(
    *,
    wire_kind: int,
    meta_blob: bytes,
    bulk_nbytes: int,
    aux_nbytes: int,
    dst: int,
    group: "ProcessGroup | None",
    device: "torch.device | str | None",
) -> None:
    torch, dist = _torch_modules()
    dev = _resolve_device(device)
    header = torch.tensor(
        [int(wire_kind), len(meta_blob), int(bulk_nbytes), int(aux_nbytes)],
        dtype=torch.long,
        device=dev,
    )
    dist.send(header, dst=dst, group=group)
    if meta_blob:
        dist.send(_bytes_to_uint8_tensor(meta_blob, device=dev), dst=dst, group=group)


def _recv_header_and_meta(
    *,
    src: int,
    group: "ProcessGroup | None",
    device: "torch.device | str | None",
) -> tuple[int, bytes, int, int]:
    torch, dist = _torch_modules()
    dev = _resolve_device(device)
    header = torch.empty(4, dtype=torch.long, device=dev)
    dist.recv(header, src=int(src), group=group)
    wire_kind, meta_n, bulk_n, aux_n = (int(x) for x in header.tolist())
    meta_blob = b""
    if meta_n > 0:
        meta_buf = torch.empty(meta_n, dtype=torch.uint8, device=dev)
        dist.recv(meta_buf, src=int(src), group=group)
        meta_blob = _uint8_tensor_to_bytes(meta_buf)
    return wire_kind, meta_blob, bulk_n, aux_n


def _send_payload_bytes(
    blob: bytes,
    *,
    dst: int,
    group: "ProcessGroup | None",
    device: "torch.device | str | None",
) -> None:
    if not blob:
        return
    torch, dist = _torch_modules()
    dist.send(_bytes_to_uint8_tensor(blob, device=device), dst=dst, group=group)


def _recv_payload_bytes(
    nbytes: int,
    *,
    src: int,
    group: "ProcessGroup | None",
    device: "torch.device | str | None",
) -> bytes:
    if nbytes <= 0:
        return b""
    torch, dist = _torch_modules()
    dev = _resolve_device(device)
    body = torch.empty(int(nbytes), dtype=torch.uint8, device=dev)
    dist.recv(body, src=int(src), group=group)
    return _uint8_tensor_to_bytes(body)


def _send_result_shm_pt2pt(
    result: Result,
    *,
    src_rank: int,
    dst: int,
    group: "ProcessGroup | None",
    device: "torch.device | str | None",
) -> None:
    assert _shm_pool is not None
    _shm_pool.write_result(
        frames_bytes=bytes(result.frames_bytes),
        mp4_bytes=result.mp4_bytes,
    )
    mp4_n = len(result.mp4_bytes) if result.mp4_bytes is not None else 0
    ref = ResultShmRef(
        meta=result_to_meta(result),
        src_rank=int(src_rank),
        slot_id=0,
        frames_nbytes=len(result.frames_bytes),
        mp4_nbytes=mp4_n,
    )
    _send_header_and_meta(
        wire_kind=WIRE_KIND_RESULT_SHM,
        meta_blob=_pickle_blob(ref),
        bulk_nbytes=0,
        aux_nbytes=0,
        dst=dst,
        group=group,
        device=device,
    )


def _recv_result_shm_pt2pt(
    meta_blob: bytes,
    *,
    src: int,
    group: "ProcessGroup | None",
    device: "torch.device | str | None",
) -> Result:
    del src, group, device
    assert _shm_pool is not None
    ref: ResultShmRef = _unpickle_blob(meta_blob)
    return _shm_pool.read_as_result(ref)


def _send_result_pt2pt(
    result: Result,
    *,
    dst: int,
    group: "ProcessGroup | None",
    device: "torch.device | str | None",
) -> None:
    frames = bytes(result.frames_bytes)
    mp4 = bytes(result.mp4_bytes) if result.mp4_bytes is not None else b""
    _send_header_and_meta(
        wire_kind=WIRE_KIND_RESULT,
        meta_blob=_pickle_blob(result_to_meta(result)),
        bulk_nbytes=len(frames),
        aux_nbytes=len(mp4),
        dst=dst,
        group=group,
        device=device,
    )
    _send_payload_bytes(frames, dst=dst, group=group, device=device)
    _send_payload_bytes(mp4, dst=dst, group=group, device=device)


def _recv_result_pt2pt(
    meta_blob: bytes,
    bulk_n: int,
    aux_n: int,
    *,
    src: int,
    group: "ProcessGroup | None",
    device: "torch.device | str | None",
) -> Result:
    meta: ResultMeta = _unpickle_blob(meta_blob)
    frames = _recv_payload_bytes(bulk_n, src=src, group=group, device=device)
    mp4 = _recv_payload_bytes(aux_n, src=src, group=group, device=device)
    return meta_to_result(
        meta,
        frames_bytes=frames,
        mp4_bytes=mp4 if mp4 else None,
    )


def _send_pickle_pt2pt(
    obj: Any,
    *,
    dst: int,
    group: "ProcessGroup | None",
    device: "torch.device | str | None",
) -> None:
    blob = _pickle_blob(obj)
    _send_header_and_meta(
        wire_kind=WIRE_KIND_PICKLE,
        meta_blob=blob,
        bulk_nbytes=0,
        aux_nbytes=0,
        dst=dst,
        group=group,
        device=device,
    )


def _recv_pickle_pt2pt(
    meta_blob: bytes,
    *,
    src: int,
    group: "ProcessGroup | None",
    device: "torch.device | str | None",
) -> Any:
    del src, group, device  # meta already received; bodies not used
    return _unpickle_blob(meta_blob)


def send_object_pt2pt(
    obj: Any,
    *,
    dst: int,
    group: "ProcessGroup | None" = None,
    device: "torch.device | str | None" = None,
    src_rank: int | None = None,
) -> None:
    """Send a single Python object to rank ``dst`` over ``group``.

    Blocking. The receiving rank must call :func:`recv_object_any_src`
    (or, for a known-source recv, mirror the protocol with
    ``dist.recv``).

    :class:`Result` payloads use the split metadata + raw-bytes path,
    or SHM when :func:`use_shm_for_results` is active.
    """
    if isinstance(obj, Result) and use_shm_for_results():
        _, dist = _torch_modules()
        rank = int(src_rank if src_rank is not None else dist.get_rank(group=group))
        _send_result_shm_pt2pt(
            obj, src_rank=rank, dst=dst, group=group, device=device
        )
        return
    if isinstance(obj, Result):
        _send_result_pt2pt(obj, dst=dst, group=group, device=device)
        return
    _send_pickle_pt2pt(obj, dst=dst, group=group, device=device)


def _recv_header_from_src(
    *,
    src: int,
    group: "ProcessGroup | None",
    device: "torch.device | str | None",
) -> tuple[int, bytes, int, int]:
    """Like :func:`_recv_header_and_meta` but header recv is pinned to ``src``."""
    torch, dist = _torch_modules()
    dev = _resolve_device(device)
    header = torch.empty(4, dtype=torch.long, device=dev)
    dist.recv(header, src=int(src), group=group)
    wire_kind, meta_n, bulk_n, aux_n = (int(x) for x in header.tolist())
    meta_blob = b""
    if meta_n > 0:
        meta_buf = torch.empty(meta_n, dtype=torch.uint8, device=dev)
        dist.recv(meta_buf, src=int(src), group=group)
        meta_blob = _uint8_tensor_to_bytes(meta_buf)
    return wire_kind, meta_blob, bulk_n, aux_n


def recv_object_from_src(
    *,
    src: int,
    group: "ProcessGroup | None" = None,
    device: "torch.device | str | None" = None,
) -> Any:
    """Receive one object from a known ``src`` rank (blocking)."""
    wire_kind, meta_blob, bulk_n, aux_n = _recv_header_from_src(
        src=int(src), group=group, device=device
    )
    if wire_kind == WIRE_KIND_RESULT_SHM:
        return _recv_result_shm_pt2pt(
            meta_blob, src=int(src), group=group, device=device
        )
    if wire_kind == WIRE_KIND_RESULT:
        return _recv_result_pt2pt(
            meta_blob,
            bulk_n,
            aux_n,
            src=int(src),
            group=group,
            device=device,
        )
    if wire_kind == WIRE_KIND_PICKLE:
        return _recv_pickle_pt2pt(
            meta_blob, src=int(src), group=group, device=device
        )
    raise RuntimeError(
        f"recv_object_from_src: unknown wire_kind={wire_kind} from rank {src}"
    )


def release_worker_slot(
    *,
    worker_rank: int,
    slot_id: int,
    dst: int,
    group: "ProcessGroup | None" = None,
    device: "torch.device | str | None" = None,
) -> None:
    """Tell a worker its SHM slot may be reused."""
    send_object_pt2pt(
        SlotRelease(rank=int(worker_rank), slot_id=int(slot_id)),
        dst=int(dst),
        group=group,
        device=device,
    )


def recv_object_any_src(
    *,
    group: "ProcessGroup | None" = None,
    device: "torch.device | str | None" = None,
) -> tuple[int, Any]:
    """Receive a single Python object from any rank in ``group``.

    Returns ``(src_rank, obj)``. Blocking. ``group`` MUST be a Gloo
    process group; NCCL does not support ANY-source recv.

    The body recvs are pinned to the source rank resolved on the header
    recv, so two concurrent senders cannot race the protocol.
    """
    torch, dist = _torch_modules()
    dev = _resolve_device(device)

    header = torch.empty(4, dtype=torch.long, device=dev)
    src_rank = dist.recv(header, src=None, group=group)
    wire_kind, meta_n, bulk_n, aux_n = (int(x) for x in header.tolist())

    meta_blob = b""
    if meta_n > 0:
        meta_buf = torch.empty(meta_n, dtype=torch.uint8, device=dev)
        body_src = dist.recv(meta_buf, src=int(src_rank), group=group)
        if int(body_src) != int(src_rank):
            raise RuntimeError(
                f"recv_object_any_src: header came from rank {src_rank} "
                f"but meta body came from rank {body_src}"
            )
        meta_blob = _uint8_tensor_to_bytes(meta_buf)

    if wire_kind == WIRE_KIND_RESULT_SHM:
        obj = _recv_result_shm_pt2pt(
            meta_blob,
            src=int(src_rank),
            group=group,
            device=device,
        )
        return int(src_rank), obj
    if wire_kind == WIRE_KIND_RESULT:
        obj = _recv_result_pt2pt(
            meta_blob,
            bulk_n,
            aux_n,
            src=int(src_rank),
            group=group,
            device=device,
        )
        return int(src_rank), obj
    if wire_kind == WIRE_KIND_PICKLE:
        return int(src_rank), _recv_pickle_pt2pt(
            meta_blob, src=int(src_rank), group=group, device=device
        )
    raise RuntimeError(
        f"recv_object_any_src: unknown wire_kind={wire_kind} from rank {src_rank}"
    )
