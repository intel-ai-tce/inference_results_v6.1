"""
Phase 2b Step 2 — slicing `LoadPlanner` for capped/sharded sparse checkpoint load.

The NVIDIA `dlrm-v3-checkpoint` saves the sparse embedding tables at their
production sizes:

    sparse_dict._embedding_collection.embeddings.item_id.weight       : (1_000_000_000, 512)  fp16  (row-sharded into 8 chunks of (125_000_000, 512))
    sparse_dict._embedding_collection.embeddings.user_id.weight       : (10_000_000, 512)     fp16  (col-sharded into 4 chunks of (10_000_000, 128))
    sparse_dict._embedding_collection.embeddings.item_category_id.weight : (128, 512)         fp16  (col-sharded into 4 chunks of (128, 128))

For Phase 2b bring-up on a single MI355 we cap `item_id` / `user_id` rows via
`DLRM_SPARSE_MAX_HASH_SIZE`. PyTorch's `DefaultLoadPlanner` refuses any
shape mismatch:

    raise ValueError(f"Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}")

`SlicingLoadPlanner` removes that strict check and instead lets the existing
chunk-overlap algorithm (`create_read_items_for_chunk_list`) narrow each
storage chunk down to the live tensor extent. The live tensor is treated as
one local chunk at offset `(0, 0, ...)` with sizes `obj.size()`; storage
chunks that fall entirely outside the live region (e.g. rows
[125_000_000, 250_000_000) of `item_id` when the live cap is 100_000) are
skipped, and partial-overlap chunks are read for just the overlapping window
into the correct dest offset.

This means:

  * `item_id` capped to 100k → single ~100 MB read from chunk 0 of
    `__0_0.distcp` (vs the full 130 GB it would normally pull).
  * `user_id` capped to 100k → 4 column reads of ~25 MB each (one per saved
    column shard), reconstructing all 512 dims of the first 100k rows.
  * `item_category_id` (already 128 rows) → unchanged, all 4 column chunks
    loaded normally.

This planner is opt-in via `DLRM_SPARSE_SLICE_CKPT=1` and is auto-enabled by
`load_sparse_checkpoint_sliced()` when the live state-dict actually disagrees
with the saved metadata.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.planner_helpers import (
    _create_chunk_list,
    _create_read_items,
    create_read_items_for_chunk_list,
)


logger: logging.Logger = logging.getLogger(__name__)


StorageOffsetMap = Dict[str, Tuple[int, ...]]


def _shapes_narrow_compatible(live: torch.Size, saved: torch.Size) -> bool:
    """True iff live can be reached from saved by per-dim narrowing.

    Same rank, every live dim <= saved dim. Used to gate the slicing path so we
    don't silently swallow an unrelated shape mismatch (which still deserves a
    loud error).
    """
    if len(live) != len(saved):
        return False
    return all(int(l) <= int(s) for l, s in zip(live, saved))


class SlicingLoadPlanner(DefaultLoadPlanner):
    """Drop-in replacement for ``DefaultLoadPlanner`` that narrows saved tensors.

    For each ``fqn`` in ``state_dict``:

    * If the live tensor's size equals the saved size: fall through to the
      default ``_create_read_items`` (same chunk-overlap math, identical
      reads).
    * If the live tensor is rank-equal and every dim is ``<=`` the saved dim:
      build a single ``ChunkStorageMetadata`` covering the live extent at
      the configured *storage offset* (default ``(0, 0, ...)``; rank-shifted
      for multi-GPU shard mode) and hand it to
      ``create_read_items_for_chunk_list``, which intersects it against the
      saved chunks and emits one read per overlapping chunk.
    * Otherwise: raise ``ValueError`` (mismatched rank or a dim that grows).

    Bytes / unsupported metadata types are passed through unchanged.

    Per-FQN ``storage_offsets`` lets each rank read its own row-shard
    directly from the saved checkpoint. For example, with
    ``storage_offsets={"...item_id.weight": (125_000_000, 0)}`` and a live
    tensor sized ``(125_000_000, 512)``, the planner emits a single read of
    chunk 1 (saved rows ``[125M, 250M)``) into local rows ``[0, 125M)``.

    Notes
    -----
    * Only meaningful for *plain* ``torch.Tensor`` parameters. ``DTensor`` and
      ``ShardedTensor`` are forwarded to ``_create_read_items`` unchanged
      (Phase 2b sparse arch lives on a single rank with plain tensors).
    * Honors ``self.allow_partial_load`` for missing keys when set; otherwise
      raises like ``DefaultLoadPlanner``.
    * The chunk-overlap math expects ``storage_offsets[i] + live_size[i] <=
      saved_size[i]``; otherwise the planner falls back to a zero offset and
      logs a warning (avoids silent over-read past the saved tensor).
    """

    def __init__(
        self,
        *args,
        storage_offsets: Optional[StorageOffsetMap] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.storage_offsets: StorageOffsetMap = storage_offsets or {}
        # FQNs where ReadItem.dest_index has a non-zero offset (because we
        # told the chunk-overlap algorithm our local chunk starts at the
        # global shard origin). The default ``find_state_dict_object`` path
        # treats those as ShardedTensor shard offsets and raises, so we have
        # to bypass that check below in ``lookup_tensor``.
        self._sharded_plain_fqns: set[str] = set(self.storage_offsets.keys())

    def lookup_tensor(self, index):  # type: ignore[override]
        """Return the plain live tensor regardless of ``index.offset``.

        ``DefaultLoadPlanner.lookup_tensor`` defers to
        ``find_state_dict_object`` which calls ``find_tensor_shard``. The
        latter rejects non-zero ``index.offset`` on plain (non-ShardedTensor)
        tensors. Our sharded path intentionally sets non-zero chunk offsets
        so the storage-side chunk-overlap math hits the right disk chunk —
        but the dest tensor is still a plain ``torch.Tensor``. Return it
        unwrapped; ``transform_tensor``'s ``narrow_tensor_by_index`` then
        slices it with ``dest_offsets`` (always local-space) and ``lengths``.
        """
        if index.fqn in self._sharded_plain_fqns:
            obj = self.state_dict[index.fqn]
            if isinstance(obj, torch.Tensor):
                return obj
        return super().lookup_tensor(index)

    def create_local_plan(self) -> LoadPlan:
        if self.metadata is None:
            raise AssertionError("SlicingLoadPlanner.create_local_plan: metadata is None")
        requests: List = []
        narrowed: int = 0
        for fqn, obj in self.state_dict.items():
            if fqn not in self.metadata.state_dict_metadata:
                if getattr(self, "allow_partial_load", False):
                    continue
                raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}")
            md = self.metadata.state_dict_metadata[fqn]
            if isinstance(md, BytesStorageMetadata) or not isinstance(
                obj, torch.Tensor
            ):
                requests += _create_read_items(fqn, md, obj)
                continue
            if not isinstance(md, TensorStorageMetadata):
                requests += _create_read_items(fqn, md, obj)
                continue
            live_size = obj.size()
            saved_size = md.size
            row_offset = self.storage_offsets.get(fqn)
            if tuple(live_size) == tuple(saved_size) and row_offset is None:
                requests += _create_read_items(fqn, md, obj)
                continue
            if not _shapes_narrow_compatible(live_size, saved_size):
                raise ValueError(
                    f"SlicingLoadPlanner cannot narrow {fqn}: "
                    f"live={tuple(live_size)} vs saved={tuple(saved_size)} "
                    "(rank must match and live <= saved per dim)"
                )
            if row_offset is not None:
                if len(row_offset) != len(live_size):
                    raise ValueError(
                        f"SlicingLoadPlanner storage_offsets[{fqn}]={row_offset!r} "
                        f"must have rank {len(live_size)} (got {len(row_offset)})"
                    )
                bad = [
                    (i, int(off), int(s), int(ss))
                    for i, (off, s, ss) in enumerate(zip(row_offset, live_size, saved_size))
                    if int(off) < 0 or int(off) + int(s) > int(ss)
                ]
                if bad:
                    logger.warning(
                        "[phase2b-slice] %s: requested storage_offsets=%s + live=%s "
                        "exceeds saved=%s at dims %s; falling back to zero offset",
                        fqn,
                        tuple(int(x) for x in row_offset),
                        tuple(live_size),
                        tuple(saved_size),
                        [b[0] for b in bad],
                    )
                    row_offset = None
            if row_offset is None:
                local_chunks = _create_chunk_list(obj)
                tag = ""
            else:
                local_chunks = [
                    ChunkStorageMetadata(
                        offsets=torch.Size(tuple(int(x) for x in row_offset)),
                        sizes=torch.Size(tuple(int(x) for x in live_size)),
                    )
                ]
                tag = f" @ storage_offset={tuple(int(x) for x in row_offset)}"
            logger.warning(
                "[phase2b-slice] narrowing %s saved=%s -> live=%s%s",
                fqn,
                tuple(saved_size),
                tuple(live_size),
                tag,
            )
            requests += create_read_items_for_chunk_list(fqn, md, local_chunks)
            narrowed += 1
        if narrowed:
            logger.warning(
                "[phase2b-slice] SlicingLoadPlanner narrowed %d tensor(s); "
                "reads will be restricted to the live extent",
                narrowed,
            )
        self.plan = LoadPlan(requests)
        return self.plan


def slice_planner_enabled() -> bool:
    """Whether to install ``SlicingLoadPlanner`` on sparse loads.

    Defaults:
      * ``DLRM_SPARSE_SLICE_CKPT=1`` → enabled
      * ``DLRM_SPARSE_SLICE_CKPT=0`` → disabled
      * unset but ``DLRM_SPARSE_MAX_HASH_SIZE`` is set → enabled (caps imply slicing)
      * unset but ``DLRM_SPARSE_WORLD`` > 1 → enabled (sharding implies slicing)
      * else → disabled
    """
    raw = os.environ.get("DLRM_SPARSE_SLICE_CKPT", "").strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    if os.environ.get("DLRM_SPARSE_MAX_HASH_SIZE", "").strip():
        return True
    try:
        if int(os.environ.get("DLRM_SPARSE_WORLD", "1")) > 1:
            return True
    except ValueError:
        pass
    return False


def _sharded_fqn_patterns() -> List[str]:
    """Substrings; an FQN is sharded iff any pattern is a substring.

    Defaults to row-sharding only ``item_id`` (the 1B-row table in the saved
    NVIDIA checkpoint). Override with ``DLRM_SPARSE_SHARDED_FQNS=item_id,user_id``.
    """
    raw = os.environ.get("DLRM_SPARSE_SHARDED_FQNS", "").strip()
    if not raw:
        return ["item_id.weight"]
    return [p.strip() for p in raw.split(",") if p.strip()]


def _is_sharded_fqn(fqn: str, patterns: Optional[List[str]] = None) -> bool:
    patterns = patterns if patterns is not None else _sharded_fqn_patterns()
    return any(p in fqn for p in patterns)


def storage_offsets_for_shard(
    state_dict: Dict[str, torch.Tensor],
    metadata,
    rank: int,
    world: int,
    sharded_fqn_patterns: Optional[List[str]] = None,
) -> StorageOffsetMap:
    """Build the per-FQN row offset map for an N-way row-sharded sparse load.

    For each FQN whose live tensor is narrower than the saved tensor and whose
    name matches one of ``sharded_fqn_patterns``: this rank reads rows
    ``[rank * live_rows, (rank+1) * live_rows)``. Other narrowed FQNs (e.g.
    ``user_id`` / ``item_category_id`` left replicated) keep ``offset = 0``.

    Caller is responsible for ensuring ``rank * live_rows <= saved_rows`` per
    sharded table; on overflow ``SlicingLoadPlanner`` falls back to a zero
    offset and warns.
    """
    if world <= 1:
        return {}
    patterns = sharded_fqn_patterns if sharded_fqn_patterns is not None else _sharded_fqn_patterns()
    out: StorageOffsetMap = {}
    for fqn, obj in state_dict.items():
        if fqn not in metadata.state_dict_metadata:
            continue
        if not _is_sharded_fqn(fqn, patterns):
            continue
        md = metadata.state_dict_metadata[fqn]
        if not isinstance(md, TensorStorageMetadata):
            continue
        if not isinstance(obj, torch.Tensor):
            continue
        live_size = obj.size()
        saved_size = md.size
        if tuple(live_size) == tuple(saved_size):
            continue
        if not _shapes_narrow_compatible(live_size, saved_size):
            continue
        # Row-sharding: dim-0 offset = rank * live_rows; other dims stay at 0.
        rows = int(live_size[0])
        offset = [rank * rows] + [0] * (len(live_size) - 1)
        out[fqn] = tuple(offset)
    return out


def make_planner_from_env(
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    metadata=None,
) -> SlicingLoadPlanner:
    """Build a ``SlicingLoadPlanner`` configured from env.

    When ``DLRM_SPARSE_WORLD > 1`` and ``state_dict`` + ``metadata`` are
    supplied, pre-computes ``storage_offsets`` for sharded FQNs unless
    Step 3b's replicated-cap path is active (the default when
    ``DLRM_SPARSE_WORLD > 1``).

    Phase 2b Step 3b: under the replicated-cap path each rank's live
    table is sized to the full ``W*S`` global cap (see
    ``inference_modules.sparse_max_hash_size``) and the planner reads
    rows ``[0, W*S)`` on every rank — i.e. a zero ``storage_offset``
    for every FQN. That guarantees every worker has a full replicated
    view of the capped sparse, so the rank-local lookup in
    ``ec_patched_forward_wo_embedding_copy`` is globally correct
    without any cross-rank collective.

    Set ``DLRM_SPARSE_REPLICATE=0`` to skip the cap bump and use the
    Step-3a per-rank-shard storage (rank-local clamp → divergent
    predictions across ranks; intended for ablation).
    """
    try:
        world = max(1, int(os.environ.get("DLRM_SPARSE_WORLD", "1")))
        rank = max(0, int(os.environ.get("DLRM_SPARSE_RANK", "0")))
    except ValueError:
        world, rank = 1, 0
    offsets: StorageOffsetMap = {}
    if world > 1 and state_dict is not None and metadata is not None:
        try:
            from sparse_routing import replicate_enabled  # noqa: WPS433
            replicate = replicate_enabled()
        except Exception:
            replicate = False
        if replicate:
            logger.warning(
                "[phase2b-3b] rank=%d/%d replicated-cap load: every rank "
                "reads rows [0, W*S) for sharded FQNs (no per-rank "
                "storage offsets, no cross-rank collective at lookup)",
                rank,
                world,
            )
        else:
            offsets = storage_offsets_for_shard(state_dict, metadata, rank, world)
            if offsets:
                logger.warning(
                    "[phase2b-slice] rank=%d/%d shard offsets: %s "
                    "(DLRM_SPARSE_REPLICATE=0; predictions will be "
                    "rank-divergent until Step 3c routing lands)",
                    rank,
                    world,
                    offsets,
                )
    return SlicingLoadPlanner(storage_offsets=offsets)
