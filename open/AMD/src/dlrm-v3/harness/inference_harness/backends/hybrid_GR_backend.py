"""
Hybrid GR (Generative Recommender) backend for DLRM inference.

Implements a custom backend that combines NVE embeddings for large tables
with optimized dense model inference, supporting distributed embedding lookups.
"""

from inference_harness.backends.base import DLRMBackend
from inference_harness.accuracy_safety import (
    should_run_optimized_compare,
    should_restore_inference_accuracy_targets,
    should_use_optimized_embed,
    should_use_uniform_targets_metadata,
)
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from generative_recommenders.modules.multitask_module import MultitaskTaskType
from torchrec.modules.embedding_configs import EmbeddingConfig
from typing import Dict, Tuple, List, Set, Optional
from dataclasses import dataclass, field
from generative_recommenders.modules.stu import STULayerConfig, STUStack
from inference_harness.backends.model.dlrm_hstu_custom import DlrmHSTUCustom
from generative_recommenders.dlrm_v3.inference.inference_modules import set_is_inference
from .ops import NVEEmbeddingCollection, NVEEmbeddingCollectionConfig
from inference_harness.backends.model.STU_custom import STULayerCustom
from torchrec.modules.embedding_modules import EmbeddingCollection
from generative_recommenders.modules.stu import STULayerConfig
from generative_recommenders.dlrm_v3.inference.inference_modules import move_sparse_output_to_device
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from generative_recommenders.dlrm_v3.datasets.dataset import Samples
from inference_harness.memory_trace import log_memory_phase

from typing import Union
from generative_recommenders.dlrm_v3.checkpoint import (
    load_nonsparse_checkpoint,
)
import nvtx
from torchrec.distributed.types import ShardedTensor
import torch
import pynve.torch.nve_layers as nve_layers
import pynve.nve as nve

from generative_recommenders.modules.dlrm_hstu import (
    DlrmHSTUConfig,
    SequenceEmbedding,
)

import os
import logging
# Suppress verbose logs from generative_recommenders (e.g., "Initialize HSTU module with configs...")
logging.getLogger("generative_recommenders.modules.dlrm_hstu").setLevel(logging.WARNING)
logging.getLogger("generative_recommenders.dlrm_v3.checkpoint").setLevel(logging.WARNING)

_logger = logging.getLogger(__name__)

# OOV-id guard at the embedding-lookup chokepoint (Plan 21 §0.4). The
# preprocessed dataset's last timestamp(s) carry a handful of embedding ids
# equal to exactly the table size (e.g. item_candidate_id == 1_000_000_000 for
# the 1e9-row table), i.e. one past the last valid row. The QSL-side clamp in
# ``mlperf_streaming_qsl._LazyTimestampSamples.get()`` is supposed to neutralize
# these, but it only protects KJTs built through that sampler path; any other
# producer (or a stale/uninitialized flag in the worker) leaks the raw id into
# ``model_impl.preprocess`` -> ``nve::GatherKeysAndDataPtrs``, which faults the
# queue with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION. Re-clamping here, on the
# device KJT that EVERY predict() path funnels through, makes the guard
# path-independent. min-clamp (id -> table_size - 1) is correct for a
# *performance* run (LoadGen does not score outputs); an *accuracy* run should
# map the OOV id to a dedicated zero row instead (revisit if/when accuracy is
# re-certified on this stack).
_CLAMP_OOB_IDS: bool = os.environ.get("DLRM_CLAMP_OOB_IDS", "0") == "1"
_FEATURE_CLAMP_MAX: Dict[str, int] = {
    "item_id": 1_000_000_000 - 1,
    "item_candidate_id": 1_000_000_000 - 1,
    "item_category_id": 128 - 1,
    "item_candidate_category_id": 128 - 1,
}
_clamp_logged = False


# Plan 51 — int8 embedding-table FAKE-QUANT emulation (the M1 GAUC-cert tool).
# M0 killed fp8/e4m3 (~2.6% recon, ~15x bf16); int8-per-row (~0.6%) is the surviving
# 1-byte candidate. This injects the *numerics* of a per-row-absmax int8 stored table
# (gather int8 -> dequant x scale -> bf16) as a fake-quant on the gather output for the
# item_id-backed keys, so run_accuracy.sh could certify GAUC *before* the real NVE
# store+dequant path existed. The real path now lives in int8_embed.py + ops.py +
# the loader (DLRM_NVE_INT8_GATHER=1); this emulation is kept for A/B + debugging and
# is MUTUALLY EXCLUSIVE with the real path (applying both would double-quantize).
# No-op unless DLRM_NVE_INT8_FAKEQUANT is set.
#   unset / "0"      -> off
#   "1" / "item_id"  -> item_id + item_candidate_id (both gather the [1e9,512] item_id table)
#   "a,b,..."        -> explicit key set
_INT8_FQ_RAW: str = os.environ.get("DLRM_NVE_INT8_FAKEQUANT", "0").strip().lower()
if _INT8_FQ_RAW in ("", "0", "false", "no"):
    _INT8_GATHER_KEYS: Set[str] = set()
elif _INT8_FQ_RAW in ("1", "item_id", "true", "yes"):
    _INT8_GATHER_KEYS = {"item_id", "item_candidate_id"}
else:
    _INT8_GATHER_KEYS = {t.strip() for t in _INT8_FQ_RAW.split(",") if t.strip()}
if _INT8_GATHER_KEYS and os.environ.get("DLRM_NVE_INT8_GATHER", "0").strip().lower() in ("1", "item_id", "true", "yes"):
    raise ValueError(
        "DLRM_NVE_INT8_FAKEQUANT and DLRM_NVE_INT8_GATHER are mutually exclusive "
        "(the real packed path already dequantizes; the fake-quant would double-quantize)")
_int8_logged = False


def _int8_fakequant(emb: torch.Tensor) -> torch.Tensor:
    """Per-row symmetric absmax int8 quant->dequant, returned as bf16.

    Emulates a stored int8 table row: scale = absmax/127 (per row), q =
    round(x/scale) clamped to [-127,127], dequant = q*scale. Math in fp32 for a
    faithful reconstruction; the network consumes bf16, so cast at the end (matching
    the real gather+dequant output dtype). An all-zero (untrained) row has absmax=0;
    the scale floor avoids div-by-zero and the row dequants back to 0.
    """
    x = emb.to(torch.float32)
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 127.0).clamp_min(1e-12)
    q = torch.round(x / scale).clamp_(-127.0, 127.0)
    return (q * scale).to(torch.bfloat16)


def _clamp_kjt_lookup_ids(kjt: KeyedJaggedTensor) -> None:
    """In-place clamp each embedding-lookup feature in ``kjt`` to its table bounds.

    KJT values are laid out key-major (each key's values for the whole batch are
    contiguous); ``length_per_key()`` gives the per-key value count. Keys without
    a configured bound are left untouched. No-op unless ``DLRM_CLAMP_OOB_IDS=1``.
    """
    if not _CLAMP_OOB_IDS:
        return
    values = kjt.values()
    offset = 0
    for key, lpk in zip(kjt.keys(), kjt.length_per_key()):
        cap = _FEATURE_CLAMP_MAX.get(key)
        if cap is not None and lpk > 0:
            values[offset:offset + lpk].clamp_(max=cap)
        offset += lpk


def is_sparse_key(k: str, v: torch.Tensor) -> bool:
    return isinstance(v, ShardedTensor) or "embedding_collection" in k


from torch.distributed.checkpoint.stateful import Stateful


class SparseState(Stateful):
    def __init__(self, model: torch.nn.Module, sparse_tensor_keys: Set[str]) -> None:
        self.model = model
        self.sparse_tensor_keys = sparse_tensor_keys

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out_dict: Dict[str, torch.Tensor] = {}
        is_sharded_tensor: Optional[bool] = None
        for k, v in self.model.state_dict().items():
            if k in self.sparse_tensor_keys:
                if is_sharded_tensor is None:
                    is_sharded_tensor = isinstance(v, ShardedTensor)
                assert is_sharded_tensor == isinstance(v, ShardedTensor)
                out_dict[k] = v
        return out_dict

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        assert not incompatible_keys.unexpected_keys


class SparseStateRankRest(Stateful):
    def __init__(self, model: torch.nn.Module, sparse_tensor_keys: Set[str]) -> None:
        self.model = model
        self.sparse_tensor_keys = sparse_tensor_keys

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out_dict: Dict[str, torch.Tensor] = {}
        is_sharded_tensor: Optional[bool] = None
        for k, v in self.model.state_dict().items():
            if k in self.sparse_tensor_keys:
                if is_sharded_tensor is None:
                    is_sharded_tensor = isinstance(v, ShardedTensor)
                assert is_sharded_tensor == isinstance(v, ShardedTensor)
                out_dict[k] = v
        return out_dict

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        assert not incompatible_keys.unexpected_keys


@dataclass
class HybridGRBackendConfig:
    batch_size: int = 1
    perf_mode: str = "performance"
    use_custom_stu: bool = True
    use_nve: bool = True
    use_multi_gpu: bool = False
    use_mpi: bool = False  # Use MPI-based multi-GPU when both use_multi_gpu and use_mpi are True
    gpu_cache_size_in_gigabytes: float = 10.0
    device_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    nve_memblock: nve.NVLMemBlock = None


@dataclass
class CustomJaggedTensor:
    values: torch.Tensor
    lengths: torch.Tensor
    offsets: torch.Tensor
    max_length: int
    embeddings: torch.Tensor


class HybridGRBackend(DLRMBackend):
    def __init__(self, model_name: str, device: torch.device = torch.device("cuda:0")):
        super().__init__(model_name=model_name)
        self.backend = "GR"
        self.model_impl: DlrmHSTUCustom = None
        self.backend_config: HybridGRBackendConfig = None
        self.device: torch.device = device
        # Opt-in per-stage timing (embedding lookup vs HSTU forward).
        # Enabled via DLRM_STAGE_TIMING=1; uses CUDA events resolved after the
        # batching_loop's stream synchronize, so it adds no extra device syncs.
        import os as _os
        self._stage_timing = _os.environ.get("DLRM_STAGE_TIMING", "0") == "1"
        self.last_stage_events = None
        self._optimized_embed_lookup = (
            _os.environ.get("DLRM_OPTIMIZED_EMBED_LOOKUP", "0") == "1"
        )
        self._optimized_embed_compare = (
            _os.environ.get("DLRM_OPTIMIZED_EMBED_COMPARE", "0") == "1"
        )
        self._optimized_embed_compare_limit = int(
            _os.environ.get("DLRM_OPTIMIZED_EMBED_COMPARE_LIMIT", "1")
        )
        self._optimized_embed_compares = 0
        self._optimized_embed_disabled = False
        self._optimized_embed_shape_skip_logged = False
        self._uniform_targets_metadata_disabled = False
        self._nve_cache_metrics = (
            _os.environ.get("DLRM_NVE_CACHE_METRICS", "0") == "1"
        )
        self._nve_cache_metrics_interval = int(
            _os.environ.get("DLRM_NVE_CACHE_METRICS_INTERVAL", "256")
        )
        self._nve_cache_metrics_batches = 0
        self._nve_prefetch_current_batch = (
            _os.environ.get("DLRM_NVE_PREFETCH_CURRENT_BATCH", "0") == "1"
        )
        self._cache_inference_zero_payloads = (
            _os.environ.get("DLRM_CACHE_INFERENCE_ZERO_PAYLOADS", "0") == "1"
        )
        self._skip_bf16_noop_cast = (
            _os.environ.get("DLRM_SKIP_BF16_NOOP_CAST", "0") == "1"
        )
        self._uniform_targets_metadata = (
            _os.environ.get("DLRM_HSTU_UNIFORM_TARGETS_METADATA", "0") == "1"
        )
        self._server_candidate_size = 2048
        self._zero_payload_cache: Dict[Tuple[int, torch.dtype, torch.device], torch.Tensor] = {}

    def initialize(self, hstu_config: DlrmHSTUConfig, embedding_table_config: Dict[str, EmbeddingConfig], backend_config: HybridGRBackendConfig):
        self.hstu_config: DlrmHSTUConfig = hstu_config
        self.embedding_table_config: Dict[str, EmbeddingConfig] = embedding_table_config
        self.backend_config: HybridGRBackendConfig = backend_config
        self.perf_mode = self.backend_config.perf_mode
        self.is_inference = (
            self.backend_config.perf_mode == "performance"
            or os.environ.get("DLRM_ACCURACY_USE_INFERENCE_MODE", "0") == "1"
        )
        self._restore_inference_accuracy_targets = (
            should_restore_inference_accuracy_targets(
                perf_mode=self.perf_mode,
                is_inference=self.is_inference,
            )
        )
        torch.cuda.set_device(self.device)
        self._server_candidate_size = int(
            getattr(self.hstu_config, "max_num_candidates_inference", 2048)
        )
        # Determinism probe/fix (env-gated). DLRM_TORCH_DETERMINISTIC=warn logs every
        # ATen op that falls back to a nondeterministic algorithm (diagnostic, does not
        # change numerics); =1 enforces deterministic algos (raises if an op lacks a
        # deterministic impl). Used to localize the intrinsic ~2% run-to-run drift that
        # persists at batch=1 (no batch-neighbor coupling).
        _determ = os.environ.get("DLRM_TORCH_DETERMINISTIC", "0").strip().lower()
        if _determ in ("1", "true", "yes", "warn"):
            _warn_only = _determ == "warn"
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=_warn_only)
                _logger.warning(
                    "[determinism] use_deterministic_algorithms(True, warn_only=%s) "
                    "CUBLAS_WORKSPACE_CONFIG=%s device=%s",
                    _warn_only, os.environ.get("CUBLAS_WORKSPACE_CONFIG"), self.device,
                )
            except Exception as _e:  # pragma: no cover - diagnostic only
                _logger.warning("[determinism] enable failed: %r", _e)
        log_memory_phase(
            _logger,
            "backend.initialize.start",
            extra={
                "batch_size": self.backend_config.batch_size,
                "use_nve": self.backend_config.use_nve,
                "use_multi_gpu": self.backend_config.use_multi_gpu,
                "use_mpi": self.backend_config.use_mpi,
                "nve_cache_gb": self.backend_config.gpu_cache_size_in_gigabytes,
            },
        )

        self.dummy_tensor = torch.ones(1, self.backend_config.batch_size * 2048, dtype=torch.bfloat16)
        # Dummy labels and weights for warmup in accuracy mode (float32 to match _serialize_result expectations)
        self.dummy_labels = torch.ones(1, self.backend_config.batch_size * 2048, dtype=torch.float32)
        self.dummy_weights = torch.ones(1, self.backend_config.batch_size * 2048, dtype=torch.float32)
        log_memory_phase(_logger, "backend.initialize.after_dummy_tensors")

        set_is_inference(is_inference=self.is_inference)
        model_dense = DlrmHSTUCustom(
            hstu_configs=self.hstu_config,
            embedding_tables=self.embedding_table_config,
            is_dense=True,
            is_inference=self.is_inference,
        )

        # Replace STU layers with custom implementation if requested
        if self.backend_config.use_custom_stu:
            stu_module = STUStack(
                stu_list=[
                    STULayerCustom(
                        config=STULayerConfig(
                            embedding_dim=hstu_config.hstu_transducer_embedding_dim,
                            num_heads=hstu_config.hstu_num_heads,
                            hidden_dim=hstu_config.hstu_attn_linear_dim,
                            attention_dim=hstu_config.hstu_attn_qk_dim,
                            output_dropout_ratio=hstu_config.hstu_linear_dropout_rate,
                            use_group_norm=hstu_config.hstu_group_norm,
                            causal=True,
                            target_aware=True,
                            max_attn_len=None,
                            attn_alpha=None,
                            recompute_normed_x=True,
                            recompute_uvqk=True,
                            recompute_y=True,
                            sort_by_length=True,
                            contextual_seq_len=0,
                        ),
                        is_inference=self.is_inference,
                        dtype=torch.bfloat16,  # bf16 is the default dtype for the model
                        device=self.device
                    )
                    for _ in range(hstu_config.hstu_attn_num_layers)
                ],
                is_inference=self.is_inference,

            )
            model_dense._hstu_transducer._stu_module = stu_module
        # Move model to device and set data type
        model_dense.eval()
        model_dense.recursive_setattr("_use_triton_cc", False)
        model_dense = model_dense.to(self.device).to(torch.bfloat16)
        model_dense.set_training_dtype(torch.bfloat16)
        log_memory_phase(_logger, "backend.initialize.after_dense_to_device")
        # Plan 14.5b (ROCm): DLRM_HSTU_KERNEL=PYTORCH forces every HammerModule op onto
        # the PyTorch path. The open-HSTU jagged-tensor Triton kernels (e.g.
        # _concat_2D_jagged) crash the gfx950 Triton backend (CanonicalizePointers
        # fat-pointer MLIR pass); the PyTorch fallback avoids the compiler bug. This is
        # broader than the attention-only DLRM_HSTU_KERNEL knob, so apply it explicitly.
        import os as _os
        if _os.environ.get("DLRM_HSTU_KERNEL", "").upper() == "PYTORCH":
            from generative_recommenders.common import HammerKernel as _HammerKernel
            model_dense.set_hammer_kernel(_HammerKernel.PYTORCH)

        # Materialize embeddings if sparse
        model_sparse = None
        if self.backend_config.use_nve:

            nve_config = []
            nve_gpu_cache_size_bytes = int(
                self.backend_config.gpu_cache_size_in_gigabytes
                * 1024
                * 1024
                * 1024
            )
            for table_name in self.embedding_table_config:
                if table_name == "item_id":
                    # Movie ID uses NVL/MPI with cache or NoCache depending on multi-GPU setting
                    if self.backend_config.use_multi_gpu:
                        # Use MPI memblock if use_mpi is enabled, otherwise use the provided memblock
                        # memblock_to_use = mpi_memblock if self.backend_config.use_mpi else self.backend_config.nve_memblock
                        memblock_to_use = self.backend_config.nve_memblock
                        nve_config.append(
                            NVEEmbeddingCollectionConfig(
                                cache_type=nve_layers.CacheType.LinearUVM,
                                gpu_cache_size_in_bytes=nve_gpu_cache_size_bytes,
                                memblock=memblock_to_use,
                                device=self.device,
                            )
                        )
                    else:
                        # Plan 14.5b (ROCm single-GPU NVE): back the full-size item_id
                        # table with LinearUVM (auto ManagedMemBlock in host memory) +
                        # a GPU cache, instead of NoCache. NoCache would try to allocate
                        # the full 1e9 x 512 x fp16 (~954 GiB) table directly in GPU
                        # memory (OOM on a 288 GiB GPU), and capping rows breaks raw-id
                        # validity (OOB lookups). LinearUVM keeps the full logical table
                        # valid in host RAM and exercises the 14.10-fixed cache path.
                        nve_config.append(
                            NVEEmbeddingCollectionConfig(
                                cache_type=nve_layers.CacheType.LinearUVM,
                                gpu_cache_size_in_bytes=nve_gpu_cache_size_bytes,
                                memblock=None,
                                device=self.device,
                            )
                        )
                elif table_name == "user_id" or table_name == "item_category_id":
                    # User ID uses NoCache (all on GPU)
                    nve_config.append(
                        NVEEmbeddingCollectionConfig(
                            cache_type=nve_layers.CacheType.NoCache,
                            gpu_cache_size_in_bytes=0,
                            memblock=None,
                            device=self.device,
                        )
                    )
                else:
                    raise ValueError(f"Unknown table name: {table_name}")

            embedding_collection = NVEEmbeddingCollection(list(self.embedding_table_config.values()), nve_config)
            model_dense._embedding_collection = embedding_collection
            log_memory_phase(
                _logger,
                "backend.initialize.after_nve_embedding_collection",
                extra={"nve_cache_size_bytes": nve_gpu_cache_size_bytes},
            )
        else:
            model_sparse = DlrmHSTUCustom(
                hstu_configs=self.hstu_config,
                embedding_tables=self.embedding_table_config,
                is_dense=False
            )
            for _, module in model_sparse.named_modules():
                if isinstance(module, EmbeddingCollection):
                    module.to_empty(device="cpu")
            model_sparse.eval()
            model_dense._embedding_collection = model_sparse._embedding_collection

        self.model_impl = model_dense
        if self._nve_cache_metrics and hasattr(self.model_impl._embedding_collection, "reset_cache_metrics"):
            self.model_impl._embedding_collection.reset_cache_metrics()
        log_memory_phase(_logger, "backend.initialize.done")

    def load_model_sparse(self, checkpoint_path: str, main_rank: bool = False, rank: int = 0):
        import os
        log_memory_phase(
            _logger,
            "backend.load_model_sparse.start",
            rank=rank,
            extra={"main_rank": main_rank},
        )
        # Plan 20: parallel sparse checkpoint load. The legacy path had rank 0 alone
        # load the full ~954 GB item_id table through the serial DCP path (no PG ->
        # single process) and scatter peer shards over xGMI (~9 min). Two opt-in
        # modes via DLRM_NVE_PARALLEL_CKPT_LOAD:
        #   "1"   Option B  -- each worker reads its own row-shard directly from
        #                      __{rank}_0.distcp (manual zip parse, no torch PG).
        #   "dcp" Option A  -- init a torch PG over the 8 workers and let one
        #                      collective dcp.load route item_id as a Shard(0)
        #                      DTensor (native DCP sharding, no manual parsing).
        mode = os.environ.get("DLRM_NVE_PARALLEL_CKPT_LOAD", "").lower()
        opt_b = mode == "1"
        opt_a = mode == "dcp"

        # Plan 51 — int8 packed item_id requires the manual per-rank loader (opt_b):
        # it quantizes+packs fp16->packed-fp16 per step. The DCP paths (opt_a / legacy)
        # load raw fp16 rows in place and would write 512-wide into the 258-wide packed
        # table -> shape mismatch. Fail fast with a clear message.
        from inference_harness.backends import int8_embed as _int8_embed
        if _int8_embed.int8_gather_enabled() and not opt_b:
            raise ValueError(
                "DLRM_NVE_INT8_GATHER=1 requires DLRM_NVE_PARALLEL_CKPT_LOAD=1 (Option B); "
                f"got mode={mode!r}. The DCP load paths cannot quantize+pack on load.")

        # Smoke modes skip the ~1 TB load entirely (tables locally initialized).
        if os.environ.get("DLRM_NVE_SMOKE_HASH_CAP", "") or os.environ.get("DLRM_NVE_SKIP_SPARSE_CKPT", "") == "1":
            logging.getLogger(__name__).warning(
                "Skipping sparse checkpoint load (NVE smoke mode, tables locally initialized)."
            )
            log_memory_phase(
                _logger,
                "backend.load_model_sparse.done",
                rank=rank,
                extra={"skipped": True},
            )
            return

        sparse_tensor_keys = {
            k for k, v in self.model_impl.state_dict().items() if is_sparse_key(k, v)
        }

        if opt_a:
            # Option A keeps item_id in the (collective) load on every rank.
            self._dcp_dtensor_load_sparse(checkpoint_path, rank, sparse_tensor_keys)
            log_memory_phase(
                _logger,
                "backend.load_model_sparse.done",
                rank=rank,
                extra={"mode": "dcp"},
            )
            return

        if not main_rank or opt_b:
            sparse_tensor_keys.discard('_embedding_collection.embeddings.item_id.weight')

        sparse_dict = {"sparse_dict": SparseState(self.model_impl, sparse_tensor_keys)}
        torch.distributed.checkpoint.load(
            sparse_dict,
            storage_reader=torch.distributed.checkpoint.FileSystemReader(checkpoint_path + "/sparse/"),
        )
        self.model_impl.state_dict()['_embedding_collection.embeddings.item_category_id.weight'] == self.model_impl._embedding_collection.embeddings.item_category_id.weight

        if opt_b:
            self._parallel_load_item_id_shard(checkpoint_path, rank)
        log_memory_phase(_logger, "backend.load_model_sparse.done", rank=rank)

    def _dcp_dtensor_load_sparse(self, checkpoint_path: str, rank: int, sparse_tensor_keys) -> None:
        """Plan 20 (Option A): native DCP sharded load over a torch process group.

        Initializes a (gloo) PG across the 8 workers, then presents item_id.weight
        as a Shard(0) DTensor whose local shard is a *view* into this worker's slice
        of the distributed NVE buffer. A single collective ``dcp.load`` then routes
        every row-chunk to its owning rank natively (DCP resolves the on-disk
        torch.save zip layout itself -- no manual parsing). The small replicated
        tables ride along as plain tensors (each rank reads them in full). DCP loads
        in place, so the bytes land directly in the NVE buffer / live tables.
        """
        import os, time
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard
        log = logging.getLogger(__name__)

        world = int(os.environ.get("DLRM_MPI_WORKER_WORLD", "") or os.environ.get("DLRM_SPARSE_WORLD", "") or 8)
        if not dist.is_initialized():
            master = os.environ.get("DLRM_SPARSE_PG_MASTER_ADDR", "127.0.0.1")
            port = os.environ.get("DLRM_SPARSE_PG_MASTER_PORT", "29501")
            backend = os.environ.get("DLRM_NVE_CKPT_PG_BACKEND", "gloo")
            dist.init_process_group(
                backend=backend, init_method=f"tcp://{master}:{port}", world_size=world, rank=rank
            )
            log.warning(f"[Plan20A dcp-dtensor] rank {rank}: PG up backend={backend} world={world}")
        mesh = init_device_mesh("cuda", (world,))

        item_key = "_embedding_collection.embeddings.item_id.weight"
        weight = self.model_impl.state_dict()[item_key]
        num_emb = int(weight.shape[0])
        assert num_emb % world == 0, f"item_id rows {num_emb} not divisible by world {world}"
        rows_per = num_emb // world
        local = weight[rank * rows_per:(rank + 1) * rows_per]
        item_dt = DTensor.from_local(local, mesh, [Shard(0)])

        # Build a nested plain dict -> FQNs "sparse_dict.<key>"; DCP loads in place.
        inner = {}
        for k, v in self.model_impl.state_dict().items():
            if k in sparse_tensor_keys:
                inner[k] = item_dt if k == item_key else v
        log.warning(
            f"[Plan20A dcp-dtensor] rank {rank}: collective sparse load "
            f"(item_id Shard(0) rows[{rank * rows_per}:{(rank + 1) * rows_per}))"
        )
        t0 = time.time()
        torch.distributed.checkpoint.load(
            {"sparse_dict": inner},
            storage_reader=torch.distributed.checkpoint.FileSystemReader(checkpoint_path + "/sparse/"),
        )
        log.warning(f"[Plan20A dcp-dtensor] rank {rank}: done in {time.time() - t0:.1f}s")

    @staticmethod
    def _locate_dcp_storage(fpath: str, blob_off: int, blob_len: int,
                            member: str = "archive/data/0"):
        """Locate the raw (uncompressed) storage bytes of a tensor inside a DCP
        shard file.

        Each DCP WriteItem is an independent ``torch.save`` zip archive embedded
        in the shard file at ``[blob_off : blob_off + blob_len)``. The tensor's
        flat storage lives in the (STORED / uncompressed) zip member
        ``archive/data/0``. Returns ``(absolute_byte_offset, storage_len)`` of
        that member so it can be streamed directly with a plain file read, with
        no decompression and without materializing the whole blob.
        """
        import io, zipfile

        class _Slice(io.RawIOBase):
            def __init__(self, path, base, length):
                self.f = open(path, "rb", buffering=0)
                self.base, self.len, self.pos = base, length, 0

            def readable(self):
                return True

            def seekable(self):
                return True

            def seek(self, o, whence=0):
                self.pos = o if whence == 0 else (self.pos + o if whence == 1 else self.len + o)
                self.f.seek(self.base + self.pos)
                return self.pos

            def tell(self):
                return self.pos

            def readinto(self, b):
                n = min(len(b), self.len - self.pos)
                if n <= 0:
                    return 0
                got = self.f.readinto(memoryview(b)[:n])
                self.pos += got
                return got

            def close(self):
                try:
                    self.f.close()
                finally:
                    super().close()

        sl = _Slice(fpath, blob_off, blob_len)
        try:
            zf = zipfile.ZipFile(sl)
            zi = zf.getinfo(member)
            assert zi.compress_type == 0, f"{member} is not STORED (uncompressed)"
            # Data starts after the local file header: 30 fixed bytes + filename + extra.
            sl.seek(zi.header_offset)
            lh = sl.read(30)
            fn_len = int.from_bytes(lh[26:28], "little")
            ex_len = int.from_bytes(lh[28:30], "little")
            data_rel = zi.header_offset + 30 + fn_len + ex_len
            return blob_off + data_rel, zi.file_size
        finally:
            sl.close()

    def _parallel_load_item_id_shard(self, checkpoint_path: str, rank: int) -> None:
        """Plan 20 (Option B): load THIS rank's row-shard of the item_id embedding
        directly from its matching DCP shard file, in parallel with every other
        worker.

        The sparse checkpoint stores item_id.weight (1e9 x 512 fp16) row-sharded
        into 8 contiguous chunks; chunk r (rows [r*N : (r+1)*N)) is an independent
        ``torch.save`` zip inside __r_0.distcp whose ``archive/data/0`` member is
        the raw, uncompressed fp16 storage. Each worker locates that member,
        streams its 128 GB through a double-buffered pinned host staging area,
        and copies it into the matching logical slice of the distributed NVE
        weight (physically resident on this worker's own GPU). No torch PG, no
        MPI, no cross-rank scatter -- 8 independent, mostly-local loads.
        """
        import os, time, numpy as np
        log = logging.getLogger(__name__)
        sparse_dir = os.path.join(checkpoint_path, "sparse")
        item_key = "sparse_dict._embedding_collection.embeddings.item_id.weight"

        # 1. Locate this rank's chunk (row range) and its shard file via metadata.
        reader = torch.distributed.checkpoint.FileSystemReader(sparse_dir)
        md = reader.read_metadata()
        tmd = md.state_dict_metadata[item_key]
        cols = int(tmd.size[1])
        dtype = tmd.properties.dtype
        itemsize = torch.empty(0, dtype=dtype).element_size()

        chunks = sorted(tmd.chunks, key=lambda c: int(c.offsets[0]))
        n_chunks = len(chunks)
        assert rank < n_chunks, f"rank {rank} >= #item_id chunks {n_chunks}"
        my = chunks[rank]
        row_start, n_rows = int(my.offsets[0]), int(my.sizes[0])
        assert int(my.offsets[1]) == 0 and int(my.sizes[1]) == cols, "unexpected column sharding"

        sinfo = None
        for idx, info in md.storage_data.items():
            off = getattr(idx, "offset", None)
            if getattr(idx, "fqn", "") == item_key and off is not None and int(off[0]) == row_start:
                sinfo = info
                break
        assert sinfo is not None, f"no storage_data for item_id chunk @row {row_start}"
        fpath = os.path.join(sparse_dir, sinfo.relative_path)
        expect = n_rows * cols * itemsize

        # 2. Resolve the raw storage byte offset inside the torch.save zip blob.
        data_off, data_len = self._locate_dcp_storage(fpath, int(sinfo.offset), int(sinfo.length))
        assert data_len == expect, f"storage bytes {data_len} != expected {expect}"

        # 3. Destination: logical slice of the full (num_embeddings x cols) NVE weight.
        #    Under int8-gather the item_id table is stored packed (fp16 x packed_width:
        #    cols int8 + fp32 scale); the read still pulls cols fp16/row from the
        #    checkpoint, then each step is quantized+packed before the write.
        weight = self.model_impl.state_dict()["_embedding_collection.embeddings.item_id.weight"]
        dst = weight[row_start:row_start + n_rows]
        from inference_harness.backends import int8_embed
        int8_pack = int8_embed.int8_gather_enabled()
        if int8_pack:
            assert dst.shape[1] == int8_embed.packed_fp16_width(cols), (
                f"int8 dst width {dst.shape[1]} != packed_fp16_width({cols})="
                f"{int8_embed.packed_fp16_width(cols)} (table not configured packed)")
            log.warning(
                f"[Plan51 int8-gather] rank {rank}: quantize+pack item_id fp16x{cols} "
                f"-> fp16x{dst.shape[1]} on load")

        # 4. Stream file -> pinned host (double-buffered) -> GPU slice on a copy stream.
        rows_per_step = int(os.environ.get("DLRM_NVE_CKPT_STEP_ROWS", str(2_000_000)))
        bufs = [torch.empty((rows_per_step, cols), dtype=dtype, pin_memory=True) for _ in range(2)]
        events = [None, None]
        copy_stream = torch.cuda.Stream(device=self.device)
        log_memory_phase(
            _logger,
            "backend.parallel_item_load.before_copy",
            rank=rank,
            extra={
                "rows": n_rows,
                "cols": cols,
                "bytes": expect,
                "rows_per_step": rows_per_step,
                "int8_pack": int8_pack,
            },
        )
        log.warning(
            f"[Plan20 parallel-ckpt] rank {rank}: loading item_id rows "
            f"[{row_start}:{row_start + n_rows}) ({expect / 1e9:.1f} GB) from "
            f"{os.path.basename(fpath)} @byte {data_off}"
        )
        t0 = time.time()
        done, bi = 0, 0
        with open(fpath, "rb", buffering=0) as f:
            f.seek(data_off)
            while done < n_rows:
                nr = min(rows_per_step, n_rows - done)
                if events[bi] is not None:
                    events[bi].synchronize()  # prior H2D out of this buffer must finish
                hb = bufs[bi][:nr]
                mv = memoryview(hb.numpy().reshape(-1).view(np.uint8))
                nbytes = nr * cols * itemsize
                got = 0
                while got < nbytes:  # readinto may return short on NFS
                    k = f.readinto(mv[got:nbytes])
                    assert k > 0, f"unexpected EOF at row {done} (+{got}/{nbytes} B)"
                    got += k
                with torch.cuda.stream(copy_stream):
                    if int8_pack:
                        # quantize+pack on GPU, then write packed rows to the dst slice
                        hb_g = hb.to(self.device, non_blocking=True)
                        packed = int8_embed.pack_int8_fp16(hb_g, cols)
                        dst[done:done + nr].copy_(packed, non_blocking=True)
                    else:
                        dst[done:done + nr].copy_(hb, non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(copy_stream)
                    events[bi] = ev
                done += nr
                bi ^= 1
            copy_stream.synchronize()
        dt = time.time() - t0
        gb = expect / 1e9
        log.warning(
            f"[Plan20 parallel-ckpt] rank {rank}: done in {dt:.1f}s "
            f"({gb / max(dt, 1e-6):.2f} GB/s)"
        )
        log_memory_phase(
            _logger,
            "backend.parallel_item_load.done",
            rank=rank,
            extra={"seconds": dt, "bytes": expect},
        )

    def load_model_dense(self, checkpoint_path: str):
        log_memory_phase(_logger, "backend.load_model_dense.start")
        load_nonsparse_checkpoint(model=self.model_impl, device=self.device, optimizer=None, path=checkpoint_path)
        log_memory_phase(_logger, "backend.load_model_dense.done")

    def _kjt_to_custom_dict(
        self,
        features: KeyedJaggedTensor,
        max_length_keys: Set[str],
        offset_keys: Set[str],
    ) -> Dict[str, CustomJaggedTensor]:
        # Compute shape metadata before moving KJT tensors to GPU. The optimized
        # GOLD path only needs max lengths for item_id/item_candidate_id, and
        # doing this after `.to(device)` introduces a per-batch device sync.
        max_lengths: Dict[str, int] = {}
        for key in max_length_keys:
            if key in features.keys():
                lengths = features[key].lengths()
                max_lengths[key] = (
                    int(lengths.max().item()) if lengths.numel() > 0 else 0
                )
        if self.backend_config.use_nve:
            features = features.to(self.device).to(torch.int64)
        else:
            features = features.to(torch.int64)
        _clamp_kjt_lookup_ids(features)

        tensor_dict: Dict[str, CustomJaggedTensor] = {}
        feature_dict = features.to_dict()
        for key in features.keys():
            jt = feature_dict[key]
            lengths = jt.lengths()
            max_length = (
                max_lengths.get(key, 0) if key in max_length_keys else 0
            )
            offsets = jt.offsets() if key in offset_keys else None
            tensor_dict[key] = CustomJaggedTensor(
                values=jt.values(),
                lengths=lengths,
                offsets=offsets,
                max_length=max_length,
                embeddings=None,
            )
        return tensor_dict

    def optimized_embedding_lookup_from_kjt(
        self, uih_features: KeyedJaggedTensor, candidates_features: KeyedJaggedTensor
    ):
        uih_key = self.hstu_config.uih_post_id_feature_name
        candidate_key = self.hstu_config.item_embedding_feature_names[0]
        tensor_dict_uih = self._kjt_to_custom_dict(
            uih_features, {uih_key}, {uih_key}
        )
        tensor_dict_candidates = self._kjt_to_custom_dict(
            candidates_features, {candidate_key}, {candidate_key}
        )
        return self.optimized_embedding_lookup(tensor_dict_uih, tensor_dict_candidates)

    def _candidate_max_length_from_kjt(self, candidates_features: KeyedJaggedTensor) -> int:
        candidate_key = self.hstu_config.item_embedding_feature_names[0]
        jt = candidates_features[candidate_key]
        lengths = jt.lengths()
        if lengths.numel() == 0:
            return 0
        return int(lengths.max().item())

    def _server_candidate_shape_from_kjt(self, candidates_features: KeyedJaggedTensor) -> bool:
        max_len = self._candidate_max_length_from_kjt(candidates_features)
        ok = max_len == self._server_candidate_size
        if (
            not ok
            and (self._optimized_embed_lookup or self._optimized_embed_compare or self._uniform_targets_metadata)
            and not self._optimized_embed_shape_skip_logged
        ):
            _logger.warning(
                "[Plan55 optimized-embed] disabling 2048-candidate shortcuts for "
                "candidate max_length=%d (expected %d)",
                max_len,
                self._server_candidate_size,
            )
            self._optimized_embed_shape_skip_logged = True
        return ok

    def _reference_targets_from_kjt(
        self,
        candidates_features: KeyedJaggedTensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Recover AccuracyOnly labels without changing inference predictions.

        TEST08 reference predictions must use the exact audited inference path.
        That path intentionally zeros supervision payloads and its multitask
        module does not return labels. The original candidate KJT still carries
        ground-truth action bitmasks, so reconstruct labels and unit weights
        before the KJT is moved to the device.
        """
        if not self._restore_inference_accuracy_targets:
            return None

        feature_name = self.hstu_config.candidates_weight_feature_name
        supervision_bitmasks = (
            candidates_features[feature_name].values().detach().to("cpu")
        )
        labels = []
        for task in self.hstu_config.multitask_configs:
            if task.task_type != MultitaskTaskType.BINARY_CLASSIFICATION:
                raise RuntimeError(
                    "Inference-mode AccuracyOnly target restoration only "
                    "supports binary classification tasks"
                )
            labels.append(
                (
                    torch.bitwise_and(supervision_bitmasks, task.task_weight) > 0
                ).to(torch.float32)
            )
        mt_labels = torch.stack(labels, dim=0).contiguous()
        return mt_labels, torch.ones_like(mt_labels)

    # WIP feature, reduce PCIe data transfer + optimize CPU here
    def optimized_embedding_lookup(
        self,
        uih_features: Dict[str, CustomJaggedTensor],
        candidates_features: Dict[str, CustomJaggedTensor],
    ):
        tensor_dict_uih = uih_features
        tensor_dict_candidates = candidates_features

        with nvtx.annotate(f"hybrid_GR_backend - optimized_embedding_lookup - embedding lookup", color="orange"):
            tensor_dict_all = {**tensor_dict_uih, **tensor_dict_candidates}
            self.model_impl._embedding_collection(tensor_dict_all)
            seq_embeddings_dict = self._pack_sequence_embeddings(tensor_dict_uih, tensor_dict_candidates)

        with nvtx.annotate(f"hybrid_GR_backend - optimized_embedding_lookup - payload features", color="orange"):
            max_uih_len = tensor_dict_uih["item_id"].max_length
            uih_seq_lengths = tensor_dict_uih["item_id"].lengths
            max_num_candidates = tensor_dict_candidates["item_candidate_id"].max_length
            num_candidates = tensor_dict_candidates["item_candidate_id"].lengths
            if should_use_uniform_targets_metadata(
                self.is_inference,
                self._uniform_targets_metadata,
                self._uniform_targets_metadata_disabled,
            ):
                total_candidates = int(num_candidates.numel()) * int(max_num_candidates)
            else:
                total_candidates = int(num_candidates.sum().item())

            payload_features = {
                "uih_offsets": tensor_dict_uih["item_id"].offsets,
                "candidate_offsets": tensor_dict_candidates["item_candidate_id"].offsets,
            }
            for (
                uih_feature_name,
                candidate_feature_name,
            ) in self.hstu_config.merge_uih_candidate_feature_mapping:
                if (
                    candidate_feature_name
                    not in self.hstu_config.item_embedding_feature_names
                    and uih_feature_name
                    not in self.hstu_config.user_embedding_feature_names
                ):
                    values_left = tensor_dict_uih[uih_feature_name].values
                    if self.is_inference and (
                        candidate_feature_name
                        == self.hstu_config.candidates_weight_feature_name
                        or candidate_feature_name
                        == self.hstu_config.candidates_watchtime_feature_name
                    ):
                        values_right = self._candidate_zero_payload(
                            total_candidates,
                            dtype=torch.int64,
                            device=values_left.device,
                        )
                    else:
                        values_right = tensor_dict_candidates[candidate_feature_name].values
                    payload_features[uih_feature_name] = values_left
                    payload_features[candidate_feature_name] = values_right
        return seq_embeddings_dict, payload_features, max_uih_len, uih_seq_lengths, max_num_candidates, num_candidates

    def _candidate_zero_payload(
        self,
        total_candidates: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not self._cache_inference_zero_payloads:
            return torch.zeros(total_candidates, dtype=dtype, device=device)
        key = (int(total_candidates), dtype, device)
        cached = self._zero_payload_cache.get(key)
        if cached is None:
            cached = torch.zeros(total_candidates, dtype=dtype, device=device)
            self._zero_payload_cache[key] = cached
            _logger.warning(
                "[Plan59 glue] cached zero candidate payload: n=%d dtype=%s device=%s",
                total_candidates,
                dtype,
                device,
            )
        return cached

    # WIP feature, reduce PCIe data transfer + optimize CPU here
    def _pack_sequence_embeddings(self, tensor_dict_uih: Dict[str, CustomJaggedTensor], tensor_dict_candidates: Dict[str, CustomJaggedTensor]) -> Dict[str, SequenceEmbedding]:
        with nvtx.annotate(f"hybrid_GR_backend - optimized_embedding_lookup - pack sequence embeddings", color="orange"):
            sequence_embeddings = {}
            for name in (
                self.hstu_config.user_embedding_feature_names
                + self.hstu_config.item_embedding_feature_names
            ):
                source = (
                    tensor_dict_uih if name in tensor_dict_uih else tensor_dict_candidates
                )
                sequence_embeddings[name] = SequenceEmbedding(
                    lengths=source[name].lengths,
                    embedding=source[name].embeddings,
                )
        return sequence_embeddings

    def _embedding_lookup_outputs_match(self, lhs, rhs) -> bool:
        (
            lhs_seq_embeddings,
            lhs_payload_features,
            lhs_max_uih_len,
            lhs_uih_seq_lengths,
            lhs_max_num_candidates,
            lhs_num_candidates,
        ) = lhs
        (
            rhs_seq_embeddings,
            rhs_payload_features,
            rhs_max_uih_len,
            rhs_uih_seq_lengths,
            rhs_max_num_candidates,
            rhs_num_candidates,
        ) = rhs

        ok = True

        def note(message: str) -> None:
            nonlocal ok
            ok = False
            _logger.warning("[Plan55 E0 optimized-embed] mismatch: %s", message)

        if lhs_max_uih_len != rhs_max_uih_len:
            note(f"max_uih_len {lhs_max_uih_len} != {rhs_max_uih_len}")
        if lhs_max_num_candidates != rhs_max_num_candidates:
            note(
                f"max_num_candidates {lhs_max_num_candidates} != {rhs_max_num_candidates}"
            )
        if not torch.equal(lhs_uih_seq_lengths, rhs_uih_seq_lengths):
            note("uih_seq_lengths differ")
        if not torch.equal(lhs_num_candidates, rhs_num_candidates):
            note("num_candidates differ")

        if set(lhs_seq_embeddings.keys()) != set(rhs_seq_embeddings.keys()):
            note(
                "seq_embedding keys "
                f"{sorted(lhs_seq_embeddings.keys())} != {sorted(rhs_seq_embeddings.keys())}"
            )
        for key in lhs_seq_embeddings.keys() & rhs_seq_embeddings.keys():
            lhs_seq = lhs_seq_embeddings[key]
            rhs_seq = rhs_seq_embeddings[key]
            if not torch.equal(lhs_seq.lengths, rhs_seq.lengths):
                note(f"{key}.lengths differ")
            if lhs_seq.embedding.shape != rhs_seq.embedding.shape:
                note(
                    f"{key}.embedding shape {tuple(lhs_seq.embedding.shape)} "
                    f"!= {tuple(rhs_seq.embedding.shape)}"
                )
            elif lhs_seq.embedding.numel() > 0:
                delta = (
                    lhs_seq.embedding.float() - rhs_seq.embedding.float()
                ).abs().max().item()
                if delta != 0.0:
                    note(f"{key}.embedding max_abs_delta={delta:.3e}")

        if set(lhs_payload_features.keys()) != set(rhs_payload_features.keys()):
            note(
                "payload keys "
                f"{sorted(lhs_payload_features.keys())} != {sorted(rhs_payload_features.keys())}"
            )
        for key in lhs_payload_features.keys() & rhs_payload_features.keys():
            lhs_value = lhs_payload_features[key]
            rhs_value = rhs_payload_features[key]
            if lhs_value.shape != rhs_value.shape:
                note(f"{key} shape {tuple(lhs_value.shape)} != {tuple(rhs_value.shape)}")
            elif not torch.equal(lhs_value, rhs_value):
                note(f"{key} values differ")
        return ok

    def embedding_lookup(self, uih_features: KeyedJaggedTensor, candidates_features: KeyedJaggedTensor):
        with nvtx.annotate(f"hybrid_GR_backend - embedding_lookup - transfer to device", color="orange"):
            if self.backend_config.use_nve:
                uih_features = uih_features.to(self.device).to(torch.int64)
                candidates_features = candidates_features.to(self.device).to(torch.int64)
            else:
                # cpu look up, base impl
                uih_features = uih_features.to(torch.int64)
                candidates_features = candidates_features.to(torch.int64)
        # Plan 21 §0.4: path-independent OOV-id guard. Clamp lookup ids to their
        # table bounds on the device KJT before they reach model_impl.preprocess
        # (-> nve::GatherKeysAndDataPtrs), so a stray id == table_size from any
        # KJT producer cannot fault the queue. No-op unless DLRM_CLAMP_OOB_IDS=1.
        global _clamp_logged
        if not _clamp_logged:
            _logger.warning(
                "[Plan21 §0.4] embedding_lookup OOV-id clamp guard: "
                "DLRM_CLAMP_OOB_IDS=%s (active=%s)",
                os.environ.get("DLRM_CLAMP_OOB_IDS", "0"), _CLAMP_OOB_IDS,
            )
            _clamp_logged = True
        _clamp_kjt_lookup_ids(uih_features)
        _clamp_kjt_lookup_ids(candidates_features)
        if self._nve_prefetch_current_batch and hasattr(self.model_impl._embedding_collection, "prefetch_features"):
            with nvtx.annotate("hybrid_GR_backend - embedding_lookup - prefetch", color="yellow"):
                self.model_impl._embedding_collection.prefetch_features(uih_features)
                self.model_impl._embedding_collection.prefetch_features(candidates_features)
        with nvtx.annotate(f"hybrid_GR_backend - embedding_lookup - embedding lookup", color="green"):
            seq_embeddings, payload_features, max_uih_len, uih_seq_lengths, max_num_candidates, num_candidates = \
                self.model_impl.preprocess(uih_features=uih_features, candidates_features=candidates_features)
        self._maybe_log_nve_cache_metrics()
        return seq_embeddings, payload_features, max_uih_len, uih_seq_lengths, max_num_candidates, num_candidates

    def _maybe_log_nve_cache_metrics(self):
        if not self._nve_cache_metrics:
            return
        self._nve_cache_metrics_batches += 1
        if (
            self._nve_cache_metrics_interval <= 0
            or self._nve_cache_metrics_batches % self._nve_cache_metrics_interval != 0
        ):
            return
        collection = self.model_impl._embedding_collection
        if not hasattr(collection, "cache_metrics"):
            return
        _logger.warning(
            "[Plan55 NVE cache] batch=%d metrics=%s",
            self._nve_cache_metrics_batches,
            collection.cache_metrics(),
        )

    def predict(self, Samples: Union[List[Dict[str, CustomJaggedTensor]], Samples]):
        uih_features = Samples.uih_features_kjt
        candidates_features = Samples.candidates_features_kjt
        reference_targets = self._reference_targets_from_kjt(candidates_features)

        _t = self._stage_timing
        if _t:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev2 = torch.cuda.Event(enable_timing=True)
            ev0.record()

        server_candidate_shape = self._server_candidate_shape_from_kjt(candidates_features)
        use_optimized_embed = should_use_optimized_embed(
            self._optimized_embed_lookup,
            self._optimized_embed_disabled,
            self.is_inference,
            server_candidate_shape,
        )
        use_optimized_compare = should_run_optimized_compare(
            self._optimized_embed_compare,
            self._optimized_embed_compares,
            self._optimized_embed_compare_limit,
            self.is_inference,
            server_candidate_shape,
        )
        if use_optimized_compare:
            baseline_lookup = self.embedding_lookup(uih_features, candidates_features)
            try:
                optimized_lookup = self.optimized_embedding_lookup_from_kjt(
                    uih_features, candidates_features
                )
                paths_match = self._embedding_lookup_outputs_match(
                    baseline_lookup, optimized_lookup
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "[Plan55 E0 optimized-embed] optimized path failed during compare: %r",
                    exc,
                )
                optimized_lookup = None
                paths_match = False
            self._optimized_embed_compares += 1
            if paths_match:
                _logger.warning(
                    "[Plan55 E0 optimized-embed] compare PASS on batch %d",
                    self._optimized_embed_compares,
                )
            else:
                _logger.warning(
                    "[Plan55 E0 optimized-embed] disabling 2048-candidate shortcuts "
                    "after compare mismatch"
                )
                self._optimized_embed_disabled = True
                self._uniform_targets_metadata_disabled = True
            lookup_outputs = (
                optimized_lookup
                if (
                    use_optimized_embed
                    and paths_match
                )
                else baseline_lookup
            )
        elif use_optimized_embed:
            try:
                lookup_outputs = self.optimized_embedding_lookup_from_kjt(
                    uih_features, candidates_features
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "[Plan55 E2 optimized-embed] optimized path failed; "
                    "falling back to baseline for this process: %r",
                    exc,
                )
                self._optimized_embed_disabled = True
                lookup_outputs = self.embedding_lookup(uih_features, candidates_features)
        else:
            lookup_outputs = self.embedding_lookup(uih_features, candidates_features)

        seq_embeddings, payload_features, max_uih_len, uih_seq_lengths, max_num_candidates, num_candidates = lookup_outputs

        if _t:
            ev1.record()

        if self.backend_config.use_nve:
            global _int8_logged
            if _INT8_GATHER_KEYS and not _int8_logged:
                _logger.warning(
                    "[Plan51 int8-fakequant] per-row int8 fake-quant on keys=%s "
                    "(DLRM_NVE_INT8_FAKEQUANT; emulation of the real packed int8 table)",
                    sorted(_INT8_GATHER_KEYS),
                )
                _int8_logged = True
            with nvtx.annotate(f"hybrid_GR_backend - converting sequence embeddings to bfloat16", color="blue"):
                seq_embeddings_bf16 = {
                    k: SequenceEmbedding(
                        lengths=seq_embeddings[k].lengths,
                        embedding=(
                            _int8_fakequant(seq_embeddings[k].embedding)
                            if k in _INT8_GATHER_KEYS
                            else (
                                seq_embeddings[k].embedding
                                if (
                                    self._skip_bf16_noop_cast
                                    and seq_embeddings[k].embedding.dtype == torch.bfloat16
                                )
                                else seq_embeddings[k].embedding.to(torch.bfloat16)
                            )
                        ),
                    )
                    for k in seq_embeddings.keys()
                }
            with nvtx.annotate(f"hybrid_GR_backend - main forward", color="yellow"):
                out = self.model_impl.main_forward(
                    seq_embeddings_bf16, payload_features, max_uih_len,
                    uih_seq_lengths, max_num_candidates, num_candidates
                )
        else:
            with nvtx.annotate(f"hybrid_GR_backend - moving to device", color="blue"):
                seq_embeddings_bf16, payload_features_bf16, uih_seq_lengths_bf16, num_candidates_bf16 = \
                    move_sparse_output_to_device(
                        seq_embeddings=seq_embeddings,
                        payload_features=payload_features,
                        uih_seq_lengths=uih_seq_lengths,
                        num_candidates=num_candidates,
                        device=self.device,
                    )
            with nvtx.annotate(f"hybrid_GR_backend - main forward", color="yellow"):
                out = self.model_impl.main_forward(
                    seq_embeddings_bf16, payload_features_bf16, max_uih_len,
                    uih_seq_lengths_bf16, max_num_candidates, num_candidates_bf16
                )
        if _t:
            ev2.record()
            # Resolved later (after the server's stream synchronize) into
            # embed_lookup_ms = ev0->ev1, hstu_forward_ms = ev1->ev2.
            self.last_stage_events = (ev0, ev1, ev2)
        if reference_targets is not None:
            preds, _, _ = out
            labels, weights = reference_targets
            if preds.shape != labels.shape:
                raise RuntimeError(
                    "Inference-mode AccuracyOnly prediction/target shape "
                    f"mismatch: predictions={tuple(preds.shape)}, "
                    f"labels={tuple(labels.shape)}"
                )
            out = preds, labels, weights
        return out

    def predict_dummy(self, Samples: Union[List[Dict[str, CustomJaggedTensor]], Samples]):
        return self.dummy_tensor, self.dummy_labels, self.dummy_weights
