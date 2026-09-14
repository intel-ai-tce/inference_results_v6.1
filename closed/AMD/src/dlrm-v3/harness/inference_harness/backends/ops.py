"""
Custom operators and embedding collections for DLRM inference.

Provides NVE (NVIDIA Embedding)-based embedding collections with GPU caching
and optimized lookup operations for large-scale recommendation models.
"""

import os
import logging
from typing import List, Dict
import torch
from torchrec import EmbeddingBagCollectionInterface
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor, JaggedTensor
import pynve.torch.nve_layers as nve_layers
import pynve.nve as nve
from torchrec.modules.embedding_configs import DataType
from dataclasses import dataclass
import nvtx as nvtx
from inference_harness.backends import int8_embed

_logger = logging.getLogger(__name__)

# Plan 21 §0.4 — OOV-id guard at the NVE embedding-lookup chokepoint.
#
# The preprocessed dataset's last timestamp(s) carry a few embedding ids equal
# to exactly the table size (e.g. item_candidate_id == 1_000_000_000 for the
# 1e9-row table), i.e. one past the last valid row. Such an id faults the NVE
# lookup with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION inside
# nve::GatherKeysAndDataPtrs (key gather / shard routing) and/or
# cuembed::EmbeddingLookUpKernel (the per-shard table read) — the gathered keys
# and the looked-up rows are two distinct consumers and BOTH must see a clamped
# id. Clamping the KJT upstream (QSL ``_LazyTimestampSamples.get`` or the
# backend ``embedding_lookup``) only protects the key-gather; the table read
# reads its index from the per-feature ``JaggedTensor`` produced here by
# ``to_dict()`` (a split that can predate an in-place upstream clamp), so the
# raw id leaks through. Clamping ``f.values()`` right before ``embedding(...)``
# — the single tensor every internal kernel derives from — makes the guard
# truly path-independent.
#
# min-clamp (id -> table_size - 1) is correct for a *performance* run (LoadGen
# does not score outputs). Keep in sync with
# ``inference_harness.dataset.mlperf_streaming_qsl._FEATURE_CLAMP_MAX``. An
# *accuracy* run should instead map the OOV id to a dedicated zero row.
_CLAMP_OOB_IDS: bool = os.environ.get("DLRM_CLAMP_OOB_IDS", "0") == "1"
_FEATURE_CLAMP_MAX: Dict[str, int] = {
    "item_id": 1_000_000_000 - 1,
    "item_candidate_id": 1_000_000_000 - 1,
    "item_category_id": 128 - 1,
    "item_candidate_category_id": 128 - 1,
    "user_id": 10_000_000 - 1,
}
_clamp_logged = False

_LOCAL_SMALL_TABLES = {"user_id", "item_category_id"}


def _local_small_table_lookup_enabled() -> bool:
    return os.environ.get("DLRM_LOCAL_SMALL_TABLE_LOOKUP", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _local_small_table_dtype(default: torch.dtype) -> torch.dtype:
    mode = os.environ.get("DLRM_LOCAL_SMALL_TABLE_DTYPE", "bf16").strip().lower()
    if mode in ("bf16", "bfloat16"):
        return torch.bfloat16
    if mode in ("fp16", "float16", "half"):
        return torch.float16
    if mode in ("table", "default"):
        return default
    raise ValueError(f"Unsupported DLRM_LOCAL_SMALL_TABLE_DTYPE={mode!r}")


def _clamp_lookup_keys(feature_name: str, keys: torch.Tensor) -> torch.Tensor:
    """Return ``keys`` clamped to ``feature_name``'s table bound (out-of-place).

    No-op (returns ``keys`` unchanged) unless ``DLRM_CLAMP_OOB_IDS=1`` and the
    feature has a configured bound. Out-of-place so we never mutate a tensor
    that may be aliased/cached upstream.
    """
    if not _CLAMP_OOB_IDS:
        return keys
    cap = _FEATURE_CLAMP_MAX.get(feature_name)
    if cap is None or keys.numel() == 0:
        return keys
    return keys.clamp(max=cap)


def torchrec_data_type_to_torch_data_type(data_type: DataType):
    """
    Convert TorchRec DataType to PyTorch dtype.

    Args:
        data_type: TorchRec DataType enum value.

    Returns:
        torch.dtype: Corresponding PyTorch data type.

    Raises:
        ValueError: If data type is not supported.
    """
    if data_type == DataType.FP32:
        return torch.float32
    elif data_type == DataType.FP16:
        return torch.float16
    elif data_type == DataType.BF16:
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid data type: {data_type}")


def torch_data_type_to_nve_data_type(data_type: torch.dtype):
    """
    Convert PyTorch dtype to NVE DataType for memblock creation.

    Args:
        data_type: PyTorch data type.

    Returns:
        nve.DataType_t: Corresponding NVE data type.

    Raises:
        ValueError: If data type is not supported by NVE.
    """
    if data_type == torch.float32:
        return nve.DataType_t.Float32
    elif data_type == torch.float16:
        return nve.DataType_t.Float16
    elif data_type == torch.bfloat16:
        return nve.DataType_t.BFloat
    else:
        raise ValueError(f"Unsupported NVE data type: {data_type}")


@dataclass
class NVEEmbeddingCollectionConfig:
    """
    Configuration for NVE embedding collection.

    Attributes:
        cache_type: Type of GPU cache to use (e.g., NoCache, LRU).
        gpu_cache_size_in_bytes: Size of GPU cache in bytes (default: 1GB).
        memblock: NVE memory block for embedding storage.
        device: CUDA device for embedding operations.
    """
    cache_type: nve_layers.CacheType = nve_layers.CacheType.NoCache
    gpu_cache_size_in_bytes: int = 1024 * 1024 * 1024  # 1GB default
    memblock: nve.MemBlock = None
    device: torch.device = None


class NVEEmbeddingCollection(EmbeddingBagCollectionInterface):
    """
    NVE-based embedding collection for large-scale recommendation models.

    Implements TorchRec's EmbeddingBagCollectionInterface using NVIDIA's
    NVE (NVIDIA Embedding) backend for optimized embedding lookups with
    GPU caching support for large embedding tables.

    Attributes:
        embeddings: Dictionary of NVEmbedding modules, one per table.
        _embedding_configs: List of embedding table configurations.
        _nve_config: List of NVE-specific configurations.
        _feature_names: List of feature names for each embedding table.
    """

    def __init__(self,
                 table_configs: List[EmbeddingConfig],
                 nve_config: List[NVEEmbeddingCollectionConfig]
                 ):
        """
        Initialize NVE embedding collection.

        Args:
            table_configs: List of embedding table configurations from TorchRec.
            nve_config: List of NVE-specific configurations (one per table).

        Raises:
            ValueError: If number of table configs doesn't match number of NVE configs.
        """
        super().__init__()
        assert len(table_configs) == len(nve_config), "Number of table configs and config must be the same"
        self.embeddings: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self._embedding_configs = table_configs
        self._nve_config = nve_config
        self._lengths_per_embedding: List[int] = []

        # Plan 51 — int8 packed-fp16 storage. For the packed tables (item_id), the
        # physical NVE table is fp16 of width packed_fp16_width(dim) (int8 data + a
        # fp32 per-row scale); the gather output is unpacked+dequanted back to a
        # dim-wide bf16 embedding in forward(), so the model still sees dim-wide.
        self._int8_enabled = int8_embed.int8_gather_enabled()
        self._local_small_table_lookup = _local_small_table_lookup_enabled()
        # logical embedding dim per table (parallel to self.embeddings.values());
        # None = not packed, int = packed (the dim to unpack to).
        self._packed_dims: List[int] = []
        self._local_lookup_tables: set[str] = set()

        table_names = set()
        for i, embedding_config in enumerate(table_configs):
            if embedding_config.name in table_names:
                raise ValueError(f"Duplicate table name {embedding_config.name}")
            table_names.add(embedding_config.name)
            config = {"kernel_mode": 1, "logging_interval": -1}
            if nve_config[i].cache_type == nve_layers.CacheType.LinearUVM:
                config["insert_threshold"] = float(
                    os.environ.get("DLRM_NVE_INSERT_THRESHOLD", "0.75")
                )
                config["min_insert_freq_gpu"] = int(
                    os.environ.get("DLRM_NVE_MIN_INSERT_FREQ_GPU", "0")
                )
                config["min_insert_size_gpu"] = int(
                    os.environ.get("DLRM_NVE_MIN_INSERT_SIZE_GPU", str(1 << 16))
                )
            packed = self._int8_enabled and embedding_config.name in int8_embed.INT8_TABLES
            if packed:
                logical_dim = embedding_config.embedding_dim
                table_width = int8_embed.packed_fp16_width(logical_dim)
                table_dtype = torch.float16
                self._packed_dims.append(logical_dim)
                _logger.warning(
                    "[Plan51 int8-gather] table %r stored packed-int8: fp16 x%d "
                    "(int8 dim=%d + fp32 scale), gather dequants to %d-wide bf16",
                    embedding_config.name, table_width, logical_dim, logical_dim,
                )
            else:
                table_width = embedding_config.embedding_dim
                table_dtype = torchrec_data_type_to_torch_data_type(embedding_config.data_type)
                self._packed_dims.append(None)
            use_local_lookup = (
                self._local_small_table_lookup
                and embedding_config.name in _LOCAL_SMALL_TABLES
                and nve_config[i].cache_type == nve_layers.CacheType.NoCache
                and not packed
            )
            if use_local_lookup:
                local_dtype = _local_small_table_dtype(table_dtype)
                self._local_lookup_tables.add(embedding_config.name)
                _logger.warning(
                    "[Plan57 local-small-table] table %r bypasses NVE NoCache: "
                    "torch.nn.Embedding dtype=%s (checkpoint dtype=%s)",
                    embedding_config.name,
                    local_dtype,
                    table_dtype,
                )
                self.embeddings[embedding_config.name] = torch.nn.Embedding(
                    num_embeddings=embedding_config.num_embeddings,
                    embedding_dim=table_width,
                    device=nve_config[i].device,
                    dtype=local_dtype,
                )
            else:
                self.embeddings[embedding_config.name] = nve_layers.NVEmbedding(
                    num_embeddings=embedding_config.num_embeddings,
                    embedding_size=table_width,
                    data_type=table_dtype,
                    cache_type=nve_config[i].cache_type,
                    gpu_cache_size=nve_config[i].gpu_cache_size_in_bytes,
                    memblock=nve_config[i].memblock,
                    weight_init=None,
                    device=nve_config[i].device,
                    config=config
                )

        self._feature_names: List[List[str]] = [table.feature_names for table in table_configs]

    def reset_cache_metrics(self) -> None:
        for embedding in self.embeddings.values():
            if hasattr(embedding, "reset_cache_metrics"):
                embedding.reset_cache_metrics()

    def cache_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for name, embedding in self.embeddings.items():
            if hasattr(embedding, "cache_metrics"):
                metrics[name] = dict(embedding.cache_metrics())
        return metrics

    def _lookup_embedding(
        self,
        table_name: str,
        feature_name: str,
        embedding: torch.nn.Module,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        keys_t = _clamp_lookup_keys(feature_name, keys)
        if table_name in self._local_lookup_tables:
            # Native local GPU lookup for small fully-resident tables. This avoids
            # NVE NoCache dispatch and can emit bf16 directly when the local
            # weight is bf16, making the later model-side .to(bf16) a no-op.
            return torch.index_select(embedding.weight, 0, keys_t)
        return embedding(keys=keys_t)

    def prefetch_features(self, features) -> None:
        """Prefetch keys through the same NVE cache policy used by lookup.

        This is an experimental hook for cache investigations. It intentionally
        does not materialize model-visible embeddings.
        """
        feature_dict = features.to_dict() if isinstance(features, KeyedJaggedTensor) else features
        for i, embedding in enumerate(self.embeddings.values()):
            if not hasattr(embedding, "prefetch"):
                continue
            for feature_name in self._feature_names[i]:
                if feature_name not in feature_dict:
                    continue
                feature = feature_dict[feature_name]
                values = feature.values() if hasattr(feature, "values") and callable(feature.values) else feature.values
                keys_t = _clamp_lookup_keys(feature_name, values)
                with nvtx.annotate(f"prefetch_{feature_name}", color="yellow"):
                    embedding.prefetch(keys_t)

    def forward(self, features: Dict[str, List[torch.Tensor]]) -> Dict[str, JaggedTensor]:
        """
        Perform embedding lookups for input features.

        Supports two input formats:
        1. KeyedJaggedTensor: Returns Dict[str, JaggedTensor] with embeddings
        2. Dict[str, CustomJaggedTensor]: Updates each tensor in-place with embeddings

        Args:
            features: Input features as KeyedJaggedTensor or Dict[str, CustomJaggedTensor].

        Returns:
            Dict[str, JaggedTensor]: Dictionary mapping feature names to embedding tensors.
        """
        flat_feature_names: List[str] = []
        for names in self._feature_names:
            flat_feature_names.extend(names)

        global _clamp_logged
        if not _clamp_logged:
            _logger.warning(
                "[Plan21 §0.4] NVE embedding-lookup OOV-id clamp guard: "
                "DLRM_CLAMP_OOB_IDS=%s (active=%s)",
                os.environ.get("DLRM_CLAMP_OOB_IDS", "0"), _CLAMP_OOB_IDS,
            )
            _clamp_logged = True

        # Handle KeyedJaggedTensor input
        if isinstance(features, KeyedJaggedTensor):
            feature_dict = features.to_dict()
            feature_embeddings: Dict[str, JaggedTensor] = {}

            for i, (table_name, embedding) in enumerate(self.embeddings.items()):
                for feature_name in self._feature_names[i]:
                    if feature_name not in feature_dict:
                        continue
                    f = feature_dict[feature_name]
                    # Clamp OOV ids on the exact tensor fed to the NVE op so both
                    # the key-gather and the table-read see in-bounds indices.
                    with nvtx.annotate(f"embedding_{feature_name}", color="blue"):
                        res = self._lookup_embedding(
                            table_name,
                            feature_name,
                            embedding,
                            f.values(),
                        )
                    if self._packed_dims[i] is not None:
                        res = int8_embed.unpack_dequant(res, self._packed_dims[i])
                    feature_embeddings[feature_name] = JaggedTensor(
                        values=res,
                        lengths=f.lengths(),
                    )
            return feature_embeddings
        else:
            # input is Dict[str, List[torch.Tensor]]
            keys = features.keys()
            for i, (table_name, embedding) in enumerate(self.embeddings.items()):
                for feature_name in self._feature_names[i]:
                    if feature_name not in keys:
                        continue
                    with nvtx.annotate(f"embedding_{feature_name}", color="blue"):
                        res = self._lookup_embedding(
                            table_name,
                            feature_name,
                            embedding,
                            features[feature_name].values,
                        )
                    if self._packed_dims[i] is not None:
                        res = int8_embed.unpack_dequant(res, self._packed_dims[i])
                    with nvtx.annotate(f"embedding_{feature_name} - packing", color="blue"):
                        features[feature_name].embeddings = res
            return features

    def embedding_bag_configs(self):
        """
        Get embedding table configurations.

        Returns:
            List[EmbeddingConfig]: List of embedding table configurations.
        """
        return self._embedding_configs

    def is_weighted(self) -> bool:
        """
        Check if embedding collection uses weighted pooling.

        Returns:
            bool: False (NVE embeddings don't use weighted pooling).
        """
        return False
