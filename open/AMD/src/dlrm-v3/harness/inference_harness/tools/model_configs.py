"""
Model configuration utilities for DLRM-HSTU inference.

Provides configuration factories for HSTU models, embedding tables, backends,
datasets, and communication settings. Supports both production and debug
configurations for flexible testing and deployment.
"""

from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from typing import Dict, Union
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig
import os
from dataclasses import dataclass

from inference_harness.mpi_utils import ZMQRequestSenderShardedConfig
from inference_harness.dataset.mlperf_streaming_qsl import DLRMv3StreamingMLPerfDataset
from inference_harness.dataset.streaming_query_sampler import StreamingQuerySamplerRef
import torch

# Model configuration constants
HSTU_EMBEDDING_DIM = 512
MOVIE_EMBEDDING_TABLE_SIZE = 500_000_000
USER_EMBEDDING_TABLE_SIZE = 3_000_000


import logging
import sys
logger = logging.getLogger(__name__)


def get_hstu_configs(model: str = "production") -> DlrmHSTUConfig:
    """
    Get HSTU model configuration.

    Args:
        model: Configuration preset ("production" or "debug").

    Returns:
        DlrmHSTUConfig: HSTU model configuration.

    Raises:
        ValueError: If model preset is not recognized.
    """
    if model == "production":
        hstu_config = DlrmHSTUConfig(
            hstu_num_heads=4,
            hstu_attn_linear_dim=128,
            hstu_attn_qk_dim=128,
            hstu_attn_num_layers=5,
            hstu_embedding_table_dim=HSTU_EMBEDDING_DIM,
            hstu_preprocessor_hidden_dim=256,
            hstu_transducer_embedding_dim=512,
            hstu_group_norm=False,
            hstu_input_dropout_ratio=0.2,
            hstu_linear_dropout_rate=0.1,
            causal_multitask_weights=0.2,
        )
        hstu_config.user_embedding_feature_names = [
            "item_id",
            "user_id",
            "item_category_id",
        ]
        hstu_config.item_embedding_feature_names = [
            "item_candidate_id",
            "item_candidate_category_id",
        ]
        hstu_config.uih_post_id_feature_name = "item_id"
        hstu_config.uih_action_time_feature_name = "action_timestamp"
        hstu_config.candidates_querytime_feature_name = "item_query_time"
        hstu_config.candidates_weight_feature_name = "item_action_weights"
        hstu_config.uih_weight_feature_name = "item_weights"
        hstu_config.candidates_watchtime_feature_name = "item_rating"
        hstu_config.action_weights = [1, 2, 4, 8, 16]
        hstu_config.action_embedding_init_std = 5.0
        hstu_config.contextual_feature_to_max_length = {"user_id": 1}
        hstu_config.contextual_feature_to_min_uih_length = {"user_id": 20}
        hstu_config.merge_uih_candidate_feature_mapping = [
            ("item_id", "item_candidate_id"),
            ("item_rating", "item_candidate_rating"),
            ("action_timestamp", "item_query_time"),
            ("item_weights", "item_action_weights"),
            ("dummy_watch_time", "item_dummy_watchtime"),
            ("item_category_id", "item_candidate_category_id"),
        ]
        hstu_config.hstu_uih_feature_names = [
            "user_id",
            "item_id",
            "item_rating",
            "action_timestamp",
            "item_weights",
            "dummy_watch_time",
            "item_category_id",
        ]
        hstu_config.hstu_candidate_feature_names = [
            "item_candidate_id",
            "item_candidate_rating",
            "item_query_time",
            "item_action_weights",
            "item_dummy_watchtime",
            "item_candidate_category_id",
        ]
        hstu_config.max_num_candidates = 32
        hstu_config.max_num_candidates_inference = 2048
        hstu_config.multitask_configs = [
            TaskConfig(
                task_name="rating",
                task_weight=1,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            )
        ]
        return hstu_config
    elif model == "debug":
        hstu_config = get_hstu_configs("production")
        return hstu_config

    else:
        raise NotImplementedError(f"Model {model} not implemented")


def get_embedding_table_config(model: str = "production") -> Dict[str, EmbeddingConfig]:
    """
    Get embedding table configurations.

    Args:
        model: Configuration preset ("production" or "debug").

    Returns:
        Dict[str, EmbeddingConfig]: Dictionary mapping table names to configurations.

    Raises:
        ValueError: If model preset is not recognized.
    """
    if model == "production":
        return {
            "item_id": EmbeddingConfig(
                num_embeddings=1_000_000_000,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="item_id",
                data_type=DataType.FP16,
                feature_names=["item_id", "item_candidate_id"],
            ),
            "item_category_id": EmbeddingConfig(
                num_embeddings=128,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="item_category_id",
                data_type=DataType.FP16,
                weight_init_max=1.0,
                weight_init_min=-1.0,
                feature_names=["item_category_id", "item_candidate_category_id"],
            ),
            "user_id": EmbeddingConfig(
                num_embeddings=10_000_000,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="user_id",
                data_type=DataType.FP16,
                feature_names=["user_id"],
            ),
        }
    elif model == "debug":
        return {
            "item_id": EmbeddingConfig(
                num_embeddings=1_000_000_000,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="item_id",
                data_type=DataType.FP16,
                feature_names=["item_id", "item_candidate_id"],
            ),
            "item_category_id": EmbeddingConfig(
                num_embeddings=128,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="item_category_id",
                data_type=DataType.FP16,
                weight_init_max=1.0,
                weight_init_min=-1.0,
                feature_names=["item_category_id", "item_candidate_category_id"],
            ),
            "user_id": EmbeddingConfig(
                num_embeddings=50000,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="user_id",
                data_type=DataType.FP16,
                feature_names=["user_id"],
            ),
        }
    else:
        raise NotImplementedError(f"Model {model} not implemented")


def _maybe_cap_embedding_tables(tables: Dict[str, "EmbeddingConfig"]) -> Dict[str, "EmbeddingConfig"]:
    """Plan 14.5b (ROCm NVE single-GPU smoke): clamp table row counts.

    The single-GPU NVE path (DLRM_ROCM_NVE=1, NoCache) allocates each table in full
    on one GPU; the production item_id table (1e9 rows x 512 x fp16 ~= 954 GiB) cannot
    fit. DLRM_NVE_SMOKE_HASH_CAP=<rows> clamps every table's num_embeddings so the
    NVE-wired path can be smoke-validated end-to-end on a single GPU. The capped tables
    no longer match the checkpoint shards, so load_model_sparse skips them in this mode
    (bit-exact lookup is certified separately by the standalone 14.6 audit).
    """
    import os
    cap_env = os.environ.get("DLRM_NVE_SMOKE_HASH_CAP", "")
    if not cap_env:
        return tables
    cap = int(cap_env)
    import dataclasses
    capped = {}
    for name, cfg in tables.items():
        if cfg.num_embeddings > cap:
            capped[name] = dataclasses.replace(cfg, num_embeddings=cap)
        else:
            capped[name] = cfg
    return capped


def _maybe_bf16_embedding_tables(tables: Dict[str, "EmbeddingConfig"]) -> Dict[str, "EmbeddingConfig"]:
    """Store selected NVE embedding tables in bf16 instead of fp16 to eliminate the
    per-iteration fp16->bf16 cast of the gathered embeddings (the HSTU dense path runs
    in bf16). Bit-exact for the kConcat (single-gather, no-reduction) NVE lookup: the
    bf16 table row equals fp16_row.to(bfloat16), identical to today's harness cast.

    Gated by DLRM_NVE_BF16_GATHER:
      unset / "0"          -> no change (fp16 tables, harness casts every gather)
      "1" / "item_id"      -> item_id only (LinearUVM byte-copy gather emits bf16 for
                              free; the harness .to(bfloat16) becomes a no-op).
      "all"                -> all tables (NoCache tables also need NVE bf16 cuembed
                              support; not enabled until that lands).
    """
    import dataclasses
    mode = os.environ.get("DLRM_NVE_BF16_GATHER", "0").strip().lower()
    if mode in ("", "0", "false", "no"):
        return tables
    if mode in ("1", "item_id", "true", "yes"):
        targets = {"item_id"}
    elif mode == "all":
        targets = set(tables.keys())
    else:
        targets = {t.strip() for t in mode.split(",") if t.strip()}
    out = {}
    for name, cfg in tables.items():
        if name in targets and cfg.data_type != DataType.BF16:
            logger.warning("[bf16-gather] storing embedding table %r in bf16 (was %s)", name, cfg.data_type)
            out[name] = dataclasses.replace(cfg, data_type=DataType.BF16)
        else:
            out[name] = cfg
    return out


@dataclass
class RocmGRBackendConfig:
    """Minimal backend config for DLRM_ROCM_GR_BACKEND (open HSTUModelFamily per worker)."""

    batch_size: int
    perf_mode: str
    use_custom_stu: bool = False
    use_nve: bool = False
    use_multi_gpu: bool = False


def get_backend_config(args, embedding_table_config: Dict[str, EmbeddingConfig], world_size: int):
    """
    Get backend configuration for hybrid GR inference.

    Args:
        args: Command-line arguments containing backend settings.
        embedding_table_config: Embedding table configurations.
        world_size: Total number of MPI processes.

    Returns:
        Backend configuration object.
    """
    # Plan 14.5: on ROCm the GR backend (Plans 4-13) is the default sparse path.
    # DLRM_ROCM_NVE=1 opts into the ported NVE lookup (Plan 14) instead — reversible,
    # and the CUDA path (DLRM_ROCM_GR_BACKEND unset) is unchanged.
    rocm_nve = os.environ.get("DLRM_ROCM_NVE", "0") == "1"
    if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1" and not rocm_nve:
        return RocmGRBackendConfig(
            batch_size=args.batch_size,
            perf_mode=args.mode,
        )

    from inference_harness.backends.hybrid_GR_backend import HybridGRBackendConfig
    from inference_harness.backends.ops import torch_data_type_to_nve_data_type
    import pynve.nve as nve

    # gpu_cache_size for the LinearUVM/MPI item_id table. NV used 10 GB/rank on B200;
    # MI355 has ~288 GB usable, so allow a larger, env-overridable default on ROCm.
    # Accept fractional GB for cache sensitivity probes, e.g. 7.5.
    nve_gpu_cache_gb = float(os.environ.get("DLRM_NVE_GPU_CACHE_GB", "10"))

    if args.use_mpi_lookup:
        partial_ranks = list(range(world_size - 1))
        local_device_ids = [i % torch.cuda.device_count() for i in range(world_size - 1)]
        item_table_config = embedding_table_config["item_id"]

        item_memblock_width = item_table_config.embedding_dim
        int8_gather = os.environ.get("DLRM_NVE_INT8_GATHER", "0").strip().lower()
        if int8_gather in ("1", "item_id", "true", "yes"):
            from inference_harness.backends import int8_embed

            item_memblock_width = int8_embed.packed_fp16_width(item_table_config.embedding_dim)

        NVL = nve.MPIMemBlock(
            item_memblock_width,
            item_table_config.num_embeddings,
            torch_data_type_to_nve_data_type(torch.float16),
            partial_ranks,
            local_device_ids,
        )
        backend_config = HybridGRBackendConfig(
            batch_size=args.batch_size,
            perf_mode=args.mode,
            # Plan 14.7b (ROCm): use_custom_stu=False — same reason as the 14.5b
            # single-GPU branch below. The custom STU is a CUDA Blackwell
            # (hstu_blackwell) kernel from /opt/FBGEMM, absent on gfx950; the ROCm
            # data-parallel NVE path uses the open Triton/PyTorch HSTU.
            use_custom_stu=False if rocm_nve else True,
            use_nve=True,
            use_multi_gpu=True,
            use_mpi=True,
            device_ids=[],
            gpu_cache_size_in_gigabytes=nve_gpu_cache_gb,
            nve_memblock=NVL,
        )
    elif rocm_nve:
        # Plan 14.5: single-GPU NVE on ROCm (item_id → NoCache, no MPIMemBlock).
        # Validates the ported NVE lookup in the harness without the cross-rank
        # MPIMemBlock sharing (that data-parallel path is the Phase 14.7 headline).
        # Plan 14.5b: use_custom_stu=False — the custom STU is a CUDA Blackwell
        # (hstu_blackwell) kernel from /opt/FBGEMM that does not exist on gfx950;
        # the ROCm path uses the open Triton HSTU (same as the GR backend).
        backend_config = HybridGRBackendConfig(
            batch_size=args.batch_size,
            perf_mode=args.mode,
            use_custom_stu=False,
            use_nve=True,
            use_multi_gpu=False,
            use_mpi=False,
            device_ids=[],
            gpu_cache_size_in_gigabytes=nve_gpu_cache_gb,
        )
    else:
        # base line meta's ref implementation, run with mpirun -n 2, 1 rank for entire backend, and modify issue_query to send to rank 0
        backend_config = HybridGRBackendConfig(
            batch_size=args.batch_size,
            perf_mode=args.mode,
            use_custom_stu=True,
            use_nve=False,
            use_multi_gpu=False,
        )
    return backend_config


def get_communicator_config(args) -> ZMQRequestSenderShardedConfig:
    """
    Get communicator configuration for inter-process communication.

    Args:
        args: Command-line arguments containing communicator settings.

    Returns:
        ZMQRequestSenderShardedConfig: Communicator configuration.

    Raises:
        NotImplementedError: If MPI communicator is selected (not yet implemented).
    """
    if args.communicator_type == "mpi":
        raise NotImplementedError("MPI communicator is not implemented yet")
    else:
        return ZMQRequestSenderShardedConfig(
            loadgen_hostname=args.loadgen_hostname,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            num_preds=2048,
            mode=args.mode,
            gpus_per_node=args.gpus_per_node,
        )


def get_dataset_latest(
    hstu_config,
    dataset_path: str,
    mode: str,
    total_queries: int,
    dataset_percentage: float,
    device: torch.device,
    scenario_name: str,
    offline_target_qps: int,
    target_duration: float,
    compute_eval: bool = False,
    batching_on_gpu: bool = False,
    max_buffer_indices: int = 1500000,
    max_buffer_lengths: int = 256,
) -> StreamingQuerySamplerRef:
    """
    Create streaming query sampler for MLPerf dataset.

    Args:
        hstu_config: HSTU model configuration.
        dataset_path: Path to the dataset directory.
        mode: Operating mode ("performance" or "accuracy").
        total_queries: Total number of queries to generate.
        dataset_percentage: Percentage of dataset to use (0.0-1.0).
        device: CUDA device for GPU batching (None for CPU-only).
        scenario_name: MLPerf scenario name ("Server" or "Offline").
        offline_target_qps: Target QPS for Offline scenario.
        target_duration: Target duration in milliseconds.
        compute_eval: Whether to compute evaluation metrics.
        batching_on_gpu: Whether to enable GPU-accelerated batching.
        max_buffer_indices: Maximum buffer size for indices.
        max_buffer_lengths: Maximum buffer size for lengths.

    Returns:
        StreamingQuerySamplerRef: Configured query sampler.
    """
    logger.info(f"Preparing dataset for mode: {mode}")
    if mode == "accuracy":
        # Normal AccuracyOnly uses eval-shaped 32-candidate samples. TEST08 is
        # different: its reference log must use the same inference candidate set
        # as the audited Server Performance log, then report labels/weights for
        # that 2048-wide response layout. Keep this opt-in so GAUC scoring keeps
        # the standard accuracy dataset semantics.
        is_inference = os.environ.get("DLRM_ACCURACY_USE_INFERENCE_DATASET", "0") == "1"
        total_queries = None
    else:
        is_inference = True
    dataset = DLRMv3StreamingMLPerfDataset(
        hstu_config=hstu_config,
        ratings_file_prefix=dataset_path,
        is_inference=is_inference,
        train_ts=90,
        total_ts=100,
        num_files=1,
        num_users=50000,
        num_items=1_000_000_000,
        num_categories=128,
        device=device,
        batching_on_gpu=batching_on_gpu,
        max_buffer_indices=max_buffer_indices,
        max_buffer_lengths=max_buffer_lengths,
    )
    streaming_query_sampler = StreamingQuerySamplerRef(
        ds=dataset,
        dataset_percentage=dataset_percentage,
        scenario_name=scenario_name,
        offline_target_qps=offline_target_qps,
        target_duration=target_duration,
        input_queries=total_queries,
        compute_eval=compute_eval,
    )
    # always load the entire dataset
    return streaming_query_sampler
