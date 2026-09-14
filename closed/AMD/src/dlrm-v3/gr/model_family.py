# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
"""
model_family for dlrm_v3.
"""

import copy
import functools
import logging
import os
import sys
import time
import uuid
from threading import Event, Lock
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torchrec
from checkpoint import (
    load_nonsparse_checkpoint,
    load_sparse_checkpoint,
)
from configs import HASH_SIZE
from datasets.dataset import Samples
from inference_modules import (
    get_hstu_model,
    HSTUSparseInferenceModule,
    move_sparse_output_to_device,
    set_is_inference,
)
from sparse_routing import replicate_enabled, route_lookup
from sparse_slicing import _is_sharded_fqn, _sharded_fqn_patterns
from timing_stats import record as timing_record
from utils import Profiler
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig, SequenceEmbedding
from pyre_extensions import none_throws
from torch import quantization as quant
from torchrec.distributed.quant_embedding import QuantEmbeddingCollection
from torchrec.modules.embedding_configs import EmbeddingConfig, QuantConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.sparse.tensor_dict import maybe_td_to_kjt
from torchrec.test_utils import get_free_port

logger: logging.Logger = logging.getLogger(__name__)

_PIPELINE_H2D_TO_WORKER: bool = os.environ.get("DLRM_PIPELINE_H2D", "1") == "1"


def _dense_gpu_ids() -> List[int]:
    """Physical GPU indices for dense workers (skip cuda:0 if it HIP-segfaults)."""
    ngpus = torch.cuda.device_count()
    skip0 = os.environ.get("DLRM_SKIP_GPU0", "0") == "1"
    if skip0 and ngpus > 1:
        gpu_ids = [i for i in range(ngpus) if i != 0]
        logger.warning(
            "DLRM_SKIP_GPU0=1: dense workers use GPU ids %s (skip cuda:0)",
            gpu_ids,
        )
        return gpu_ids
    return list(range(ngpus))


def _clone_dense_worker_batch(
    item: Tuple[
        uuid.UUID,
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
    ],
) -> Tuple[
    uuid.UUID,
    Dict[str, SequenceEmbedding],
    Dict[str, torch.Tensor],
    int,
    torch.Tensor,
    int,
    torch.Tensor,
]:
    """Clone GPU batch tensors only (much faster than copy.deepcopy)."""
    (
        batch_id,
        seq_embeddings,
        payload_features,
        max_uih_len,
        uih_seq_lengths,
        max_num_candidates,
        num_candidates,
    ) = item
    seq_embeddings = {
        k: SequenceEmbedding(
            lengths=seq_embeddings[k].lengths.clone(),
            embedding=seq_embeddings[k].embedding.clone(),
        )
        for k in seq_embeddings
    }
    payload_features = {k: v.clone() for k, v in payload_features.items()}
    return (
        batch_id,
        seq_embeddings,
        payload_features,
        max_uih_len,
        uih_seq_lengths.clone(),
        max_num_candidates,
        num_candidates.clone(),
    )


class HSTUModelFamily:
    """
    High-level interface for the HSTU model family.

    Manages both sparse (embedding) and dense (transformer) components of the
    HSTU model, supporting distributed inference across multiple GPUs.

    Args:
        hstu_config: Configuration object for the HSTU model.
        table_config: Dictionary of embedding table configurations.
        output_trace: Whether to enable profiling trace output.
        sparse_quant: Whether to quantize sparse embeddings.
        compute_eval: Whether to compute evaluation metrics (includes labels).
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
        sparse_quant: bool = False,
        compute_eval: bool = False,
    ) -> None:
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.sparse: ModelFamilySparseDist = ModelFamilySparseDist(
            hstu_config=hstu_config,
            table_config=table_config,
            quant=sparse_quant,
        )

        assert torch.cuda.is_available(), "CUDA is required for this benchmark."
        ngpus = torch.cuda.device_count()
        self.world_size = int(os.environ.get("WORLD_SIZE", str(ngpus)))
        logger.warning(f"Using {self.world_size} GPU(s)...")
        dense_model_family_clazz = (
            ModelFamilyDenseDist
            if self.world_size > 1
            else ModelFamilyDenseSingleWorker
        )
        self.dense: Union[ModelFamilyDenseDist, ModelFamilyDenseSingleWorker] = (
            dense_model_family_clazz(
                hstu_config=hstu_config,
                table_config=table_config,
                output_trace=output_trace,
                compute_eval=compute_eval,
            )
        )
        # Sparse runs on CPU in the main process; serialize concurrent predict() calls.
        self._sparse_lock: Lock = Lock()

    def version(self) -> str:
        """Return the PyTorch version string."""
        return torch.__version__

    def name(self) -> str:
        """Return the model family name identifier."""
        return "model-family-hstu"

    def load(self, model_path: str) -> None:
        """
        Load model checkpoints from disk.

        Args:
            model_path: Base path to the model checkpoint directory.
        """
        self.sparse.load(model_path=model_path)
        self.dense.load(model_path=model_path)

    def predict(
        self, samples: Optional[Samples]
    ) -> Optional[
        Tuple[
            torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, float
        ]
    ]:
        """
        Run inference on a batch of samples.

        Processes samples through sparse embeddings, then dense forward pass.

        Args:
            samples: Input samples containing features. If None, signals shutdown.

        Returns:
            Tuple of (predictions, labels, weights, sparse_time, dense_time) or None.
        """
        with torch.inference_mode():
            if samples is None:
                self.dense.predict(None, None, 0, None, 0, None)
                return None
            with self._sparse_lock:
                (
                    seq_embeddings,
                    payload_features,
                    max_uih_len,
                    uih_seq_lengths,
                    max_num_candidates,
                    num_candidates,
                    dt_sparse,
                ) = self.sparse.predict(samples)
            timing_record("predict.sparse", dt_sparse)
            out = self.dense.predict(
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            )
            (  # pyre-ignore [23]
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
                dt_dense,
            ) = out
            return (
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
                dt_sparse,
                dt_dense,
            )

    def shutdown(self) -> None:
        """Stop dense worker processes (required before profile/benchmark exit)."""
        self.dense.shutdown()


def ec_patched_forward_wo_embedding_copy(
    ec_module: torchrec.EmbeddingCollection,
    features: KeyedJaggedTensor,  # can also take TensorDict as input
) -> Dict[str, JaggedTensor]:
    """
    Run the EmbeddingBagCollection forward pass. This method takes in a `KeyedJaggedTensor`
    and returns a `Dict[str, JaggedTensor]`, which is the result of the individual embeddings for each feature.

    Args:
        features (KeyedJaggedTensor): KJT of form [F X B X L].

    Returns:
        Dict[str, JaggedTensor]
    """
    features = maybe_td_to_kjt(features, None)
    feature_embeddings: Dict[str, JaggedTensor] = {}
    jt_dict: Dict[str, JaggedTensor] = features.to_dict()
    # Phase 2b Step 3c: when the replicated-cap path is *off* and the
    # FQN is sharded (rank holds only its own [r*S, (r+1)*S) slice),
    # route every batch through ``all_to_all_single`` so any global
    # index resolves to the correct embedding regardless of which
    # rank actually holds the row. Computed once per forward.
    routing_active = not replicate_enabled()
    sharded_patterns = _sharded_fqn_patterns() if routing_active else []
    for i, (table_name, emb_module) in enumerate(ec_module.embeddings.items()):
        feature_names = ec_module._feature_names[i]
        embedding_names = ec_module._embedding_names_by_table[i]
        # Phase 2b: when the table weights live on a different device than the
        # incoming indices (e.g. CPU input + cuda:0 weights), bounce indices to
        # the weight device for the lookup and bring the result back so the
        # rest of preprocess (cat / cumsum / payload zeros) stays single-device.
        # Clamp against this table's actual row count, not the global
        # HASH_SIZE, so DLRM_SPARSE_MAX_HASH_SIZE-capped tables don't
        # index past their end (GPU OOB → HSA hardware exception).
        #
        # Phase 2b mode selection for this table:
        #   * Replicate ON (Step 3b, default for WORLD>1): ``table_rows``
        #     is the full global cap ``W*S`` and rows ``[0, W*S)`` are
        #     populated on every rank → rank-local clamp + lookup is
        #     globally correct.
        #   * Replicate OFF + sharded FQN (Step 3c): ``table_rows == S``;
        #     route each global index to its owner via
        #     ``all_to_all_single`` and reassemble; OOV → zero.
        #   * Replicate OFF + non-sharded FQN: replicated table on every
        #     rank, rank-local clamp + lookup is correct.
        weight_device = emb_module.weight.device
        table_rows = int(emb_module.weight.shape[0])
        clamp_max = min(HASH_SIZE - 1, table_rows - 1)
        use_routing = routing_active and _is_sharded_fqn(
            f"{table_name}.weight", sharded_patterns
        )
        for j, embedding_name in enumerate(embedding_names):
            feature_name = feature_names[j]
            f = jt_dict[feature_name]
            raw_values = f.values()
            src_device = raw_values.device
            if use_routing:
                # Hand un-clamped global indices to the router; it does
                # its own range mask + zero-fill for OOV positions.
                lookup = route_lookup(
                    emb_module, raw_values, shard_rows=table_rows
                )
            else:
                indices = torch.clamp(raw_values, min=0, max=clamp_max)
                if indices.device != weight_device:
                    indices = indices.to(weight_device, non_blocking=True)
                lookup = emb_module(
                    input=indices
                )  # remove the dtype cast at https://github.com/meta-pytorch/torchrec/blob/0a2cebd5472a7edc5072b3c912ad8aaa4179b9d9/torchrec/modules/embedding_modules.py#L486
            if lookup.device != src_device:
                lookup = lookup.to(src_device, non_blocking=True)
            feature_embeddings[embedding_name] = JaggedTensor(
                values=lookup,
                lengths=f.lengths(),
                weights=f.values() if ec_module._need_indices else None,
            )
    return feature_embeddings


class ModelFamilySparseDist:
    """
    Sparse Arch module manager.

    Handles loading and inference of sparse embedding lookups, optionally
    with quantization for memory efficiency.

    Args:
        hstu_config: HSTU model configuration.
        table_config: Embedding table configurations.
        quant: Whether to apply dynamic quantization to embeddings.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        quant: bool = False,
    ) -> None:
        super(ModelFamilySparseDist, self).__init__()
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.module: Optional[torch.nn.Module] = None
        self.quant: bool = quant

    def load(self, model_path: str) -> None:
        """
        Load sparse model checkpoint and optionally apply quantization.

        Args:
            model_path: Path to the model checkpoint directory.
        """
        logger.warning(f"Loading sparse module from {model_path}")

        sparse_arch: HSTUSparseInferenceModule = HSTUSparseInferenceModule(
            table_config=self.table_config,
            hstu_config=self.hstu_config,
        )
        # Phase 2b sparse load mode selection:
        #
        # Default (no cap, no override) → DefaultLoadPlanner: full ~1 TB load.
        # DLRM_SPARSE_MAX_HASH_SIZE=N    → SlicingLoadPlanner (in checkpoint.py)
        #                                  narrows saved tensors to the live
        #                                  capped shape and loads real values.
        # DLRM_SPARSE_SKIP_CKPT=1        → Step-1-style probe: skip the load
        #                                  entirely and zero-init embeddings
        #                                  (fast smoke, predictions garbage).
        # DLRM_SPARSE_SLICE_CKPT=0/1     → Explicit override on the slicing
        #                                  planner; respected by checkpoint.py.
        skip_env = os.environ.get("DLRM_SPARSE_SKIP_CKPT", "").strip().lower()
        skip_ckpt = skip_env in {"1", "true", "yes"}
        if skip_ckpt:
            logger.warning(
                "[phase2b-probe] Skipping sparse checkpoint load "
                "(DLRM_SPARSE_SKIP_CKPT=1); zero-init embedding tables",
            )
            for name, p in sparse_arch.named_parameters():
                if "embedding" in name.lower() and p.is_meta is False:
                    p.data.zero_()
            for name, b in sparse_arch.named_buffers():
                if "embedding" in name.lower() and b.is_meta is False:
                    b.data.zero_()
        else:
            load_sparse_checkpoint(model=sparse_arch._hstu_model, path=model_path)
        sparse_arch.eval()
        if self.quant:
            self.module = quant.quantize_dynamic(
                sparse_arch,
                qconfig_spec={
                    torchrec.EmbeddingCollection: QuantConfig(
                        activation=quant.PlaceholderObserver.with_args(
                            dtype=torch.float
                        ),
                        weight=quant.PlaceholderObserver.with_args(
                            dtype=torch.int8),
                    ),
                },
                mapping={
                    torchrec.EmbeddingCollection: QuantEmbeddingCollection,
                },
                inplace=False,
            )
        else:
            sparse_arch._hstu_model._embedding_collection.forward = (  # pyre-ignore[8]
                functools.partial(
                    ec_patched_forward_wo_embedding_copy,
                    sparse_arch._hstu_model._embedding_collection,
                )
            )
            self.module = sparse_arch
        # Phase 2b Step 3b: log the chosen multi-worker correctness mode so
        # operators can tell at a glance whether predictions are guaranteed
        # globally correct (replicated cap) or rank-divergent (Step-3a
        # shard storage without Step-3c routing).
        try:
            world = int(os.environ.get("DLRM_SPARSE_WORLD", "1"))
        except ValueError:
            world = 1
        if world > 1:
            per_table = {
                name: int(emb.weight.shape[0])
                for name, emb in (
                    sparse_arch._hstu_model._embedding_collection.embeddings.items()
                )
            }
            if replicate_enabled():
                logger.warning(
                    "[phase2b-3b] replicated-cap mode: WORLD=%d RANK=%s "
                    "per-rank live rows=%s (each rank holds the full "
                    "global capped sparse; rank-local lookup is "
                    "globally correct)",
                    world,
                    os.environ.get("DLRM_SPARSE_RANK", "0"),
                    per_table,
                )
            else:
                # Step 3c: pre-warm the worker PG so the first warmup
                # forward doesn't pay the init RTT under the routing
                # path. ``get_worker_process_group`` is lazy so this is
                # the *only* call that pays the cost.
                from sparse_routing import (  # noqa: WPS433
                    get_worker_process_group,
                    routing_backend,
                )
                pg = get_worker_process_group()
                logger.warning(
                    "[phase2b-3c] sharded-routing mode: WORLD=%d RANK=%s "
                    "per-rank live rows=%s pg=%s backend=%s "
                    "(every sparse forward routes via all_to_all_single "
                    "across worker ranks; each rank holds only its own "
                    "shard [r*S, (r+1)*S))",
                    world,
                    os.environ.get("DLRM_SPARSE_RANK", "0"),
                    per_table,
                    "ready" if pg is not None else "unavailable",
                    routing_backend() or "n/a",
                )
        logger.warning(f"sparse module is {self.module}")

    def predict(
        self, samples: Samples
    ) -> Tuple[
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
        float,
    ]:
        """
        Run sparse forward pass (embedding lookups).

        Args:
            samples: Input samples with feature tensors.

        Returns:
            Tuple of (seq_embeddings, payload_features, max_uih_len, uih_seq_lengths,
            max_num_candidates, num_candidates, elapsed_time).
        """
        with torch.profiler.record_function("sparse forward"):
            module: torch.nn.Module = none_throws(self.module)
            assert self.module is not None
            uih_features = samples.uih_features_kjt
            candidates_features = samples.candidates_features_kjt
            t0: float = time.time()
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = module(
                uih_features=uih_features,
                candidates_features=candidates_features,
            )
            dt_sparse: float = time.time() - t0
            return (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
                dt_sparse,
            )


def _dense_worker_main(
    rank: int,
    gpu_id: int,
    world_size: int,
    model_path: str,
    samples_q: mp.Queue,
    result_q: mp.Queue,
    ready_q: mp.Queue,
    hstu_config: DlrmHSTUConfig,
    table_config: Dict[str, EmbeddingConfig],
    output_trace: bool,
    compute_eval: bool,
) -> None:
    """
    Top-level dense worker entry (must be picklable for mp.spawn).

    Do not use a bound method here: pickling ModelFamilyDenseDist fails with
    TypeError: cannot pickle 'weakref.ReferenceType' when starting rank > 0.
    """
    import sys

    def _wlog(msg: str) -> None:
        print(
            f"[dense-worker rank={rank} pid={os.getpid()}] {msg}",
            flush=True,
            file=sys.stderr,
        )

    _wlog("worker started")
    from generative_recommenders.ops.rocm_compat import apply_rocm_fbgemm_cumsum_patch

    apply_rocm_fbgemm_cumsum_patch()
    set_is_inference(is_inference=not compute_eval)
    _wlog("building HSTU model on CPU ...")
    model = get_hstu_model(
        table_config=table_config,
        hstu_config=hstu_config,
        table_device="cpu",
        max_hash_size=100,
        is_dense=True,
    ).to(torch.bfloat16)
    model.set_training_dtype(torch.bfloat16)
    from generative_recommenders.common import HammerKernel

    hk = os.environ.get("DLRM_HAMMER_KERNEL", "").upper()
    if hk == "PYTORCH":
        model.set_hammer_kernel(HammerKernel.PYTORCH)
        _wlog("HammerKernel.PYTORCH (DLRM_HAMMER_KERNEL)")
    device = torch.device(f"cuda:{gpu_id}")
    _wlog(f"set_device cuda:{gpu_id} (worker rank={rank})")
    torch.cuda.set_device(device)
    _wlog("loading non_sparse.ckpt ...")
    load_nonsparse_checkpoint(
        model=model, device=device, optimizer=None, path=model_path
    )
    _wlog("moving model to GPU ...")
    model = model.to(device)
    model.eval()
    _wlog("ready — signaling parent")
    ready_q.put(rank)
    profiler = Profiler(rank) if output_trace else None

    try:
        _dense_worker_loop(
            rank,
            device,
            model,
            samples_q,
            result_q,
            profiler,
            output_trace,
        )
    except Exception:
        import traceback

        _wlog("worker crashed:\n" + traceback.format_exc())
        raise


def _dense_worker_loop(
    rank: int,
    device: torch.device,
    model: torch.nn.Module,
    samples_q: mp.Queue,
    result_q: mp.Queue,
    profiler: Optional[Profiler],
    output_trace: bool,
) -> None:
    with torch.inference_mode():
        while True:
            item = samples_q.get()
            if item == -1:
                break
            if output_trace:
                assert profiler is not None
                profiler.step()
            with torch.profiler.record_function("get_item_from_queue"):
                (
                    id,
                    seq_embeddings,
                    payload_features,
                    max_uih_len,
                    uih_seq_lengths,
                    max_num_candidates,
                    num_candidates,
                ) = item
                assert seq_embeddings is not None
                if _PIPELINE_H2D_TO_WORKER:
                    seq_embeddings, payload_features, uih_seq_lengths, num_candidates = (
                        move_sparse_output_to_device(
                            seq_embeddings=seq_embeddings,
                            payload_features=payload_features,
                            uih_seq_lengths=uih_seq_lengths,
                            num_candidates=num_candidates,
                            device=device,
                        )
                    )
                if os.environ.get("DLRM_SKIP_DENSE_BATCH_CLONE", "0") != "1":
                    (
                        id,
                        seq_embeddings,
                        payload_features,
                        max_uih_len,
                        uih_seq_lengths,
                        max_num_candidates,
                        num_candidates,
                    ) = _clone_dense_worker_batch(
                        (
                            id,
                            seq_embeddings,
                            payload_features,
                            max_uih_len,
                            uih_seq_lengths,
                            max_num_candidates,
                            num_candidates,
                        )
                    )
            with torch.profiler.record_function("dense forward"):
                if os.environ.get("DLRM_DENSE_WORKER_VERBOSE", "0") == "1":
                    print(
                        f"[dense-worker rank={rank}] main_forward start",
                        flush=True,
                        file=sys.stderr,
                    )
                (
                    _,
                    _,
                    _,
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                ) = model.main_forward(
                    seq_embeddings=seq_embeddings,
                    payload_features=payload_features,
                    max_uih_len=max_uih_len,
                    uih_seq_lengths=uih_seq_lengths,
                    max_num_candidates=max_num_candidates,
                    num_candidates=num_candidates,
                )
                if os.environ.get("DLRM_DENSE_WORKER_VERBOSE", "0") == "1":
                    print(
                        f"[dense-worker rank={rank}] main_forward done",
                        flush=True,
                        file=sys.stderr,
                    )
                assert mt_target_preds is not None
                mt_target_preds = mt_target_preds.detach().to(device="cpu")
                if mt_target_labels is not None:
                    mt_target_labels = mt_target_labels.detach().to(device="cpu")
                if mt_target_weights is not None:
                    mt_target_weights = mt_target_weights.detach().to(device="cpu")
                result_q.put(
                    (id, mt_target_preds, mt_target_labels, mt_target_weights)
                )


class ModelFamilyDenseDist:
    """
    Distributed dense module manager for multi-GPU inference.

    Spawns worker processes for each GPU to run dense forward passes in parallel,
    with samples distributed via inter-process queues.

    Args:
        hstu_config: HSTU model configuration.
        table_config: Embedding table configurations.
        output_trace: Whether to enable profiling traces.
        compute_eval: Whether to compute evaluation metrics.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
        compute_eval: bool = False,
    ) -> None:
        super(ModelFamilyDenseDist, self).__init__()
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.output_trace = output_trace
        self.compute_eval = compute_eval

        self._gpu_ids = _dense_gpu_ids()
        requested = int(os.environ.get("WORLD_SIZE", str(len(self._gpu_ids))))
        self.world_size = min(requested, len(self._gpu_ids))
        self._gpu_ids = self._gpu_ids[: self.world_size]
        self.rank = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        self.dist_backend = "nccl"

        ctx = mp.get_context("spawn")
        self.samples_q: List[mp.Queue] = [ctx.Queue()
                                          for _ in range(self.world_size)]
        self.result_q: List[mp.Queue] = [ctx.Queue()
                                         for _ in range(self.world_size)]
        self.ready_q: mp.Queue = ctx.Queue()
        self._dense_processes: List[mp.Process] = []

    def load(self, model_path: str) -> None:
        """
        Load dense model and spawn worker processes for distributed inference.

        Args:
            model_path: Path to the model checkpoint directory.
        """
        logger.warning(f"Loading dense module from {model_path}")

        ctx = mp.get_context("spawn")
        self._dense_processes = []
        ready_timeout_s = float(
            os.environ.get("DLRM_DENSE_WORKER_READY_TIMEOUT", "3600")
        )
        for rank in range(self.world_size):
            gpu_id = self._gpu_ids[rank]
            p = ctx.Process(
                target=_dense_worker_main,
                args=(
                    rank,
                    gpu_id,
                    self.world_size,
                    model_path,
                    self.samples_q[rank],
                    self.result_q[rank],
                    self.ready_q,
                    self.hstu_config,
                    self.table_config,
                    self.output_trace,
                    self.compute_eval,
                ),
            )
            p.start()
            self._dense_processes.append(p)
            logger.warning(
                "dense worker rank %d (cuda:%d) started pid=%s",
                rank,
                gpu_id,
                p.pid,
            )

        t_all = time.time()
        ready_ranks: set[int] = set()
        while len(ready_ranks) < self.world_size:
            try:
                ready_rank = self.ready_q.get(timeout=ready_timeout_s)
            except Exception as e:
                alive = sum(1 for p in self._dense_processes if p.is_alive())
                raise RuntimeError(
                    f"Dense workers ready {len(ready_ranks)}/{self.world_size} "
                    f"after {time.time() - t_all:.0f}s (alive={alive}): {e}"
                ) from e
            ready_ranks.add(int(ready_rank))
            logger.warning(
                "dense worker rank %d ready (%d/%d)",
                ready_rank,
                len(ready_ranks),
                self.world_size,
            )
        logger.warning(
            "All %d dense workers ready in %.1fs",
            self.world_size,
            time.time() - t_all,
        )
        dead = [i for i, p in enumerate(self._dense_processes) if not p.is_alive()]
        if dead:
            raise RuntimeError(
                f"Dense worker(s) died during startup: ranks {dead}"
            )
        self.rank = 0

    def capture_output(
        self, id: uuid.UUID, rank: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve inference results from a worker process.

        Args:
            id: Unique identifier for the request.
            rank: Worker rank to retrieve from.

        Returns:
            Tuple of (predictions, labels, weights).
        """
        timeout_s = float(os.environ.get("DLRM_DENSE_RESULT_TIMEOUT", "600"))
        deadline = time.time() + timeout_s
        poll_s = float(os.environ.get("DLRM_DENSE_RESULT_POLL_S", "2.0"))
        debug = os.environ.get("DLRM_DEBUG_DENSE", "0") == "1"
        if debug:
            logger.warning(
                "capture_output rank=%d id=%s (timeout=%.0fs)",
                rank,
                id,
                timeout_s,
            )
        while True:
            try:
                item = self.result_q[rank].get(
                    timeout=min(poll_s, max(0.1, deadline - time.time()))
                )
            except Exception as e:
                if debug:
                    logger.warning(
                        "capture_output rank=%d poll (alive=%s)",
                        rank,
                        self._dense_processes[rank].is_alive(),
                    )
                if not self._dense_processes[rank].is_alive():
                    raise RuntimeError(
                        f"Dense worker rank {rank} (cuda:{self._gpu_ids[rank]}) "
                        f"died while waiting for result"
                    ) from e
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"Dense worker rank {rank} (cuda:{self._gpu_ids[rank]}) "
                        f"timed out after {timeout_s}s"
                    ) from e
                continue
            recv_id, preds, labels, weights = item
            assert recv_id == id
            return preds, labels, weights

    def _alive_ranks(self) -> List[int]:
        return [i for i, p in enumerate(self._dense_processes) if p.is_alive()]

    def get_rank(self) -> int:
        """
        Get the next worker rank for load balancing.

        Returns:
            Rank index, cycling through available workers.
        """
        alive = self._alive_ranks()
        if not alive:
            raise RuntimeError("All dense worker processes have exited")
        for _ in range(self.world_size):
            rank = self.rank
            self.rank = (self.rank + 1) % self.world_size
            if rank in alive:
                return rank
        return alive[0]

    def predict(
        self,
        seq_embeddings: Optional[Dict[str, SequenceEmbedding]],
        payload_features: Optional[Dict[str, torch.Tensor]],
        max_uih_len: int,
        uih_seq_lengths: Optional[torch.Tensor],
        max_num_candidates: int,
        num_candidates: Optional[torch.Tensor],
    ) -> Optional[
        Tuple[torch.Tensor, Optional[torch.Tensor],
              Optional[torch.Tensor], float]
    ]:
        """
        Run distributed dense forward pass.

        Dispatches work to a worker process and collects results.

        Args:
            seq_embeddings: Sequence embeddings from sparse module.
            payload_features: Additional feature tensors.
            max_uih_len: Maximum UIH sequence length.
            uih_seq_lengths: Per-sample UIH lengths.
            max_num_candidates: Maximum candidates per sample.
            num_candidates: Per-sample candidate counts.

        Returns:
            Tuple of (predictions, labels, weights, elapsed_time) or None if shutdown.
        """
        id = uuid.uuid4()
        # If none is received terminate all subprocesses
        if seq_embeddings is None:
            for rank in range(self.world_size):
                self.samples_q[rank].put(-1)
            return None
        assert (
            payload_features is not None
            and num_candidates is not None
            and uih_seq_lengths is not None
        )
        rank: Optional[int] = None
        for _ in range(self.world_size):
            candidate = self.get_rank()
            if self._dense_processes[candidate].is_alive():
                rank = candidate
                break
        if rank is None:
            raise RuntimeError("No alive dense workers")
        device = torch.device(f"cuda:{self._gpu_ids[rank]}")
        t0: float = time.time()
        if not _PIPELINE_H2D_TO_WORKER:
            t_h2d: float = time.time()
            seq_embeddings, payload_features, uih_seq_lengths, num_candidates = (
                move_sparse_output_to_device(
                    seq_embeddings=seq_embeddings,
                    payload_features=payload_features,
                    uih_seq_lengths=uih_seq_lengths,
                    num_candidates=num_candidates,
                    device=device,
                )
            )
            timing_record("predict.h2d", time.time() - t_h2d)
        self.samples_q[rank].put(
            (
                id,
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            )
        )
        t_wait: float = time.time()
        (mt_target_preds, mt_target_labels, mt_target_weights) = self.capture_output(
            id, rank
        )
        timing_record("predict.dense_queue_wait", time.time() - t_wait)
        dt_dense = time.time() - t0
        timing_record("predict.dense_e2e", dt_dense)
        return (
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
            dt_dense,
        )

    def shutdown(self) -> None:
        """Send shutdown to all dense workers and join."""
        if not self._dense_processes:
            return
        for rank in range(self.world_size):
            try:
                self.samples_q[rank].put(-1, timeout=10)
            except Exception as e:
                logger.warning("shutdown: rank %d queue put failed: %s", rank, e)
        join_timeout = float(os.environ.get("DLRM_DENSE_WORKER_JOIN_TIMEOUT", "60"))
        for rank, p in enumerate(self._dense_processes):
            p.join(timeout=join_timeout)
            if p.is_alive():
                logger.warning(
                    "shutdown: terminating dense worker rank %d pid=%s",
                    rank,
                    p.pid,
                )
                p.terminate()
                p.join(timeout=10)
        self._dense_processes = []


class ModelFamilyDenseSingleWorker:
    """
    Single-worker dense module manager for single-GPU inference.

    Simpler alternative to ModelFamilyDenseDist for single-GPU setups.

    Args:
        hstu_config: HSTU model configuration.
        table_config: Embedding table configurations.
        output_trace: Whether to enable profiling traces.
        compute_eval: Whether to compute evaluation metrics.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
        compute_eval: bool = False,
    ) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.output_trace = output_trace
        self.device: torch.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.profiler: Optional[Profiler] = (
            Profiler(rank=0) if self.output_trace else None
        )

    def shutdown(self) -> None:
        """No-op for in-process single-GPU dense (no worker processes)."""
        self.model = None

    def load(self, model_path: str) -> None:
        """
        Load dense model for single-GPU inference.

        Args:
            model_path: Path to the model checkpoint directory.
        """
        logger.warning(f"Loading dense module from {model_path}")
        self.model = (
            get_hstu_model(
                table_config=self.table_config,
                hstu_config=self.hstu_config,
                table_device="cpu",
                is_dense=True,
            )
            .to(self.device)
            .to(torch.bfloat16)
        )
        self.model.set_training_dtype(torch.bfloat16)
        load_nonsparse_checkpoint(
            model=self.model, device=self.device, optimizer=None, path=model_path
        )
        assert self.model is not None
        self.model.eval()

    def predict(
        self,
        seq_embeddings: Optional[Dict[str, SequenceEmbedding]],
        payload_features: Optional[Dict[str, torch.Tensor]],
        max_uih_len: int,
        uih_seq_lengths: Optional[torch.Tensor],
        max_num_candidates: int,
        num_candidates: Optional[torch.Tensor],
    ) -> Optional[
        Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            float,
        ]
    ]:
        """
        Run dense forward pass on single GPU.

        Args:
            seq_embeddings: Sequence embeddings from sparse module.
            payload_features: Additional feature tensors.
            max_uih_len: Maximum UIH sequence length.
            uih_seq_lengths: Per-sample UIH lengths.
            max_num_candidates: Maximum candidates per sample.
            num_candidates: Per-sample candidate counts.

        Returns:
            Tuple of (predictions, labels, weights, elapsed_time).
        """
        if self.output_trace:
            assert self.profiler is not None
            self.profiler.step()
        assert (
            payload_features is not None
            and uih_seq_lengths is not None
            and num_candidates is not None
            and seq_embeddings is not None
        )
        t0: float = time.time()
        with torch.profiler.record_function("dense forward"):
            seq_embeddings, payload_features, uih_seq_lengths, num_candidates = (
                move_sparse_output_to_device(
                    seq_embeddings=seq_embeddings,
                    payload_features=payload_features,
                    uih_seq_lengths=uih_seq_lengths,
                    num_candidates=num_candidates,
                    device=self.device,
                )
            )
            assert self.model is not None
            (
                _,
                _,
                _,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = self.model.main_forward(  # pyre-ignore [29]
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                max_uih_len=max_uih_len,
                uih_seq_lengths=uih_seq_lengths,
                max_num_candidates=max_num_candidates,
                num_candidates=num_candidates,
            )
            assert mt_target_preds is not None
            mt_target_preds = mt_target_preds.detach().to(device="cpu")
            if mt_target_labels is not None:
                mt_target_labels = mt_target_labels.detach().to(device="cpu")
            if mt_target_weights is not None:
                mt_target_weights = mt_target_weights.detach().to(device="cpu")
            dt_dense: float = time.time() - t0
            return mt_target_preds, mt_target_labels, mt_target_weights, dt_dense
