import os

from .rocm_bootstrap import apply_rocm_env

if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1":
    apply_rocm_env()

from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from .dataset.streaming_query_sampler import StreamingQuerySamplerRef
from .mpi_utils import MPIDataPacketDSIndex, MPIDataPacketTSRequest
from generative_recommenders.dlrm_v3.datasets.dataset import (
    Samples,
)
import gc
import torch
import queue
from threading import Event, Thread
from typing import Dict
from dataclasses import dataclass
import nvtx
import time
import logging
import sys
import csv
import json

from inference_harness.rocm_timing import maybe_report, record
from inference_harness.zmq_trace import packet_summary, trace_wk, zmq_trace_enabled
from inference_harness.memory_trace import log_memory_phase
from .mpi_utils import ZMQRequestSenderShardedConfig, ZMQRequestSenderSharded


# Plan 06 Phase 6 — opt-in Layer A trace tags (high-frequency events that
# would dominate the trace if always on: every lockstep tick_start /
# zmq_poll_done is ~1 ms granularity, multiplied by W ranks).
_TRACE_LAYER_A = os.environ.get("DLRM_TRACE_LAYER_A", "0") == "1"


@dataclass
class StreamDataPacket:
    """
    Data packet for producer-consumer pattern with CUDA streams.

    The producer (listener thread) prepares data on data_stream and records
    an event when done. The consumer (dispatch thread) waits on this event
    before running inference on the default stream.
    """
    query_ids: list
    batch: any  # The prepared batch (Samples or tensor dicts)
    transfer_done_event: torch.cuda.Event  # Signals when data prep is complete
    is_warmup: bool = False
    results: any = None


logger = logging.getLogger(__name__)


def sample_to_batch_pcie(sample: Samples, device: torch.cuda.device, non_blocking: bool = True) -> Dict:
    from .backends.hybrid_GR_backend import CustomJaggedTensor
    """
    Transfer sample data from CPU to GPU via PCIe for inference.

    Converts KeyedJaggedTensor (KJT) features into CustomJaggedTensor format
    and transfers them to the specified GPU device.

    Args:
        sample: Sample data containing UIH and candidate features as KJTs.
        device: Target CUDA device for data transfer.
        non_blocking: Whether to use non-blocking (asynchronous) transfers.

    Returns:
        tuple: (tensor_dict_uih, tensor_dict_candidates) containing CustomJaggedTensors
            on the target GPU device.
    """
    tensor_dict_uih = {}
    tensor_dict_candidates = {}

    with nvtx.annotate(f"hybrid_GR_backend - optimized_embedding_lookup - tensor prepare", color="orange"):
        for i in sample.uih_features_kjt.keys():
            tensor_dict_uih[i] = CustomJaggedTensor(
                values=sample.uih_features_kjt[i].values().to(device, non_blocking=non_blocking),
                lengths=sample.uih_features_kjt[i].lengths().to(device, non_blocking=non_blocking),
                offsets=sample.uih_features_kjt[i].offsets().to(device, non_blocking=non_blocking),
                max_length=sample.uih_features_kjt[i].lengths().max().item(),
                embeddings=None,
            )
        for i in sample.candidates_features_kjt.keys():
            tensor_dict_candidates[i] = CustomJaggedTensor(
                values=sample.candidates_features_kjt[i].values().to(device, non_blocking=non_blocking),
                lengths=sample.candidates_features_kjt[i].lengths().to(device, non_blocking=non_blocking),
                offsets=sample.candidates_features_kjt[i].offsets().to(device, non_blocking=non_blocking),
                max_length=sample.candidates_features_kjt[i].lengths().max().item(),
                embeddings=None,
            )
    return tensor_dict_uih, tensor_dict_candidates


class DLRMInferenceServer:
    """
    Distributed inference server for DLRM models with MLPerf LoadGen integration.

    Implements a producer-consumer pattern with separate threads for:
    - Listening for inference requests (producer)
    - Processing batches and running inference (consumer)

    Supports optional CUDA streams for overlapped data preparation and inference,
    and communicates with LoadGen via MPI or ZMQ for distributed benchmarking.

    Attributes:
        backend: HybridGRBackend instance for model inference.
        query_streaming_sampler: Dataset sampler for loading queries.
        device: CUDA device for inference.
        batch_size: Number of samples per inference batch.
        mode: Operating mode ("performance" or "accuracy").
        use_cuda_streams: Whether to use CUDA streams for overlapped execution.
    """

    def __init__(
        self,
        query_streaming_sampler: StreamingQuerySamplerRef,
        device: torch.cuda.device,
        local_rank: int = 0,
        batch_size: int = 16,
        warmup_steps: int = 5000,
        verbose: int = 0,
        loadgen_rank: int = 0,
        mode: str = "performance",
        use_cuda_streams: bool = False,
        worker_comm=None,
    ):
        """
        Initialize the DLRM inference server.

        Args:
            query_streaming_sampler: Dataset sampler for loading queries.
            device: CUDA device for inference.
            local_rank: MPI rank of this worker process.
            batch_size: Number of samples per inference batch.
            warmup_steps: Number of warmup iterations (not used in current implementation).
            verbose: Logging verbosity level (0=INFO, 1=DEBUG, -1=WARNING).
            loadgen_rank: MPI rank of the LoadGen process.
            mode: Operating mode ("performance" or "accuracy").
            use_cuda_streams: Enable overlapped batching/inference with CUDA streams.
            worker_comm: Optional MPI sub-communicator containing only worker
                ranks (LoadGen rank excluded). Required for the Phase 2b
                Step 3d ``DLRM_SPARSE_LOCKSTEP_DISPATCH=1`` path that
                broadcasts each LoadGen batch to every worker so the Step 3c
                ``sparse_routing.route_lookup`` all_to_all_single doesn't
                deadlock on an asymmetric ZMQ fair-queue dispatch.
        """
        # Core components
        self.backend: HybridGRBackend = None
        self.query_streaming_sampler = query_streaming_sampler
        self.device = device
        self.local_rank = local_rank
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        self.loadgen_rank = loadgen_rank
        self.mode = mode
        self.use_cuda_streams = use_cuda_streams
        # Phase 2b Step 3d: worker-only MPI subcomm (LoadGen excluded). Used
        # by the lockstep-dispatch listener loop to broadcast each ZMQ batch
        # to every worker so route_lookup's all_to_all_single is symmetric.
        self.worker_comm = worker_comm

        # Model configuration storage (set during init_backend)
        self.hstu_config = None
        self.embedding_table_config = None
        self.backend_config = None

        # Producer-consumer queue for inference requests (bounded for backpressure)
        self.request_queue = queue.Queue(maxsize=10)
        torch.cuda.empty_cache()
        torch.cuda.set_device(self.device)
        log_memory_phase(
            logger,
            "server.__init__.after_empty_cache",
            rank=self.local_rank,
            extra={"batch_size": self.batch_size, "use_cuda_streams": self.use_cuda_streams},
        )

        # Disable garbage collection to reduce latency variance (affects all threads)
        gc.disable()
        self.cudart = torch.cuda.cudart()

        # Threading controls
        self.stop_event = Event()
        self.batching_thread = None

        # ========== CUDA Streams for Overlapped Execution ==========
        # data_stream: Used by producer (listener) for batching/data prep
        # default_stream: Used for inference (cutlass kernels use default_stream internally)
        # This allows overlapping data preparation with inference
        if self.use_cuda_streams:
            self.data_stream = torch.cuda.Stream(device=self.device)
            # Note: We use the default stream for inference because cutlass/CuTe DSL kernels
            # internally use cutlass_torch.default_stream(), not the PyTorch current stream.
            # Using default_stream avoids stream mismatch issues and hangs.
            logger.info(f"[Worker Comm: {self.local_rank}] CUDA streams enabled for overlapped batching/inference (data_stream + default_stream)")
        else:
            self.data_stream = None

        # Configure logger level based on verbosity setting
        if self.verbose >= 1:
            logger.setLevel(logging.DEBUG)
        elif self.verbose == 0:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # Performance tracking
        self.num_batch_processed = 0

        # Opt-in per-stage predict timing (embed lookup vs HSTU forward).
        # Resolved from backend CUDA events after each batch's stream sync.
        self._stage_timing = os.environ.get("DLRM_STAGE_TIMING", "0") == "1"
        self._predict_stage_latencies = []  # (batch_index, embed_ms, forward_ms, predict_wall_ms)
        self._stage_idx = 0
        self._last_pred_wall_ms = 0.0

    def init_communicator(self, shard_id=0, communicator_config: ZMQRequestSenderShardedConfig = None):
        """
        Initialize inter-process communication for the inference server.

        Sets up communication channels with LoadGen using either MPI or ZMQ (sharded).
        Sharded ZMQ reduces communication fan-in by grouping workers on the same node.

        Args:
            shard_id: Shard identifier (typically node ID). Workers on the same node share a shard
                to reduce communication fan-in from N:1 to N/shards:1.
            communicator_config: Configuration for the communication backend (ZMQ or MPI).
        """
        # Initialize batching latency tracking
        self._batching_latencies = []  # List of (batch_index, latency_ms)

        # Initialize communication backend (ZMQ or MPI)
        if isinstance(communicator_config, ZMQRequestSenderShardedConfig):
            self.request_sender = ZMQRequestSenderSharded(
                is_loadgen=False,
                shard_id=shard_id,
                rank=self.local_rank,
                config=communicator_config
            )
        else:
            # MPI communication path (not yet implemented)
            self.worker_comm = communicator_config.worker_comm
            self.loadgen_comm = communicator_config.loadgen_comm
            self.use_async_mpi = communicator_config.use_async_mpi
            raise NotImplementedError("MPI communication is not yet implemented")

        # Request listener thread controls
        self._request_listener_thread: Thread = None
        self._request_stop_event = Event()
        self._packet_counter = 0

        # Start listening for inference requests
        self._start_request_listener()

    def init_backend(self,
                     hstu_config: DlrmHSTUConfig,
                     embedding_table_config: Dict[str, EmbeddingConfig],
                     backend_config,
                     checkpoint_path: str = "/raid/data/zihaok_1/89/"
                     ):
        """
        Initialize the hybrid GR backend and load model weights.

        Creates the backend instance, initializes it with the model configuration,
        and loads both dense and sparse model components from checkpoints.
        Rank 0 is designated as the main rank for coordinating sparse table loading.

        Args:
            hstu_config: Configuration for the HSTU model architecture.
            embedding_table_config: Configuration for embedding tables.
            backend_config: Configuration for the hybrid GR backend.
            checkpoint_path: Path to the model checkpoint directory.
        """
        # Store configs for later use in multiprocessing
        self.hstu_config = hstu_config
        self.embedding_table_config = embedding_table_config
        self.backend_config = backend_config

        # Plan 14.5b: DLRM_ROCM_NVE=1 opts into the ported NVE HybridGRBackend even
        # when DLRM_ROCM_GR_BACKEND=1 (the GR launcher sets the latter). This must
        # mirror the gate in model_configs.get_backend_config, which already returns a
        # HybridGRBackendConfig for the NVE path — otherwise the config is NVE but the
        # backend dispatched here stays the GR backend.
        _rocm_nve = os.environ.get("DLRM_ROCM_NVE", "0") == "1"
        if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1" and not _rocm_nve:
            from .backends.generative_recommener_backend import (
                GenerativeRecommenderBackend,
            )

            apply_rocm_env()
            local_gpu = int(self.device.index) if self.device.index is not None else 0
            os.environ["DLRM_MPI_LOCAL_RANK"] = str(self.local_rank)
            self.backend = GenerativeRecommenderBackend(
                model_name="dlrm_hstu",
                perf_mode=backend_config.perf_mode,
                device_id=[local_gpu],
            )
            self.backend.initialize(
                hstu_config=hstu_config,
                embedding_table_config=embedding_table_config,
            )
            self.backend.load_model(checkpoint_path=checkpoint_path)
        else:
            from .backends.hybrid_GR_backend import HybridGRBackend

            # Initialize backend for this device
            self.backend = HybridGRBackend(model_name="dlrm_hstu", device=self.device)
            self.backend.initialize(
                hstu_config=hstu_config,
                embedding_table_config=embedding_table_config,
                backend_config=backend_config,
            )

            # Load model weights (dense on all ranks, sparse with coordination)
            self.backend.load_model_dense(checkpoint_path=checkpoint_path)
            if self.local_rank == 0:
                self.backend.load_model_sparse(
                    checkpoint_path=checkpoint_path, main_rank=True, rank=self.local_rank
                )
            else:
                self.backend.load_model_sparse(
                    checkpoint_path=checkpoint_path, main_rank=False, rank=self.local_rank
                )

        # Start the batching thread for processing inference requests
        self.batching_thread = Thread(target=self.batching_loop)
        self.batching_thread.start()

    def warmup(self, warmup_steps: int = 100, batching_warmup_steps: int | None = None):
        """
        Warm up the inference server with dummy predictions.

        Runs inference on sample data to initialize CUDA kernels and stabilize
        performance before benchmarking. Resets the dataset sampler to the
        starting timestamp after warmup completes.

        Args:
            warmup_steps: Number of warmup inference iterations to perform.
            batching_warmup_steps: Batches to run through the batching thread
                (default: env DLRM_BATCHING_WARMUP_STEPS or warmup_steps).
        """
        # Phase 3 Mode B: under sharded sparse routing every
        # ``backend.predict()`` call traverses ``all_to_all_single``
        # (``route_lookup``). If each worker runs warmup independently the
        # collective send/recv counts drift across ranks within a few
        # iterations and gloo aborts with "Connection reset by peer".
        # Force per-step lockstep entry via a worker-subcomm barrier so
        # every rank enters every collective on the same step (verified
        # 2026-05-26: Mode B 2W warmup deadlock).
        #
        # Plan 07 Phase 7.1b (2026-05-28): gating split into two flags so
        # the Barrier survives switching off lockstep dispatch.
        # ``needs_collective_warmup`` is the property we actually need
        # (collective symmetry); it's True whenever sharded sparse is
        # the live routing path (``DLRM_SPARSE_REPLICATE=0``) and the
        # worker subcomm exists. The original ``lockstep_warmup`` is
        # kept for the ``_warmup_batching_thread`` skip below, which
        # specifically banks on the lockstep listener broadcasting the
        # first ZMQ batches.
        lockstep_warmup = self._lockstep_enabled() and self.worker_comm is not None
        needs_collective_warmup = (
            self.worker_comm is not None
            and os.environ.get("DLRM_SPARSE_REPLICATE", "") == "0"
        )
        for i in range(warmup_steps):
            if i == 0:
                log_memory_phase(
                    logger,
                    "server.warmup.direct.start",
                    rank=self.local_rank,
                    extra={"warmup_steps": warmup_steps},
                )
            if needs_collective_warmup:
                self.worker_comm.Barrier()
            with torch.inference_mode():
                # Log progress periodically
                if i % 50 == 0 or i == warmup_steps - 1:
                    logger.info(f"[Worker Comm: {self.local_rank}] Warmup step {i} / {warmup_steps}")

                # Rotate query ids so Triton JITs the shape variety LoadGen will see.
                base = (i * self.batch_size) % max(
                    1, self.query_streaming_sampler.total_requests - self.batch_size
                )
                query_ids = list(range(base, base + self.batch_size))
                outputs_ts = self.query_streaming_sampler.get_samples_indices(query_ids)
                warmup_samples = self.query_streaming_sampler.ds.get_samples_with_ts_updated(outputs_ts)
                _ = self.backend.predict(warmup_samples)

                # Reset when reaching end of dataset
                if self.query_streaming_sampler.ts_processed_cnt >= self.query_streaming_sampler.total_requests:
                    self.query_streaming_sampler.init_sut()
        log_memory_phase(
            logger,
            "server.warmup.direct.done",
            rank=self.local_rank,
            extra={"warmup_steps": warmup_steps},
        )
        if os.environ.get("DLRM_MEMORY_EMPTY_CACHE_AFTER_DIRECT_WARMUP", "0") == "1":
            log_memory_phase(
                logger,
                "server.warmup.direct.before_empty_cache",
                rank=self.local_rank,
            )
            for _dev_idx in range(torch.cuda.device_count()):
                try:
                    torch.cuda.synchronize(_dev_idx)
                except Exception:  # noqa: BLE001
                    pass
            torch.cuda.empty_cache()
            log_memory_phase(
                logger,
                "server.warmup.direct.after_empty_cache",
                rank=self.local_rank,
            )

        if batching_warmup_steps is None:
            batching_steps = int(
                os.environ.get("DLRM_BATCHING_WARMUP_STEPS", str(warmup_steps))
            )
        else:
            batching_steps = batching_warmup_steps
        # Phase 3 Mode B: _warmup_batching_thread enqueues N batches and
        # waits for the batching thread to drain them, but the per-batch
        # predict in that thread hits all_to_all_single under sharded
        # routing, and the main thread cannot coordinate per-step
        # collectives with the worker side (different python thread).
        # The lockstep listener already JIT-warms the batching thread on
        # the first real ZMQ batches it broadcasts, so skipping the
        # async batching warmup here is safe and avoids the mid-warmup
        # gloo "Connection reset by peer" abort.
        if lockstep_warmup and batching_steps > 0:
            logger.info(
                f"[Worker Comm: {self.local_rank}] Skipping batching-thread "
                f"warmup ({batching_steps} steps) under lockstep dispatch; "
                f"first real listener batches will JIT the batching thread."
            )
            batching_steps = 0
        if batching_steps > 0 and self.batching_thread is not None:
            log_memory_phase(
                logger,
                "server.warmup.batching_thread.start",
                rank=self.local_rank,
                extra={"batching_steps": batching_steps},
            )
            self._warmup_batching_thread(batching_steps)
            log_memory_phase(
                logger,
                "server.warmup.batching_thread.done",
                rank=self.local_rank,
                extra={"batching_steps": batching_steps},
            )
            log_memory_phase(
                logger,
                "server.warmup.partial_flush.start",
                rank=self.local_rank,
            )
            self._warmup_partial_flush_batch()
            log_memory_phase(
                logger,
                "server.warmup.partial_flush.done",
                rank=self.local_rank,
            )

        # Plan 04: synchronize every visible device before tearing down the
        # warmup-era cached slabs. Empirically this is not sufficient to fix
        # the rank-1 ``Memory access fault by GPU node-N`` (see
        # plans/04_FbgemmGPU_Trace_and_Localize.md), but it is the right shape
        # of safety: ``empty_cache`` returns ~700 MiB of slabs back to HIP via
        # ``segment_free``, so any kernels still queued on those slabs must be
        # done first. ``DLRM_PLAN04_SKIP_EMPTY_CACHE=1`` keeps the cache for
        # diagnostic A/B runs.
        for _dev_idx in range(torch.cuda.device_count()):
            try:
                torch.cuda.synchronize(_dev_idx)
            except Exception:  # noqa: BLE001
                pass
        if os.environ.get("DLRM_PLAN04_SKIP_EMPTY_CACHE", "0") != "1":
            log_memory_phase(logger, "server.warmup.before_empty_cache", rank=self.local_rank)
            torch.cuda.empty_cache()
            log_memory_phase(logger, "server.warmup.after_empty_cache", rank=self.local_rank)
        if os.environ.get("DLRM_HSTU_STU_GRAPH_DEFER_CAPTURE", "0") == "1":
            capture_steps = int(
                os.environ.get("DLRM_HSTU_STU_GRAPH_DEFER_CAPTURE_STEPS", "128")
            )
            if capture_steps > 0 and self.batching_thread is not None:
                from generative_recommenders.modules.hstu_transducer import (
                    set_stu_graph_defer_capture,
                )

                log_memory_phase(
                    logger,
                    "server.warmup.deferred_stu_graph_capture.start",
                    rank=self.local_rank,
                    extra={"capture_steps": capture_steps},
                )
                set_stu_graph_defer_capture(False)
                self._warmup_batching_thread(capture_steps)
                for _dev_idx in range(torch.cuda.device_count()):
                    try:
                        torch.cuda.synchronize(_dev_idx)
                    except Exception:  # noqa: BLE001
                        pass
                log_memory_phase(
                    logger,
                    "server.warmup.deferred_stu_graph_capture.done",
                    rank=self.local_rank,
                    extra={"capture_steps": capture_steps},
                )
                if os.environ.get("DLRM_PLAN04_SKIP_EMPTY_CACHE", "0") != "1":
                    log_memory_phase(
                        logger,
                        "server.warmup.deferred_stu_graph_capture.before_empty_cache",
                        rank=self.local_rank,
                    )
                    torch.cuda.empty_cache()
                    log_memory_phase(
                        logger,
                        "server.warmup.deferred_stu_graph_capture.after_empty_cache",
                        rank=self.local_rank,
                    )
        if os.environ.get("DLRM_HSTU_STU_GRAPH_FREEZE_AFTER_WARMUP", "0") == "1":
            from generative_recommenders.modules.hstu_transducer import (
                set_stu_graph_freeze_capture,
            )

            set_stu_graph_freeze_capture(True)
        # Reset the sampler to the start timestamp for benchmarking
        self.query_streaming_sampler.init_sut()
        log_memory_phase(logger, "server.warmup.done", rank=self.local_rank)

    def _warmup_batching_thread(self, steps: int, timeout_s: float = 600.0) -> None:
        """
        Prime the batching thread + listener path (same as ZMQ production).

        Direct predict() on the main thread does not JIT Triton in the batching
        thread; without this, the first ZMQ batches can stall for minutes.
        """
        before = self.num_batch_processed
        logger.info(
            f"[Worker Comm: {self.local_rank}] Batching-thread warmup: {steps} steps"
        )
        for i in range(steps):
            base = (i * self.batch_size) % max(
                1, self.query_streaming_sampler.total_requests - self.batch_size
            )
            query_ids = list(range(base, base + self.batch_size))
            outputs_ts = self.query_streaming_sampler.get_samples_indices(query_ids)
            sample = self.query_streaming_sampler.ds.get_samples_with_ts_updated(
                outputs_ts
            )
            self.enqueue_batch(
                MPIDataPacketTSRequest(
                    query_ids=query_ids,
                    ts_request_pairs=outputs_ts,
                    batch=sample,
                    is_warmup=False,
                    skip_result_send=True,
                )
            )

        deadline = time.time() + timeout_s
        while self.num_batch_processed - before < steps:
            if time.time() > deadline:
                raise TimeoutError(
                    f"Batching-thread warmup timed out after {timeout_s}s: "
                    f"got {self.num_batch_processed - before}/{steps} batches"
                )
            time.sleep(0.001)
        logger.info(
            f"[Worker Comm: {self.local_rank}] Batching-thread warmup complete "
            f"({steps} batches)"
        )

    def _warmup_partial_flush_batch(self, timeout_s: float = 600.0) -> None:
        """
        Prime ROCm flush padding (partial batch padded to batch_size).

        LoadGen flush sends e.g. 6 queries padded to batch_size 10; without this
        the first padded flush batch can stall Triton JIT for minutes.
        """
        if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") != "1":
            return
        real_n = int(
            os.environ.get("DLRM_FLUSH_PAD_WARMUP_REAL_N", str(self.batch_size - 4))
        )
        if real_n <= 0 or real_n >= self.batch_size:
            return
        pad = self.batch_size - real_n
        base = (777 * self.batch_size) % max(
            1, self.query_streaming_sampler.total_requests - self.batch_size
        )
        query_ids = list(range(base, base + real_n))
        outputs_ts = self.query_streaming_sampler.get_samples_indices(query_ids)
        padded_ids = query_ids + [query_ids[-1]] * pad
        padded_ts = outputs_ts + [outputs_ts[-1]] * pad
        padded_sample = self.query_streaming_sampler.ds.get_samples_with_ts_updated(
            padded_ts
        )
        before = self.num_batch_processed
        logger.info(
            f"[Worker Comm: {self.local_rank}] Partial-flush batching warmup: "
            f"{real_n} -> {self.batch_size} queries"
        )
        self.enqueue_batch(
            MPIDataPacketTSRequest(
                query_ids=padded_ids,
                ts_request_pairs=padded_ts,
                batch=padded_sample,
                is_warmup=False,
                skip_result_send=True,
            )
        )
        deadline = time.time() + timeout_s
        while self.num_batch_processed - before < 1:
            if time.time() > deadline:
                raise TimeoutError(
                    f"Partial-flush batching warmup timed out after {timeout_s}s"
                )
            time.sleep(0.001)
        logger.info(
            f"[Worker Comm: {self.local_rank}] Partial-flush batching warmup complete"
        )

    def enqueue_batch(self, batch_data: Dict):
        """
        Enqueue a batch of inference requests to the processing queue.

        Args:
            batch_data: Dictionary containing batch information and data.
        """
        self.request_queue.put(batch_data)

    def batching_loop(self):
        """
        Consumer thread: Runs inference on default_stream.

        With CUDA streams enabled:
            - Waits on transfer_done_event before reading batch data
            - Runs inference on default_stream (overlaps with next batch prep on data_stream)
            - Note: cutlass/CuTe DSL kernels use default_stream internally, so we use it here
              to avoid stream mismatch issues

        Timeline (overlapped):
            data_stream:     [Prep B0][Prep B1][Prep B2][Prep B3]...
            default_stream:         [Infer B0][Infer B1][Infer B2]...
        """
        # CRITICAL: Set CUDA device context for this thread
        # Each thread in PyTorch needs its own device context
        torch.cuda.set_device(self.device)

        # Ensure garbage collection is disabled in this thread for consistent latency
        gc.disable()
        logger.info(f"[Worker Comm: {self.local_rank}] [Batching Thread] Started batching thread and set CUDA device to {self.device}")

        # Plan 06 Phase 6.5 — PyTorch Profiler hook around steady-state
        # predict() calls. Off by default (zero overhead). Set
        # ``DLRM_TORCH_PROFILER=1`` to enable. The profiler activates after
        # ``DLRM_TORCH_PROFILER_SKIP`` (default 100) production batches —
        # past warmup and pipeline fill — then captures the next
        # ``DLRM_TORCH_PROFILER_N`` (default 20) batches and exports a
        # Chrome trace to ``DLRM_TORCH_PROFILER_DIR``
        # (default /tmp/dlrm_torchprof) as ``rank{R}_b{B}.json``. Only the
        # rank matching ``DLRM_TORCH_PROFILER_RANK`` (default 0) profiles;
        # other ranks no-op. The trace covers default_stream (predict),
        # data_stream (collate H2D when use_cuda_streams=True), and the
        # RCCL comm stream.
        _profiler = None
        _profile_steps_done = 0
        _profile_started = False
        if os.environ.get("DLRM_TORCH_PROFILER", "0") == "1":
            try:
                _profile_rank = int(os.environ.get("DLRM_TORCH_PROFILER_RANK", "0"))
            except ValueError:
                _profile_rank = 0
            if self.local_rank == _profile_rank:
                try:
                    _profile_skip = int(
                        os.environ.get("DLRM_TORCH_PROFILER_SKIP", "100")
                    )
                except ValueError:
                    _profile_skip = 100
                try:
                    _profile_n = int(os.environ.get("DLRM_TORCH_PROFILER_N", "20"))
                except ValueError:
                    _profile_n = 20
                _profile_dir = os.environ.get(
                    "DLRM_TORCH_PROFILER_DIR", "/tmp/dlrm_torchprof"
                )
                os.makedirs(_profile_dir, exist_ok=True)
                _profile_out = os.path.join(
                    _profile_dir,
                    f"rank{self.local_rank}_b{_profile_n}_skip{_profile_skip}.json",
                )
                logger.info(
                    f"[Worker Comm: {self.local_rank}] [Phase 6.5 profiler] "
                    f"will activate at batch {_profile_skip + 1}, capture "
                    f"{_profile_n} batches, write {_profile_out}"
                )

                def _on_trace_ready(prof):  # noqa: ARG001
                    try:
                        prof.export_chrome_trace(_profile_out)
                        logger.info(
                            f"[Worker Comm: {self.local_rank}] "
                            f"[Phase 6.5 profiler] wrote {_profile_out}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            f"[Worker Comm: {self.local_rank}] "
                            f"[Phase 6.5 profiler] export failed: {exc!r}"
                        )

                try:
                    _profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch.profiler.schedule(
                            wait=0, warmup=0, active=_profile_n, repeat=1
                        ),
                        on_trace_ready=_on_trace_ready,
                        record_shapes=os.environ.get(
                            "DLRM_TORCH_PROFILER_SHAPES", "0"
                        ) == "1",
                        with_stack=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"[Worker Comm: {self.local_rank}] "
                        f"[Phase 6.5 profiler] failed to create profiler: {exc!r}; "
                        f"continuing without profiling"
                    )
                    _profiler = None

        # Process requests until stop signal and queue is empty
        _last_dequeue_t = None  # Plan 21: inter-batch period (period - batch.total = starvation gap)
        while not self.stop_event.is_set() or not self.request_queue.empty():
            if self.request_queue.empty():
                time.sleep(0.0001)  # 100μs sleep to avoid busy-waiting
                continue

            data_packet = self.request_queue.get()
            t_batch = time.perf_counter()
            if _last_dequeue_t is not None and not getattr(data_packet, "is_warmup", False):
                # Period between consecutive batches on this worker. With sharded
                # dispatch, period >> batch.total means the worker is starved
                # (queue empty), pointing at dispatch/route_lookup coordination
                # rather than GPU compute. (Plan 21 Phase 21.0.)
                record("batch.period", t_batch - _last_dequeue_t)
            _last_dequeue_t = t_batch
            seq = getattr(data_packet, "zmq_seq", -1)
            trace_wk(
                self.local_rank,
                seq if seq >= 0 else self.num_batch_processed + 1,
                "batch_dequeue",
                processed=self.num_batch_processed,
                packet=self._packet_counter,
                is_warmup=data_packet.is_warmup,
                skip_send=getattr(data_packet, "skip_result_send", False),
            )

            # ========== CUDA Streams Path: Overlapped Execution ==========
            if self.use_cuda_streams:
                # Plan 07 Phase 7.1a: guard against
                # ``transfer_done_event=None``. ``_warmup_batching_thread``
                # / ``_warmup_partial_flush_batch`` build synthetic packets
                # without recording a data_stream event (they precede the
                # listener path that records one). Under
                # ``use_cuda_streams=True`` (Plan 06 launcher default),
                # ``wait_event(None)`` crashes those warmup batches with
                # ``AttributeError: 'NoneType' object has no attribute
                # 'wait'``. Only wait when an event was actually recorded;
                # the warmup synthetic batches are CPU-collated and have
                # no pending H2D anyway. Real ZMQ-listener packets always
                # set ``transfer_done_event`` (``_listen_loop`` ~:1248,
                # ``_lockstep_listen_loop`` ~:1162 under Plan 06 Change B).
                if data_packet.transfer_done_event is not None:
                    # Wait for data prep to complete on data_stream before running inference
                    # We use default_stream for inference because cutlass/CuTe DSL kernels
                    # internally use cutlass_torch.default_stream(), not PyTorch's current stream
                    torch.cuda.current_stream(self.device).wait_event(data_packet.transfer_done_event)

                # Run inference on default_stream (can overlap with next batch prep on data_stream)
                with nvtx.annotate(f"Inference-{self.num_batch_processed}", color="yellow"):
                    if data_packet.is_warmup:
                        predict, labels, weights = self.backend.predict_dummy(data_packet.batch)
                    else:
                        t_pred = time.perf_counter()
                        with torch.inference_mode():
                            predict, labels, weights = self.backend.predict(data_packet.batch)
                        record("batch.predict", time.perf_counter() - t_pred)
                        self._last_pred_wall_ms = (time.perf_counter() - t_pred) * 1000.0

                # Plan 04 hypothesis: full-device sync (covers RCCL comm
                # stream, fbgemm data stream, and any other stream that the
                # current-stream-only sync misses). Gated for A/B testing.
                _full_sync = os.environ.get("DLRM_PLAN04_FULL_DEVICE_SYNC", "1") == "1"
                if _full_sync:
                    torch.cuda.synchronize(self.device)
                else:
                    torch.cuda.current_stream(self.device).synchronize()
                trace_wk(
                    self.local_rank,
                    seq if seq >= 0 else self.num_batch_processed + 1,
                    "cuda_sync_done",
                    full_device=bool(_full_sync),
                    stream_path=True,
                )
            else:
                # ========== Non-Stream Path: Sequential Execution ==========
                if data_packet.is_warmup:
                    predict, labels, weights = self.backend.predict_dummy(data_packet.batch)
                else:
                    trace_wk(
                        self.local_rank,
                        seq if seq >= 0 else self.num_batch_processed + 1,
                        "predict_start",
                        **packet_summary(
                            data_packet.query_ids, data_packet.ts_request_pairs
                        ),
                    )
                    t_pred = time.perf_counter()
                    with torch.inference_mode():
                        predict, labels, weights = self.backend.predict(data_packet.batch)
                    pred_ms = (time.perf_counter() - t_pred) * 1000
                    record("batch.predict", time.perf_counter() - t_pred)
                    self._last_pred_wall_ms = pred_ms
                    trace_wk(
                        self.local_rank,
                        seq if seq >= 0 else self.num_batch_processed + 1,
                        "predict_done",
                        predict_ms=round(pred_ms, 2),
                    )

                # Plan 04 hypothesis: full-device sync (see stream-path
                # branch above for rationale). Gated for A/B testing.
                _full_sync = os.environ.get("DLRM_PLAN04_FULL_DEVICE_SYNC", "1") == "1"
                if _full_sync:
                    torch.cuda.synchronize(self.device)
                else:
                    torch.cuda.current_stream(self.device).synchronize()
                trace_wk(
                    self.local_rank,
                    seq if seq >= 0 else self.num_batch_processed + 1,
                    "cuda_sync_done",
                    full_device=bool(_full_sync),
                    stream_path=False,
                )

            if self._stage_timing and not data_packet.is_warmup:
                self._collect_stage_timing(self._last_pred_wall_ms)

            self.num_batch_processed += 1

            # Send results back to LoadGen (skip during batching-thread warmup)
            data_packet.results = (predict, labels, weights)
            if not getattr(data_packet, "skip_result_send", False):
                result_data_packet = MPIDataPacketDSIndex(
                    query_ids=data_packet.query_ids,
                    ts_request_pairs=data_packet.ts_request_pairs,
                    results=data_packet.results,
                    zmq_seq=seq,
                )
                trace_wk(
                    self.local_rank,
                    seq if seq >= 0 else self.num_batch_processed,
                    "lg_send_start",
                    processed=self.num_batch_processed,
                )
                t_send = time.perf_counter()
                with nvtx.annotate(f"send to loadgen rank", color="orange"):
                    self.request_sender.send_to_loadgen(result_data_packet)
                trace_wk(
                    self.local_rank,
                    seq if seq >= 0 else self.num_batch_processed,
                    "lg_send_done",
                    send_ms=round((time.perf_counter() - t_send) * 1000, 2),
                )
                if not data_packet.is_warmup:
                    record("batch.send_zmq", time.perf_counter() - t_send)
                    record("batch.total", time.perf_counter() - t_batch)
                    maybe_report()
            elif getattr(data_packet, "skip_result_send", False):
                trace_wk(
                    self.local_rank,
                    seq if seq >= 0 else self.num_batch_processed,
                    "skip_result_send",
                    processed=self.num_batch_processed,
                )

            self._batching_latencies.append(
                (self.num_batch_processed, (time.perf_counter() - t_batch) * 1000)
            )

            # Plan 06 Phase 6.5 — drive the PyTorch profiler one step per
            # batch when active. Skip warmup batches (is_warmup=True) so the
            # skip-count reflects production batches only.
            if (
                _profiler is not None
                and not data_packet.is_warmup
            ):
                if (
                    not _profile_started
                    and self.num_batch_processed >= _profile_skip
                ):
                    _profiler.start()
                    _profile_started = True
                    logger.info(
                        f"[Worker Comm: {self.local_rank}] "
                        f"[Phase 6.5 profiler] STARTED at batch "
                        f"{self.num_batch_processed}"
                    )
                if _profile_started:
                    _profiler.step()
                    _profile_steps_done += 1
                    if _profile_steps_done >= _profile_n:
                        try:
                            _profiler.stop()
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                f"[Worker Comm: {self.local_rank}] "
                                f"[Phase 6.5 profiler] stop() failed: {exc!r}"
                            )
                        # one-shot — drop the reference so we don't restart
                        _profiler = None

        logger.debug(f"[Worker Comm: {self.local_rank}] [Batching Thread] Stopped successfully.")

    def stop_batching(self, timeout: float = 10.0):
        """
        Gracefully stop the batching thread.

        Args:
            timeout: Maximum time to wait for the thread to finish (in seconds)
        """

        logger.debug(f"[Worker Comm: {self.local_rank}] [Batching Thread] Stopping batching thread on {self.device}...")
        self.stop_event.set()  # Signal the thread to stop

        # Stop listener thread if running
        if self._request_listener_thread is not None and self._request_listener_thread.is_alive():
            self._stop_request_listener(timeout=timeout)

        # Wait for the thread to finish
        self.batching_thread.join(timeout=timeout)

        if self.batching_thread.is_alive():
            logger.warning(f"[Worker Comm: {self.local_rank}] [Batching Thread] Warning: Batching thread on {self.device} did not stop within {timeout}s")
        else:
            logger.debug(f"[Worker Comm: {self.local_rank}] [Batching Thread] Batching thread on {self.device} stopped successfully.")

        if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1" and hasattr(
            self.backend, "shutdown"
        ):
            self.backend.shutdown()

    # ====================================================================
    # Phase 2b Step 3d: lockstep-dispatch helpers
    # ====================================================================

    def _lockstep_enabled(self) -> bool:
        """Whether to use the Step 3d coordinated dispatch path.

        Auto-on when the Step 3c routing primitive is active
        (``DLRM_SPARSE_REPLICATE=0`` with multiple worker ranks) and a
        worker subcomm was passed in; explicit override via
        ``DLRM_SPARSE_LOCKSTEP_DISPATCH={0,1}``.

        Falls back silently (returns False) if mpi4py is not importable
        or the subcomm is missing — the default ZMQ fair-queue listener
        is used in that case, matching pre-3d behavior.
        """
        env = os.environ.get("DLRM_SPARSE_LOCKSTEP_DISPATCH", "").strip()
        if env == "0":
            return False
        if self.worker_comm is None:
            return False
        try:
            world = self.worker_comm.Get_size()
        except Exception:  # noqa: BLE001
            return False
        if world < 2:
            return False
        if env == "1":
            return True
        # Auto-on when sparse routing (Step 3c) is the live path.
        replicate = os.environ.get("DLRM_SPARSE_REPLICATE", "")
        return replicate == "0"

    def _rocm_serial_wait(self, seq: int) -> None:
        """ROCm per-listener-tick pipeline-depth gate.

        Plan 05 (listener pipelining, ``plans/05_Listener_Pipelining.md``):
        was previously a strict serial gate (block until
        ``num_batch_processed >= _packet_counter``, i.e. pipeline depth = 1);
        now a queue-depth gate that lets the listener thread collate batch
        N+1 while the batching thread predicts batch N. Pipeline depth is
        controlled by ``DLRM_LISTENER_QUEUE_DEPTH`` (default 2 — one in
        flight + one queued ahead; ``=1`` reverts to pre-Plan-05 strict
        behavior). The hard upper bound stays at ``request_queue.maxsize``
        (10, see ``__init__``); this env only sets the soft pipeline depth.
        """
        if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") != "1":
            return
        try:
            cap = int(os.environ.get("DLRM_LISTENER_QUEUE_DEPTH", "2"))
        except ValueError:
            cap = 2
        cap = max(1, cap)
        deadline = time.time() + float(
            os.environ.get("DLRM_ROCM_BATCH_SERIAL_TIMEOUT_S", "600")
        )
        stall_s = float(os.environ.get("DLRM_ZMQ_TRACE_STALL_S", "5"))
        last_stall = time.time()
        trace_wk(
            self.local_rank,
            seq,
            "serial_wait_start",
            processed=self.num_batch_processed,
            packet=self._packet_counter,
            cap=cap,
        )
        # Gate semantics: ``in_flight = _packet_counter - num_batch_processed``
        # counts batches enqueued-but-not-predict-complete. Block while
        # ``in_flight > cap - 1`` so at most ``cap`` batches are in the
        # listener->batching pipeline at any moment. ``cap=1`` reproduces
        # the original strict serial behavior; ``cap=2`` permits one-batch
        # lookahead (the Plan 05 default).
        while (self._packet_counter - self.num_batch_processed) > (cap - 1):
            if time.time() > deadline:
                trace_wk(
                    self.local_rank,
                    seq,
                    "serial_wait_timeout",
                    processed=self.num_batch_processed,
                    packet=self._packet_counter,
                    cap=cap,
                )
                logger.warning(
                    f"[Worker Comm: {self.local_rank}] Batch pipeline gate "
                    f"timed out: processed {self.num_batch_processed}/"
                    f"{self._packet_counter} (cap={cap})"
                )
                return
            if time.time() - last_stall >= stall_s:
                trace_wk(
                    self.local_rank,
                    seq,
                    "serial_wait_stall",
                    processed=self.num_batch_processed,
                    packet=self._packet_counter,
                    cap=cap,
                )
                last_stall = time.time()
            time.sleep(0.0001)
        trace_wk(
            self.local_rank,
            seq,
            "serial_wait_done",
            processed=self.num_batch_processed,
            packet=self._packet_counter,
            cap=cap,
        )

    def _lockstep_listen_loop(self) -> None:
        """Coordinated dispatch listener for Phase 2b Step 3d.

        Algorithm per tick (every worker rank in lockstep):

          1. Each worker briefly polls ZMQ for a batch
             (``DLRM_SPARSE_LOCKSTEP_POLL_MS``, default 100 ms).
          2. ``worker_comm.Allreduce(rank if got_batch else -1, MAX)``
             discovers which worker (if any) received a batch.
          3. If a winner exists, the receiver serializes the request
             packet bytes and broadcasts to every worker; everyone
             reconstructs the same ``MPIDataPacketTSRequest``.
          4. Every worker collates samples + enqueues to its batching
             thread. ``skip_result_send`` is set on non-receivers so
             only the ZMQ receiver replies to LoadGen — LoadGen still
             sees exactly one result per dispatched batch.
          5. Standard ROCm per-tick serial wait + trace events.

        This trades per-worker compute parallelism (every worker runs
        the full forward, including dense+HSTU) for cross-rank
        correctness of the sparse path: ``route_lookup`` /
        ``all_to_all_single`` now sees every worker in every batch and
        cannot deadlock on the otherwise asymmetric ZMQ fair-queue
        dispatch. Memory sharding is preserved (each rank still holds
        only its 1/W slice of the sparse table under
        ``DLRM_SPARSE_REPLICATE=0``), which is the actual scale-out
        constraint for the production embedding (1 B rows × 512 D ×
        fp16 ≈ 1 TB — does not fit replicated on any single GPU).
        """
        from mpi4py import MPI  # noqa: WPS433
        import numpy as np  # noqa: WPS433

        torch.cuda.set_device(self.device)
        gc.disable()

        wcomm = self.worker_comm
        my_wrank = wcomm.Get_rank()
        world = wcomm.Get_size()
        use_rocm_recv = os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1"
        poll_ms = int(os.environ.get("DLRM_SPARSE_LOCKSTEP_POLL_MS", "100"))

        logger.info(
            f"[Worker Comm: {self.local_rank}] [Step3d] lockstep dispatch "
            f"active (worker_world={world}, poll_ms={poll_ms}); broadcasting "
            f"each LoadGen batch via MPI so route_lookup all_to_all_single "
            f"is symmetric"
        )

        send_buf = np.zeros(1, dtype=np.int64)
        recv_buf = np.zeros(1, dtype=np.int64)
        size_buf = np.zeros(1, dtype=np.int64)

        # World cap. We Allreduce a packed int64 bitmask (one bit per rank)
        # so every worker knows exactly which peers received a ZMQ batch
        # this tick, and we then Bcast each of those batches in rank order.
        # This subsumes the previous Allreduce(MAX) single-winner protocol,
        # which silently dropped the lower-rank's packet when two workers
        # both received a batch within the same poll tick (typical at test
        # start when LoadGen bursts the first W batches before either
        # worker polls; see Phase 3 Mode B drain hang).
        if world > 63:  # pragma: no cover - real production cap is 8
            raise RuntimeError(
                f"[Step3d] lockstep dispatch supports up to 63 workers "
                f"(int64 bitmask) but got world={world}"
            )

        _tick_counter = 0
        while not self._request_stop_event.is_set():
            _tick_counter += 1
            # Plan 21 Phase 21.0 — per-tick budget probes. record() is a no-op
            # unless DLRM_TIMING is on; n-count ratios across the tick.* vs
            # batch.* names reveal batches-per-tick under saturation.
            _t_tick0 = time.perf_counter()
            if _TRACE_LAYER_A:
                trace_wk(
                    self.local_rank,
                    -_tick_counter,
                    "tick_start",
                    tick=_tick_counter,
                    poll_ms=poll_ms,
                )
            # Step 1: each rank polls ZMQ briefly. Anyone may receive.
            packet = None
            if use_rocm_recv:
                try:
                    packet = self.request_sender.try_receive_from_loadgen(
                        timeout_ms=poll_ms
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"[Worker Comm: {self.local_rank}] [Step3d] "
                        f"ZMQ recv error: {exc!r}; continuing"
                    )
                    packet = None
            else:
                if self.request_sender.probe_from_loadgen():
                    packet = self.request_sender.receive_from_loadgen()
                else:
                    # Sleep poll_ms so peers in Allreduce don't burn CPU.
                    time.sleep(poll_ms / 1000.0)
            if _TRACE_LAYER_A:
                trace_wk(
                    self.local_rank,
                    -_tick_counter,
                    "zmq_poll_done",
                    tick=_tick_counter,
                    got_packet=bool(packet is not None),
                )

            # Step 2: exchange a packed bitmask so every rank knows which
            # peers received a packet this tick (and will Bcast it).
            send_buf[0] = (1 << my_wrank) if packet is not None else 0
            wcomm.Allreduce(send_buf, recv_buf, op=MPI.SUM)
            mask = int(recv_buf[0])
            if _TRACE_LAYER_A:
                trace_wk(
                    self.local_rank,
                    -_tick_counter,
                    "allreduce_done",
                    tick=_tick_counter,
                    mask=mask,
                )
            if mask == 0:
                continue  # nobody had work this tick
            # Coordination cost before any useful work: ZMQ poll (incl. the
            # wait for peers in lockstep) + the winner-discovery Allreduce.
            # (tick.coord n vs tick.bcast n gives batches-per-tick.)
            record("tick.coord", time.perf_counter() - _t_tick0)

            # Step 3-5: process every packet this tick in ascending rank
            # order. Each src does its own Bcast(size) + Bcast(payload),
            # then every worker collates + enqueues + serial_waits. The
            # batching thread is itself serial per rank so back-to-back
            # batches on the same tick are pipelined cleanly into it.
            for src in range(world):
                if not (mask & (1 << src)):
                    continue
                _t_bcast0 = time.perf_counter()
                if my_wrank == src:
                    payload_bytes = self.request_sender._serialize_request(
                        packet
                    )
                    size_buf[0] = len(payload_bytes)
                else:
                    payload_bytes = None
                    size_buf[0] = 0
                wcomm.Bcast(size_buf, root=src)
                size = int(size_buf[0])
                if _TRACE_LAYER_A:
                    trace_wk(
                        self.local_rank,
                        -_tick_counter,
                        "bcast_size_done",
                        tick=_tick_counter,
                        src=src,
                        size=size,
                    )
                if size <= 0:
                    continue
                if my_wrank == src:
                    payload_arr = np.frombuffer(
                        payload_bytes, dtype=np.uint8
                    ).copy()
                else:
                    payload_arr = np.empty(size, dtype=np.uint8)
                wcomm.Bcast(payload_arr, root=src)
                if _TRACE_LAYER_A:
                    trace_wk(
                        self.local_rank,
                        -_tick_counter,
                        "bcast_payload_done",
                        tick=_tick_counter,
                        src=src,
                    )
                local_packet = self.request_sender._deserialize_request(
                    bytes(payload_arr)
                )
                # Per-batch payload broadcast cost (size + payload Bcast over
                # the worker comm + deserialize) — the lockstep transport tax.
                record("tick.bcast", time.perf_counter() - _t_bcast0)

                self._packet_counter += 1
                seq = getattr(local_packet, "zmq_seq", -1)
                trace_wk(
                    self.local_rank,
                    seq if seq >= 0 else self._packet_counter,
                    "lg_recv",
                    packet=self._packet_counter,
                    processed=self.num_batch_processed,
                    is_warmup=local_packet.is_warmup,
                    lockstep_root=src,
                    **packet_summary(
                        local_packet.query_ids,
                        local_packet.ts_request_pairs,
                    ),
                )

                # Plan 05 plan Change B (Plan 06 Phase 6.3) — wire
                # ``use_cuda_streams=True`` into the lockstep listener so the
                # collate H2D for batch N+1 overlaps with batch N's predict
                # on default_stream. The default (single-receiver) listener
                # at ``_listen_loop`` does this at :1028-1043; the lockstep
                # path was missing it, so even with ``DLRM_USE_CUDA_STREAMS=1``
                # the streams were unengaged on Mode B (Plan 06 §6.3 RCA).
                transfer_done_event = None
                if self.use_cuda_streams:
                    transfer_done_event = torch.cuda.Event()
                    with torch.cuda.stream(self.data_stream):
                        if local_packet.is_warmup:
                            sample = None
                        else:
                            trace_wk(
                                self.local_rank,
                                seq if seq >= 0 else self._packet_counter,
                                "collate_start",
                                packet=self._packet_counter,
                            )
                            batch_start = time.perf_counter()
                            sample = (
                                self.query_streaming_sampler.ds.get_samples_with_ts_updated(
                                    local_packet.ts_request_pairs
                                )
                            )
                            batch_end = time.perf_counter()
                            record("batch.collate", batch_end - batch_start)
                            batch_latency_ms = (batch_end - batch_start) * 1000
                            self._batching_latencies.append(
                                (self._packet_counter, batch_latency_ms)
                            )
                            trace_wk(
                                self.local_rank,
                                seq if seq >= 0 else self._packet_counter,
                                "collate_done",
                                collate_ms=round(batch_latency_ms, 2),
                            )
                    transfer_done_event.record(self.data_stream)
                else:
                    if local_packet.is_warmup:
                        sample = None
                    else:
                        trace_wk(
                            self.local_rank,
                            seq if seq >= 0 else self._packet_counter,
                            "collate_start",
                            packet=self._packet_counter,
                        )
                        batch_start = time.perf_counter()
                        sample = (
                            self.query_streaming_sampler.ds.get_samples_with_ts_updated(
                                local_packet.ts_request_pairs
                            )
                        )
                        batch_end = time.perf_counter()
                        record("batch.collate", batch_end - batch_start)
                        batch_latency_ms = (batch_end - batch_start) * 1000
                        self._batching_latencies.append(
                            (self._packet_counter, batch_latency_ms)
                        )
                        trace_wk(
                            self.local_rank,
                            seq if seq >= 0 else self._packet_counter,
                            "collate_done",
                            collate_ms=round(batch_latency_ms, 2),
                        )

                skip_send = my_wrank != src
                data_packet_result = MPIDataPacketTSRequest(
                    query_ids=local_packet.query_ids,
                    ts_request_pairs=local_packet.ts_request_pairs,
                    batch=sample,
                    is_warmup=local_packet.is_warmup,
                    transfer_done_event=transfer_done_event,
                    zmq_seq=seq,
                    skip_result_send=skip_send,
                )
                self.enqueue_batch(data_packet_result)
                trace_wk(
                    self.local_rank,
                    seq if seq >= 0 else self._packet_counter,
                    "enqueue_done",
                    qdepth=self.request_queue.qsize(),
                    skip_send=skip_send,
                )

                # ROCm serial wait per dispatched batch (matches default
                # listener). Pipelines back-to-back batches into the
                # batching thread; the next src's Bcast can only start
                # after the previous batch finishes here.
                _t_sw0 = time.perf_counter()
                self._rocm_serial_wait(
                    seq if seq >= 0 else self._packet_counter
                )
                record("tick.serial_wait", time.perf_counter() - _t_sw0)

            # Whole work-tick wall time (coord + every batch's bcast + collate
            # + serial_wait-on-predict). Sums the budget for this tick.
            record("tick.total", time.perf_counter() - _t_tick0)

        logger.debug(
            f"[Worker Comm: {self.local_rank}] [Step3d] lockstep listener stopped."
        )

    def _start_request_listener(self, tag: int = None, poll_interval_sec: float = 0.0001):
        """
        Producer thread: Listens for MPI messages and prepares batches on data_stream.

        With CUDA streams enabled:
            - Batching/data prep happens on data_stream
            - Records event when prep is done
            - Consumer (dispatch thread) waits on event before inference on default_stream

        Timeline (overlapped):
            data_stream:     [Prep B0][Prep B1][Prep B2][Prep B3]...
            default_stream:         [Infer B0][Infer B1][Infer B2]...

        Args:
            tag: Message tag to match. If None, listens for any tag.
            poll_interval_sec: Sleep between probe cycles to avoid busy-waiting
        """
        self._request_stop_event.clear()
        self.sample = None

        def _listen_loop():
            # CRITICAL: Set CUDA device context for this thread
            # Each thread in PyTorch needs its own device context
            torch.cuda.set_device(self.device)

            # Ensure garbage collection is disabled in this thread for consistent latency
            gc.disable()

            use_rocm_recv = os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1"
            # Phase 7.5 idle-driven drain handshake state (Plan 07).
            # When LoadGen finishes issuing queries the listener sees no new
            # packets, while batches still in the worker's queue traverse
            # ``route_lookup`` global all_to_all_single collectives. If the
            # per-rank batch counts differ (always at end-of-test under
            # round-robin when total_batches % num_shards != 0), the
            # trailing rank blocks on the next collective whose peer side
            # will never be enqueued. gloo's 300 s watchdog then fires
            # inside the worker, blowing up the LG-side pre-flush drain.
            # Fix: on a sustained idle window AND batching caught up, run
            # ``_drain_pad_collective`` from the listener thread to
            # equalize per-rank packet counts before the batching thread
            # hits the asymmetric collective. mpi4py's ``worker_comm``
            # Allreduce is on a transport disjoint from RCCL's worker_pg
            # so it can run concurrently with any in-flight route_lookup
            # without rank-pairing conflict.
            needs_idle_drain = self._needs_drain_pad()
            drain_pad_idle_s = float(
                os.environ.get("DLRM_SPARSE_DRAIN_PAD_IDLE_MS", "1500")
            ) / 1000.0
            last_recv_t = time.perf_counter()

            def _maybe_drain_pad():
                # Fires on every sustained idle window (no new packets for
                # ``drain_pad_idle_s``). Critical: the guard is purely on
                # listener-side staleness — we deliberately do NOT gate on
                # batching progress because the batching thread may be
                # BLOCKED in the very route_lookup collective we're trying
                # to unblock (in which case ``num_batch_processed <
                # _packet_counter`` is exactly the symptom of the hang).
                #
                # Re-fires every ``drain_pad_idle_s`` during continued idle
                # to handle MLPerf's two-stage flush (pre-flush drain ->
                # send partial -> post-flush drain), where a fresh
                # asymmetry can appear AFTER the first handshake.
                # ``_drain_pad_collective`` is a cheap no-op when both
                # ranks are already equalized (single MPI Allreduce on a
                # ~8-byte payload, ~1 ms on local NICs).
                nonlocal last_recv_t
                if not needs_idle_drain:
                    return
                if (time.perf_counter() - last_recv_t) <= drain_pad_idle_s:
                    return
                self._drain_pad_collective()
                last_recv_t = time.perf_counter()

            while not self._request_stop_event.is_set():
                if use_rocm_recv:
                    data_packet = self.request_sender.try_receive_from_loadgen()
                    if data_packet is None:
                        _maybe_drain_pad()
                        continue
                else:
                    if not self.request_sender.probe_from_loadgen():
                        _maybe_drain_pad()
                        time.sleep(poll_interval_sec)
                        continue
                    with nvtx.annotate(f"receive from loadgen rank", color="blue"):
                        data_packet = self.request_sender.receive_from_loadgen()

                last_recv_t = time.perf_counter()
                self._packet_counter += 1
                seq = getattr(data_packet, "zmq_seq", -1)
                trace_wk(
                    self.local_rank,
                    seq if seq >= 0 else self._packet_counter,
                    "lg_recv",
                    packet=self._packet_counter,
                    processed=self.num_batch_processed,
                    is_warmup=data_packet.is_warmup,
                    **packet_summary(
                        data_packet.query_ids, data_packet.ts_request_pairs
                    ),
                )

                # ========== CUDA Streams Path: Prepare Data on data_stream ==========
                transfer_done_event = None
                if self.use_cuda_streams:
                    transfer_done_event = torch.cuda.Event()
                    with torch.cuda.stream(self.data_stream):
                        if data_packet.is_warmup:
                            sample = None
                        else:
                            with nvtx.annotate(f"Producer-Prep-batching-{self._packet_counter}", color="blue"):
                                batch_start = time.perf_counter()
                                sample = self.query_streaming_sampler.ds.get_samples_with_ts_updated(data_packet.ts_request_pairs)
                                batch_end = time.perf_counter()
                                record("batch.collate", batch_end - batch_start)
                                batch_latency_ms = (batch_end - batch_start) * 1000
                                self._batching_latencies.append((self._packet_counter, batch_latency_ms))
                    transfer_done_event.record(self.data_stream)
                else:
                    with nvtx.annotate(f"MPI Listener - batching dataset index packet and sending to dispatch thread", color="green"):
                        if data_packet.is_warmup:
                            sample = None
                        else:
                            trace_wk(
                                self.local_rank,
                                seq if seq >= 0 else self._packet_counter,
                                "collate_start",
                                packet=self._packet_counter,
                            )
                            batch_start = time.perf_counter()
                            sample = self.query_streaming_sampler.ds.get_samples_with_ts_updated(data_packet.ts_request_pairs)
                            batch_end = time.perf_counter()
                            record("batch.collate", batch_end - batch_start)
                            batch_latency_ms = (batch_end - batch_start) * 1000
                            self._batching_latencies.append((self._packet_counter, batch_latency_ms))
                            trace_wk(
                                self.local_rank,
                                seq if seq >= 0 else self._packet_counter,
                                "collate_done",
                                collate_ms=round(batch_latency_ms, 2),
                            )

                data_packet_result = MPIDataPacketTSRequest(
                    query_ids=data_packet.query_ids,
                    ts_request_pairs=data_packet.ts_request_pairs,
                    batch=sample,
                    is_warmup=data_packet.is_warmup,
                    transfer_done_event=transfer_done_event,
                    zmq_seq=seq,
                )
                self.enqueue_batch(data_packet_result)
                trace_wk(
                    self.local_rank,
                    seq if seq >= 0 else self._packet_counter,
                    "enqueue_done",
                    qdepth=self.request_queue.qsize(),
                )
                self._rocm_serial_wait(
                    seq if seq >= 0 else self._packet_counter
                )

            logger.debug(f"[Worker Comm: {self.local_rank}] [Backend Request Listener] Stopped.")

        if zmq_trace_enabled():
            log_path = os.environ.get("DLRM_ZMQ_TRACE_LOG", "")
            logger.info(
                f"[Worker Comm: {self.local_rank}] ZMQ trace ON"
                + (f" -> {log_path}" if log_path else " (stderr)")
            )
        # Phase 2b Step 3d: pick the lockstep dispatch path when sharded
        # routing is the live sparse mode; otherwise keep the legacy
        # single-receiver ZMQ fair-queue listener (zero behavior change
        # for REPLICATE=1 / Step 3b smokes and CPU-sparse runs).
        if self._lockstep_enabled():
            loop_fn = self._lockstep_listen_loop
        else:
            loop_fn = _listen_loop
        self._request_listener_thread = Thread(target=loop_fn, daemon=True)
        self._request_listener_thread.start()

    def _stop_request_listener(self, timeout: float = 10.0):
        """
        Gracefully stop the request listener thread.

        Signals the listener thread to stop, waits for it to finish, and
        shuts down the communication channel.

        Phase 7.5 (2026-05-28): under sharded dispatch we run the
        drain-handshake collective in this method, BETWEEN the listener
        thread join and the ZMQ shutdown. See ``_drain_pad_collective``.

        Args:
            timeout: Maximum time to wait for the thread to finish (in seconds).
        """
        if self._request_listener_thread is None:
            return

        self._request_stop_event.set()
        self._request_listener_thread.join(timeout=timeout)
        # Phase 7.5: equalize per-rank packet counts before the batching
        # thread drains. Required under sharded dispatch when total LG
        # batches don't divide evenly into W workers (e.g. 5831 = 8*728 + 7
        # at W=8 b=6 qps=290 / 120 s) — the trailing rank otherwise blocks
        # the others on a route_lookup collective that will never have its
        # other side enqueued. Gated on the same condition that turns
        # ``route_lookup`` into a global collective: sharded dispatch +
        # sharded sparse + worker subcomm exists.
        if self._needs_drain_pad():
            self._drain_pad_collective()
        self.request_sender.shutdown()
        logger.info(f"[Worker Comm: {self.local_rank}] [Backend Request Listener] received {self._packet_counter} packets.")

    def _needs_drain_pad(self) -> bool:
        """Phase 7.5 — gate the drain handshake.

        Active only when sharded dispatch is the live path (sharded sparse
        with a worker subcomm and lockstep dispatch off). Lockstep already
        equalizes per-rank batch counts via its Allreduce-driven listener
        so no handshake is needed there. Allow explicit override via
        ``DLRM_SPARSE_DRAIN_PAD={0,1}``.
        """
        env = os.environ.get("DLRM_SPARSE_DRAIN_PAD", "").strip()
        if env == "0":
            return False
        if env == "1":
            return self.worker_comm is not None
        if self.worker_comm is None:
            return False
        if self._lockstep_enabled():
            return False
        # Auto-on when sparse routing (Step 3c) is the live path AND
        # lockstep dispatch is off (Plan 07 sharded dispatch path).
        return os.environ.get("DLRM_SPARSE_REPLICATE", "") == "0"

    def _drain_pad_collective(self) -> None:
        """Phase 7.5 (Plan 07) — equalize per-rank batch counts before drain.

        Under sharded dispatch every batch enters ``route_lookup``'s global
        ``all_to_all_single`` collectives (3 per route × 2 routes per
        batch = 6 collectives / batch). When LoadGen's total query count
        doesn't divide evenly into ``num_shards`` (always the case for
        non-integer-multiple test windows), the trailing rank has one
        fewer batch than the leading ranks. On drain, the leading ranks
        exit while the trailing ranks block forever waiting for the
        collective rendezvous of the next batch that will never come.

        Fix: ``worker_comm.allreduce(_packet_counter, op=MAX)`` to find
        the global max ``K``, then short ranks enqueue ``K - local_count``
        synthetic ``skip_result_send`` batches. The batches flow through
        ``batching_loop`` → ``backend.predict()`` → ``route_lookup`` just
        like production batches, so the collectives have symmetric
        participation. The fast ranks' "extra" collective on their last
        batch is satisfied by the slow ranks' padding-batch collective.

        Synthetic batches use the same shape as
        ``_warmup_batching_thread`` (real sampler-built batches with
        ``is_warmup=False`` + ``skip_result_send=True``). Warmup batches
        have ``transfer_done_event=None`` and the Phase 7.1a
        ``batching_loop`` guard already handles that.
        """
        try:
            from mpi4py import MPI
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[Worker Comm: {self.local_rank}] "
                f"[Phase 7.5 drain] mpi4py unavailable: {exc!r}; "
                f"skipping drain handshake (drain may hang)"
            )
            return

        local_count = self._packet_counter
        try:
            target = self.worker_comm.allreduce(local_count, op=MPI.MAX)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[Worker Comm: {self.local_rank}] "
                f"[Phase 7.5 drain] allreduce(MAX) failed: {exc!r}"
            )
            return
        needed = target - local_count
        if needed <= 0:
            # No-op idle heartbeat (other ranks also idle, equalized) —
            # log at debug to keep production logs clean. We re-fire
            # every drain_pad_idle_s so this can happen multiple times.
            logger.debug(
                f"[Worker Comm: {self.local_rank}] "
                f"[Phase 7.5 drain] local={local_count} max={target} pad=0 (no-op)"
            )
            return
        logger.info(
            f"[Worker Comm: {self.local_rank}] "
            f"[Phase 7.5 drain] local={local_count} max={target} pad={needed}"
        )

        total_requests = max(1, self.query_streaming_sampler.total_requests - self.batch_size)
        for i in range(needed):
            # Build a real-shape batch identical to _warmup_batching_thread.
            # The Plan 05 patch ensures route_lookup symmetric participation
            # even with skewed bucket counts, so the exact indices used here
            # don't matter for collective correctness — only the COUNT of
            # batches needs to match across ranks.
            base = (i * self.batch_size) % total_requests
            query_ids = list(range(base, base + self.batch_size))
            outputs_ts = self.query_streaming_sampler.get_samples_indices(query_ids)
            sample = self.query_streaming_sampler.ds.get_samples_with_ts_updated(outputs_ts)
            padding = MPIDataPacketTSRequest(
                query_ids=query_ids,
                ts_request_pairs=outputs_ts,
                batch=sample,
                is_warmup=False,
                transfer_done_event=None,
                skip_result_send=True,
                zmq_seq=-(local_count + i + 1),
            )
            self.enqueue_batch(padding)
            self._packet_counter += 1
        logger.info(
            f"[Worker Comm: {self.local_rank}] "
            f"[Phase 7.5 drain] enqueued {needed} pad batches; "
            f"counter now {self._packet_counter}"
        )

    def _collect_stage_timing(self, predict_wall_ms: float):
        """
        Resolve the CUDA events recorded by backend.predict into per-stage ms.

        Called right after the batching_loop's stream synchronize, so the events
        are guaranteed complete and elapsed_time() is valid (no extra sync added).
        """
        events = getattr(self.backend, "last_stage_events", None)
        if events is None:
            return
        ev0, ev1, ev2 = events
        embed_ms = ev0.elapsed_time(ev1)      # embedding_lookup (NVE route_lookup)
        forward_ms = ev1.elapsed_time(ev2)    # HSTU main_forward
        self._predict_stage_latencies.append((self._stage_idx, embed_ms, forward_ms, predict_wall_ms))
        self._stage_idx += 1
        self.backend.last_stage_events = None

    def dump_stage_latency(self, output_dir: str = "batching_latency"):
        """
        Dump per-stage predict latencies (embed lookup vs HSTU forward) to CSV.

        Only writes when DLRM_STAGE_TIMING=1 produced samples. Columns:
            batch_index, embed_lookup_ms, hstu_forward_ms, predict_wall_ms
        """
        self.dump_nve_cache_metrics(output_dir)
        if not getattr(self, "_predict_stage_latencies", None):
            return
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"backend_stage_latency_{self.local_rank}.csv")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["batch_index", "embed_lookup_ms", "hstu_forward_ms", "predict_wall_ms"])
            for batch_index, embed_ms, forward_ms, wall_ms in self._predict_stage_latencies:
                writer.writerow([batch_index, f"{embed_ms:.4f}", f"{forward_ms:.4f}", f"{wall_ms:.4f}"])

        logger.info(f"[Worker Comm: {self.local_rank}] Stage latency saved to {output_path} ({len(self._predict_stage_latencies)} batches)")

    def dump_nve_cache_metrics(self, output_dir: str = "batching_latency"):
        if os.environ.get("DLRM_NVE_CACHE_METRICS", "0") != "1":
            return
        collection = getattr(getattr(self.backend, "model_impl", None), "_embedding_collection", None)
        if collection is None or not hasattr(collection, "cache_metrics"):
            return
        metrics = collection.cache_metrics()
        if not metrics:
            return
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"nve_cache_metrics_{self.local_rank}.json")
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        logger.info(f"[Worker Comm: {self.local_rank}] NVE cache metrics saved to {output_path}")

    def dump_latency(self, output_dir: str = "batching_latency"):
        """
        Dump batching latencies to a CSV file.

        Args:
            output_dir: Directory to save the latency file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"backend_batching_latency_{self.local_rank}.csv")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["batch_index", "latency_ms"])
            for batch_index, latency_ms in self._batching_latencies:
                writer.writerow([batch_index, f"{latency_ms:.4f}"])

        logger.info(f"[Worker Comm: {self.local_rank}] Batching latency saved to {output_path} ({len(self._batching_latencies)} batches)")
