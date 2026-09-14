import logging
import os
import sys

from inference_harness.rocm_bootstrap import apply_rocm_env
from inference_harness.rocm_timing import maybe_report, record
from inference_harness.zmq_trace import packet_summary, trace_lg, zmq_trace_enabled
from inference_harness.accuracy_safety import (
    accuracy_response_candidate_size,
    accuracy_response_copy_size,
)

if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1":
    apply_rocm_env()

from inference_harness.mpi_utils import MPIDataPacketDSIndex, MPIDataPacketTSRequest, ZMQRequestSenderShardedConfig, ZMQRequestSenderSharded
from inference_harness.dataset.streaming_query_sampler import StreamingQuerySamplerRef
import time
import threading
from typing import Dict, List, Optional, Tuple
import argparse
import mlperf_loadgen as lg
import numpy as np
import torch
import nvtx
import gc

# QuerySamplesComplete pool (C++ TestPybind or Python fallback on ROCm)
try:
    sys.path.insert(0, "/opt/ffi_utils/build")
    import TestPybind  # type: ignore
except ImportError:
    from inference_harness import testpybind_pool as TestPybind  # type: ignore

logger = logging.getLogger(__name__)
_VECTORIZE_RESPONSE_BUFFER = (
    os.environ.get("DLRM_VECTORIZE_RESPONSE_BUFFER", "0") == "1"
)
_REUSE_LOADGEN_BUFFERS = (
    os.environ.get("DLRM_REUSE_LOADGEN_BUFFERS", "0") == "1"
)
_RESPONSE_BUFFER_RING_SIZE = int(
    os.environ.get("DLRM_RESPONSE_BUFFER_RING_SIZE", "512")
)
_ACCURACY_RESPONSE_CANDIDATE_SIZE = int(
    os.environ.get("DLRM_ACCURACY_RESPONSE_CANDIDATE_SIZE", "0")
)
# End-of-stream diagnostic tracing. When enabled (``DLRM_EOS_TRACE=1``) the
# rank-8 result path logs the exact per-query metadata and candidate-size
# arithmetic for any batch whose padded results do not evenly divide the
# (possibly flush-trimmed) query count. Used to pin the TEST08 ts=9 residual
# to the flush-batch candidate_size miscompute. Off by default (zero overhead).
_EOS_TRACE = os.environ.get("DLRM_EOS_TRACE", "0") == "1"
# TEST08 correctness fix (default ON). A ROCm partial flush batch is padded
# real_n -> batch_size (``_pad_flush_batch_rocm``) so the worker sees a warm
# shape; the worker therefore returns results with width batch_size * cs. The
# recv path trims query_ids / ts_request_pairs back to real_n but historically
# left ``results`` padded, so ``report_loadgen_best_perf`` computed
# candidate_size = width // real_n (inflated above the true per-query width) and
# misaligned every per-query slice (start = candidate_size * i, i >= 1). The pad
# rows are duplicates of the last real query appended at the end, so dropping the
# trailing (padded - real_n) query blocks from results restores alignment. Set
# ``DLRM_FLUSH_TRIM_RESULTS=0`` to reproduce the legacy (buggy) behaviour.
_FLUSH_TRIM_RESULTS = os.environ.get("DLRM_FLUSH_TRIM_RESULTS", "1") == "1"
_response_buffer_rings: Dict[Tuple[int, int], List[np.ndarray]] = {}
_response_buffer_next: Dict[Tuple[int, int], int] = {}


# ========== Global Thread Pool for QuerySamplesComplete ==========
# Lazy initialization - created on first use
_qsc_pool = None
_QSC_POOL_NUM_THREADS = 10


def get_qsc_pool():
    """
    Get or create the global QuerySamplesComplete thread pool.

    Returns:
        TestPybind.QuerySamplesCompletePool: Thread pool for async query completion.
    """
    global _qsc_pool
    if _qsc_pool is None:
        _qsc_pool = TestPybind.QuerySamplesCompletePool(
            num_threads=_QSC_POOL_NUM_THREADS,
            test_mode=False
        )
    return _qsc_pool


# ========== Buffer Management for Response Buffers ==========
# Prevents use-after-free by keeping buffers alive until thread pool processes them
_accuracy_buffers = []
# Maximum buffers to retain before cleanup. <=0 disables cleanup entirely
# (retain all), used to test end-of-stream use-after-free hypotheses where the
# async QSC pool may lag behind buffer truncation at stream drain.
_ACCURACY_BUFFER_MAX_SIZE = int(
    os.environ.get("DLRM_ACCURACY_BUFFER_MAX_SIZE", "100")
)


def _store_accuracy_buffer(buf):
    """
    Store buffer reference to prevent garbage collection until thread pool processes it.

    In performance/accuracy mode, we pass buffer pointers to a C++ thread pool that
    processes them asynchronously. We must keep Python references alive to prevent
    garbage collection.

    Args:
        buf: NumPy array or buffer to keep alive.
    """
    global _accuracy_buffers
    _accuracy_buffers.append(buf)

    # Cleanup old buffers when we exceed max size (thread pool should have processed them).
    # A non-positive max disables cleanup so every buffer stays alive until process exit.
    if _ACCURACY_BUFFER_MAX_SIZE > 0 and len(_accuracy_buffers) > _ACCURACY_BUFFER_MAX_SIZE:
        # Keep only the last half to avoid unbounded growth
        _accuracy_buffers = _accuracy_buffers[_ACCURACY_BUFFER_MAX_SIZE // 2:]


def clear_accuracy_buffers():
    """Clear all stored accuracy buffers. Call after benchmarking completes."""
    global _accuracy_buffers
    _accuracy_buffers.clear()


def _get_reused_response_2d(
    pool,
    num_queries: int,
    floats_per_query: int,
) -> Tuple[np.ndarray, bool]:
    """Return a reusable response buffer when safe, else a fresh one.

    The production C++ QSC pool queues pointers and calls LoadGen later, so
    reuse must be a ring, not a single buffer. A 512-slot ring is >2 seconds
    of q11.9k Server traffic at b64, far beyond normal QSC lag. If the pool's
    own queue grows close to the ring size, fall back to a fresh allocation.
    """
    shape = (num_queries, floats_per_query)
    if not _REUSE_LOADGEN_BUFFERS or _RESPONSE_BUFFER_RING_SIZE <= 0:
        return np.empty(shape, dtype=np.float32), False

    qsize_fn = getattr(pool, "queue_size", None)
    if callable(qsize_fn):
        try:
            if qsize_fn() >= max(1, _RESPONSE_BUFFER_RING_SIZE - _QSC_POOL_NUM_THREADS - 1):
                return np.empty(shape, dtype=np.float32), False
        except Exception:  # noqa: BLE001
            pass

    ring = _response_buffer_rings.get(shape)
    if ring is None:
        ring = [np.empty(shape, dtype=np.float32) for _ in range(_RESPONSE_BUFFER_RING_SIZE)]
        _response_buffer_rings[shape] = ring
        _response_buffer_next[shape] = 0
    idx = _response_buffer_next[shape]
    _response_buffer_next[shape] = (idx + 1) % len(ring)
    return ring[idx], True


# MLPerf scenario mapping
SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

# MLPerf test mode mapping
MODE_MAP = {
    "performance": lg.TestMode.PerformanceOnly,
    "accuracy": lg.TestMode.AccuracyOnly,
    "find_peak": lg.TestMode.FindPeakPerformance,
}


def parse_user_conf(
    args: argparse.Namespace,
) -> Tuple[lg.TestSettings, lg.LogSettings, str]:
    """
    Parse MLPerf user.conf and construct LoadGen settings and log settings.
    """
    settings = lg.TestSettings()
    settings.scenario = SCENARIO_MAP[args.scenario]
    settings.mode = MODE_MAP[args.mode]

    settings.FromConfig(args.user_conf, "dlrm-v3", args.scenario)
    # settings.min_query_count = streaming_query_sampler.num_queries

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.abspath(".")
    os.makedirs(output_dir, exist_ok=True)

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = output_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = False

    return settings, log_settings


@nvtx.annotate(f"report_loadgen_best_perf", color="yellow")
def report_loadgen_best_perf(
    data_packet: MPIDataPacketDSIndex,
    mt_target_preds: torch.Tensor,
    mt_target_labels: torch.Tensor,
    mt_target_weights: torch.Tensor,
    compute_eval: bool = False,
    ts_idx_list: Optional[List[float]] = None,
    query_idx_list: Optional[List[float]] = None,
):
    """
    Report inference results back to MLPerf LoadGen using multi-threaded completion.

    Uses a C++ thread pool (TestPybind.QuerySamplesCompletePool) to handle
    QuerySamplesComplete calls asynchronously, allowing the main thread to
    continue processing without blocking on LoadGen API calls.

    Args:
        data_packet: Data packet containing query IDs for result correlation.
        mt_target_preds: Model predictions tensor.
        mt_target_labels: Ground truth labels tensor.
        mt_target_weights: Sample weights tensor.
        compute_eval: If True, includes labels and weights for accuracy evaluation;
            if False, only predictions are reported (performance mode).
    """
    pool = get_qsc_pool()

    num_queries = len(data_packet.query_ids)
    if ts_idx_list is None or len(ts_idx_list) != num_queries:
        ts_idx_list = [-1.0] * num_queries
    if query_idx_list is None or len(query_idx_list) != num_queries:
        query_idx_list = [-1.0] * num_queries

    if not compute_eval:
        # Performance mode: include ts_idx/query_idx metadata + predictions
        assert mt_target_preds.is_contiguous(), "mt_target_preds is not contiguous"
        candidate_size = mt_target_preds.size(1) // num_queries
        with nvtx.annotate(f"convert to fp32", color="yellow"):
            all_preds = mt_target_preds[0].contiguous().float().numpy()  # pyre-ignore [61]

        floats_per_query = candidate_size + 2
        bytes_per_query = floats_per_query * 4
        with nvtx.annotate(f"build response buffer", color="purple"):
            if _VECTORIZE_RESPONSE_BUFFER:
                response_2d, reused_response_buffer = _get_reused_response_2d(
                    pool,
                    num_queries,
                    floats_per_query,
                )
                response_2d[:, 0] = ts_idx_list
                response_2d[:, 1] = query_idx_list
                response_2d[:, 2:] = all_preds[
                    : num_queries * candidate_size
                ].reshape(num_queries, candidate_size)
                response_buffer = response_2d.reshape(-1)
            else:
                reused_response_buffer = False
                response_buffer = np.empty(num_queries * floats_per_query, dtype=np.float32)
                for i in range(num_queries):
                    start = candidate_size * i
                    end = candidate_size * (i + 1)
                    buf_offset = i * floats_per_query
                    response_buffer[buf_offset] = float(ts_idx_list[i])
                    response_buffer[buf_offset + 1] = float(query_idx_list[i])
                    response_buffer[buf_offset + 2: buf_offset + 2 + candidate_size] = all_preds[start:end]

        with nvtx.annotate(f"report back to loadgen", color="yellow"):
            t_qsc = time.perf_counter()
            base_ptr = response_buffer.ctypes.data
            pool.enqueue_batch(data_packet.query_ids, base_ptr, bytes_per_query)
            record("loadgen.qsc_enqueue", time.perf_counter() - t_qsc)
            if zmq_trace_enabled() and getattr(data_packet, "zmq_seq", -1) >= 0:
                trace_lg(
                    -1,
                    data_packet.zmq_seq,
                    "qsc_enqueue",
                    n=num_queries,
                    bytes_per_query=bytes_per_query,
                )

        if not reused_response_buffer:
            _store_accuracy_buffer(response_buffer)
    else:
        # Accuracy mode: predictions + labels + weights + candidate_size
        candidate_size = mt_target_preds.size(1) // num_queries

        if _EOS_TRACE:
            _predsz = mt_target_preds.size(1)
            _rem = _predsz % num_queries
            if _rem != 0:
                # Padded flush results were NOT trimmed to match the
                # flush-trimmed query count: candidate_size is miscomputed and
                # every per-query slice (start = candidate_size * i) is shifted
                # by (candidate_size - true_cs) * i, reading neighbour data.
                _reqids = [
                    (int(ts_idx_list[i]), int(query_idx_list[i]))
                    for i in range(num_queries)
                ]
                logger.info(
                    f"[EOS-TRACE] emit MISALIGN nq={num_queries} "
                    f"predsz={_predsz} candidate_size={candidate_size} "
                    f"rem={_rem} reqids={_reqids}"
                )

        with nvtx.annotate(f"convert to fp32", color="yellow"):
            all_preds = mt_target_preds[0].contiguous().float().numpy()  # pyre-ignore [61]
            all_labels = mt_target_labels[0].contiguous().float().numpy()  # pyre-ignore [16,61]
            all_weights = mt_target_weights[0].contiguous().float().numpy()  # pyre-ignore [61]

        # TEST08 compares audited PerformanceOnly predictions with an AccuracyOnly
        # reference. Server perf emits the inference candidate shape (2048), while
        # AccuracyOnly normally scores 32 eval candidates. When requested, emit the
        # Server width and leave padded labels/weights at zero so verifier NE only
        # counts the real labeled candidates.
        emit_candidate_size = accuracy_response_candidate_size(
            candidate_size,
            _ACCURACY_RESPONSE_CANDIDATE_SIZE,
        )
        copy_candidate_size = accuracy_response_copy_size(
            candidate_size,
            emit_candidate_size,
        )

        # Each query: ts_idx + query_idx + preds + labels + weights + candidate_size
        floats_per_query = 3 * emit_candidate_size + 3
        bytes_per_query = floats_per_query * 4  # float32 = 4 bytes

        with nvtx.annotate(f"build response buffer", color="purple"):
            # Build contiguous buffer with interleaved data for each query
            response_buffer = np.zeros(num_queries * floats_per_query, dtype=np.float32)

            for i in range(num_queries):
                start = candidate_size * i
                end = start + copy_candidate_size
                buf_offset = i * floats_per_query

                response_buffer[buf_offset] = float(ts_idx_list[i])
                response_buffer[buf_offset + 1] = float(query_idx_list[i])
                response_buffer[buf_offset + 2: buf_offset + 2 + copy_candidate_size] = all_preds[start:end]
                response_buffer[
                    buf_offset + 2 + emit_candidate_size: buf_offset + 2 + emit_candidate_size + copy_candidate_size
                ] = all_labels[start:end]
                response_buffer[
                    buf_offset + 2 + 2 * emit_candidate_size: buf_offset + 2 + 2 * emit_candidate_size + copy_candidate_size
                ] = all_weights[start:end]
                response_buffer[buf_offset + 2 + 3 * emit_candidate_size] = float(emit_candidate_size)

        with nvtx.annotate(f"report back to loadgen", color="yellow"):
            t_qsc = time.perf_counter()
            base_ptr = response_buffer.ctypes.data
            pool.enqueue_batch(data_packet.query_ids, base_ptr, bytes_per_query)
            record("loadgen.qsc_enqueue", time.perf_counter() - t_qsc)

        # CRITICAL: Store buffer reference to prevent garbage collection
        # The thread pool processes asynchronously, so buffer must stay alive
        _store_accuracy_buffer(response_buffer)


class TestRunner:
    """
    MLPerf LoadGen test runner for distributed DLRM inference benchmarking.

    Manages the LoadGen process, coordinating query issuance to distributed workers
    and collecting results for performance/accuracy evaluation. Supports both Server
    and Offline scenarios with configurable batch sizes and communication backends.

    The runner uses a producer-consumer pattern:
    - Producer: LoadGen issues queries via issue_queries callback
    - Consumer: Results listener thread collects responses from workers

    Attributes:
        streaming_query_sampler: Dataset sampler for loading queries.
        batch_size: Number of samples per batch sent to workers.
        rank: MPI rank of this LoadGen process.
        mode: Operating mode ("performance" or "accuracy").
        scenario: MLPerf scenario ("server" or "offline").
        request_sender: Communication backend for worker coordination.
    """

    def __init__(self,
                 streaming_query_sampler: StreamingQuerySamplerRef,
                 batch_size: int = 128,
                 rank: int = 0,
                 worker_world_size: int = 1,
                 verbose: int = 0,
                 mode: str = "performance",
                 scenario: str = "offline",
                 communicator_config: ZMQRequestSenderShardedConfig = None):
        """
        Initialize the MLPerf LoadGen test runner.

        Args:
            streaming_query_sampler: Dataset sampler for loading queries.
            batch_size: Number of samples per batch sent to workers.
            rank: MPI rank of this LoadGen process.
            worker_world_size: Total number of inference worker processes.
            verbose: Logging verbosity level (0=INFO, 1=DEBUG, -1=WARNING).
            mode: Operating mode ("performance" or "accuracy").
            scenario: MLPerf scenario ("server" or "offline").
            communicator_config: Configuration for inter-process communication.
        """
        # Disable garbage collection for consistent latency
        gc.disable()

        # Core configuration
        self.streaming_query_sampler = streaming_query_sampler
        self.batch_size = batch_size
        self.rank = rank
        self.mode = mode
        self.scenario = scenario
        self._worker_ranks = list(range(worker_world_size))
        # Diagnostic-only: rotate round-robin worker assignment without changing
        # rank/GPU mapping. This separates data-residue skew from a true rank0
        # or physical-GPU0 straggler.
        if self._worker_ranks:
            rotate = int(os.environ.get("DLRM_WORKER_RANK_ROTATE", "0"))
            rotate %= len(self._worker_ranks)
            if rotate:
                self._worker_ranks = self._worker_ranks[rotate:] + self._worker_ranks[:rotate]
                logger.info(
                    "[Loadgen Rank: %s] Worker round-robin order rotated by %s: %s",
                    rank,
                    rotate,
                    self._worker_ranks,
                )
        self._num_workers = worker_world_size

        # Initialize communication backend
        if isinstance(communicator_config, ZMQRequestSenderShardedConfig):
            logger.info(f"[Loadgen Rank: {rank}] Loadgen hostname: {communicator_config.loadgen_hostname}, num_shards: {communicator_config.num_shards}")
            # Sharded ZMQ: Creates num_shards socket pairs for better scalability
            # Workers connect to ports based on their shard_id for reduced fan-in
            self.request_sender = ZMQRequestSenderSharded(
                is_loadgen=True,
                shard_id=0,  # LoadGen doesn't use shard_id
                rank=rank,
                config=communicator_config
            )
        else:
            # MPI communication path (not yet implemented)
            self.worker_world_size = worker_world_size
            self.use_async_mpi = communicator_config.use_async_mpi
            pass

        # Debugging and testing utilities
        self._dummy_response_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
        self.debug_accuracy = False
        self._accuracy_dump_file = None

        # Results collection thread and query samples complete pool
        self.query_samples_complete_pool = TestPybind.QuerySamplesCompletePool(
            num_threads=_QSC_POOL_NUM_THREADS,
            test_mode=False
        )
        self.result_stop_event = threading.Event()
        self.results_thread = threading.Thread(target=self._results_listener, daemon=True)
        self.results_thread.start()

        # Performance tracking counters
        self.out_batch_counter = 0  # Number of batches sent to workers
        self.in_batch_counter = 0   # Number of batches received from workers
        self.num_issue_queries_called = 0  # Number of times issue_queries was called

        # Latency testing mode (for debugging/profiling, not MLPerf benchmarking)
        self.test_latency_mode = False
        self._latency_send_times: Dict[int, float] = {}  # request_idx -> send timestamp
        self._latency_results: list = []  # list of (request_idx, latency_ms)

        # Current batch accumulation
        self.current_query_ids = []
        self.current_qsl_ids = []
        # ROCm flush padding: map first query id -> real (unpadded) batch size
        self._flush_real_query_counts: Dict[int, int] = {}
        self._zmq_warmup_mode: bool = False

        # Plan 48 Probe A.3 — hybrid ΣLᵢ² dispatch: cap each batch at batch_size AND cut
        # early when accumulated ΣLᵢ² >= T (T = alpha * batch_size * mean(seq_len²)). This
        # clips heavy-batch (long-history) service-time tails with no b>batch_size downside.
        # Default off => fixed batch_size cut (bit-identical).
        self._l2_dispatch = os.environ.get("DLRM_L2_DISPATCH", "0") == "1"
        self._l2_alpha = float(os.environ.get("DLRM_L2_DISPATCH_ALPHA", "1.0"))
        self._l2_T = None  # lazily computed on first Server batch
        self._l2_cur_target = batch_size
        if self._l2_dispatch:
            logger.info(f"[l2-dispatch] DLRM_L2_DISPATCH=1 alpha={self._l2_alpha} "
                        f"cap={batch_size} (cut at min(cap, count s.t. ΣLᵢ²>=T))")

        # Configure logging verbosity
        self.verbose = verbose
        if self.verbose >= 1:
            logger.setLevel(logging.DEBUG)
        elif self.verbose == 0:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

    def setup_loadgen(self, args: argparse.Namespace):
        """
        Set up MLPerf LoadGen components: SUT, QSL, and settings.

        Constructs the System Under Test (SUT) with query issuance callbacks,
        the Query Sample Library (QSL) with dataset access, and configures
        test settings from user.conf and command-line arguments.

        Args:
            args: Command-line arguments containing scenario, mode, and configuration paths.

        Returns:
            tuple: (sut, qsl, settings, log_settings) for use with LoadGen.StartTest.
        """
        logger.info(f"[Loadgen Rank: {self.rank}] Setting up LoadGen...")
        self.test_latency_mode = False

        self.mode = args.mode

        sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

        count = self.streaming_query_sampler.get_item_count()
        print("QSL has count: ", count)
        qsl = lg.ConstructQSL(
            count,
            count,
            self.streaming_query_sampler.load_query_samples,
            self.streaming_query_sampler.unload_query_samples,
        )

        return sut, qsl

    def issue_queries(self, query_samples):
        """
        MLPerf LoadGen callback to issue queries to inference workers.

        Routes query issuance to scenario-specific handlers (Server or Offline).
        This method is called by LoadGen whenever queries need to be processed.

        Args:
            query_samples: List of QuerySample objects from LoadGen.

        Raises:
            ValueError: If an unsupported scenario is specified.
        """
        if self.scenario.lower() == "server":
            self.issue_queries_server(query_samples)
        elif self.scenario.lower() == "offline":
            self.issue_queries_offline(query_samples)
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}")

    def issue_queries_server(self, query_samples):
        """
        Issue queries to workers in Server scenario.

        In Server mode, queries arrive dynamically and are batched until
        batch_size is reached, then sent to workers via round-robin.

        Args:
            query_samples: List of QuerySample objects from LoadGen.
        """
        with nvtx.annotate(f"issue_queries", color="blue"):
            self.num_issue_queries_called += 1
            if zmq_trace_enabled() and (
                self.num_issue_queries_called <= 3
                or self.num_issue_queries_called % 50 == 0
            ):
                s0 = query_samples[0]
                trace_lg(
                    self.rank,
                    0,
                    "issue_queries_cb",
                    call=self.num_issue_queries_called,
                    n_samples=len(query_samples),
                    buf=len(self.current_query_ids),
                    out=self.out_batch_counter,
                    in_=self.in_batch_counter,
                    sample0_id=int(s0.id),
                    sample0_index=int(s0.index),
                )

            # Plan 48 A.3 — hybrid ΣLᵢ² dispatch: when a fresh batch starts, set this
            # batch's cut count = min(batch_size, #upcoming queries until ΣLᵢ² >= T),
            # peeked from the sampler cursor (read-only; get_samples_indices below advances
            # it by the actual count sent, so peek/advance stay aligned). Off => batch_size.
            if self._l2_dispatch and len(self.current_query_ids) == 0:
                if self._l2_T is None:
                    self._l2_T = self._l2_alpha * self.batch_size * \
                        self.streaming_query_sampler.get_l2_mean()
                self._l2_cur_target = self.streaming_query_sampler.peek_l2_batch_count(
                    self._l2_T, self.batch_size)

            # Accumulate queries into current batch
            for sample in query_samples:
                self.current_query_ids.append(int(sample.id))
                self.current_qsl_ids.append(int(sample.index))

            # Send batch when full (or when the ΣLᵢ² target count is reached)
            _cut = self._l2_cur_target if self._l2_dispatch else self.batch_size
            if len(self.current_query_ids) >= _cut:
                # Get timestamp-request_id pairs from the query sampler
                # Plan 21 Phase 21.1 — time the single-threaded issue stages.
                _t_gs = time.perf_counter()
                outputs_ts = self.streaming_query_sampler.get_samples_indices(self.current_query_ids)
                record("loadgen.issue_getsamples", time.perf_counter() - _t_gs)

                # Create data packet with timestamp-request pairs
                data_packet_ts_request = MPIDataPacketTSRequest(
                    query_ids=self.current_query_ids.copy(),
                    ts_request_pairs=outputs_ts,  # List of (ts, request_id) tuples
                    is_warmup=False
                )

                # Round-robin distribution to workers
                target_rank = self._worker_ranks[self.out_batch_counter % self._num_workers]
                next_seq = self.out_batch_counter + 1
                _t_bp = time.perf_counter()
                self._wait_issue_backpressure(next_seq, label="server")
                record("loadgen.issue_backpressure", time.perf_counter() - _t_bp)
                self.out_batch_counter += 1
                data_packet_ts_request.zmq_seq = self.out_batch_counter
                trace_lg(
                    self.rank,
                    data_packet_ts_request.zmq_seq,
                    "issue_send",
                    target_rank=target_rank,
                    out=self.out_batch_counter,
                    in_=self.in_batch_counter,
                    **packet_summary(
                        data_packet_ts_request.query_ids,
                        data_packet_ts_request.ts_request_pairs,
                    ),
                )

                logger.debug(f"[Loadgen Rank: {self.rank}] sending ts_request packet to worker rank {target_rank}, out_batch_counter: {self.out_batch_counter}, num_pairs: {len(outputs_ts)}")

                # Send to worker
                _t_sw = time.perf_counter()
                self.request_sender.send_to_worker(data_packet_ts_request)
                record("loadgen.issue_send", time.perf_counter() - _t_sw)

                # Reset batch accumulation
                self.current_query_ids = []
                self.current_qsl_ids = []

    def issue_queries_offline(self, query_samples):
        """
        Issue queries to workers in Offline scenario.

        In Offline mode, all queries are issued at once. They are batched
        and distributed to workers via round-robin as batches are filled.

        Args:
            query_samples: List of QuerySample objects from LoadGen.
        """
        logger.info(f"[Loadgen Rank: {self.rank}] [Offline Mode] query_samples length: {len(query_samples)}")
        with nvtx.annotate(f"issue_queries", color="blue"):
            self.num_issue_queries_called += 1

            # Process each sample, sending batches as they fill up
            for sample in query_samples:
                self.current_query_ids.append(int(sample.id))
                self.current_qsl_ids.append(int(sample.index))

                if len(self.current_query_ids) >= self.batch_size:
                    # Get timestamp-request_id pairs from the query sampler
                    outputs_ts = self.streaming_query_sampler.get_samples_indices(self.current_query_ids)

                    # Create data packet with timestamp-request pairs
                    data_packet_ts_request = MPIDataPacketTSRequest(
                        query_ids=self.current_query_ids.copy(),
                        ts_request_pairs=outputs_ts,  # List of (ts, request_id) tuples
                        is_warmup=False
                    )

                    # Round-robin distribution to workers
                    target_rank = self._worker_ranks[self.out_batch_counter % self._num_workers]
                    next_seq = self.out_batch_counter + 1
                    self._wait_issue_backpressure(next_seq, label="offline")
                    self.out_batch_counter += 1
                    data_packet_ts_request.zmq_seq = self.out_batch_counter
                    trace_lg(
                        self.rank,
                        data_packet_ts_request.zmq_seq,
                        "issue_send",
                        target_rank=target_rank,
                        out=self.out_batch_counter,
                        in_=self.in_batch_counter,
                        offline=True,
                        **packet_summary(
                            data_packet_ts_request.query_ids,
                            data_packet_ts_request.ts_request_pairs,
                        ),
                    )

                    logger.debug(f"[Loadgen Rank: {self.rank}] sending ts_request packet to worker rank {target_rank}, out_batch_counter: {self.out_batch_counter}, num_pairs: {len(outputs_ts)}")

                    # Send to worker
                    self.request_sender.send_to_worker(data_packet_ts_request)

                    # Reset batch accumulation
                    self.current_query_ids = []
                    self.current_qsl_ids = []

    def _pad_flush_batch_rocm(
        self, query_ids: List[int], outputs_ts: List[Tuple[int, int]]
    ) -> Tuple[List[int], List[Tuple[int, int]], int]:
        """Pad partial flush batches to full batch_size so Triton sees a warm shape."""
        real_n = len(query_ids)
        if (
            os.environ.get("DLRM_ROCM_GR_BACKEND", "0") != "1"
            or real_n >= self.batch_size
            or real_n == 0
        ):
            return query_ids, outputs_ts, real_n
        pad = self.batch_size - real_n
        padded_ids = query_ids + [query_ids[-1]] * pad
        padded_ts = outputs_ts + [outputs_ts[-1]] * pad
        self._flush_real_query_counts[padded_ids[0]] = real_n
        logger.info(
            f"[Loadgen Rank: {self.rank}] ROCm flush padding {real_n} -> {self.batch_size} queries"
        )
        return padded_ids, padded_ts, real_n

    def _wait_issue_backpressure(self, next_seq: int, label: str = "issue") -> None:
        """ROCm: cap in-flight ZMQ request batches (drops seen when LG outruns worker)."""
        if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") != "1":
            return
        max_inflight = int(os.environ.get("DLRM_ZMQ_MAX_INFLIGHT", "1"))
        if max_inflight <= 0:
            return
        timeout_s = float(os.environ.get("DLRM_ISSUE_BACKPRESSURE_TIMEOUT_S", "300"))
        deadline = time.time() + timeout_s
        stall_s = float(os.environ.get("DLRM_ZMQ_TRACE_STALL_S", "5"))
        last_stall = time.time()
        while self.out_batch_counter - self.in_batch_counter >= max_inflight:
            if time.time() > deadline:
                trace_lg(
                    self.rank,
                    next_seq,
                    "issue_backpressure_timeout",
                    label=label,
                    out=self.out_batch_counter,
                    in_=self.in_batch_counter,
                    max_inflight=max_inflight,
                )
                logger.warning(
                    f"[Loadgen Rank: {self.rank}] Issue backpressure timeout ({label}) "
                    f"after {timeout_s}s: in={self.in_batch_counter} out={self.out_batch_counter}"
                )
                return
            if time.time() - last_stall >= stall_s:
                trace_lg(
                    self.rank,
                    next_seq,
                    "issue_backpressure_stall",
                    label=label,
                    out=self.out_batch_counter,
                    in_=self.in_batch_counter,
                    pending=self.out_batch_counter - self.in_batch_counter,
                )
                last_stall = time.time()
            time.sleep(0.0001)

    def _drain_worker_batches(self, label: str, timeout_s: float | None = None) -> bool:
        """Wait until all sent batches have been received from workers (ROCm)."""
        pending = self.out_batch_counter - self.in_batch_counter
        if pending <= 0:
            return True
        if timeout_s is None:
            timeout_s = float(os.environ.get("DLRM_FLUSH_DRAIN_TIMEOUT_S", "600"))
        logger.info(
            f"[Loadgen Rank: {self.rank}] Draining {pending} outstanding batch(es) "
            f"({label})..."
        )
        deadline = time.time() + timeout_s
        last_log = time.time()
        trace_lg(
            self.rank,
            0,
            "drain_start",
            label=label,
            out=self.out_batch_counter,
            in_=self.in_batch_counter,
            pending=pending,
        )
        while self.in_batch_counter < self.out_batch_counter:
            if time.time() > deadline:
                trace_lg(
                    self.rank,
                    0,
                    "drain_timeout",
                    label=label,
                    out=self.out_batch_counter,
                    in_=self.in_batch_counter,
                )
                logger.warning(
                    f"[Loadgen Rank: {self.rank}] Drain timeout ({label}) after "
                    f"{timeout_s}s: received {self.in_batch_counter}/"
                    f"{self.out_batch_counter} batches"
                )
                return False
            if time.time() - last_log >= 10.0:
                trace_lg(
                    self.rank,
                    0,
                    "drain_stall",
                    label=label,
                    out=self.out_batch_counter,
                    in_=self.in_batch_counter,
                    pending=self.out_batch_counter - self.in_batch_counter,
                )
                logger.info(
                    f"[Loadgen Rank: {self.rank}] Still draining ({label}): "
                    f"{self.in_batch_counter}/{self.out_batch_counter} batches"
                )
                last_log = time.time()
            time.sleep(0.01)
        trace_lg(
            self.rank,
            0,
            "drain_done",
            label=label,
            out=self.out_batch_counter,
            in_=self.in_batch_counter,
        )
        logger.info(
            f"[Loadgen Rank: {self.rank}] Drain complete ({label}): "
            f"{self.in_batch_counter} batches"
        )
        return True

    def flush_queries(self):
        """
        MLPerf LoadGen callback to flush any remaining queries in the batch buffer.

        Called by LoadGen at the end of a test to ensure all queries are processed,
        even if the final batch is not full. Sends partial batches to workers.
        """
        logger.info(f"[Loadgen Rank: {self.rank}] Flushing queries...")
        trace_lg(
            self.rank,
            0,
            "flush_begin",
            out=self.out_batch_counter,
            in_=self.in_batch_counter,
            partial=len(self.current_query_ids),
        )
        if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1":
            self._drain_worker_batches("pre-flush")

        if len(self.current_query_ids) > 0:
            # Get timestamp-request pairs for remaining queries
            outputs_ts = self.streaming_query_sampler.get_samples_indices(self.current_query_ids)
            query_ids, outputs_ts, real_n = self._pad_flush_batch_rocm(
                self.current_query_ids.copy(), outputs_ts
            )

            # Create data packet for partial (possibly padded) batch
            data_packet_ts_request = MPIDataPacketTSRequest(
                query_ids=query_ids,
                ts_request_pairs=outputs_ts,
                is_warmup=False
            )

            # Send to next worker in round-robin
            target_rank = self._worker_ranks[self.out_batch_counter % self._num_workers]
            next_seq = self.out_batch_counter + 1
            self._wait_issue_backpressure(next_seq, label="flush")
            self.out_batch_counter += 1
            data_packet_ts_request.zmq_seq = self.out_batch_counter
            trace_lg(
                self.rank,
                data_packet_ts_request.zmq_seq,
                "flush_send",
                target_rank=target_rank,
                padded=len(query_ids),
                out=self.out_batch_counter,
                in_=self.in_batch_counter,
                **packet_summary(query_ids, outputs_ts, real_n=real_n),
            )
            logger.info(
                f"[Loadgen Rank: {self.rank}] Flushing {real_n} remaining queries "
                f"(batch {len(query_ids)}) to worker {target_rank}"
            )
            self.request_sender.send_to_worker(data_packet_ts_request)

            # Clear batch buffer
            self.current_query_ids = []
            self.current_qsl_ids = []

        if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1":
            self._drain_worker_batches("post-flush")

        # Allow workers to finish partial batches (ROCm Triton JIT can lag flush)
        flush_wait_s = float(os.environ.get("DLRM_FLUSH_WAIT_S", "0.5"))
        if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1":
            flush_wait_s = float(os.environ.get("DLRM_FLUSH_WAIT_S", "5.0"))
        time.sleep(flush_wait_s)
        trace_lg(
            self.rank,
            0,
            "flush_end",
            out=self.out_batch_counter,
            in_=self.in_batch_counter,
        )
        logger.info(f"[Loadgen Rank: {self.rank}] All queries flushed")

    def zmq_real_warmup(self, num_batches: int, timeout_s: float = 900.0) -> None:
        """
        Warm up LoadGen→ZMQ→worker→ZMQ with real preprocessed samples (ROCm).

        Direct worker warmup JITs Triton but skips listener/batching threads.
        """
        if num_batches <= 0:
            return
        self._zmq_warmup_mode = True
        start_in = self.in_batch_counter
        logger.info(
            f"[Loadgen Rank: {self.rank}] ZMQ real warmup: {num_batches} batches "
            f"(batch_size={self.batch_size})"
        )
        for i in range(num_batches):
            base = (i * self.batch_size) % max(
                1, self.streaming_query_sampler.total_requests - self.batch_size
            )
            query_ids = list(range(base, base + self.batch_size))
            outputs_ts = self.streaming_query_sampler.get_samples_indices(query_ids)
            data_packet_ts_request = MPIDataPacketTSRequest(
                query_ids=query_ids,
                ts_request_pairs=outputs_ts,
                is_warmup=False,
            )
            self.out_batch_counter += 1
            self.request_sender.send_to_worker(data_packet_ts_request)
            batch_deadline = time.time() + timeout_s
            while self.in_batch_counter - start_in <= i:
                if time.time() > batch_deadline:
                    raise TimeoutError(
                        f"ZMQ real warmup timed out on batch {i + 1}/{num_batches} "
                        f"after {timeout_s}s"
                    )
                time.sleep(0.001)

        self.streaming_query_sampler.init_sut()
        self.out_batch_counter = 0
        self.in_batch_counter = 0
        self.current_query_ids = []
        self.current_qsl_ids = []
        self._flush_real_query_counts.clear()
        self._zmq_warmup_mode = False
        logger.info(f"[Loadgen Rank: {self.rank}] ZMQ real warmup complete")

    def warmup_partial_flush_zmq(self, timeout_s: float = 120.0) -> None:
        """
        Warm LoadGen→ZMQ→listener→batching for ROCm padded flush batches.

        Worker-side enqueue_batch warmup does not exercise the listener collate path.
        """
        if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") != "1":
            return
        if os.environ.get("DLRM_ZMQ_FLUSH_WARMUP", "1") != "1":
            return
        real_n = int(
            os.environ.get("DLRM_FLUSH_PAD_WARMUP_REAL_N", str(self.batch_size - 4))
        )
        if real_n <= 0 or real_n >= self.batch_size:
            return
        self._zmq_warmup_mode = True
        start_in = self.in_batch_counter
        base = (888 * self.batch_size) % max(
            1, self.streaming_query_sampler.total_requests - self.batch_size
        )
        query_ids = list(range(base, base + real_n))
        outputs_ts = self.streaming_query_sampler.get_samples_indices(query_ids)
        padded_ids, padded_ts, real_n = self._pad_flush_batch_rocm(
            query_ids, outputs_ts
        )
        logger.info(
            f"[Loadgen Rank: {self.rank}] ZMQ partial-flush warmup: "
            f"{real_n} -> {self.batch_size} queries"
        )
        data_packet_ts_request = MPIDataPacketTSRequest(
            query_ids=padded_ids,
            ts_request_pairs=padded_ts,
            is_warmup=False,
        )
        self.out_batch_counter += 1
        data_packet_ts_request.zmq_seq = self.out_batch_counter
        trace_lg(
            self.rank,
            data_packet_ts_request.zmq_seq,
            "warmup_partial_send",
            **packet_summary(padded_ids, padded_ts, real_n=real_n),
        )
        self.request_sender.send_to_worker(data_packet_ts_request)
        deadline = time.time() + timeout_s
        stall_s = float(os.environ.get("DLRM_ZMQ_TRACE_STALL_S", "5"))
        last_stall = time.time()
        while self.in_batch_counter <= start_in:
            if time.time() > deadline:
                trace_lg(
                    self.rank,
                    data_packet_ts_request.zmq_seq,
                    "warmup_partial_timeout",
                    in_=self.in_batch_counter,
                    out=self.out_batch_counter,
                )
                self._zmq_warmup_mode = False
                raise TimeoutError(
                    f"ZMQ partial-flush warmup timed out after {timeout_s}s "
                    f"(in={self.in_batch_counter} out={self.out_batch_counter})"
                )
            if time.time() - last_stall >= stall_s:
                trace_lg(
                    self.rank,
                    data_packet_ts_request.zmq_seq,
                    "warmup_partial_stall",
                    in_=self.in_batch_counter,
                    out=self.out_batch_counter,
                )
                last_stall = time.time()
            time.sleep(0.001)
        self.streaming_query_sampler.init_sut()
        self.out_batch_counter = 0
        self.in_batch_counter = 0
        self.current_query_ids = []
        self.current_qsl_ids = []
        self._flush_real_query_counts.clear()
        self._zmq_warmup_mode = False
        trace_lg(self.rank, 0, "warmup_partial_done", out=0, in_=0)
        logger.info(f"[Loadgen Rank: {self.rank}] ZMQ partial-flush warmup complete")

    def warmup_production_zmq_batches(
        self, num_batches: int, timeout_per_batch_s: float = 180.0
    ) -> None:
        """
        Warm the full LoadGen→ZMQ→listener→batching path for N stream batches.

        Worker-only batching warmup does not JIT the listener collate path used in
        production; late batches (e.g. 34–36 in a 30 s run) can stall without this.
        """
        if num_batches <= 0:
            return
        if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") != "1":
            return
        if os.environ.get("DLRM_ZMQ_STREAM_WARMUP", "1") != "1":
            return
        self._zmq_warmup_mode = True
        start_in = self.in_batch_counter
        logger.info(
            f"[Loadgen Rank: {self.rank}] ZMQ stream warmup: {num_batches} full batches"
        )
        for i in range(num_batches):
            base = (i * self.batch_size) % max(
                1, self.streaming_query_sampler.total_requests - self.batch_size
            )
            query_ids = list(range(base, base + self.batch_size))
            outputs_ts = self.streaming_query_sampler.get_samples_indices(query_ids)
            data_packet_ts_request = MPIDataPacketTSRequest(
                query_ids=query_ids,
                ts_request_pairs=outputs_ts,
                is_warmup=False,
            )
            self.out_batch_counter += 1
            self.request_sender.send_to_worker(data_packet_ts_request)
            batch_deadline = time.time() + timeout_per_batch_s
            while self.in_batch_counter - start_in <= i:
                if time.time() > batch_deadline:
                    self._zmq_warmup_mode = False
                    raise TimeoutError(
                        f"ZMQ stream warmup timed out on batch {i + 1}/{num_batches} "
                        f"after {timeout_per_batch_s}s"
                    )
                time.sleep(0.001)

        self.streaming_query_sampler.init_sut()
        self.out_batch_counter = 0
        self.in_batch_counter = 0
        self.current_query_ids = []
        self.current_qsl_ids = []
        self._flush_real_query_counts.clear()
        self._zmq_warmup_mode = False
        logger.info(f"[Loadgen Rank: {self.rank}] ZMQ stream warmup complete")

    def benchmark_zmq_latency(self, num_requests: int = 50000):
        """
        Run a latency profiling test (for debugging, not MLPerf benchmarking).

        Sends warmup requests to workers and measures round-trip latency,
        collecting percentile statistics for performance analysis.

        Args:
            num_requests: Number of test requests to send.
        """
        self.test_latency_mode = True
        self._latency_send_times.clear()
        self._latency_results.clear()

        logger.info(f"[Loadgen Rank: {self.rank}] Starting latency test with {num_requests} requests")
        for i in range(num_requests):
            # fake dataset index for benchmark latency
            query_ids = list(range(i * self.batch_size, (i + 1) * self.batch_size))
            outputs_ts = [(j, j) for j in range(self.batch_size)]
            data_packet_ts_request = MPIDataPacketTSRequest(
                query_ids=query_ids,
                ts_request_pairs=outputs_ts,
                is_warmup=True
            )
            # Record send time
            self._latency_send_times[i] = time.perf_counter()
            self.request_sender.send_to_worker(data_packet_ts_request)
            time.sleep(0.0001)

        # Wait for all responses
        timeout = 10  # seconds
        start_wait = time.time()
        while len(self._latency_results) < num_requests:
            if time.time() - start_wait > timeout:
                logger.warning(f"Timeout waiting for responses. Got {len(self._latency_results)}/{num_requests}")
                break
            time.sleep(0.01)

        self._report_latency()

        self.test_latency_mode = False

    def _results_listener(self):
        """
        Background thread that listens for inference results from workers.

        Continuously polls for results from workers and reports them back to
        LoadGen via QuerySamplesComplete. Runs until result_stop_event is set.
        """
        # Ensure garbage collection is disabled for consistent latency
        gc.disable()
        if zmq_trace_enabled():
            log_path = os.environ.get("DLRM_ZMQ_TRACE_LOG", "")
            logger.info(
                f"[Loadgen Rank: {self.rank}] ZMQ trace ON"
                + (f" -> {log_path}" if log_path else " (stderr)")
            )
        logger.info(f"[Loadgen Rank: {self.rank}] Results listener started")

        while not self.result_stop_event.is_set():
            # Check for incoming results from workers
            if self.request_sender.probe_from_worker():
                data_packet = self.request_sender.receive_from_worker()
                if self.test_latency_mode:
                    self._record_test_latency(data_packet)
                    continue
                if self._zmq_warmup_mode:
                    self.in_batch_counter += 1
                    continue

                self.in_batch_counter += 1
                seq = getattr(data_packet, "zmq_seq", -1)
                trace_lg(
                    self.rank,
                    seq if seq >= 0 else self.in_batch_counter,
                    "recv_result",
                    in_=self.in_batch_counter,
                    out=self.out_batch_counter,
                    **packet_summary(
                        data_packet.query_ids, data_packet.ts_request_pairs
                    ),
                )
                if self.in_batch_counter % 500 == 0:
                    logger.info(f"[Loadgen Rank: {self.rank}] Received {self.in_batch_counter} batches from workers. data shape: {data_packet.results[0].shape}")

                real_n = self._flush_real_query_counts.pop(data_packet.query_ids[0], None)
                if real_n is not None and real_n < len(data_packet.query_ids):
                    trace_lg(
                        self.rank,
                        seq if seq >= 0 else self.in_batch_counter,
                        "flush_trim",
                        real_n=real_n,
                        padded=len(data_packet.query_ids),
                    )
                    if _EOS_TRACE:
                        # PIN the ts=9 residual: results were computed for the
                        # PADDED batch (batch_size queries) but query_ids /
                        # ts_request_pairs are about to be trimmed to real_n.
                        # report_loadgen_best_perf then divides the padded
                        # preds width by real_n -> candidate_size is inflated
                        # from the true per-query width and every slice for
                        # these real_n queries is misaligned. Log the exact
                        # req_ids that get corrupted + the arithmetic.
                        _padded = len(data_packet.query_ids)
                        _res0 = data_packet.results[0]
                        _predsz = (
                            _res0.size(1) if hasattr(_res0, "dim") and _res0.dim() > 1 else -1
                        )
                        _start_ts = self.streaming_query_sampler.start_ts
                        _pairs = data_packet.ts_request_pairs or []
                        _kept = [
                            (int(ts - _start_ts), int(req)) for ts, req in _pairs[:real_n]
                        ]
                        logger.info(
                            f"[EOS-TRACE] flush_trim seq={seq} real_n={real_n} "
                            f"padded={_padded} preds_shape={tuple(_res0.shape)} "
                            f"cs_if_padded={_predsz // _padded if _padded else -1} "
                            f"cs_after_trim={_predsz // real_n if real_n else -1} "
                            f"kept_reqids={_kept}"
                        )
                    if _FLUSH_TRIM_RESULTS and data_packet.results is not None:
                        # Drop the padded query blocks from results so the per-
                        # query width divides evenly by real_n. Pad rows are the
                        # trailing (padded - real_n) blocks (last real query
                        # repeated), so keeping the first real_n * cs columns of
                        # each result tensor preserves the real queries in order.
                        _padded_n = len(data_packet.query_ids)
                        _trimmed = []
                        for _t in data_packet.results:
                            if (
                                hasattr(_t, "dim")
                                and _t.dim() == 2
                                and _padded_n > 0
                                and _t.size(1) % _padded_n == 0
                            ):
                                _cs = _t.size(1) // _padded_n
                                _trimmed.append(_t[:, : real_n * _cs].contiguous())
                            else:
                                _trimmed.append(_t)
                        data_packet.results = tuple(_trimmed)
                        if _EOS_TRACE:
                            logger.info(
                                f"[EOS-TRACE] flush_fix seq={seq} trimmed results "
                                f"padded={_padded_n} -> real_n={real_n} "
                                f"new_preds_shape={tuple(data_packet.results[0].shape)}"
                            )
                    data_packet.query_ids = data_packet.query_ids[:real_n]
                    if data_packet.ts_request_pairs:
                        data_packet.ts_request_pairs = data_packet.ts_request_pairs[:real_n]

                # Extract results
                # Plan 21 Phase 21.1 — time the single-threaded synchronous
                # result-processing stage (fp32 convert + buffer build + QSC
                # enqueue). loadgen.result_proc mean × rate ≈ this thread's
                # utilization; if it saturates while workers idle, it's the cap.
                _t_rp = time.perf_counter()
                preds = data_packet.results[0]
                labels = data_packet.results[1]
                weights = data_packet.results[2]
                ts_idx_list, query_idx_list = self._build_query_metadata(data_packet)

                # Normal mode: Report results back to LoadGen
                report_loadgen_best_perf(
                    data_packet,
                    preds,
                    labels,
                    weights,
                    compute_eval=True if self.mode == "accuracy" else False,
                    ts_idx_list=ts_idx_list,
                    query_idx_list=query_idx_list,
                )
                if not getattr(data_packet, "is_warmup", False):
                    record("loadgen.result_proc", time.perf_counter() - _t_rp)
                # Debug mode: Dump tensors to file for accuracy verification
                if self.debug_accuracy:
                    self._record_accuracy(preds, labels, weights)

            else:
                # No results available, sleep briefly to avoid busy-waiting
                time.sleep(0.001)  # 1ms sleep

        logger.info(f"[Loadgen Rank: {self.rank}] Results listener stopped")

        # Verify all batches were received
        assert self.in_batch_counter == self.out_batch_counter, \
            f"Batch count mismatch: received {self.in_batch_counter}, sent {self.out_batch_counter}"

    def _record_test_latency(self, data_packet):
        # Latency profiling mode: Compute round-trip latency
        recv_time = time.perf_counter()
        first_qid = data_packet.query_ids[0]
        request_idx = first_qid // self.batch_size  # Derive request index from query_id pattern
        if request_idx in self._latency_send_times:
            latency_ms = (recv_time - self._latency_send_times[request_idx]) * 1000
            self._latency_results.append((request_idx, latency_ms))

    def _build_query_metadata(
        self, data_packet: MPIDataPacketDSIndex
    ) -> Tuple[List[float], List[float]]:
        num_queries = len(data_packet.query_ids)
        pairs = data_packet.ts_request_pairs
        if not pairs or len(pairs) != num_queries:
            return [-1.0] * num_queries, [-1.0] * num_queries
        start_ts = self.streaming_query_sampler.start_ts
        ts_idx_list = [float(ts - start_ts) for ts, _ in pairs]
        query_idx_list = [float(req_id) for _, req_id in pairs]
        return ts_idx_list, query_idx_list

    def _report_latency(self):
        latencies = [lat for _, lat in self._latency_results]
        logger.info(f"[Latency Test Results] n={len(latencies)}")
        logger.info(f"  Min:    {min(latencies):.3f} ms")
        logger.info(f"  Max:    {max(latencies):.3f} ms")
        logger.info(f"  Avg:    {sum(latencies) / len(latencies):.3f} ms")
        sorted_lat = sorted(latencies)
        p50_idx = int(len(sorted_lat) * 0.5)
        p75_idx = int(len(sorted_lat) * 0.75)
        p95_idx = int(len(sorted_lat) * 0.95)
        p98_idx = int(len(sorted_lat) * 0.98)
        p99_idx = int(len(sorted_lat) * 0.99)
        p999_idx = int(len(sorted_lat) * 0.999)
        logger.info(f"  P50:    {sorted_lat[p50_idx]:.3f} ms")
        logger.info(f"  P75:    {sorted_lat[p75_idx]:.3f} ms")
        logger.info(f"  P95:    {sorted_lat[min(p95_idx, len(sorted_lat) - 1)]:.3f} ms")
        logger.info(f"  P98:    {sorted_lat[min(p98_idx, len(sorted_lat) - 1)]:.3f} ms")
        logger.info(f"  P99:    {sorted_lat[min(p99_idx, len(sorted_lat) - 1)]:.3f} ms")
        logger.info(f"  P99.9:  {sorted_lat[min(p999_idx, len(sorted_lat) - 1)]:.3f} ms")

    def _record_accuracy(self, preds, labels, weights):
        if self._accuracy_dump_file is None:
            self._accuracy_dump_file = open("tensor_dump_mpi_recv.txt", "a")
        f = self._accuracy_dump_file
        f.write(f"iteration: {self.in_batch_counter - 1} ----------\n\n")

        # Output predictions
        f.write(f"iteration: {self.in_batch_counter - 1} mt_target_preds,\n\n")
        f.write(f"{preds.cpu()[0].tolist()[0:128]}\n\n")
        f.write(f"iteration: {self.in_batch_counter - 1} mt_target_preds shape: {preds.cpu()[0].shape}\n\n")

        # Output labels
        f.write(f"iteration: {self.in_batch_counter - 1} mt_target_labels,\n\n")
        f.write(f"{labels.cpu()[0].tolist()[0:128]}\n\n")

        # Output weights
        f.write(f"iteration: {self.in_batch_counter - 1} mt_target_weights,\n\n")
        f.write(f"{weights.cpu()[0].tolist()[0:128]}\n\n\n")

    def shutdown(self):
        """
        Gracefully shutdown the TestRunner and its background threads.

        Stops the results listener thread, waits for it to complete, and
        closes communication channels with workers.
        """
        logger.info(f"[Loadgen Rank: {self.rank}] Shutting down TestRunner...")

        # Signal the results listener thread to stop
        self.result_stop_event.set()

        # Wait for thread to finish (with timeout to avoid hanging)
        if self.results_thread.is_alive():
            logger.info(f"[Loadgen Rank: {self.rank}] Waiting for results listener thread to stop...")
            self.results_thread.join(timeout=2.0)
            if self.results_thread.is_alive():
                logger.warning(f"[Loadgen Rank: {self.rank}] Results listener thread did not stop within timeout")

        # Close communication channels (after thread has stopped)
        self.request_sender.shutdown(num_workers=self._num_workers)

        if self._accuracy_dump_file is not None:
            self._accuracy_dump_file.close()
            self._accuracy_dump_file = None

        logger.info(f"[Loadgen Rank: {self.rank}] TestRunner shutdown complete.")
