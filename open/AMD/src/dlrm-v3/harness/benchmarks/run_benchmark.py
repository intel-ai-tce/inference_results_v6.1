#!/usr/bin/env python3
"""
Multi-GPU benchmark script for DLRMv3 with MLPerf LoadGen.
Uses a "small world" setup where loadgen runs as rank 8 in the same MPI world.
Automatically adds project root to Python path for module imports.
"""

import sys
import os
import warnings
from pathlib import Path

# ROCm bootstrap before torch / backend imports on each rank.
_script_dir_boot = Path(__file__).resolve().parent
_project_root_boot = _script_dir_boot.parent
if str(_project_root_boot) not in sys.path:
    sys.path.insert(0, str(_project_root_boot))

if os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1":
    # Resolve local rank from any of OpenMPI / generic PMI / MVAPICH envs.
    _mpi_local_rank_pin = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("PMI_RANK")
        or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK")
        or 0
    )

    # GPU visibility strategy (two modes):
    #
    #   * DEFAULT (Phase 2b Step 3a behaviour) — narrow HIP_VISIBLE_DEVICES
    #     to a single physical GPU per rank BEFORE `import torch`. Required
    #     because fbgemm / torchrec / triton touch cuda:0 from every rank's
    #     process at import time and the cross-rank cuda:0 collision
    #     produces an HSA `Memory access fault by GPU node-N` on rank N's
    #     first real kernel launch after warmup (Phase 2b multi-worker
    #     repro). Pinning each rank to exactly one device makes the
    #     cross-rank collision structurally impossible.
    #
    #   * DLRM_HIP_FULL_VISIBILITY=1 (Plan 01 / RCCL path) — keep the full
    #     HIP_VISIBLE_DEVICES list, but call `torch.cuda.set_device(local_rank)`
    #     IMMEDIATELY after `import torch` (before fbgemm imports), so every
    #     rank pins its compute to its local rank's device while still
    #     letting torch.distributed (RCCL backend) see every peer's GPU for
    #     the on-device `route_lookup` all_to_all_single primitive. Required
    #     because per-rank narrowing leaves peer GPUs invisible to RCCL,
    #     which falls back to gloo and forces the CPU↔GPU bounce that
    #     dominates the Mode B comms cost
    #     (documents/11_Phase3_ModeB_Comms_Cost.md). The early `set_device`
    #     call below preempts the fbgemm/triton cuda:0 touch.
    _full_visibility = os.environ.get("DLRM_HIP_FULL_VISIBILITY", "0") == "1"

    _pin_visible = os.environ.get("HIP_VISIBLE_DEVICES", "")
    _pin_devs = [d.strip() for d in _pin_visible.split(",") if d.strip()]

    if _full_visibility:
        # Leave HIP_VISIBLE_DEVICES at the full pinned set so torch and
        # RCCL both see every peer GPU. Call `torch.cuda.set_device` on
        # this rank's index IMMEDIATELY after `import torch` (before
        # fbgemm/triton imports) so per-rank compute still pins
        # correctly to the right physical device. ROCR_VISIBLE_DEVICES
        # is intentionally left untouched here; if it was unset by the
        # launcher (Phase 2b Step 3a fix), RCCL's HSA-level topology
        # discovery sees every GPU on the host, which is harmless as
        # long as RCCL is initialized via torch.distributed with
        # explicit device IDs (we call torch.cuda.set_device on every
        # rank before `init_process_group(backend='nccl', …)`). The
        # alternative — setting ROCR == HIP pre-mpirun — reproducibly
        # fails inside the first `set_device` with "No HIP GPUs are
        # available" on this image (verified Plan 01 Phase A4).
        if len(_pin_devs) > 1:
            import torch as _t_boot  # noqa: WPS433
            _t_boot.cuda.set_device(_mpi_local_rank_pin % len(_pin_devs))
            del _t_boot
    elif len(_pin_devs) > 1:
        # Narrow HIP_VISIBLE_DEVICES per rank (legacy stable path).
        os.environ["HIP_VISIBLE_DEVICES"] = _pin_devs[_mpi_local_rank_pin % len(_pin_devs)]
        # Do NOT also narrow ROCR_VISIBLE_DEVICES — if both HIP_VISIBLE
        # and ROCR_VISIBLE are pre-set to a multi-device list and we
        # then narrow them via Python before `import torch`, the HSA
        # runtime fails on the first set_device with "No HIP GPUs are
        # available" (verified reproducibly under `mpirun ... bash -lc
        # 'exec python3 ...'`). Narrowing only HIP_VISIBLE works fine
        # — torch sees 1 device, set_device(0) succeeds, and the
        # actual physical pin still takes effect.
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)

    from inference_harness.rocm_bootstrap import apply_rocm_env
    apply_rocm_env()
else:
    from inference_harness.rocm_bootstrap import apply_rocm_env  # noqa: F401

# Suppress TorchScript dtype annotation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit.annotations")

import torch
from typing import Dict
import socket
# Per-rank Triton JIT cache. Originally keyed only on SLURM_PROCID, which is
# unset under mpirun -> every rank shared /tmp/triton_cache_rank_0 and JIT
# writes collided, producing garbage .hsaco -> HSA `Memory access fault by
# GPU node-N` on the first kernel launch after warmup (Phase 2b Step 3a
# multi-worker repro). Fall back to OpenMPI / generic PMI rank envs so each
# MPI rank gets its own cache directory.
#
# Plan 64: allow a versioned cache epoch so stale per-rank autotune products
# cannot silently survive into a new GOLD stack. Default preserves the legacy
# path exactly; set DLRM_TRITON_CACHE_EPOCH=<tag> to use a fresh per-rank tree.
rank = int(
    os.environ.get("SLURM_PROCID")
    or os.environ.get("OMPI_COMM_WORLD_RANK")
    or os.environ.get("PMI_RANK")
    or os.environ.get("MV2_COMM_WORLD_RANK")
    or 0
)
_cache_epoch = os.environ.get("DLRM_TRITON_CACHE_EPOCH", "").strip()
if _cache_epoch:
    _safe_epoch = "".join(c if c.isalnum() or c in "._-" else "_" for c in _cache_epoch)
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{_safe_epoch}_rank_{rank}"
else:
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_rank_{rank}"

# Add project root to Python path to allow imports from inference_harness
# This handles the case where the script is run directly without PYTHONPATH set
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mpi4py import MPI
from inference_harness.inference_server import DLRMInferenceServer
from inference_harness.tools.model_configs import get_hstu_configs, get_embedding_table_config, get_backend_config, get_dataset_latest, get_communicator_config
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from inference_harness.mpi_utils import ZMQRequestSenderShardedConfig
from inference_harness.test_runner import TestRunner, clear_accuracy_buffers
from inference_harness.test_runner import parse_user_conf
from inference_harness.accuracy_safety import (
    server_accuracy_duration_batches,
    should_run_zmq_warmups,
    worker_warmup_steps_for_mode,
)


import logging
import argparse

import mlperf_loadgen as lg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    """
    Parse command line arguments for the multi-GPU benchmark script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing dataset paths,
            model configuration, LoadGen settings, and performance tuning options.
    """
    parser = argparse.ArgumentParser(
        description="MLPerf LoadGen integration for DLRMv3 with DLRMv3MLPerfDataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the preprocessed dataset directory"
    )
    parser.add_argument(
        "--dataset-percentage",
        type=float,
        default=1,
        help="Percentage of dataset to load (0.0-1.0)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the checkpoint file"
    )
    # LoadGen configuration args
    parser.add_argument(
        "--scenario",
        type=str,
        default="Server",
        choices=["Server", "Offline"],
        help="MLPerf scenario to run"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="performance",
        choices=["performance", "accuracy"],
        help="MLPerf test mode"
    )
    parser.add_argument(
        "--user-conf",
        type=str,
        default="",
        help="Path to user.conf file (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for results"
    )
    # NVE configuration args
    parser.add_argument(
        "--use-mpi-lookup",
        action="store_true",
        help="Enable MPI-based embedding lookup with NVE"
    )
    # Performance tuning args
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (Server mode only)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup inference steps before benchmarking"
    )

    # Communicator configuration arguments
    parser.add_argument(
        "--communicator-type",
        type=str,
        choices=["mpi", "zmq"],
        default="zmq",
        help="Communicator type: 'mpi' (MPI) or 'zmq' (ZMQ), MPI is not supported yet"
    )

    # ZMQ-specific configuration (not applicable when using MPI communicator)
    parser.add_argument(
        "--loadgen-hostname",
        type=str,
        default=None,
        help="Hostname of the LoadGen rank (for ZMQ communicator)"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Number of ZMQ socket shards (typically equals num_nodes). Reduces communication fan-in from N:1 to N/shards:1"
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
        help="Number of GPUs per node (used for shard calculation)"
    )

    # Dataset GPU batching arguments
    parser.add_argument(
        "--batching-on-gpu",
        action="store_true",
        help="Enable GPU batching for KJT collation in dataset (faster but uses more GPU memory; Server mode only)"
    )
    parser.add_argument(
        "--use-cuda-streams",
        action="store_true",
        help="Enable CUDA streams for overlapped batching and inference (Server mode only)"
    )
    return parser.parse_args()


def _estimate_server_batches(settings, batch_size: int, scenario: str, worker_world_size: int = 1) -> int:
    """Estimate full batches per WORKER for duration-driven Server tests (ROCm warmup sizing).

    Phase 3 multi-worker: divide by worker_world_size since ZMQ PUSH/PULL
    fair-queues batches across workers — each worker only handles 1/W of
    the aggregate query budget at run time, so warming up each one for
    the full aggregate count just wastes ~ (W-1)/W of the warmup time.
    Floor at args.warmup_steps via the outer max(...) keeps cold JIT
    paths covered.
    """
    if scenario.lower() != "server":
        return 0
    min_duration_ms = int(getattr(settings, "min_duration_ms", 0) or 0)
    if min_duration_ms <= 1000:
        return 0
    qps = float(
        getattr(settings, "server_target_qps", 0)
        or getattr(settings, "target_qps", 0)
        or 12
    )
    est_queries = int(qps * min_duration_ms / 1000) + batch_size
    per_worker = max(1, worker_world_size)
    return max(1, (est_queries + batch_size - 1) // batch_size // per_worker)


def initialize_inference_server(args,
                                local_rank,
                                local_device_id,
                                loadgen_rank,
                                hstu_config: DlrmHSTUConfig,
                                embedding_table_config: Dict[str, EmbeddingConfig],
                                backend_config,
                                communicator_config: ZMQRequestSenderShardedConfig,
                                settings,
                                worker_comm=None):
    """
    Initialize and configure the DLRMInferenceServer for worker processes.

    Args:
        args: Command line arguments containing dataset and model configuration.
        local_rank: MPI rank of the current worker process.
        local_device_id: CUDA device ID for the current worker.
        loadgen_rank: MPI rank of the LoadGen process.
        hstu_config: Configuration for the HSTU model architecture.
        embedding_table_config: Configuration for embedding tables.
        backend_config: Configuration for the hybrid GR backend.
        communicator_config: Configuration for inter-process communication.
        settings: LoadGen test settings parsed from user.conf.

    Returns:
        DLRMInferenceServer: Initialized inference server ready to process queries.
    """
    current_device = torch.device(f"cuda:{local_device_id}")
    # hack, in dataset instantiation, we need to know
    # how many sample user is trying to run, in order to form this dataset
    # hence, we get offline target qps and min duration from loadgen user.conf, which is used to calculate
    # total number of queries loadgen will issue.

    # Plan 10 §10.1 — env-var override for gpu-batching, parallel to
    # ``DLRM_USE_CUDA_STREAMS`` so the modeB launcher can drive both from
    # the same ``export`` block.  When either the CLI flag
    # (``--batching-on-gpu``) or ``DLRM_BATCHING_ON_GPU=1`` is set,
    # enable the pre-allocated KJTBatchBufferPool path in the worker's
    # streaming dataset.  LoadGen rank below stays at False (LG never
    # collates, only emits ts_request_pairs).
    _env_batching_on_gpu = os.environ.get("DLRM_BATCHING_ON_GPU", "0") == "1"
    _effective_batching_on_gpu = bool(args.batching_on_gpu) or _env_batching_on_gpu

    streaming_query_sampler = get_dataset_latest(
        hstu_config=hstu_config,
        dataset_path=args.dataset_path,
        mode=args.mode,
        total_queries=settings.min_query_count,
        dataset_percentage=args.dataset_percentage,
        device=current_device,
        scenario_name=args.scenario,
        offline_target_qps=settings.offline_expected_qps,
        target_duration=settings.min_duration_ms,
        compute_eval=True if args.mode == "accuracy" else False,
        batching_on_gpu=_effective_batching_on_gpu,
        max_buffer_indices=args.batch_size * 48000,  # 48000 is per sample's rough index vector's size
        max_buffer_lengths=args.batch_size * 8,  # 8 is per sample's length vector's size
    )
    streaming_query_sampler.load_query_samples_preprocessed(args.dataset_path)

    # Calculate shard_id based on node assignment
    # Workers on the same node share a shard to reduce communication fan-in
    shard_id = local_rank // args.gpus_per_node

    # Plan 06 Phase 6.3 — env-var override for CUDA streams plumbing. The
    # CLI flag (--use-cuda-streams) lives in the launcher script, but env
    # is easier to wire from ``docker exec -e`` / the Mode B launcher's
    # default block in run_perf_server_scaleout_modeB.sh. When either the
    # flag or DLRM_USE_CUDA_STREAMS=1 is set, enable the stream-overlapped
    # path in both the default and lockstep listeners.
    _env_use_cuda_streams = os.environ.get("DLRM_USE_CUDA_STREAMS", "0") == "1"
    _effective_use_cuda_streams = bool(args.use_cuda_streams) or _env_use_cuda_streams

    inf_server = DLRMInferenceServer(
        query_streaming_sampler=streaming_query_sampler,
        device=current_device,
        local_rank=local_rank,
        batch_size=args.batch_size,
        verbose=0,
        loadgen_rank=loadgen_rank,
        mode=args.mode,
        use_cuda_streams=_effective_use_cuda_streams,
        # Phase 2b Step 3d: pass the worker-only MPI subcomm so the
        # lockstep listener can MPI-broadcast each ZMQ batch when
        # sharded sparse routing is the live path.
        worker_comm=worker_comm,
    )
    inf_server.init_communicator(
        shard_id=shard_id,
        communicator_config=communicator_config
    )
    inf_server.init_backend(hstu_config=hstu_config, embedding_table_config=embedding_table_config, backend_config=backend_config, checkpoint_path=args.checkpoint_path)

    # Model must be loaded on every worker rank for distributed inference
    logger.info(f"[Worker Comm: {local_rank}] Inference server initialized successfully.")
    return inf_server


def initialize_loadgen_runner(args, local_rank, worker_world_size, hstu_config: DlrmHSTUConfig, communicator_config: ZMQRequestSenderShardedConfig, settings):
    """
    Initialize the MLPerf LoadGen test runner.

    Args:
        args: Command line arguments containing dataset and LoadGen configuration.
        local_rank: MPI rank of the LoadGen process.
        worker_world_size: Total number of worker processes (excluding LoadGen).
        hstu_config: Configuration for the HSTU model architecture.
        communicator_config: Configuration for inter-process communication.
        settings: LoadGen test settings parsed from user.conf.

    Returns:
        TestRunner: Initialized test runner ready to execute MLPerf LoadGen tests.
    """
    streaming_query_sampler = get_dataset_latest(
        hstu_config=hstu_config,
        dataset_path=args.dataset_path,
        mode=args.mode,
        total_queries=settings.min_query_count,
        dataset_percentage=args.dataset_percentage,
        device=None,
        scenario_name=args.scenario,
        offline_target_qps=settings.offline_expected_qps,
        target_duration=settings.min_duration_ms,
        compute_eval=True if args.mode == "accuracy" else False,
        batching_on_gpu=False,  # Loadgen doesn't use GPU batching
        max_buffer_indices=args.batch_size * 48000,  # 48000 is per sample's rough index vector's size
        max_buffer_lengths=args.batch_size * 8,  # 8 is per sample's length vector's size
    )
    streaming_query_sampler.sync_preprocessed_metadata(args.dataset_path)
    runner = TestRunner(
        streaming_query_sampler=streaming_query_sampler,
        batch_size=args.batch_size,
        rank=local_rank,
        worker_world_size=worker_world_size,
        verbose=0,
        mode=args.mode,
        scenario=args.scenario,
        communicator_config=communicator_config
    )
    return runner


def run_loadgen(runner: TestRunner, args, settings, log_settings, local_rank):
    """
    Execute the MLPerf LoadGen benchmark test.

    Sets up and runs the LoadGen test with the configured scenario and mode,
    then cleans up resources after completion.

    Args:
        runner: Configured TestRunner instance ready to execute the test.
        args: Command line arguments.
        settings: LoadGen test settings.
        log_settings: LoadGen log settings.
        local_rank: MPI rank of the LoadGen process.
    """
    sut, qsl = runner.setup_loadgen(args)

    logger.info(f"[Loadgen Child Comm: {local_rank}] Starting LoadGen test: {args.scenario} / {args.mode}")
    logger.info(f"[Loadgen Child Comm: {local_rank}] {'=' * 80}")

    from inference_harness.zmq_trace import trace_lg, zmq_trace_enabled

    if zmq_trace_enabled():
        trace_lg(
            local_rank,
            0,
            "start_test_enter",
            scenario=args.scenario,
            mode=args.mode,
            out=runner.out_batch_counter,
            in_=runner.in_batch_counter,
        )
    # Execute the MLPerf LoadGen benchmark
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    if zmq_trace_enabled():
        trace_lg(
            local_rank,
            0,
            "start_test_return",
            out=runner.out_batch_counter,
            in_=runner.in_batch_counter,
        )

    # Clear accuracy buffers to free memory held by completion thread pool
    clear_accuracy_buffers()

    lg.DestroySUT(sut)
    lg.DestroyQSL(qsl)

    # Stop the results listener thread
    runner.result_stop_event.set()
    runner.results_thread.join(timeout=5.0)


if __name__ == '__main__':
    # ========== MPI Initialization ==========
    comm = MPI.COMM_WORLD
    local_rank = comm.Get_rank()
    local_device_id = local_rank % torch.cuda.device_count()
    world_size = comm.Get_size()

    # LoadGen always runs on the last rank; all other ranks are workers
    loadgen_rank = world_size - 1
    worker_world_size = world_size - 1

    # Phase 2b Step 3d: build a worker-only MPI subcomm (LoadGen rank
    # excluded via MPI.UNDEFINED) so the inference-server listener can do
    # cross-worker broadcasts when sharded sparse routing is active. We
    # always split — the comm is None on LoadGen and unused on workers
    # unless DLRM_SPARSE_LOCKSTEP_DISPATCH=1 (auto-on under REPLICATE=0
    # + WORLD>1).
    if local_rank != loadgen_rank:
        _worker_color = 0
        _worker_key = local_rank
    else:
        _worker_color = MPI.UNDEFINED
        _worker_key = 0
    worker_comm = comm.Split(_worker_color, _worker_key)

    # ========== Configuration Initialization ==========
    # Parse arguments and load model/dataset configurations
    args = get_args()

    # Plan 12 Phase 12.3 — on the ROCm path, the CLI ``--use-mpi-lookup`` flag
    # has no NVE memblock to wire up (NVE is CUDA-kernel-only). Instead the
    # ROCm port (Plan 12) exports ``DLRM_USE_MPI_LOOKUP=1`` so
    # ``sparse_routing._use_mpi_lookup_env()`` flips ``route_lookup`` to its
    # MPI-substitute transport at the three ``all_to_all_single`` sites. The
    # registered route_lookup_comm is set up by the server's
    # ``_register_route_lookup_comm()`` (Phase 12.3 server-side). Env wins if
    # already set externally — keeps the launcher-script override behavior.
    if args.use_mpi_lookup and os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1":
        if os.environ.get("DLRM_USE_MPI_LOOKUP", "") == "":
            os.environ["DLRM_USE_MPI_LOOKUP"] = "1"
            if local_rank == 0:
                logger.info(
                    "[plan12] --use-mpi-lookup on ROCm path → exporting "
                    "DLRM_USE_MPI_LOOKUP=1 (route_lookup will use MPI "
                    "substitute transport once worker_comm is registered)"
                )

    model_config = "production"
    hstu_config = get_hstu_configs(model_config)
    embedding_table_config = get_embedding_table_config(model_config)
    # Plan 14.5b: ROCm single-GPU NVE smoke clamps table sizes to fit one GPU.
    from inference_harness.tools.model_configs import _maybe_cap_embedding_tables, _maybe_bf16_embedding_tables
    embedding_table_config = _maybe_cap_embedding_tables(embedding_table_config)
    # bf16-gather: optionally store NVE tables in bf16 to drop the fp16->bf16 cast.
    embedding_table_config = _maybe_bf16_embedding_tables(embedding_table_config)
    communicator_config = get_communicator_config(args)

    # Parse user.conf once and reuse settings throughout
    settings, log_settings = parse_user_conf(args)

    # Backend config must be created on every rank, even those not storing embedding tables
    # This is required for proper MPI memory block allocation
    backend_config = get_backend_config(args=args, embedding_table_config=embedding_table_config, world_size=world_size)

    # ========== Worker and LoadGen Initialization ==========
    inf_server = None
    runner = None

    if local_rank != loadgen_rank:
        os.environ["DLRM_MPI_LOCAL_RANK"] = str(local_rank)
        os.environ["DLRM_MPI_WORKER_WORLD"] = str(worker_world_size)
        os.environ["DLRM_OUTPUT_DIR"] = os.path.abspath(args.output_dir)
        inf_server = initialize_inference_server(args,
                                                 local_rank,
                                                 local_device_id,
                                                 loadgen_rank,
                                                 hstu_config,
                                                 embedding_table_config,
                                                 backend_config,
                                                 communicator_config,
                                                 settings,
                                                 worker_comm=worker_comm,
                                                 )
        comm.Barrier()
        query_batches = max(
            1, (settings.min_query_count + args.batch_size - 1) // args.batch_size
        )
        duration_batches = server_accuracy_duration_batches(
            args.mode,
            _estimate_server_batches(
                settings, args.batch_size, args.scenario, worker_world_size
            ),
        )
        worker_warmup_steps = worker_warmup_steps_for_mode(
            args.mode,
            args.warmup_steps,
        )
        if args.mode == "accuracy":
            # AccuracyOnly is scored, not latency-sensitive, and the Server path can
            # expose 32-candidate evaluation batches. Avoid pre-LoadGen worker
            # warmup here; LoadGen will drive the scored queries directly.
            batching_warmup_steps = 0
        else:
            batching_warmup_steps = max(
                args.warmup_steps, query_batches, duration_batches
            )
        logger.info(
            f"[Worker: {local_rank}] Batching warmup steps={batching_warmup_steps} "
            f"(min_query_batches={query_batches}, duration_batches={duration_batches})"
        )
        inf_server.warmup(
            warmup_steps=worker_warmup_steps,
            batching_warmup_steps=batching_warmup_steps,
        )
        comm.Barrier()
        try:
            from inference_harness.rocm_timing import reset as timing_reset

            timing_reset()
            logger.info(
                f"[Worker: {local_rank}] Timing stats reset after {args.warmup_steps} warmup steps"
            )
        except Exception as exc:
            logger.warning(f"[Worker: {local_rank}] timing reset skipped: {exc!r}")
        comm.Barrier()
    else:
        # LoadGen rank: Initialize test runner
        runner = initialize_loadgen_runner(args,
                                           local_rank,
                                           worker_world_size,
                                           hstu_config,
                                           communicator_config,
                                           settings
                                           )

        # Wait for all workers to complete set up including loading weights
        comm.Barrier()
        # wait for all workers to complete warmup
        comm.Barrier()
        rocm_backend = os.environ.get("DLRM_ROCM_GR_BACKEND", "0") == "1"
        if should_run_zmq_warmups(rocm_backend, args.mode):
            if os.environ.get("DLRM_ZMQ_FLUSH_WARMUP", "1") == "1":
                runner.warmup_partial_flush_zmq(
                    timeout_s=float(os.environ.get("DLRM_ZMQ_FLUSH_WARMUP_TIMEOUT_S", "120"))
                )
            zmq_real = int(os.environ.get("DLRM_ZMQ_REAL_WARMUP_BATCHES", "0"))
            if zmq_real > 0:
                per_batch_timeout = float(
                    os.environ.get("DLRM_ZMQ_REAL_WARMUP_TIMEOUT_S", "120")
                )
                runner.zmq_real_warmup(
                    num_batches=zmq_real, timeout_s=per_batch_timeout
                )
        elif rocm_backend:
            logger.info("[Loadgen Rank: %s] Skipping ZMQ warmups in AccuracyOnly mode", local_rank)
        else:
            zmq_bench = int(os.environ.get("DLRM_ZMQ_LATENCY_REQUESTS", "50000"))
            if zmq_bench > 0:
                runner.benchmark_zmq_latency(num_requests=zmq_bench)
        comm.Barrier()

    # ========== Run MLPerf LoadGen Benchmark ==========
    # LoadGen starts measuring performance; workers listen for queries
    if local_rank == loadgen_rank:
        run_loadgen(runner, args, settings, log_settings, local_rank)
    comm.Barrier()

    # ========== Cleanup and Shutdown ==========
    if local_rank != loadgen_rank:
        inf_server._stop_request_listener(timeout=10.0)
        inf_server.stop_batching(timeout=10.0)
        inf_server.dump_latency(output_dir=args.output_dir)
        inf_server.dump_stage_latency(output_dir=args.output_dir)
        try:
            from inference_harness.rocm_timing import format_summary, maybe_report, summarize

            maybe_report(force=True)
            summary = summarize()
            if summary:
                logger.info(format_summary(summary))
        except Exception as exc:
            logger.warning(f"[Worker: {local_rank}] timing summary failed: {exc!r}")
        logger.info(f"[Worker: {local_rank}] Inference server stopped.")
    else:
        runner.shutdown()
        try:
            from inference_harness.rocm_timing import format_summary, maybe_report, summarize

            maybe_report(force=True)
            summary = summarize()
            if summary:
                logger.info(f"[Loadgen: {local_rank}] {format_summary(summary)}")
        except Exception as exc:
            logger.warning(f"[Loadgen: {local_rank}] timing summary failed: {exc!r}")
        logger.info(f"[Loadgen: {local_rank}] Loadgen runner stopped.")

    # Ensure all ranks complete before exiting
    comm.Barrier()
