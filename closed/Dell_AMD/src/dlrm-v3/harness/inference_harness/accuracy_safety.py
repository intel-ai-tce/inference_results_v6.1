"""Dependency-free guards for accuracy-safe optimized inference shortcuts."""

from __future__ import annotations


def should_run_optimized_compare(
    optimized_compare: bool,
    optimized_compares: int,
    optimized_compare_limit: int,
    is_inference: bool,
    server_candidate_shape: bool,
) -> bool:
    """Return whether to run the one-time optimized-vs-baseline compare."""
    return (
        optimized_compare
        and optimized_compares < optimized_compare_limit
        and is_inference
        and server_candidate_shape
    )


def should_use_optimized_embed(
    optimized_lookup: bool,
    optimized_disabled: bool,
    is_inference: bool,
    server_candidate_shape: bool,
) -> bool:
    """Optimized embedding is inference/performance-only and 2048-shape-only."""
    return optimized_lookup and not optimized_disabled and is_inference and server_candidate_shape


def should_use_uniform_targets_metadata(
    is_inference: bool,
    uniform_targets_metadata: bool,
    uniform_targets_metadata_disabled: bool,
) -> bool:
    """Uniform target metadata is an inference-only shortcut."""
    return is_inference and uniform_targets_metadata and not uniform_targets_metadata_disabled


def server_accuracy_duration_batches(mode: str, duration_batches: int) -> int:
    """Accuracy mode should not inherit Server min-duration warmup sizing."""
    return 0 if mode == "accuracy" else duration_batches


def worker_warmup_steps_for_mode(mode: str, warmup_steps: int) -> int:
    """Accuracy mode should not run pre-LoadGen worker predict warmup."""
    return 0 if mode == "accuracy" else warmup_steps


def should_run_zmq_warmups(rocm_backend: bool, mode: str) -> bool:
    """ZMQ warmups are performance-only for the ROCm path."""
    return rocm_backend and mode != "accuracy"


def accuracy_response_candidate_size(candidate_size: int, requested_size: int) -> int:
    """Return the AccuracyOnly response width requested by TEST08, if any."""
    return requested_size if requested_size > 0 else candidate_size


def accuracy_response_copy_size(candidate_size: int, emit_candidate_size: int) -> int:
    """Only real candidates should be copied; padded candidates stay zeroed."""
    return min(candidate_size, emit_candidate_size)
