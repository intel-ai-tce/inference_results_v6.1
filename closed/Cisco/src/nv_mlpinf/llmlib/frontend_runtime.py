# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Runtime supervision for the Dynamo disaggregated frontend process."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from io import TextIOBase
from pathlib import Path
import subprocess
import time


DISCOVERY_STREAM_PANIC = "Unfold must not be polled after it returned"
DISCOVERY_RECOVERY_EXHAUSTED = "Dynamo frontend discovery recovery exhausted"
FRONTEND_RUNTIME_FAILURE = "Dynamo frontend runtime failure"
FRONTEND_SUPERVISOR_ACTIVE = "frontend_supervisor.active"
FRONTEND_LOG_POLL_SECONDS = 0.25
FRONTEND_TERMINATE_TIMEOUT_SECONDS = 5.0


def _read_discovery_stream_panic(
    log_stream: TextIOBase,
    previous_tail: str,
) -> tuple[bool, str]:
    """Read newly appended log data and retain enough text for split markers."""
    new_text = log_stream.read()
    if not new_text:
        return False, previous_tail
    combined = previous_tail + new_text
    tail_length = max(len(DISCOVERY_STREAM_PANIC) - 1, 0)
    return DISCOVERY_STREAM_PANIC in combined, combined[-tail_length:]


def _terminate_process(
    process: subprocess.Popen,
    timeout_seconds: float,
) -> int:
    returncode = process.poll()
    if returncode is not None:
        return returncode

    try:
        process.terminate()
    except ProcessLookupError:
        pass

    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=timeout_seconds)


def _wait_for_exit_or_discovery_stream_panic(
    process: subprocess.Popen,
    log_path: Path,
    *,
    sleep: Callable[[float], None],
    poll_interval_seconds: float,
    terminate_timeout_seconds: float,
) -> tuple[int, bool]:
    """Return on process exit or stop a live process after the known stream panic."""
    previous_tail = ""
    with log_path.open("r", errors="replace") as log_stream:
        while True:
            returncode = process.poll()
            panicked, previous_tail = _read_discovery_stream_panic(
                log_stream, previous_tail
            )
            if panicked:
                if returncode is None:
                    returncode = _terminate_process(process, terminate_timeout_seconds)
                return returncode, True

            if returncode is not None:
                # A final read after poll() observes all output flushed at process exit.
                panicked, previous_tail = _read_discovery_stream_panic(
                    log_stream, previous_tail
                )
                return returncode, panicked

            sleep(poll_interval_seconds)


def run_frontend_with_discovery_recovery(
    command: Sequence[str],
    environment: dict[str, str],
    log_path: Path,
    *,
    max_restarts: int = 1,
    restart_delay_seconds: float = 1.0,
    log_poll_seconds: float = FRONTEND_LOG_POLL_SECONDS,
    terminate_timeout_seconds: float = FRONTEND_TERMINATE_TIMEOUT_SECONDS,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    on_start: Callable[[subprocess.Popen, int], None] | None = None,
    on_panic_restart: Callable[[int], None] | None = None,
) -> int:
    """Run a frontend, retrying only the known Dynamo Unfold stream panic.

    The surrounding NATS and etcd processes intentionally remain alive across the
    retry so the replacement frontend can rediscover workers. Callers that keep
    etcd state across restarts should pass ``on_panic_restart`` to reset stale
    discovery registrations before the next frontend attempt.
    """
    if max_restarts < 0:
        raise ValueError("max_restarts must be non-negative")

    log_path = Path(log_path)
    failure_path = log_path.with_name("frontend_runtime_failure.log")
    active_path = log_path.with_name(FRONTEND_SUPERVISOR_ACTIVE)
    try:
        failure_path.unlink()
    except FileNotFoundError:
        pass

    active_path.write_text("Dynamo frontend child is supervised\n")
    try:
        for attempt in range(max_restarts + 1):
            with log_path.open("w") as log_stream:
                process = popen(
                    list(command),
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                )
                if on_start is not None:
                    on_start(process, attempt)
                try:
                    returncode, panicked = _wait_for_exit_or_discovery_stream_panic(
                        process,
                        log_path,
                        sleep=sleep,
                        poll_interval_seconds=log_poll_seconds,
                        terminate_timeout_seconds=terminate_timeout_seconds,
                    )
                except KeyboardInterrupt:
                    _terminate_process(process, terminate_timeout_seconds)
                    raise

            if not panicked:
                failure_path.write_text(
                    f"{FRONTEND_RUNTIME_FAILURE}: frontend exited with code {returncode} "
                    "without the recoverable discovery-stream panic\n"
                )
                return returncode

            if attempt == max_restarts:
                failure_path.write_text(
                    f"{FRONTEND_RUNTIME_FAILURE}: {DISCOVERY_RECOVERY_EXHAUSTED} "
                    f"after {attempt} restart(s); "
                    f"last frontend exit code: {returncode}\n"
                )
                return returncode

            archived_log = log_path.with_name(f"disagg_frontend.attempt_{attempt}.log")
            log_path.replace(archived_log)
            if on_panic_restart is not None:
                on_panic_restart(attempt + 1)
            sleep(restart_delay_seconds)

        raise AssertionError("unreachable")
    finally:
        try:
            active_path.unlink()
        except FileNotFoundError:
            pass
