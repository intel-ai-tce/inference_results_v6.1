"""Translating backend output into LoadGen responses.

LoadGen's Python API takes a list of ``QuerySampleResponse(query_id, ptr,
size)`` tuples where ``ptr`` is a raw memory address. The buffer the pointer
references **must outlive the call to ``QuerySamplesComplete``** because
LoadGen copies it asynchronously on its own threads.

This module encapsulates two responsibilities:

  1. Building the ``array.array("B", ...)`` buffer that owns the bytes.
  2. Keeping a reference to that buffer alive until the SUT explicitly
     declares the batch is drained (typically inside ``flush_queries``).

A ``ResponseWriter`` interface is provided so tests can substitute a
recording implementation that does not need ``mlperf_loadgen`` installed.
"""

from __future__ import annotations

import array
import logging
from dataclasses import dataclass, field
from typing import Protocol

_log = logging.getLogger(__name__)

__all__ = [
    "ResponseWriter",
    "LoadgenResponseWriter",
    "RecordingResponseWriter",
    "build_response_writer",
]


class ResponseWriter(Protocol):
    """Sends one finished sample's bytes back to whatever consumes them."""

    def complete(self, query_id: int, payload: bytes) -> None:
        ...

    def release(self) -> None:
        """Free any buffers held to keep ``ptr`` valid for LoadGen."""


@dataclass
class LoadgenResponseWriter:
    """Real ResponseWriter that calls into ``mlperf_loadgen``.

    Construction imports ``mlperf_loadgen`` lazily so unit tests on machines
    without loadgen can still ``import wan_harness.response``.
    """

    _retained: list[array.array] = field(default_factory=list)

    def __post_init__(self) -> None:
        import mlperf_loadgen as lg  # type: ignore[import-not-found]

        self._lg = lg

    def complete(self, query_id: int, payload: bytes) -> None:
        buf = array.array("B", payload)
        self._retained.append(buf)
        ptr, size = buf.buffer_info()
        self._lg.QuerySamplesComplete(
            [self._lg.QuerySampleResponse(query_id, ptr, size)]
        )

    def release(self) -> None:
        # Once LoadGen has logged or copied the payload we can drop our
        # references. The SUT is expected to call this exactly once per
        # `flush_queries` cycle.
        n = len(self._retained)
        self._retained.clear()
        if n:
            _log.debug("LoadgenResponseWriter released %d retained buffers", n)


@dataclass
class RecordingResponseWriter:
    """Test double – records ``(query_id, payload)`` pairs in order."""

    responses: list[tuple[int, bytes]] = field(default_factory=list)

    def complete(self, query_id: int, payload: bytes) -> None:
        self.responses.append((int(query_id), bytes(payload)))

    def release(self) -> None:
        # Nothing to release; the bytes are owned by ``self.responses``.
        return


def build_response_writer(*, backend_name: str | None = None) -> ResponseWriter:
    """Pick the right writer for the runtime.

    The default is :class:`LoadgenResponseWriter` because the harness expects
    LoadGen to be installed in production runs. ``backend_name`` is accepted
    for symmetry but is currently unused; we keep the parameter so we can
    swap in alternative writers (e.g. a JSON dumper) later without changing
    callers.
    """
    del backend_name  # currently unused but kept for forward compatibility
    return LoadgenResponseWriter()
