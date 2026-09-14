"""System Under Test: bridges LoadGen <-> Dispatcher <-> Backend.

The SUT is the only thing LoadGen directly touches via callbacks:

  - :meth:`issue_queries(query_samples)`: LoadGen hands us samples to run.
  - :meth:`flush_queries()`: LoadGen tells us to drain in-flight work.

The class is intentionally small and free of model/torch imports so it can
be unit-tested against the :class:`RecordingResponseWriter` and the
:class:`MockBackend` on a CPU laptop.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

if TYPE_CHECKING:
    from .artefacts import ArtefactWriter
    from .dispatcher import Dispatcher
    from .qsl import WanQSL
    from .response import ResponseWriter

from .post_run_overhead import (
    PHASE_SUT_ARTEFACT_WRITE,
    PHASE_SUT_RESPONSE_COMPLETE,
    get_collector,
)

_log = logging.getLogger(__name__)

__all__ = [
    "WanSUT",
    "QuerySampleLike",
]


@dataclass(frozen=True)
class QuerySampleLike:
    """Duck-type of ``mlperf_loadgen.QuerySample`` for tests.

    The real LoadGen object exposes ``index`` and ``id`` attributes; tests
    use this dataclass.
    """

    index: int
    id: int


class WanSUT:
    """Wires LoadGen callbacks to the dispatcher and (optionally) sidecar
    artefact writing.

    Threading model:
        - LoadGen may call :meth:`issue_queries` from one or more threads
          (depending on the scenario). In Offline it is typically a single
          large batch; in SingleStream it is many small (size-1) batches in
          quick succession.
        - We serialise dispatcher access with a single lock so the
          single-process Mock path is trivially correct; the multi-rank path
          will keep this same contract.
        - Response-writer buffers are released only inside
          :meth:`flush_queries` (LoadGen guarantees no in-flight queries at
          that point).
    """

    def __init__(
        self,
        dispatcher: "Dispatcher",
        qsl: "WanQSL",
        response_writer: "ResponseWriter",
        *,
        artefact_writer: "ArtefactWriter | None" = None,
        name: str = "wan-2.2-t2v-a14b-sut",
    ) -> None:
        self._dispatcher = dispatcher
        self._qsl = qsl
        self._response_writer = response_writer
        self._artefact_writer = artefact_writer
        self._name = name
        self._lock = threading.Lock()
        self._completed = 0
        self._issued = 0

    # ------------------------------------------------------------------
    # Read-only telemetry (handy in tests and in `--print-config` output).
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def completed_count(self) -> int:
        return self._completed

    @property
    def issued_count(self) -> int:
        return self._issued

    # ------------------------------------------------------------------
    # LoadGen callbacks.
    # ------------------------------------------------------------------
    def issue_queries(self, query_samples: Iterable[Any]) -> None:
        """Hand off a LoadGen batch to the dispatcher and stream completions.

        ``query_samples`` is whatever LoadGen passes in (a list of objects
        with ``.index`` and ``.id``). We accept any iterable of duck-typed
        objects so unit tests can use :class:`QuerySampleLike`.
        """
        # Snapshot the batch into plain Python ints – we do not want to hold
        # references to LoadGen's internal C++ owned objects across the
        # generation loop.
        samples = list(query_samples)
        indices: list[int] = [int(s.index) for s in samples]
        query_ids: list[int] = [int(s.id) for s in samples]
        if not samples:
            return

        with self._lock:
            self._issued += len(samples)

        _log.debug("issue_queries: %d samples (first=%s)", len(samples), indices[0])

        prompts = self._qsl.get_prompts(indices)

        # LoadGen Offline replicates QSL indices to satisfy
        # ``samples_per_query`` when it exceeds the loaded sample count, so
        # the same ``sample_index`` may appear multiple times in the batch
        # with *distinct* ``query_id`` values. We must call
        # ``QuerySamplesComplete`` exactly once per ``query_id`` (LoadGen
        # raises "Attempted to complete a sample twice" and stalls the
        # post-test phase otherwise). Maintain a FIFO queue of pending
        # ``(query_id, prompt)`` slots per ``sample_index`` and pop one slot
        # for every dispatcher completion, in the order they were issued.
        pending_for_index: dict[int, deque[tuple[int, str]]] = {}
        for idx, qid, prompt in zip(indices, query_ids, prompts):
            pending_for_index.setdefault(idx, deque()).append((qid, prompt))

        for video in self._dispatcher.generate(prompts=prompts, indices=indices):
            video_idx = int(video.sample_index)
            queue = pending_for_index.get(video_idx)
            if not queue:
                raise RuntimeError(
                    f"Backend returned sample_index={video.sample_index!r} "
                    f"with no pending query_id (already completed all "
                    f"occurrences in batch {indices!r})"
                )
            qid, prompt = queue.popleft()

            if self._dispatcher.is_response_owner():
                oc = get_collector()
                nbytes = len(video.frames_bytes)
                t_resp = time.perf_counter()
                self._response_writer.complete(qid, video.frames_bytes)
                oc.record(
                    PHASE_SUT_RESPONSE_COMPLETE,
                    time.perf_counter() - t_resp,
                    sample_index=video_idx,
                    rank=0,
                    nbytes=nbytes,
                )
                if self._artefact_writer is not None:
                    t_art = time.perf_counter()
                    self._artefact_writer.write(video, prompt=prompt)
                    oc.record(
                        PHASE_SUT_ARTEFACT_WRITE,
                        time.perf_counter() - t_art,
                        sample_index=video_idx,
                        rank=0,
                        nbytes=nbytes,
                    )

            with self._lock:
                self._completed += 1

    def flush_queries(self) -> None:
        """LoadGen signal that no more queries are in flight for now."""
        _log.debug("flush_queries: issued=%d completed=%d", self._issued, self._completed)
        self._dispatcher.flush()
        # Now it is safe to drop retained response buffers – LoadGen has
        # finished reading from them.
        self._response_writer.release()


def build_query_sample_factory() -> Callable[[int, int], Any]:
    """Return a constructor for the LoadGen QuerySample type.

    Used by the runner when synthesising offline test inputs. Falls back to
    :class:`QuerySampleLike` when loadgen is not importable so the dry-run
    test suite still runs.
    """
    try:
        import mlperf_loadgen as lg  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - exercised only without loadgen
        return lambda index, query_id: QuerySampleLike(index=index, id=query_id)
    return lambda index, query_id: lg.QuerySample(query_id, index)


def _ensure_sequence(samples: Any) -> Sequence[Any]:
    """Defensive helper: LoadGen sometimes hands us iterators, sometimes
    lists. Materialise once so we can size + index.
    """
    if isinstance(samples, (list, tuple)):
        return samples
    return list(samples)
