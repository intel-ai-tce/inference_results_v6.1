"""Backend contract – the only interface the LoadGen-side of the harness sees.

Keeping this surface minimal is what lets us swap the Mock backend for a real
xDiT/Wan 2.2 backend later without touching ``sut.py`` or ``loadgen_runner.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Sequence

if TYPE_CHECKING:
    from ..config import HarnessConfig


__all__ = [
    "Backend",
    "BackendBuildError",
    "GeneratedVideo",
]


class BackendBuildError(RuntimeError):
    """Raised when a backend cannot be instantiated or set up."""


@dataclass
class GeneratedVideo:
    """One backend-produced sample, ready for LoadGen completion.

    Attributes:
        sample_index:
            The QSL index (as supplied by LoadGen in
            ``mlperf_loadgen.QuerySample.index``).
        frames_bytes:
            Raw frame buffer, exactly as the SUT will hand it to
            ``mlperf_loadgen.QuerySampleResponse`` (the buffer's lifetime is
            managed by the SUT, not the backend).
        frame_count: Number of frames in ``frames_bytes``.
        height: Frame height in pixels.
        width: Frame width in pixels.
        mp4_bytes:
            Optional pre-encoded MP4 payload, populated in accuracy mode so
            ``artefacts.py`` can write per-sample sidecar files without
            re-encoding. Performance mode leaves this ``None``.
    """

    sample_index: int
    frames_bytes: bytes
    frame_count: int
    height: int
    width: int
    mp4_bytes: bytes | None = None


class Backend(ABC):
    """Abstract base class for inference backends.

    Subclasses MUST be cheap to construct: heavy work (loading model weights,
    initialising ``torch.distributed`` groups, running ``torch.compile``)
    belongs in :meth:`setup`, not ``__init__``. That way ``--print-config`` and
    ``--help`` invocations don't pay the cost of bringing up CUDA.
    """

    #: A short, stable name surfaced to the CLI (``--backend <name>``).
    name: str = "base"

    def __init__(self, config: "HarnessConfig") -> None:
        self._config = config
        self._is_set_up = False

    @property
    def config(self) -> "HarnessConfig":
        return self._config

    @property
    def is_set_up(self) -> bool:
        return self._is_set_up

    @abstractmethod
    def setup(self, *, rank: int = 0, world_size: int = 1) -> None:
        """Load weights, build pipelines, run warmup. Idempotent."""

    def warmup_settings(self) -> tuple[int, str] | None:
        """Return ``(num_prompts_per_rank, prompt)`` for a pre-LoadGen warmup,
        or ``None`` to skip warmup entirely.

        The dispatcher consumes this in ``loadgen_runner.run`` *before*
        ``lg.StartTest`` so first-call costs (``torch.compile``, kernel
        compilation, CUDA-graph capture) are excluded from measured timing.

        Default is ``None`` so backends without those costs (the Mock,
        for instance) automatically opt out.
        """
        return None

    @abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        indices: Sequence[int],
    ) -> Iterator[GeneratedVideo]:
        """Generate one or more videos.

        The backend MUST yield :class:`GeneratedVideo` instances **in the
        order they finish**, not necessarily in the order they appeared in
        ``prompts`` / ``indices``. This is what allows the SUT to call
        ``QuerySamplesComplete`` per finished sample and report accurate
        per-sample latencies.

        Args:
            prompts: Positive prompts, one per video to generate.
            indices: QSL sample indices, parallel to ``prompts``.

        Yields:
            One :class:`GeneratedVideo` per completed sample. ``sample_index``
            on each yielded item must be one of the indices in ``indices``.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Release GPU memory, shut down ``torch.distributed``, etc.

        Idempotent and must not raise on a backend that was never set up.
        """

    # ------------------------------------------------------------------
    # Context-manager sugar so callers can use ``with backend:``.
    # ------------------------------------------------------------------
    def __enter__(self) -> "Backend":
        if not self._is_set_up:
            self.setup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.teardown()
