"""QSL: the Query Sample Library for the wan-2.2-t2v-a14b benchmark.

The MLPerf rules and ``mlperf.conf`` pin the QSL count at 248. The QSL is
small enough that holding everything in CPU RAM is trivial; we still
implement :meth:`load_query_samples` / :meth:`unload_query_samples` per the
LoadGen QSL contract.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Sequence

from .data.prompts import PromptDataset

_log = logging.getLogger(__name__)

__all__ = ["WanQSL"]


class WanQSL:
    """In-memory QSL backed by a :class:`PromptDataset`.

    The class exposes the four methods LoadGen needs (``load_query_samples``,
    ``unload_query_samples``, plus the ``total_sample_count`` and
    ``performance_sample_count`` properties used by the runner), as well as a
    small ``get_prompts`` helper for the SUT.

    The ``loaded_indices`` set is used by tests to assert that LoadGen is
    honouring the QSL contract.
    """

    def __init__(
        self,
        dataset: PromptDataset,
        *,
        performance_sample_count: int | None = None,
    ) -> None:
        self._dataset = dataset
        if performance_sample_count is None:
            performance_sample_count = len(dataset)
        if performance_sample_count <= 0:
            raise ValueError(
                f"performance_sample_count must be positive, got {performance_sample_count!r}"
            )
        if performance_sample_count > len(dataset):
            raise ValueError(
                f"performance_sample_count={performance_sample_count} "
                f"> total_sample_count={len(dataset)}"
            )
        self._performance_sample_count = performance_sample_count
        self._loaded: set[int] = set()

    # ------------------------------------------------------------------
    # Properties used by the LoadGen runner.
    # ------------------------------------------------------------------
    @property
    def dataset(self) -> PromptDataset:
        return self._dataset

    @property
    def total_sample_count(self) -> int:
        return len(self._dataset)

    @property
    def performance_sample_count(self) -> int:
        return self._performance_sample_count

    @property
    def loaded_indices(self) -> frozenset[int]:
        return frozenset(self._loaded)

    # ------------------------------------------------------------------
    # LoadGen callbacks.
    # ------------------------------------------------------------------
    def load_query_samples(self, indices: Iterable[int]) -> None:
        """LoadGen tells us which samples will be queried next.

        Records which sample indices LoadGen has loaded.
        """
        added = [int(i) for i in indices]
        for i in added:
            if not 0 <= i < len(self._dataset):
                raise IndexError(
                    f"load_query_samples received out-of-range index {i} "
                    f"(dataset has {len(self._dataset)} entries)"
                )
        self._loaded.update(added)
        _log.debug("load_query_samples(%d): now %d loaded", len(added), len(self._loaded))

    def unload_query_samples(self, indices: Iterable[int]) -> None:
        """LoadGen tells us we can release these samples."""
        removed = [int(i) for i in indices]
        for i in removed:
            self._loaded.discard(i)
        _log.debug(
            "unload_query_samples(%d): %d loaded after release",
            len(removed),
            len(self._loaded),
        )

    # ------------------------------------------------------------------
    # SUT-side helpers.
    # ------------------------------------------------------------------
    def get_prompts(self, indices: Sequence[int]) -> list[str]:
        """Return the prompts at ``indices``, in the same order."""
        return self._dataset.get_many(indices)
