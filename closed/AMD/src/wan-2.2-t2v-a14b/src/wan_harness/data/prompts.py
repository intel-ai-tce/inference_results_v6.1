"""Prompt loading and deterministic indexing for the QSL.

The MLPerf submission expects exactly 248 prompts in stable order (the VBench
prompt list). This module loads them from disk, returns a hashable
:class:`PromptDataset`, and exposes a synthetic fallback so tests can run
without the upstream data files present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = ["PromptDataset", "load_prompts", "synthetic_prompts"]


@dataclass(frozen=True)
class PromptDataset:
    """Immutable, indexable container of positive prompts.

    Prompts are stored in the exact order they appear in the source file
    (``vbench_prompts.txt``). The QSL uses the row index as the LoadGen
    ``sample_index``.
    """

    prompts: tuple[str, ...]
    source: str

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]

    def get_many(self, indices: Sequence[int]) -> list[str]:
        try:
            return [self.prompts[i] for i in indices]
        except IndexError as exc:
            raise IndexError(
                f"Prompt index out of range: requested {indices!r}, "
                f"dataset has {len(self.prompts)} prompts"
            ) from exc


def load_prompts(path: Path) -> PromptDataset:
    """Read a prompts text file (one prompt per non-empty line)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {path}. "
            f"Place `vbench_prompts.txt` at this path or pass --prompts."
        )
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    prompts = tuple(line.strip() for line in raw_lines if line.strip())
    if not prompts:
        raise ValueError(f"Prompts file is empty: {path}")
    return PromptDataset(prompts=prompts, source=str(path))


def synthetic_prompts(count: int = 16, *, prefix: str = "synthetic") -> PromptDataset:
    """Create a deterministic synthetic prompt set for tests / dry-runs.

    The strings are short and unique so any per-prompt logging stays readable.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count!r}")
    return PromptDataset(
        prompts=tuple(f"{prefix}-prompt-{i:04d}" for i in range(count)),
        source=f"<synthetic:{count}>",
    )
