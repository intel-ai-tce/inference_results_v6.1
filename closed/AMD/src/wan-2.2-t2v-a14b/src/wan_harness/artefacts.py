"""Sidecar artefact writer for the accuracy mode.

In accuracy mode each finished sample is written to disk so the (separate)
VBench evaluation step can score them. Filenames are derived from the QSL
sample index, **not** from the prompt text, to avoid the filesystem-safety
landmines we hit in the v6.0 implementation (slashes, NULs, long names,
collisions on identical prompts).

Why ``{index}.mp4`` and not ``{prompt}-0.mp4`` *here*
-----------------------------------------------------

VBench actually requires the ``{prompt}-{index}.mp4`` layout to score
the MLPerf reference dimensions (``scene``, ``appearance_style``, etc.):
those dimensions read structured ``auxiliary_info`` from VBench's
built-in ``VBench_full_info.json``, which is only consulted in
*vbench_standard* mode -- and that mode parses the prompt out of the
filename via ``vbench.utils.get_prompt_from_filename``. The alternative
*custom_input* mode (which we initially aimed for) blanket-refuses six
dimensions including two of the MLPerf six; see
``vbench/__init__.py:check_dimension_requires_extra_info``.

We still keep the harness's on-disk artefacts numeric -- ``{index}.mp4``
-- and emit a ``prompts.json`` sidecar mapping filename -> prompt. The
:mod:`wan_harness.vbench` evaluator then *stages* the videos into a
side directory of symlinks named ``{prompt}-{index}.mp4`` for VBench to
read. This split keeps the slash/NUL/NAME_MAX safety of the v6.0 retro
right at the artefact-writing path -- only the evaluator (which knows
the active prompt set is safe) pays the cost of the rename.

If the backend chose not to pre-encode an MP4 (``GeneratedVideo.mp4_bytes``
is ``None`` – typically a Mock dry-run), we fall back to a ``.bin`` raw
frame buffer with a JSON sidecar describing its shape. Such directories
are not VBench-runnable, but they keep the dry-run useful for debugging
the artefact-writing path.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from .backends.base import GeneratedVideo

_log = logging.getLogger(__name__)

__all__ = ["ArtefactWriter", "NullArtefactWriter"]


@dataclass
class _WrittenArtefact:
    sample_index: int
    path: Path
    kind: str  # "mp4" or "bin"


class ArtefactWriter:
    """Writes one file per ``GeneratedVideo`` into ``out_dir``.

    Filenames are bare sample indices – ``0.mp4``, ``1.mp4``, …, ``247.mp4``
    – matching MLPerf's ``{numeric_id}.mp4`` convention from
    ``data/samples_filename_ids.txt``. No zero-padding so a directory
    listing matches the reference one-for-one.

    A small index file (``artefacts.jsonl``) records every written sample,
    and on :meth:`finalize` a ``prompts.json`` is emitted in the shape
    VBench's ``--mode=custom_input`` consumes.
    """

    PROMPTS_JSON_NAME = "prompts.json"
    ARTEFACTS_JSONL_NAME = "artefacts.jsonl"

    def __init__(self, out_dir: Path) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._out_dir / self.ARTEFACTS_JSONL_NAME
        self._prompts_path = self._out_dir / self.PROMPTS_JSON_NAME
        self._lock = threading.Lock()
        # Truncate the index + sidecar on construction so re-runs do not
        # append stale data.
        self._index_path.write_text("", encoding="utf-8")
        # filename -> prompt, accumulated for finalize().
        self._prompt_map: dict[str, str] = {}

    @property
    def out_dir(self) -> Path:
        return self._out_dir

    @property
    def prompts_json_path(self) -> Path:
        """Where :meth:`finalize` will write the VBench prompt mapping."""
        return self._prompts_path

    def write(self, video: GeneratedVideo, *, prompt: str) -> Path:
        """Persist one sample. ``prompt`` is required so the finalize
        step has the data VBench's custom_input mode needs."""
        stem = str(int(video.sample_index))
        if video.mp4_bytes is not None:
            path = self._out_dir / f"{stem}.mp4"
            path.write_bytes(video.mp4_bytes)
            kind = "mp4"
            record = {
                "sample_index": int(video.sample_index),
                "path": path.name,
                "kind": kind,
                "prompt": prompt,
                "frame_count": video.frame_count,
                "height": video.height,
                "width": video.width,
                "bytes": len(video.mp4_bytes),
            }
        else:
            path = self._out_dir / f"{stem}.bin"
            path.write_bytes(video.frames_bytes)
            kind = "bin"
            record = {
                "sample_index": int(video.sample_index),
                "path": path.name,
                "kind": kind,
                "prompt": prompt,
                "frame_count": video.frame_count,
                "height": video.height,
                "width": video.width,
                "bytes": len(video.frames_bytes),
                "shape": [video.frame_count, video.height, video.width, 3],
                "dtype": "uint8",
            }
        with self._lock:
            self._prompt_map[path.name] = prompt
            with self._index_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        _log.debug("artefact: wrote %s (%d bytes)", path, record["bytes"])
        return path

    def finalize(self) -> Path:
        """Flush the accumulated ``filename -> prompt`` mapping to
        ``prompts.json`` so VBench's custom_input mode can find it.

        Idempotent: a second call simply rewrites the file with the
        current state. Returns the path that was written.
        """
        with self._lock:
            payload = dict(self._prompt_map)
        self._prompts_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        _log.info(
            "artefacts: wrote %s with %d entries", self._prompts_path, len(payload)
        )
        return self._prompts_path


class NullArtefactWriter:
    """Drop-in writer that does nothing; used in performance mode."""

    out_dir: Path = Path("/dev/null")

    def write(self, video: GeneratedVideo, *, prompt: str) -> Path:  # noqa: D401, ARG002
        return self.out_dir

    def finalize(self) -> Path:  # noqa: D401
        return self.out_dir
