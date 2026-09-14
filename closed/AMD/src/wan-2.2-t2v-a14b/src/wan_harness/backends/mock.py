"""Mock backend: drives the entire LoadGen path with no GPU or model weights.

This backend deliberately depends only on the Python standard library so the
dry-run is exercisable on a laptop, in CI, or inside any container that has
nothing more than ``mlperf_loadgen`` + this package installed.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Iterator, Sequence

from ..wire import WorkUnit
from .base import Backend, GeneratedVideo

_log = logging.getLogger(__name__)


class MockBackend(Backend):
    """Deterministic, fast-by-default stand-in for a real video backend.

    The frame buffer it returns has exactly the same shape and dtype layout
    a real backend would produce (``num_frames * height * width * 3`` bytes
    of ``uint8``-cast ``[0, 255]`` data), so the SUT, the response writer,
    the artefact writer, and LoadGen all see realistic payload sizes.
    """

    name = "mock"

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------
    def setup(self, *, rank: int = 0, world_size: int = 1) -> None:
        cfg = self.config
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._dist_initialized = False

        if world_size > 1 and cfg.mock_dispatch is not None:
            import torch.distributed as dist  # noqa: WPS433

            if not dist.is_initialized():
                dist.init_process_group(
                    backend="gloo",
                    rank=self._rank,
                    world_size=self._world_size,
                )
                self._dist_initialized = True
            _log.info(
                "MockBackend: Gloo process group ready (rank=%d world_size=%d "
                "dispatch=%s)",
                self._rank,
                self._world_size,
                cfg.mock_dispatch,
            )
        elif rank != 0 or world_size != 1:
            _log.warning(
                "MockBackend: rank=%d/world_size=%d without mock_dispatch; "
                "only rank 0 will run (set mock_dispatch=async|wave for "
                "multi-rank Offline profiling)",
                rank,
                world_size,
            )
        self._delay_s = max(cfg.mock_delay_ms, 0) / 1000.0
        self._frame_count = cfg.num_frames
        self._height = cfg.height
        self._width = cfg.width
        self._frame_size = cfg.height * cfg.width * 3
        self._payload_kind = cfg.mock_payload
        if self._payload_kind == "zeros":
            self._frame = bytes(self._frame_size)  # all zero pixels
        else:
            # `noise` payload: per-sample-deterministic, computed on demand
            # in `_payload_for` so we don't hold one buffer per sample.
            self._frame = b""
        self._is_set_up = True
        _log.info(
            "MockBackend ready: %d frames %dx%d, payload=%s, delay=%dms",
            self._frame_count,
            self._height,
            self._width,
            self._payload_kind,
            cfg.mock_delay_ms,
        )

    def teardown(self) -> None:
        if getattr(self, "_dist_initialized", False):
            try:
                import torch.distributed as dist  # noqa: WPS433

                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as exc:  # pragma: no cover - best effort
                _log.warning("MockBackend.teardown: dist destroy failed: %s", exc)
        self._is_set_up = False

    # ------------------------------------------------------------------
    # Generation.
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: Sequence[str],
        indices: Sequence[int],
    ) -> Iterator[GeneratedVideo]:
        if not self._is_set_up:
            raise RuntimeError("MockBackend.generate() called before setup()")
        if len(prompts) != len(indices):
            raise ValueError(
                f"len(prompts)={len(prompts)} != len(indices)={len(indices)}"
            )
        for prompt, idx in zip(prompts, indices):
            yield self.run_unit(
                self.build_work_unit(prompt=prompt, sample_index=int(idx))
            )

    def build_work_unit(self, *, prompt: str, sample_index: int) -> WorkUnit:
        """Pack one prompt for the multi-rank Offline dispatchers."""
        return WorkUnit(
            sample_index=int(sample_index),
            prompt=prompt,
            input_args={},
        )

    def run_unit(self, unit: WorkUnit) -> GeneratedVideo:
        """Generate one sample. Used by Offline DP dispatchers on every rank."""
        if not self._is_set_up:
            raise RuntimeError("MockBackend.run_unit() called before setup()")
        if self._delay_s:
            time.sleep(self._delay_s)
        return GeneratedVideo(
            sample_index=int(unit.sample_index),
            frames_bytes=self._payload_for(int(unit.sample_index), unit.prompt),
            frame_count=self._frame_count,
            height=self._height,
            width=self._width,
            mp4_bytes=None,
        )

    # ------------------------------------------------------------------
    # Payload builders.
    # ------------------------------------------------------------------
    def _payload_for(self, idx: int, prompt: str) -> bytes:
        """Return the raw frame buffer for one sample.

        For the ``zeros`` payload kind we reuse a single shared buffer to
        keep peak memory tiny – the SUT will copy it into an ``array.array``
        anyway when it builds the LoadGen response.
        """
        if self._payload_kind == "zeros":
            return self._frame * self._frame_count

        # Deterministic, prompt+index-derived pseudo-random bytes. Slow-ish
        # for very large frames but only used for the `noise` ablation path.
        seed = hashlib.blake2b(
            f"{idx}:{prompt}".encode("utf-8"),
            digest_size=16,
        ).digest()
        # Tile the seed up to one frame, then repeat per frame.
        repeats = (self._frame_size + len(seed) - 1) // len(seed)
        frame = (seed * repeats)[: self._frame_size]
        return frame * self._frame_count
