"""Mock backend ``run_unit`` contract for Offline dispatchers."""

from __future__ import annotations

from wan_harness.backends.mock import MockBackend
from wan_harness.config import HarnessConfig


def test_mock_run_unit_realistic_payload_size() -> None:
    cfg = HarnessConfig(
        backend="mock",
        height=720,
        width=1280,
        num_frames=81,
        mock_payload="zeros",
    )
    backend = MockBackend(cfg)
    backend.setup()
    unit = backend.build_work_unit(prompt="test prompt", sample_index=3)
    video = backend.run_unit(unit)
    expected = 720 * 1280 * 81 * 3
    assert len(video.frames_bytes) == expected
    assert video.sample_index == 3
