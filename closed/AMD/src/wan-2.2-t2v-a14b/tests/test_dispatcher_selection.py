"""Test that :func:`build_dispatcher` picks the right class for each topology.

These tests do not import xfuser or torch.distributed; they stub out the
heavy bits of :class:`WanBackend` so the dispatcher selection logic is
exercised in isolation.
"""

from __future__ import annotations

import pytest

from wan_harness.backends.mock import MockBackend
from wan_harness.backends.wan22_config import (
    ParallelismConfig,
    WanBackendConfig,
)
from wan_harness.config import HarnessConfig
from wan_harness.dispatcher import (
    AsyncDPDispatcher,
    SingleProcessDispatcher,
    UlyssesDispatcher,
    WaveDispatcher,
    build_dispatcher,
)


def _mock_backend(**overrides: object) -> MockBackend:
    cfg = HarnessConfig(
        backend="mock",
        height=8,
        width=16,
        num_frames=2,
        **overrides,
    )
    return MockBackend(cfg)


class _FakeWanBackend:
    """Minimal stand-in for :class:`WanBackend` exposing what the dispatcher reads."""

    name = "wan22"

    def __init__(self, mode: str, *, dispatch: str = "wave") -> None:
        self.backend_config = WanBackendConfig(
            parallelism=ParallelismConfig(
                mode=mode,
                ulysses_degree=8 if mode == "ulysses" else 1,
                data_parallel_workers=8 if mode == "data_parallel" else 1,
                dispatch=dispatch,
            ),
        )
        self.distributed_device = "cpu"


def test_mock_backend_always_single_process() -> None:
    backend = _mock_backend()
    disp = build_dispatcher(backend, rank=0, world_size=1)
    assert isinstance(disp, SingleProcessDispatcher)
    assert disp.is_response_owner() is True


def test_mock_backend_with_world_size_returns_single_process() -> None:
    backend = _mock_backend()
    disp = build_dispatcher(backend, rank=0, world_size=4)
    assert isinstance(disp, SingleProcessDispatcher)


def test_mock_backend_with_mock_dispatch_wave() -> None:
    backend = _mock_backend(mock_dispatch="wave")
    disp = build_dispatcher(backend, rank=0, world_size=4)
    assert isinstance(disp, WaveDispatcher)


def test_mock_backend_with_mock_dispatch_async() -> None:
    backend = _mock_backend(mock_dispatch="async")
    disp = build_dispatcher(backend, rank=0, world_size=4)
    assert isinstance(disp, AsyncDPDispatcher)


def test_world_size_one_returns_single_process_even_for_wan() -> None:
    # The world_size==1 short-circuit avoids importing WanBackend, so we
    # can pass anything with .name set here.
    backend = _FakeWanBackend(mode="ulysses")
    disp = build_dispatcher(backend, rank=0, world_size=1)  # type: ignore[arg-type]
    assert isinstance(disp, SingleProcessDispatcher)


@pytest.fixture
def patch_wan_backend(monkeypatch: pytest.MonkeyPatch):
    """Make ``isinstance(backend, WanBackend)`` accept :class:`_FakeWanBackend`.

    ``build_dispatcher`` does a local ``from .backends.wan22 import WanBackend``
    which we replace with our fake. Tests don't load xfuser this way.
    """
    monkeypatch.setattr(
        "wan_harness.backends.wan22.WanBackend", _FakeWanBackend
    )
    return monkeypatch


def test_ulysses_mode_picks_ulysses_dispatcher(patch_wan_backend) -> None:
    backend = _FakeWanBackend(mode="ulysses")
    disp = build_dispatcher(backend, rank=0, world_size=8)  # type: ignore[arg-type]
    assert isinstance(disp, UlyssesDispatcher)
    assert disp.is_response_owner() is True
    assert disp.info.world_size == 8


def test_ulysses_dispatcher_rank_nonzero_not_owner(patch_wan_backend) -> None:
    backend = _FakeWanBackend(mode="ulysses")
    disp = build_dispatcher(backend, rank=3, world_size=8)  # type: ignore[arg-type]
    assert isinstance(disp, UlyssesDispatcher)
    assert disp.is_response_owner() is False


def test_data_parallel_mode_picks_wave_dispatcher(patch_wan_backend) -> None:
    backend = _FakeWanBackend(mode="data_parallel")
    disp = build_dispatcher(backend, rank=0, world_size=8)  # type: ignore[arg-type]
    assert isinstance(disp, WaveDispatcher)
    assert disp.is_response_owner() is True


def test_data_parallel_wave_dispatcher_rank_nonzero_not_owner(patch_wan_backend) -> None:
    backend = _FakeWanBackend(mode="data_parallel")
    disp = build_dispatcher(backend, rank=5, world_size=8)  # type: ignore[arg-type]
    assert isinstance(disp, WaveDispatcher)
    assert disp.is_response_owner() is False


def test_data_parallel_async_picks_async_dispatcher(patch_wan_backend) -> None:
    backend = _FakeWanBackend(mode="data_parallel", dispatch="async")
    disp = build_dispatcher(backend, rank=0, world_size=8)  # type: ignore[arg-type]
    assert isinstance(disp, AsyncDPDispatcher)
    assert disp.is_response_owner() is True
    assert disp.info.name == "async-dp"
    assert disp.info.world_size == 8


def test_data_parallel_async_dispatcher_rank_nonzero_not_owner(patch_wan_backend) -> None:
    backend = _FakeWanBackend(mode="data_parallel", dispatch="async")
    disp = build_dispatcher(backend, rank=5, world_size=8)  # type: ignore[arg-type]
    assert isinstance(disp, AsyncDPDispatcher)
    assert disp.is_response_owner() is False


def test_default_dispatch_is_wave(patch_wan_backend) -> None:
    """Constructing a DP backend without setting ``dispatch`` keeps the
    wave path so existing configs / tests are unchanged."""
    backend = _FakeWanBackend(mode="data_parallel")  # default dispatch="wave"
    assert backend.backend_config.parallelism.dispatch == "wave"
    disp = build_dispatcher(backend, rank=0, world_size=8)  # type: ignore[arg-type]
    assert isinstance(disp, WaveDispatcher)


def test_unknown_backend_rejects() -> None:
    class _OtherBackend:
        name = "other"
    with pytest.raises(TypeError):
        build_dispatcher(_OtherBackend(), rank=0, world_size=2)  # type: ignore[arg-type]


def test_expected_world_size_matches_config() -> None:
    cfg = WanBackendConfig(
        parallelism=ParallelismConfig(mode="ulysses", ulysses_degree=8),
    )
    assert cfg.expected_world_size == 8

    cfg = WanBackendConfig(
        parallelism=ParallelismConfig(
            mode="data_parallel", data_parallel_workers=8
        ),
    )
    assert cfg.expected_world_size == 8


# ----------------------------------------------------------------------
# Warmup behaviour.
# ----------------------------------------------------------------------


def test_warmup_dispatches_correct_count_single_process() -> None:
    """SingleProcessDispatcher should dispatch exactly ``num_prompts`` units."""
    backend = _mock_backend()
    backend.setup()
    disp = build_dispatcher(backend, rank=0, world_size=1)

    seen_indices: list[int] = []
    real_generate = backend.generate

    def spy_generate(prompts, indices):  # type: ignore[no-untyped-def]
        for v in real_generate(prompts=prompts, indices=indices):
            seen_indices.append(v.sample_index)
            yield v

    backend.generate = spy_generate  # type: ignore[method-assign]
    disp.warmup(3, "warmup-prompt")

    # SingleProcess multiplier == 1; expect exactly 3 dispatched.
    assert len(seen_indices) == 3
    # Warmup uses negative indices so they cannot clash with QSL indices.
    assert all(i < 0 for i in seen_indices)


def test_warmup_zero_or_negative_is_noop() -> None:
    backend = _mock_backend()
    backend.setup()
    disp = build_dispatcher(backend, rank=0, world_size=1)

    calls: list[int] = []
    real_generate = backend.generate

    def spy_generate(prompts, indices):  # type: ignore[no-untyped-def]
        calls.append(len(prompts))
        yield from real_generate(prompts=prompts, indices=indices)

    backend.generate = spy_generate  # type: ignore[method-assign]
    disp.warmup(0, "x")
    disp.warmup(-5, "x")
    assert calls == []  # never invoked


def test_warmup_multiplier_per_dispatcher(patch_wan_backend) -> None:
    """Ulysses multiplier=1 (all ranks see every prompt); Wave + async DP
    both = world_size so a single ``num_prompts=1`` warmup pass covers
    every rank exactly once."""
    ulysses_disp = build_dispatcher(
        _FakeWanBackend(mode="ulysses"), rank=0, world_size=8
    )  # type: ignore[arg-type]
    wave_disp = build_dispatcher(
        _FakeWanBackend(mode="data_parallel"), rank=0, world_size=8
    )  # type: ignore[arg-type]
    async_disp = build_dispatcher(
        _FakeWanBackend(mode="data_parallel", dispatch="async"),
        rank=0,
        world_size=8,
    )  # type: ignore[arg-type]
    sp_disp = build_dispatcher(_mock_backend(), rank=0, world_size=1)

    assert ulysses_disp._warmup_multiplier() == 1  # noqa: SLF001 (test internals)
    assert wave_disp._warmup_multiplier() == 8     # noqa: SLF001
    assert async_disp._warmup_multiplier() == 8    # noqa: SLF001
    assert sp_disp._warmup_multiplier() == 1       # noqa: SLF001
