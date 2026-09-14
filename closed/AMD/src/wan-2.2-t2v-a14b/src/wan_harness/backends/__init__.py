"""Backends produce generated videos for the SUT.

The set of backends is registered lazily so importing ``wan_harness.backends``
does not pull in heavy dependencies (torch, xDiT, diffusers). The Mock backend
is always available; the Wan 2.2 backend is loaded on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .base import Backend, BackendBuildError, GeneratedVideo

if TYPE_CHECKING:
    from ..config import HarnessConfig

__all__ = [
    "Backend",
    "BackendBuildError",
    "GeneratedVideo",
    "build_backend",
    "registered_backends",
]


# Lazily-resolved factories so we don't import torch/xDiT for a mock dry-run.
_FACTORIES: dict[str, Callable[[], type[Backend]]] = {}


def _register(name: str, loader: Callable[[], type[Backend]]) -> None:
    _FACTORIES[name] = loader


def _load_mock() -> type[Backend]:
    from .mock import MockBackend
    return MockBackend


def _load_wan22() -> type[Backend]:
    from .wan22 import WanBackend
    return WanBackend


_register("mock", _load_mock)
_register("wan22", _load_wan22)


def registered_backends() -> list[str]:
    """Return the list of backend names that can be passed to ``--backend``."""
    return sorted(_FACTORIES)


def build_backend(name: str, config: "HarnessConfig") -> Backend:
    """Instantiate (but do not ``setup``) the backend named ``name``.

    Callers are responsible for calling :meth:`Backend.setup` and
    :meth:`Backend.teardown` themselves; this two-step pattern keeps
    construction cheap and makes the lifecycle visible.
    """
    if name not in _FACTORIES:
        raise BackendBuildError(
            f"Unknown backend {name!r}. Registered backends: {registered_backends()}"
        )
    cls = _FACTORIES[name]()
    return cls(config)
