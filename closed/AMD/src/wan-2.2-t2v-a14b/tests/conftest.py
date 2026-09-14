"""Shared pytest fixtures and src-path bootstrap."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Make `src/` importable for tests run via plain `pytest` without an editable
# install (handy on systems whose pip is too old for PEP 660).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def loadgen_available() -> bool:
    return importlib.util.find_spec("mlperf_loadgen") is not None


@pytest.fixture(scope="session")
def has_loadgen() -> bool:
    return loadgen_available()


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``loadgen`` when the module isn't importable."""
    if loadgen_available():
        return
    skip_marker = pytest.mark.skip(reason="mlperf_loadgen not installed")
    for item in items:
        if "loadgen" in item.keywords:
            item.add_marker(skip_marker)
