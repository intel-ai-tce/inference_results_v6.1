"""Lightweight logging setup shared by CLI and runner."""

from __future__ import annotations

import logging
import os
import sys

__all__ = ["configure_logging"]


def configure_logging(level: str | int = "INFO") -> None:
    """Configure a single, sane root logger.

    Re-running this is safe; it removes existing handlers first so test
    fixtures don't accumulate duplicates.
    """
    if isinstance(level, str):
        level = level.upper()
        numeric = getattr(logging, level, logging.INFO)
    else:
        numeric = int(level)

    rank = int(os.environ.get("RANK", "0"))
    prefix = f"[rank={rank}] " if rank else ""

    fmt = f"%(asctime)s {prefix}%(levelname)-7s %(name)s :: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(handler)
    root.setLevel(numeric)
