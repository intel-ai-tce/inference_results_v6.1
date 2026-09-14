"""
Inference backends for the inference harness.
"""

from .base import DLRMBackend

__all__ = ["DLRMBackend", "GenerativeRecommenderBackend", "HybridGRBackend"]


def __getattr__(name: str):
    if name == "GenerativeRecommenderBackend":
        from .generative_recommener_backend import GenerativeRecommenderBackend

        return GenerativeRecommenderBackend
    if name == "HybridGRBackend":
        from .hybrid_GR_backend import HybridGRBackend

        return HybridGRBackend
    raise AttributeError(name)
