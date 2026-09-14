"""Unified configuration for the wan-2.2-t2v-a14b harness.

There are four logical sources of configuration, in increasing precedence:

  1. Built-in defaults on the :class:`HarnessConfig` dataclass below.
  2. ``configs/inference_config.yaml`` (model-side generation parameters).
  3. Environment variables (prefixed with ``WAN_HARNESS_``).
  4. CLI flags (parsed by :mod:`wan_harness.cli`).

This module owns merging and validation. Nothing else in the harness should
re-parse YAML or read ``os.environ`` for these knobs.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "HarnessConfig",
    "coerce_field_value",
    "load_harness_config",
    "resolve_backend_config_path",
    "REPO_ROOT",
]


REPO_ROOT = Path(__file__).resolve().parents[2]
"""Project root – the directory containing ``configs/``, ``data/``, ``scripts/``."""


_VALID_SCENARIOS = ("Offline", "SingleStream")
_VALID_MODES = ("performance", "accuracy")
_VALID_BACKENDS = ("mock", "wan22")
_MODEL_NAME = "wan-2.2-t2v-a14b"
_ENV_PREFIX = "WAN_HARNESS_"


@dataclass
class HarnessConfig:
    """One source of truth for everything that controls a benchmark run.

    The dataclass is intentionally flat – there is no nesting – so it is
    trivially serialisable to YAML/JSON for the ``--print-config`` dump and
    for ``harness_metadata.json``.
    """

    # ------------------------------------------------------------------
    # Model / generation parameters (sourced from inference_config.yaml).
    # These must not be changed for a closed-division submission; we still
    # surface them here so a developer can override them for ablations.
    # ------------------------------------------------------------------
    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 16
    sample_steps: int = 20
    seed: int = 42
    guidance_scale: float = 4.0
    guidance_scale_2: float = 3.0
    boundary_ratio: float = 0.875
    negative_prompt: str = ""

    # ------------------------------------------------------------------
    # Harness / runtime parameters.
    # ------------------------------------------------------------------
    model_name: str = _MODEL_NAME
    backend: str = "mock"
    scenario: str = "Offline"
    mode: str = "performance"

    # Filesystem inputs.
    prompts_path: Path = field(
        default_factory=lambda: REPO_ROOT / "data" / "vbench_prompts.txt"
    )
    fixed_latent_path: Path | None = field(
        default_factory=lambda: REPO_ROOT / "data" / "fixed_latent.pt"
    )
    inference_config_path: Path = field(
        default_factory=lambda: REPO_ROOT / "configs" / "inference_config.yaml"
    )
    user_conf_path: Path = field(
        default_factory=lambda: REPO_ROOT / "configs" / "user.conf"
    )
    # Per-backend config file (currently only the ``wan22`` backend uses it).
    # When unset and ``backend == 'wan22'``, defaults to
    # ``configs/wan22/<scenario>.yaml`` – see
    # :func:`resolve_backend_config_path` below.
    backend_config_path: Path | None = None
    # Ignored at runtime: LoadGen loads its built-in mlperf.conf automatically
    # on the first FromConfig(user.conf) call. Kept for CLI compatibility only.
    mlperf_conf_path: Path | None = None
    audit_conf_path: Path | None = None

    # Filesystem outputs.
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "runs")

    # LoadGen knobs that we sometimes want to override at the CLI for testing.
    # When None they are taken from mlperf.conf / user.conf.
    performance_sample_count: int | None = None
    min_query_count: int | None = None
    min_duration_ms: int | None = None
    max_duration_ms: int | None = None
    target_qps: float | None = None
    target_latency_ns: int | None = None

    # ------------------------------------------------------------------
    # Backend-specific knobs (prefix-namespaced to keep the dataclass flat).
    # ------------------------------------------------------------------
    mock_delay_ms: int = 0
    """Per-sample artificial latency for the Mock backend (used to test
    scheduler behaviour and to make dry-run timing visible in logs)."""

    mock_payload: str = "zeros"
    """``zeros`` returns an all-zero frame buffer; ``noise`` returns a
    deterministic pseudo-random buffer derived from the sample index. Both
    are byte-identical between runs."""

    mock_dispatch: str | None = None
    """When set to ``wave`` or ``async`` and ``backend == 'mock'`` with
    ``world_size > 1``, route the mock backend through the corresponding
    Offline data-parallel dispatcher so post-``run_unit`` wire overhead
    can be profiled without a GPU model."""

    result_transport: str | None = None
    """Bulk Result data plane override: ``shm`` (default when unset on
    mock) or ``gloo``. Wan22 runs inherit
    ``parallelism.result_transport`` from the backend YAML unless set."""

    # ------------------------------------------------------------------
    # Debug / observability.
    # ------------------------------------------------------------------
    enable_loadgen_trace: bool = False
    measure_post_run_overhead: bool = False
    """Record per-phase timings for work after ``backend.run_unit`` returns
    (result packaging, cross-rank transfer, LoadGen completion)."""
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Validation.
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.scenario not in _VALID_SCENARIOS:
            raise ValueError(
                f"scenario must be one of {_VALID_SCENARIOS!r}, got {self.scenario!r}"
            )
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES!r}, got {self.mode!r}"
            )
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {_VALID_BACKENDS!r}, got {self.backend!r}"
            )
        if self.height <= 0 or self.height % 8 != 0:
            raise ValueError(f"height must be a positive multiple of 8, got {self.height!r}")
        if self.width <= 0 or self.width % 8 != 0:
            raise ValueError(f"width must be a positive multiple of 8, got {self.width!r}")
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames!r}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps!r}")
        if self.sample_steps <= 0:
            raise ValueError(f"sample_steps must be positive, got {self.sample_steps!r}")
        if not (0.0 < self.boundary_ratio < 1.0):
            raise ValueError(
                f"boundary_ratio must be in (0, 1), got {self.boundary_ratio!r}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed!r}")
        if self.mock_delay_ms < 0:
            raise ValueError(
                f"mock_delay_ms must be non-negative, got {self.mock_delay_ms!r}"
            )
        if self.mock_payload not in ("zeros", "noise"):
            raise ValueError(
                f"mock_payload must be 'zeros' or 'noise', got {self.mock_payload!r}"
            )
        if self.mock_dispatch is not None and self.mock_dispatch not in ("wave", "async"):
            raise ValueError(
                "mock_dispatch must be 'wave', 'async', or None, "
                f"got {self.mock_dispatch!r}"
            )
        if self.result_transport is not None and self.result_transport not in (
            "shm",
            "gloo",
        ):
            raise ValueError(
                "result_transport must be 'shm', 'gloo', or None, "
                f"got {self.result_transport!r}"
            )

        for path_field in (
            "prompts_path",
            "inference_config_path",
            "user_conf_path",
            "output_dir",
        ):
            value = getattr(self, path_field)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, path_field, Path(value))
        for opt_path_field in (
            "fixed_latent_path",
            "mlperf_conf_path",
            "audit_conf_path",
            "backend_config_path",
        ):
            value = getattr(self, opt_path_field)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, opt_path_field, Path(value))

    # ------------------------------------------------------------------
    # Serialisation helpers.
    # ------------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict representation (Path -> str, etc.)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                out[f.name] = str(value)
            else:
                out[f.name] = value
        return out

    def merged(self, **overrides: Any) -> "HarnessConfig":
        """Return a new HarnessConfig with the given fields overridden.

        Values of ``None`` are ignored so partial CLI overrides work cleanly.
        """
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)


# ----------------------------------------------------------------------
# Loader.
# ----------------------------------------------------------------------


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def _env_overrides() -> dict[str, Any]:
    """Pick up ``WAN_HARNESS_*`` env vars and map them onto dataclass fields.

    Only fields that are present on :class:`HarnessConfig` are picked up; an
    unknown ``WAN_HARNESS_FOO`` is silently ignored to allow shell exports
    that aren't meant for us.
    """
    field_names = {f.name for f in fields(HarnessConfig)}
    out: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        name = key[len(_ENV_PREFIX):].lower()
        if name not in field_names:
            continue
        out[name] = coerce_field_value(name, raw)
    return out


def coerce_field_value(name: str, raw: str) -> Any:
    """Coerce a string value to the type of the named :class:`HarnessConfig` field.

    Used by both the environment-variable loader and the CLI's
    ``--set key=value`` escape hatch. Unknown field names raise ``KeyError``
    so callers can produce useful error messages.
    """
    type_map = {f.name: f.type for f in fields(HarnessConfig)}
    if name not in type_map:
        raise KeyError(name)
    field_type = type_map[name]
    # `field_type` is a string under `from __future__ import annotations`,
    # so we match against known names.
    if field_type in ("int", "int | None"):
        return int(raw)
    if field_type in ("float", "float | None"):
        return float(raw)
    if field_type == "bool":
        return raw.lower() in ("1", "true", "yes", "y", "on")
    if field_type in ("Path", "Path | None"):
        return Path(raw)
    return raw


def load_harness_config(
    *,
    inference_config_path: Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> HarnessConfig:
    """Build a fully-resolved :class:`HarnessConfig`.

    Precedence (highest wins):
        1. ``cli_overrides``
        2. ``WAN_HARNESS_*`` environment variables
        3. ``inference_config.yaml`` (or whatever ``inference_config_path`` points at)
        4. dataclass defaults
    """
    base = HarnessConfig()
    if inference_config_path is not None:
        base = replace(base, inference_config_path=Path(inference_config_path))

    yaml_data = dict(_read_yaml(base.inference_config_path))
    # Only inject keys that are valid HarnessConfig fields.
    valid_fields = {f.name for f in fields(HarnessConfig)}
    yaml_kwargs = {k: v for k, v in yaml_data.items() if k in valid_fields}
    if yaml_kwargs:
        base = base.merged(**yaml_kwargs)

    env_kwargs = _env_overrides()
    if env_kwargs:
        base = base.merged(**env_kwargs)

    if cli_overrides:
        base = base.merged(**{k: v for k, v in cli_overrides.items() if v is not None})

    # Final validation pass (replace() re-invokes __post_init__).
    return base


def resolve_backend_config_path(config: HarnessConfig) -> Path | None:
    """Resolve the per-backend YAML path.

    For the ``wan22`` backend, returns ``config.backend_config_path`` if
    set, else falls back to the conventional ``configs/wan22/<scenario>.yaml``
    location. For backends that have no per-backend YAML (``mock``), returns
    ``None``.
    """
    if config.backend != "wan22":
        return None
    if config.backend_config_path is not None:
        return Path(config.backend_config_path)
    return REPO_ROOT / "configs" / "wan22" / f"{config.scenario}.yaml"
