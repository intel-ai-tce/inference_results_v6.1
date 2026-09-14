"""Configuration schema for the real Wan 2.2 T2V-A14B backend.

The harness uses *one* :class:`WanBackendConfig` per scenario, loaded from a
YAML file under ``configs/wan22/<scenario>.yaml``. Keeping this config
separate from :class:`HarnessConfig` lets the LoadGen-side knobs
(scenario, mode, prompts path, output dir, ...) stay flat and
JSON-serialisable while the backend-specific knobs grow as needed.

The YAML is structured into a handful of sections that mirror the natural
groupings of ``xfuser.xFuserArgs``:

  * ``model``       – which checkpoint to load and what dtype to cast to.
  * ``parallelism`` – the xDiT parallelism mode and per-axis degrees.
  * ``vae``         – DistVAE knobs and VAE memory tricks.
  * ``generation``  – non-xfuser generation-side options (e.g. fixed latent).
  * ``compile``     – ``torch.compile`` toggles (deferred for v1).
  * ``xfuser_extra`` – escape hatch: any extra ``xFuserArgs`` keyword arg
    we have not promoted to a dedicated field.

Anything in ``xfuser_extra`` is splatted into ``xFuserArgs(...)`` as-is, so
it must use the xfuser field names (not our re-namings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "WanBackendConfig",
    "ModelConfig",
    "ParallelismConfig",
    "VaeConfig",
    "GenerationConfig",
    "CompileConfig",
    "WarmupConfig",
    "load_wan_backend_config",
    "default_config_path_for_scenario",
    "PARALLELISM_MODES",
    "DP_DISPATCH_MODES",
]


PARALLELISM_MODES = ("ulysses", "data_parallel")
"""Selects which dispatcher class :func:`build_dispatcher` instantiates.

* ``ulysses``       – all world-size ranks run xfuser's Ulysses SP for one
  prompt at a time. ``ulysses_degree`` must equal ``world_size``.
* ``data_parallel`` – ``data_parallel_workers`` independent ranks, each
  with its own model copy. The harness owns the prompt dispatch via one
  of the data-parallel dispatchers in :mod:`wan_harness.dispatcher`;
  xfuser itself sees a single-rank world.
"""


DP_DISPATCH_MODES = ("wave", "async")


_DEFAULT_MODEL_PATH = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
_DEFAULT_MODEL_REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"


@dataclass
class ModelConfig:
    path: str = _DEFAULT_MODEL_PATH
    revision: str | None = _DEFAULT_MODEL_REVISION
    transformer_dtype: str = "bfloat16"


@dataclass
class ParallelismConfig:
    mode: str = "ulysses"
    ulysses_degree: int = 1
    ring_degree: int = 1
    tensor_parallel_degree: int = 1
    pipefusion_parallel_degree: int = 1
    data_parallel_degree: int = 1
    """Maps onto xfuser's own data-parallel degree (NOT our DP dispatcher).
    Leave at 1 when ``mode == 'data_parallel'`` – we manage DP ourselves."""

    data_parallel_workers: int = 1
    """Number of independent ranks in our custom DP dispatcher. Only used
    when ``mode == 'data_parallel'``. Must equal ``world_size``."""

    dispatch: str = "wave"
    """Which DP scheduler to use. Only meaningful when
    ``mode == 'data_parallel'``; must be one of :data:`DP_DISPATCH_MODES`.

    * ``wave``  (default) – :class:`~wan_harness.dispatcher.WaveDispatcher`.
      Backwards-compatible lockstep waves.
    * ``async`` – :class:`~wan_harness.dispatcher.AsyncDPDispatcher`.
      Pull-style scheduler that hands the next prompt to whichever rank
      finished first. Use this when per-rank latency varies wave-to-wave
      (e.g. prompt-length-dependent attention) and the wave dispatcher's
      straggler tax dominates.
    """

    result_transport: str = "shm"
    """Bulk :class:`~wan_harness.wire.Result` data plane for DP dispatchers.

    * ``shm`` (default) – POSIX shared memory on single-node runs; Gloo
      carries only metadata and slot handoff.
    * ``gloo`` – legacy path: raw ``uint8`` tensor send/recv over Gloo.
    """

    use_cfg_parallel: bool = False


@dataclass
class VaeConfig:
    use_parallel: bool = False
    """xfuser ``use_parallel_vae``: install DistVAE adapters on the VAE."""

    parallel_size: int = 0
    """xfuser ``vae_parallel_size``: number of ranks dedicated to a
    pipelined VAE. ``0`` means VAE runs on the same ranks as DiT (using
    the world group as the VAE process group)."""

    enable_tiling: bool = False
    """VAE memory trick: chunk the latent spatially during decode."""

    enable_slicing: bool = False
    """VAE memory trick: chunk the batch dim during decode."""


# ``scheduler/scheduler_config.json`` on Wan-AI/Wan2.2-T2V-A14B-Diffusers.
# The MLPerf reference leaves this untouched via ``from_pretrained``; xfuser
# still requires the key in ``input_args``, so the harness passes it through.
CHECKPOINT_FLOW_SHIFT: float = 3.0


@dataclass
class GenerationConfig:
    use_fixed_latent: bool = True
    """Inject the fixed initial latent from ``HarnessConfig.fixed_latent_path``
    into each ``_run_pipe`` call so two runs of the same prompt are
    bit-identical (a requirement for the MLPerf accuracy check)."""

    flow_shift: float = CHECKPOINT_FLOW_SHIFT
    """Scheduler ``flow_shift``. Defaults to the Diffusers checkpoint value
    (same as the MLPerf reference ``from_pretrained`` path)."""


@dataclass
class CompileConfig:
    use_torch_compile: bool = True
    """Enable ``torch.compile`` on the DiT transformer(s). First call pays a
    multi-minute compilation cost; subsequent same-shape calls are fast.

    The :class:`WarmupConfig` below is responsible for absorbing that
    one-time cost *before* LoadGen starts measuring."""


@dataclass
class WarmupConfig:
    """Configurable warmup pass that runs before LoadGen ``StartTest``.

    The warmup dispatches one or more synthetic prompts at the configured
    generation shape so that:

    * ``torch.compile`` compiles its shape-specialised graphs.
    * AITER (or whatever attention backend is selected) emits and caches
      its first-call HSACO kernels.
    * Diffusers' CUDA-graph capture (if enabled) records its replay.

    Without this, the first measured query pays all of the above and skews
    the throughput / per-query latency numbers.
    """

    enabled: bool = True

    num_prompts: int = 1
    """Per-rank warmup count. The dispatcher multiplies this by its
    topology-dependent factor (1 for Ulysses lockstep, ``world_size`` for
    data-parallel) so every rank actually executes ``num_prompts`` warmups.
    One is usually enough for compile + kernel caches; bump to 2 if you
    want to also warm the steady-state graphs."""

    prompt: str = (
        "A peaceful mountain landscape with flowing waterfalls under a clear blue sky."
    )
    """Synthetic prompt used for every warmup pass. Kept identical so
    shape-specialised compile caches hit on the first measured prompt."""


@dataclass
class WanBackendConfig:
    """Per-scenario configuration for the real Wan 2.2 backend.

    Construct one of these per LoadGen test. The dispatcher reads
    ``parallelism.mode`` to decide its topology; the backend reads
    ``model`` + ``vae`` + ``compile`` + ``xfuser_extra`` to build the
    underlying ``xFuserArgs``.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    vae: VaeConfig = field(default_factory=VaeConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    xfuser_extra: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments forwarded verbatim to ``xFuserArgs(...)``."""

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------
    # Validation.
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        p = self.parallelism
        if p.mode not in PARALLELISM_MODES:
            raise ValueError(
                f"parallelism.mode must be one of {PARALLELISM_MODES!r}, "
                f"got {p.mode!r}"
            )
        for name in (
            "ulysses_degree",
            "ring_degree",
            "tensor_parallel_degree",
            "pipefusion_parallel_degree",
            "data_parallel_degree",
            "data_parallel_workers",
        ):
            value = getattr(p, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"parallelism.{name} must be a positive int, got {value!r}"
                )
        if p.mode == "data_parallel" and p.data_parallel_workers < 2:
            raise ValueError(
                "parallelism.data_parallel_workers must be >= 2 when "
                "parallelism.mode == 'data_parallel'"
            )

        if p.dispatch not in DP_DISPATCH_MODES:
            raise ValueError(
                f"parallelism.dispatch must be one of {DP_DISPATCH_MODES!r}, "
                f"got {p.dispatch!r}"
            )
        if p.result_transport not in ("shm", "gloo"):
            raise ValueError(
                f"parallelism.result_transport must be 'shm' or 'gloo', "
                f"got {p.result_transport!r}"
            )
        if p.dispatch != "wave" and p.mode != "data_parallel":
            raise ValueError(
                f"parallelism.dispatch={p.dispatch!r} is only valid when "
                f"parallelism.mode == 'data_parallel'; got mode={p.mode!r}"
            )

        if self.vae.parallel_size < 0:
            raise ValueError(
                f"vae.parallel_size must be >= 0, got {self.vae.parallel_size!r}"
            )

        if self.warmup.num_prompts < 0:
            raise ValueError(
                f"warmup.num_prompts must be >= 0, got {self.warmup.num_prompts!r}"
            )
        if self.warmup.enabled and not self.warmup.prompt:
            raise ValueError("warmup.prompt must be a non-empty string when warmup.enabled")

    # ------------------------------------------------------------------
    # Topology helpers used by the dispatcher and runner.
    # ------------------------------------------------------------------
    @property
    def expected_world_size(self) -> int:
        """Return the expected ``torchrun --nproc-per-node`` value."""
        p = self.parallelism
        if p.mode == "ulysses":
            dit = (
                p.ulysses_degree
                * p.ring_degree
                * p.tensor_parallel_degree
                * p.pipefusion_parallel_degree
                * p.data_parallel_degree
            )
            return dit + self.vae.parallel_size
        # data_parallel mode: our custom dispatcher owns the topology.
        return p.data_parallel_workers


# ----------------------------------------------------------------------
# Loaders.
# ----------------------------------------------------------------------


def _coerce_section(section_cls, raw: Any):
    """Build a section dataclass from a dict, ignoring unknown keys.

    Unknown keys are not silently dropped: they raise so a typo in the
    YAML is caught early.
    """
    if raw is None:
        return section_cls()
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"expected mapping for section {section_cls.__name__}, got {type(raw).__name__}"
        )
    valid = {f.name for f in section_cls.__dataclass_fields__.values()}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"unknown keys in {section_cls.__name__}: {sorted(unknown)!r}"
        )
    return section_cls(**dict(raw))


def load_wan_backend_config(path: Path | str) -> WanBackendConfig:
    """Parse a per-scenario ``WanBackendConfig`` YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WanBackendConfig YAML not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: expected a mapping at the top level")

    valid_sections = {
        "model", "parallelism", "vae", "generation", "compile", "warmup",
        "xfuser_extra",
    }
    unknown = set(raw) - valid_sections
    if unknown:
        raise ValueError(
            f"{path}: unknown top-level sections: {sorted(unknown)!r}; "
            f"valid sections are {sorted(valid_sections)!r}"
        )

    extra = raw.get("xfuser_extra") or {}
    if not isinstance(extra, Mapping):
        raise TypeError(f"{path}: xfuser_extra must be a mapping")

    return WanBackendConfig(
        model=_coerce_section(ModelConfig, raw.get("model")),
        parallelism=_coerce_section(ParallelismConfig, raw.get("parallelism")),
        vae=_coerce_section(VaeConfig, raw.get("vae")),
        generation=_coerce_section(GenerationConfig, raw.get("generation")),
        compile=_coerce_section(CompileConfig, raw.get("compile")),
        warmup=_coerce_section(WarmupConfig, raw.get("warmup")),
        xfuser_extra=dict(extra),
    )


def default_config_path_for_scenario(scenario: str, repo_root: Path) -> Path:
    """Return the conventional YAML path for ``scenario``."""
    return repo_root / "configs" / "wan22" / f"{scenario}.yaml"
