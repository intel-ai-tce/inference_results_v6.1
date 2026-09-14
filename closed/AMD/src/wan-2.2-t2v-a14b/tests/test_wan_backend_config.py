"""Tests for :mod:`wan_harness.backends.wan22_config`.

These tests run on CPU and do not touch xfuser or torch – the config
layer is intentionally side-effect free.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from wan_harness.backends.wan22_config import (
    PARALLELISM_MODES,
    WanBackendConfig,
    default_config_path_for_scenario,
    load_wan_backend_config,
)
from wan_harness.config import (
    REPO_ROOT,
    HarnessConfig,
    resolve_backend_config_path,
)


# ----------------------------------------------------------------------
# Schema-level tests.
# ----------------------------------------------------------------------


def test_defaults_validate() -> None:
    cfg = WanBackendConfig()
    assert cfg.parallelism.mode == "ulysses"
    assert cfg.parallelism.ulysses_degree == 1
    assert cfg.expected_world_size == 1
    # torch.compile is on by default (warmup absorbs the cost).
    assert cfg.compile.use_torch_compile is True
    # Warmup is on by default with a non-trivial synthetic prompt.
    assert cfg.warmup.enabled is True
    assert cfg.warmup.num_prompts == 1
    assert cfg.warmup.prompt


def test_unknown_parallelism_mode_rejected() -> None:
    from wan_harness.backends.wan22_config import ParallelismConfig
    with pytest.raises(ValueError):
        WanBackendConfig(parallelism=ParallelismConfig(mode="garbage"))


def test_negative_vae_size_rejected() -> None:
    from wan_harness.backends.wan22_config import VaeConfig
    with pytest.raises(ValueError):
        WanBackendConfig(vae=VaeConfig(parallel_size=-1))


def test_data_parallel_requires_two_workers() -> None:
    from wan_harness.backends.wan22_config import ParallelismConfig
    with pytest.raises(ValueError):
        WanBackendConfig(
            parallelism=ParallelismConfig(mode="data_parallel", data_parallel_workers=1)
        )


def test_modes_constant() -> None:
    assert PARALLELISM_MODES == ("ulysses", "data_parallel")


def test_dispatch_modes_constant() -> None:
    from wan_harness.backends.wan22_config import DP_DISPATCH_MODES
    assert DP_DISPATCH_MODES == ("wave", "async")


def test_default_dispatch_is_wave() -> None:
    """Existing YAMLs (which omit ``dispatch``) must keep the wave path."""
    from wan_harness.backends.wan22_config import ParallelismConfig
    cfg = WanBackendConfig(
        parallelism=ParallelismConfig(mode="data_parallel", data_parallel_workers=4)
    )
    assert cfg.parallelism.dispatch == "wave"


def test_dispatch_async_accepted_under_data_parallel() -> None:
    from wan_harness.backends.wan22_config import ParallelismConfig
    cfg = WanBackendConfig(
        parallelism=ParallelismConfig(
            mode="data_parallel",
            data_parallel_workers=4,
            dispatch="async",
        )
    )
    assert cfg.parallelism.dispatch == "async"


def test_dispatch_unknown_value_rejected() -> None:
    from wan_harness.backends.wan22_config import ParallelismConfig
    with pytest.raises(ValueError, match="parallelism.dispatch"):
        WanBackendConfig(
            parallelism=ParallelismConfig(
                mode="data_parallel",
                data_parallel_workers=4,
                dispatch="bogus",
            )
        )


def test_dispatch_async_rejected_under_ulysses_mode() -> None:
    """The sub-knob is only meaningful when DP owns the topology."""
    from wan_harness.backends.wan22_config import ParallelismConfig
    with pytest.raises(ValueError, match="only valid when"):
        WanBackendConfig(
            parallelism=ParallelismConfig(
                mode="ulysses",
                ulysses_degree=4,
                dispatch="async",
            )
        )


def test_dispatch_yaml_roundtrip(tmp_path: Path) -> None:
    p = _write_yaml(
        tmp_path / "ok.yaml",
        """
        parallelism:
          mode: data_parallel
          data_parallel_workers: 4
          dispatch: async
        """,
    )
    cfg = load_wan_backend_config(p)
    assert cfg.parallelism.mode == "data_parallel"
    assert cfg.parallelism.dispatch == "async"


# ----------------------------------------------------------------------
# YAML round-trip.
# ----------------------------------------------------------------------


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_load_singlestream_yaml() -> None:
    path = REPO_ROOT / "configs" / "wan22" / "SingleStream.yaml"
    cfg = load_wan_backend_config(path)
    assert cfg.parallelism.mode == "ulysses"
    assert cfg.parallelism.ulysses_degree == 8
    assert cfg.vae.use_parallel is True
    assert cfg.vae.parallel_size == 0
    assert cfg.expected_world_size == 8
    # SingleStream ships with compile + warmup enabled in the YAML.
    assert cfg.compile.use_torch_compile is True
    assert cfg.warmup.enabled is True
    assert cfg.warmup.num_prompts >= 1
    assert cfg.generation.flow_shift == 3.0


def test_load_offline_yaml() -> None:
    path = REPO_ROOT / "configs" / "wan22" / "Offline.yaml"
    cfg = load_wan_backend_config(path)
    assert cfg.parallelism.mode == "data_parallel"
    assert cfg.parallelism.data_parallel_workers == 8
    assert cfg.vae.use_parallel is False
    assert cfg.expected_world_size == 8
    # Offline ships with compile + warmup enabled in the YAML.
    assert cfg.compile.use_torch_compile is True
    assert cfg.warmup.enabled is True
    assert cfg.warmup.num_prompts >= 1
    assert cfg.generation.flow_shift == 3.0


_FP8_ATTN_ALIASES = frozenset({"AITER_FP8", "aiter_fp8"})
_LOW_ATTN_ALIASES = frozenset({
    "AITER_SPARGE_ASM_V2",
    "aiter_sparge_asm_v2",
    "AITER_MXFP4",
    "aiter_mxfp4",
})


def _schedule_tokens(schedule: str) -> list[str]:
    return [token.strip().upper() for token in schedule.split(",") if token.strip()]


def _schedule_uses_fp8_and_low_precision(schedule: str) -> bool:
    """True when an explicit per-step schedule mentions both recipe families."""
    tokens = _schedule_tokens(schedule)
    fp8 = {alias.upper() for alias in _FP8_ATTN_ALIASES}
    low = {alias.upper() for alias in _LOW_ATTN_ALIASES}
    return bool(tokens) and any(token in fp8 for token in tokens) and any(
        token in low for token in tokens
    )


def test_shipped_yamls_carry_attention_recipe() -> None:
    """Shipped YAMLs must opt into AITER attention + MXFP4 GEMMs.

    Attention tuning changes often (step counts, explicit per-step
    schedules, backend renames), so this test only guards against
    accidentally dropping the whole recipe – not against specific tuning
    choices.

    Accepts either:
    - a dense ``attention_backend`` FP8 recipe, or
    - hybrid mode via legacy backend-pair knobs *or* an explicit
      ``hybrid_attn_schedule`` string.
    """
    for scenario in ("SingleStream", "Offline"):
        cfg = load_wan_backend_config(
            REPO_ROOT / "configs" / "wan22" / f"{scenario}.yaml"
        )
        xf = cfg.xfuser_extra
        if xf.get("use_hybrid_attn_schedule"):
            assert xf.get("attention_backend") is None, (
                f"{scenario}.yaml: hybrid mode must not set attention_backend"
            )
            explicit_schedule = xf.get("hybrid_attn_schedule")
            if explicit_schedule:
                assert xf.get("hybrid_attn_high_precision_backend") is None, (
                    f"{scenario}.yaml: explicit hybrid_attn_schedule must not "
                    "also set hybrid_attn_high_precision_backend"
                )
                assert xf.get("hybrid_attn_low_precision_backend") is None, (
                    f"{scenario}.yaml: explicit hybrid_attn_schedule must not "
                    "also set hybrid_attn_low_precision_backend"
                )
                assert _schedule_uses_fp8_and_low_precision(explicit_schedule), (
                    f"{scenario}.yaml: hybrid_attn_schedule must mention at "
                    f"least one FP8 and one low-precision backend "
                    f"({sorted(_FP8_ATTN_ALIASES)} + {sorted(_LOW_ATTN_ALIASES)})"
                )
            else:
                assert xf.get("hybrid_attn_high_precision_backend") in _FP8_ATTN_ALIASES, (
                    f"{scenario}.yaml: hybrid high-precision backend must be FP8 "
                    f"({sorted(_FP8_ATTN_ALIASES)}), got "
                    f"{xf.get('hybrid_attn_high_precision_backend')!r}"
                )
                assert xf.get("hybrid_attn_low_precision_backend") in _LOW_ATTN_ALIASES, (
                    f"{scenario}.yaml: hybrid low-precision backend must be a known "
                    f"mxfp4-class AITER name ({sorted(_LOW_ATTN_ALIASES)}), got "
                    f"{xf.get('hybrid_attn_low_precision_backend')!r}"
                )
        else:
            ab = xf.get("attention_backend")
            assert ab in _FP8_ATTN_ALIASES, (
                f"{scenario}.yaml: xfuser_extra.attention_backend must be a known "
                f"dense-AITER FP8 recipe ({sorted(_FP8_ATTN_ALIASES)}), got {ab!r}"
            )
        assert xf.get("cross_attention_backend") == "aiter"
        assert xf.get("use_fp4_gemms") is True


def test_hybrid_attn_schedule_recipe_helper() -> None:
    assert _schedule_uses_fp8_and_low_precision(
        "AITER_MXFP4,AITER_FP8,AITER_MXFP4"
    )
    assert not _schedule_uses_fp8_and_low_precision("AITER_FP8,AITER_FP8")
    assert not _schedule_uses_fp8_and_low_precision("")


def test_xfuser_extra_accepts_explicit_hybrid_attn_schedule(tmp_path: Path) -> None:
    p = _write_yaml(
        tmp_path / "ok.yaml",
        """
        xfuser_extra:
          use_hybrid_attn_schedule: true
          num_hybrid_attn_high_precision_steps: 0
          hybrid_attn_schedule: "AITER_MXFP4,AITER_FP8"
        """,
    )
    cfg = load_wan_backend_config(p)
    assert cfg.xfuser_extra["hybrid_attn_schedule"] == "AITER_MXFP4,AITER_FP8"
    assert cfg.xfuser_extra.get("hybrid_attn_high_precision_backend") is None


def test_hybrid_step_counts_forwarded_when_in_xfuser_extra() -> None:
    """If the user enables the hybrid schedule via ``xfuser_extra``, the
    per-call step-count knobs must reach the input_args dict. xfuser's
    runner reads them from there via ``__getitem__``, not from
    ``self.config``, so missing the forward would KeyError at setup.
    """
    from wan_harness.backends.wan22 import WanBackend
    from wan_harness.backends.wan22_config import WanBackendConfig

    cfg = HarnessConfig(backend="wan22", scenario="SingleStream")
    backend = WanBackend(cfg)
    backend._bcfg = WanBackendConfig(  # noqa: SLF001
        xfuser_extra={
            "use_hybrid_attn_schedule": True,
            "hybrid_attn_low_precision_backend": "AITER_SAGE_V2",
            "hybrid_attn_high_precision_backend": "AITER_SAGE",
            "num_hybrid_attn_high_precision_steps": 5,
            "num_hybrid_gemm_high_precision_steps": 3,
        }
    )
    args = backend._build_input_args(prompt="x")  # noqa: SLF001
    assert args["num_hybrid_attn_high_precision_steps"] == 5
    assert args["num_hybrid_gemm_high_precision_steps"] == 3


def test_hybrid_step_counts_absent_in_default_recipe() -> None:
    """When ``xfuser_extra`` omits hybrid step keys, ``input_args`` must not
    add them – they're only forwarded when present in the backend YAML.
    """
    from wan_harness.backends.wan22 import WanBackend
    from wan_harness.backends.wan22_config import WanBackendConfig

    cfg = HarnessConfig(backend="wan22", scenario="SingleStream")
    backend = WanBackend(cfg)
    backend._bcfg = WanBackendConfig(  # noqa: SLF001
        xfuser_extra={"attention_backend": "aiter_fp8"}
    )
    args = backend._build_input_args(prompt="x")  # noqa: SLF001
    assert args["flow_shift"] == 3.0
    assert "num_hybrid_attn_high_precision_steps" not in args
    assert "num_hybrid_gemm_high_precision_steps" not in args


def test_warmup_section_yaml(tmp_path: Path) -> None:
    p = _write_yaml(
        tmp_path / "ok.yaml",
        """
        warmup:
          enabled: false
          num_prompts: 3
          prompt: "custom warmup"
        """,
    )
    cfg = load_wan_backend_config(p)
    assert cfg.warmup.enabled is False
    assert cfg.warmup.num_prompts == 3
    assert cfg.warmup.prompt == "custom warmup"


def test_warmup_rejects_negative_num_prompts() -> None:
    from wan_harness.backends.wan22_config import WarmupConfig
    with pytest.raises(ValueError):
        WanBackendConfig(warmup=WarmupConfig(num_prompts=-1))


def test_warmup_rejects_empty_prompt_when_enabled() -> None:
    from wan_harness.backends.wan22_config import WarmupConfig
    with pytest.raises(ValueError):
        WanBackendConfig(warmup=WarmupConfig(enabled=True, prompt=""))


def test_load_unknown_top_level_section_rejected(tmp_path: Path) -> None:
    p = _write_yaml(
        tmp_path / "bad.yaml",
        """
        model:
          path: Wan-AI/Wan2.2-T2V-A14B-Diffusers
        weird_top_level: 1
        """,
    )
    with pytest.raises(ValueError):
        load_wan_backend_config(p)


def test_load_unknown_section_key_rejected(tmp_path: Path) -> None:
    p = _write_yaml(
        tmp_path / "bad.yaml",
        """
        parallelism:
          mode: ulysses
          mystery_field: 42
        """,
    )
    with pytest.raises(ValueError):
        load_wan_backend_config(p)


def test_xfuser_extra_passthrough(tmp_path: Path) -> None:
    p = _write_yaml(
        tmp_path / "ok.yaml",
        """
        parallelism:
          mode: ulysses
          ulysses_degree: 2
        xfuser_extra:
          attention_backend: sage_dense
          use_fp8_gemms: true
        """,
    )
    cfg = load_wan_backend_config(p)
    assert cfg.xfuser_extra == {
        "attention_backend": "sage_dense",
        "use_fp8_gemms": True,
    }


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_wan_backend_config(tmp_path / "does_not_exist.yaml")


# ----------------------------------------------------------------------
# expected_world_size derivation.
# ----------------------------------------------------------------------


def test_ulysses_world_size_with_vae_split() -> None:
    from wan_harness.backends.wan22_config import (
        ParallelismConfig,
        VaeConfig,
    )
    cfg = WanBackendConfig(
        parallelism=ParallelismConfig(
            mode="ulysses",
            ulysses_degree=4,
        ),
        vae=VaeConfig(parallel_size=2),
    )
    # dit=4 + vae=2 == 6
    assert cfg.expected_world_size == 6


def test_data_parallel_world_size() -> None:
    from wan_harness.backends.wan22_config import ParallelismConfig
    cfg = WanBackendConfig(
        parallelism=ParallelismConfig(mode="data_parallel", data_parallel_workers=4)
    )
    assert cfg.expected_world_size == 4


# ----------------------------------------------------------------------
# HarnessConfig wiring.
# ----------------------------------------------------------------------


def test_resolve_backend_config_path_for_wan22() -> None:
    cfg = HarnessConfig(backend="wan22", scenario="Offline")
    p = resolve_backend_config_path(cfg)
    assert p is not None
    assert p.name == "Offline.yaml"
    assert p.parent.name == "wan22"


def test_resolve_backend_config_path_for_mock() -> None:
    cfg = HarnessConfig(backend="mock")
    assert resolve_backend_config_path(cfg) is None


def test_resolve_explicit_backend_config_path(tmp_path: Path) -> None:
    cfg = HarnessConfig(
        backend="wan22",
        scenario="SingleStream",
        backend_config_path=tmp_path / "custom.yaml",
    )
    p = resolve_backend_config_path(cfg)
    assert p == tmp_path / "custom.yaml"


def test_default_config_path_for_scenario() -> None:
    p = default_config_path_for_scenario("SingleStream", REPO_ROOT)
    assert p.name == "SingleStream.yaml"
    assert p.parent.name == "wan22"
