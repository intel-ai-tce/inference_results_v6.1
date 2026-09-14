"""Real Wan 2.2 T2V-A14B backend, driven by xfuser's MODEL_REGISTRY.

The class is registered under ``--backend wan22``. Construction is cheap
(read the per-scenario YAML and validate); the heavy work happens in
:meth:`setup`:

  1. Initialise ``torch.distributed`` via xfuser's helper (xfuser respects
     ``torchrun`` env vars).
  2. Construct an ``xFuserArgs`` from the :class:`WanBackendConfig` and
     ``HarnessConfig``.
  3. Look up the registered class for ``Wan-AI/Wan2.2-T2V-A14B-Diffusers``
     in ``xfuser.model_executor.models.runner_models.base_model.MODEL_REGISTRY``.
  4. Instantiate the class and call ``.initialize(seed_input_args)`` once.
     This is what loads the two transformers, the VAE, applies the
     Ulysses / DistVAE wrappers, and runs ``torch.compile`` warmup if
     enabled.
  5. (Optional) Load the fixed initial latent from
     ``HarnessConfig.fixed_latent_path`` so MLPerf accuracy comparisons
     are bit-identical across runs.

The dispatcher calls :meth:`build_work_unit` to turn (prompt, sample
index) pairs into wire-friendly :class:`WorkUnit` objects, then calls
:meth:`run_unit` on every rank to drive one diffusion. This split keeps
the dispatcher backend-agnostic – it does not need to know about
``HarnessConfig`` or xfuser.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..config import REPO_ROOT, resolve_backend_config_path
from ..video_encoder import (
    VideoEncoderError,
    encode_frames_to_mp4,
    ffmpeg_available,
)
from ..wire import WorkUnit
from .base import Backend, BackendBuildError, GeneratedVideo
from .wan22_config import (
    WanBackendConfig,
    default_config_path_for_scenario,
    load_wan_backend_config,
)

_log = logging.getLogger(__name__)

__all__ = ["WanBackend"]


class WanBackend(Backend):
    """xfuser-driven backend for the Wan 2.2 T2V-A14B model.

    The dispatcher (one of :class:`UlyssesDispatcher` or
    :class:`WaveDispatcher`) owns the rank topology; this class owns
    the model.
    """

    name = "wan22"

    def __init__(self, config) -> None:  # type: ignore[override]
        super().__init__(config)
        self._bcfg = _load_backend_config(config)
        # Populated in setup():
        self._rank = 0
        self._world_size = 1
        self._device: str | None = None
        self._model: Any = None
        self._fixed_latent: Any = None

    # ------------------------------------------------------------------
    # Public surface used by the dispatcher.
    # ------------------------------------------------------------------
    @property
    def backend_config(self) -> WanBackendConfig:
        """The fully-resolved :class:`WanBackendConfig` for this run."""
        return self._bcfg

    @property
    def distributed_device(self) -> str | None:
        """Device string the dispatcher should use for ``torch.distributed``
        send/recv tensors (typically ``"cuda:<local_rank>"`` once setup is
        done; ``None`` before setup or in single-process mode).
        """
        return self._device

    def warmup_settings(self) -> tuple[int, str] | None:
        """Expose the YAML ``warmup`` section to :mod:`loadgen_runner`.

        Returns ``None`` when warmup is disabled or num_prompts==0 so the
        runner can skip the pass entirely.
        """
        w = self._bcfg.warmup
        if not w.enabled or w.num_prompts <= 0:
            return None
        return (int(w.num_prompts), w.prompt)

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------
    def setup(self, *, rank: int = 0, world_size: int = 1) -> None:
        if self._is_set_up:
            return
        self._rank = rank
        self._world_size = world_size

        expected_world = self._bcfg.expected_world_size
        if world_size != expected_world:
            raise BackendBuildError(
                f"WanBackend ({self._bcfg.parallelism.mode}): world_size={world_size} "
                f"does not match the per-scenario YAML expected_world_size={expected_world}. "
                f"Check parallelism.* / vae.parallel_size in "
                f"{resolve_backend_config_path(self._config)}."
            )

        # Accuracy mode encodes every sample to MP4 for the downstream
        # VBench evaluator. We require ffmpeg up front (rather than at
        # first sample) so a missing binary fails the run during setup,
        # before any GPU work has been done.
        if self._config.mode == "accuracy" and not ffmpeg_available():
            raise BackendBuildError(
                "WanBackend(mode=accuracy): ffmpeg binary not found on PATH. "
                "Install ffmpeg (apt-get install -y ffmpeg) or run with "
                "--mode performance."
            )

        # Heavy imports kept inside setup so --print-config and the Mock
        # dry-run never pull in torch / xfuser.
        import torch  # noqa: WPS433 (intentional local import)

        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            # We want a distinct GPU per rank. If the visible-device set is
            # smaller than the world (e.g. ROCR_VISIBLE_DEVICES masks too
            # many devices), warn loudly so the user doesn't silently get
            # two ranks racing on one GPU – xfuser would also wrap via
            # `rank % device_count()`.
            if world_size > device_count:
                _log.warning(
                    "WanBackend: world_size=%d but only %d CUDA/ROCm devices visible; "
                    "ranks will share GPUs (rank %% device_count). Check "
                    "CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES.",
                    world_size, device_count,
                )
            if local_rank >= device_count:
                _log.warning(
                    "WanBackend: local_rank=%d >= device_count=%d; wrapping to %d",
                    local_rank, device_count, local_rank % device_count,
                )
                local_rank = local_rank % device_count
            torch.cuda.set_device(local_rank)
            self._device = f"cuda:{local_rank}"
            dev_props = torch.cuda.get_device_properties(local_rank)
            dev_name = dev_props.name
        else:
            self._device = "cpu"
            dev_name = "cpu"
        _log.info(
            "WanBackend.setup: rank=%d local_rank=%d world_size=%d device=%s (%s)",
            rank, local_rank, world_size, self._device, dev_name,
        )

        xfuser_args = self._build_xfuser_args()
        seed_input_args = self._build_input_args(prompt="warmup")

        self._model = _instantiate_xfuser_model(xfuser_args)
        self._model.initialize(seed_input_args)

        if self._bcfg.generation.use_fixed_latent:
            self._fixed_latent = self._maybe_load_fixed_latent()

        self._is_set_up = True
        _log.info(
            "WanBackend ready (mode=%s, ulysses=%d, dp=%d, vae_parallel=%s)",
            self._bcfg.parallelism.mode,
            self._bcfg.parallelism.ulysses_degree,
            self._bcfg.parallelism.data_parallel_workers,
            self._bcfg.vae.use_parallel,
        )

    def teardown(self) -> None:
        if not self._is_set_up:
            return
        try:
            import torch.distributed as dist  # noqa: WPS433
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as exc:  # pragma: no cover - best effort
            _log.warning("WanBackend.teardown: dist destroy failed: %s", exc)
        self._model = None
        self._fixed_latent = None
        self._is_set_up = False

    # ------------------------------------------------------------------
    # Generation entry points.
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: Sequence[str],
        indices: Sequence[int],
    ) -> Iterator[GeneratedVideo]:
        """Single-process generate path.

        Only used when the harness is forced into single-rank mode (e.g.
        a smoke test on a 1-GPU box with ``ulysses_degree=1``). The
        multi-rank dispatchers call :meth:`run_unit` directly.
        """
        if not self._is_set_up:
            raise RuntimeError("WanBackend.generate called before setup()")
        for prompt, idx in zip(prompts, indices):
            unit = self.build_work_unit(prompt=prompt, sample_index=int(idx))
            yield self.run_unit(unit)

    def build_work_unit(self, *, prompt: str, sample_index: int) -> WorkUnit:
        """Pack ``(prompt, index)`` into a :class:`WorkUnit` for the dispatcher."""
        return WorkUnit(
            sample_index=int(sample_index),
            prompt=prompt,
            input_args=self._build_input_args(prompt=prompt),
        )

    def run_unit(self, unit: WorkUnit) -> GeneratedVideo:
        """Run one diffusion step. Called by the dispatcher on every rank.

        In Ulysses mode the output is the same on every rank (the
        sequence-parallel collective ops gather to all ranks); the
        dispatcher uses rank 0's copy and discards the others. In DP
        mode each worker rank produces its own canonical output and
        sends it back to rank 0 over the wire.
        """
        if not self._is_set_up:
            raise RuntimeError("WanBackend.run_unit called before setup()")

        import torch  # noqa: WPS433

        args = dict(unit.input_args)
        pipe = self._model.pipe

        gen_seed = int(args.pop("seed", self._config.seed))
        device = (
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        generator = torch.Generator(device=device).manual_seed(gen_seed)

        pipe_kwargs: dict[str, Any] = {
            "prompt": unit.prompt,
            "negative_prompt": args.get("negative_prompt") or None,
            "height": args["height"],
            "width": args["width"],
            "num_frames": args["num_frames"],
            "num_inference_steps": args["num_inference_steps"],
            "guidance_scale": args["guidance_scale"],
            "generator": generator,
            "output_type": "np",
        }
        if args.get("guidance_scale_2") is not None:
            pipe_kwargs["guidance_scale_2"] = args["guidance_scale_2"]
        if self._fixed_latent is not None:
            pipe_kwargs["latents"] = self._fixed_latent

        # Time the diffusion + VAE-decode call. ``pipe(...)`` returns
        # ``output_type="np"`` which forces a CPU sync inside diffusers,
        # so ``perf_counter`` around the call captures the real
        # GPU-resident wall time without an extra cuda.synchronize().
        t_start = time.perf_counter()
        output = pipe(**pipe_kwargs)
        frames = _first_video(output)
        t_pipe = time.perf_counter() - t_start

        frame_count, height, width = _video_shape(frames)
        frames_uint8 = _frames_as_uint8(frames)
        frames_bytes = frames_uint8.tobytes()

        mp4_bytes: bytes | None = None
        t_encode = 0.0
        if self._config.mode == "accuracy":
            t_enc_start = time.perf_counter()
            try:
                mp4_bytes = encode_frames_to_mp4(
                    frames_uint8, fps=int(self._config.fps)
                )
            except VideoEncoderError:
                # Fail loudly: the artefact writer would otherwise silently
                # fall back to .bin and produce a directory VBench can't read.
                raise
            t_encode = time.perf_counter() - t_enc_start

        t_total = time.perf_counter() - t_start

        _log.info(
            "run_unit rank=%d sample=%d size=%dx%d frames=%d steps=%d "
            "pipe=%.3fs%s total=%.3fs",
            self._rank,
            int(unit.sample_index),
            int(height),
            int(width),
            int(frame_count),
            int(pipe_kwargs["num_inference_steps"]),
            t_pipe,
            f" encode={t_encode:.3f}s" if mp4_bytes is not None else "",
            t_total,
        )

        return GeneratedVideo(
            sample_index=int(unit.sample_index),
            frames_bytes=frames_bytes,
            frame_count=int(frame_count),
            height=int(height),
            width=int(width),
            mp4_bytes=mp4_bytes,
        )

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------
    def _build_xfuser_args(self):
        """Translate :class:`WanBackendConfig` + :class:`HarnessConfig`
        into a ``xfuser.xFuserArgs`` instance.
        """
        from xfuser import xFuserArgs  # noqa: WPS433

        bcfg = self._bcfg
        cfg = self._config
        p = bcfg.parallelism

        if p.mode == "ulysses":
            data_parallel_degree = p.data_parallel_degree
        else:
            # Custom DP: we own per-wave routing via WaveDispatcher,
            # but xfuser still needs `dit_parallel_size + vae_parallel_size`
            # to equal the actual torch.distributed world_size or its
            # ParallelConfig assertion fires. Telling xfuser the world is
            # data-parallel of size `workers` satisfies the check while
            # keeping each rank functionally a single-rank instance
            # (ulysses=ring=tp=pp=1, no intra-group collectives during
            # generation).
            data_parallel_degree = p.data_parallel_workers

        kwargs: dict[str, Any] = {
            "model": bcfg.model.path,
            "use_torch_compile": bcfg.compile.use_torch_compile,
            "use_cfg_parallel": p.use_cfg_parallel,
            "ulysses_degree": p.ulysses_degree,
            "ring_degree": p.ring_degree,
            "tensor_parallel_degree": p.tensor_parallel_degree,
            "pipefusion_parallel_degree": p.pipefusion_parallel_degree,
            "data_parallel_degree": data_parallel_degree,
            "vae_parallel_size": bcfg.vae.parallel_size,
            "use_parallel_vae": bcfg.vae.use_parallel,
            "enable_tiling": bcfg.vae.enable_tiling,
            "enable_slicing": bcfg.vae.enable_slicing,
            # Generation defaults (used by xfuser warmup; the per-call
            # input_args we pass into _run_pipe overrides these anyway).
            "height": cfg.height,
            "width": cfg.width,
            "num_frames": cfg.num_frames,
            "num_inference_steps": cfg.sample_steps,
            "guidance_scale": cfg.guidance_scale,
            "guidance_scale_2": cfg.guidance_scale_2,
            "flow_shift": bcfg.generation.flow_shift,
            "seed": cfg.seed,
            "negative_prompt": cfg.negative_prompt,
            # NOTE: `output_type` here is the engine-config default and
            # xfuser only accepts {"pil", "latent"} on this surface.
            # The per-call np buffer is requested directly on `pipe(...)`
            # via pipe_kwargs in `run_unit` instead.
        }
        # Splat any unmapped knobs (sage / fp8 / fp4 / attention backend / ...).
        kwargs.update(bcfg.xfuser_extra)
        return xFuserArgs(**kwargs)

    def _build_input_args(self, *, prompt: str) -> dict[str, Any]:
        cfg = self._config
        bcfg = self._bcfg
        args: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": cfg.negative_prompt or None,
            "height": int(cfg.height),
            "width": int(cfg.width),
            "num_frames": int(cfg.num_frames),
            "num_inference_steps": int(cfg.sample_steps),
            "guidance_scale": float(cfg.guidance_scale),
            "guidance_scale_2": (
                float(cfg.guidance_scale_2) if cfg.guidance_scale_2 is not None else None
            ),
            "flow_shift": float(bcfg.generation.flow_shift),
            "seed": int(cfg.seed),
        }
        # Future-proofing: forward the hybrid-schedule step counts when the
        # user enables the schedule via ``xfuser_extra``. xfuser is
        # split-brained about these knobs – they are declared as
        # ``xFuserArgs`` CLI fields (so they pass validation when set under
        # ``xfuser_extra``), but the runner reads them from ``input_args[...]``
        # via ``__getitem__`` in ``_setup_hybrid_attn_schedule`` /
        # ``_setup_hybrid_gemm_schedule``. Without this forwarding the
        # runner would KeyError as soon as the schedule is turned on.
        # The guard makes this a no-op for the default non-hybrid recipe.
        for k in (
            "num_hybrid_attn_high_precision_steps",
            "num_hybrid_gemm_high_precision_steps",
        ):
            if k in bcfg.xfuser_extra:
                args[k] = bcfg.xfuser_extra[k]
        return args

    def _maybe_load_fixed_latent(self):
        """Load the fixed-latent tensor from disk, if present.

        Returns ``None`` (and logs a warning) when the file does not
        exist – this keeps in-container smoke tests possible before
        ``data/fixed_latent.pt`` has been fetched.
        """
        path = self._config.fixed_latent_path
        if path is None or not Path(path).exists():
            _log.warning(
                "fixed_latent_path %s missing; running without fixed latent",
                path,
            )
            return None

        import torch  # noqa: WPS433

        tensor = torch.load(path, map_location=self._device or "cpu")
        _log.info("loaded fixed latent from %s (shape=%s)", path, tuple(tensor.shape))
        return tensor


# ----------------------------------------------------------------------
# Module-private helpers.
# ----------------------------------------------------------------------


def _load_backend_config(harness_config) -> WanBackendConfig:
    """Resolve the YAML path then parse it."""
    path = resolve_backend_config_path(harness_config)
    if path is None or not Path(path).exists():
        # Default by scenario when nothing was specified.
        path = default_config_path_for_scenario(
            harness_config.scenario, REPO_ROOT
        )
    if not Path(path).exists():
        raise BackendBuildError(
            f"WanBackend config YAML not found: {path}. "
            f"Pass --backend-config PATH or create configs/wan22/{harness_config.scenario}.yaml."
        )
    return load_wan_backend_config(path)


def _instantiate_xfuser_model(xfuser_args):
    """Look up the registered class and instantiate it."""
    from xfuser.model_executor.models.runner_models.base_model import (  # noqa: WPS433
        MODEL_REGISTRY,
    )

    key = xfuser_args.model
    if key not in MODEL_REGISTRY:
        raise BackendBuildError(
            f"xfuser MODEL_REGISTRY has no entry for {key!r}. "
            f"Available keys: {sorted(MODEL_REGISTRY)!r}"
        )
    cls = MODEL_REGISTRY[key]
    _log.info("instantiating xfuser model %s (class=%s)", key, cls.__name__)
    return cls(xfuser_args)


def _first_video(output):
    """Pull the first video out of a diffusers ``WanPipelineOutput`` (or
    xfuser :class:`DiffusionOutput`) regardless of which container shape
    we get.
    """
    frames = getattr(output, "frames", None)
    if frames is None:
        frames = getattr(output, "videos", None)
    if frames is None:
        raise RuntimeError(
            f"Could not extract frames from pipeline output {type(output).__name__}"
        )
    # diffusers wraps multi-batch frames in a list/tuple/ndarray.
    if isinstance(frames, (list, tuple)):
        return frames[0]
    # 5-D ndarray: (batch, num_frames, h, w, c).
    if hasattr(frames, "ndim") and frames.ndim == 5:
        return frames[0]
    return frames


def _video_shape(frames):
    """Return ``(num_frames, height, width)`` from whatever shape we got."""
    import numpy as np  # noqa: WPS433

    arr = np.asarray(frames)
    if arr.ndim == 4:  # (T, H, W, C)
        return arr.shape[0], arr.shape[1], arr.shape[2]
    if arr.ndim == 5:  # (B, T, H, W, C) – shouldn't happen post _first_video.
        return arr.shape[1], arr.shape[2], arr.shape[3]
    raise RuntimeError(
        f"Unexpected frames shape {arr.shape}; expected (T,H,W,C) or (B,T,H,W,C)"
    )


def _frames_as_uint8(frames):
    """Return a contiguous ``uint8 (T, H, W, 3)`` view of ``frames``.

    diffusers commonly returns ``float32`` in ``[0, 1]`` when
    ``output_type="np"``; the harness needs uint8 both for the LoadGen
    byte buffer and for the MP4 encoder, so the renormalisation happens
    once here.
    """
    import numpy as np  # noqa: WPS433

    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)
