"""MP4 encoder for accuracy-mode artefacts.

The harness keeps the raw frame buffer (``frames_bytes``) as the canonical
LoadGen response payload – that's what MLPerf actually scores against and
what the reference v6.1 implementation puts on the wire too. For accuracy
mode we additionally encode the same frames into an MP4 so the downstream
VBench evaluator can score the video the way it normally does (it walks
a directory of ``.mp4`` files; nothing else).

Design decisions:

* **ffmpeg via subprocess, not a Python library.** ``diffusers``,
  ``imageio``, and ``torchvision`` all delegate to ffmpeg under the hood,
  so we save a Python dep and gain explicit control over codec / pixel
  format / quality.
* **H.264 with autodetected encoder.** ``libx264`` is the MLPerf
  reference's choice and what we prefer, but several pytorch base images
  ship a GPL-free ffmpeg build that omits it. We discover what the
  available ffmpeg actually has (``ffmpeg -encoders``) once at startup
  and fall back to ``libopenh264`` (BSD-licensed) when libx264 is gone,
  with a loud warning. Users who explicitly pass a non-default codec
  get no fallback – that's intentional, an explicit codec is a request,
  not a hint.
* **Temp file, not a pipe.** MP4 needs to rewrite the moov atom at the
  end of muxing, which is impossible on a non-seekable stdout. We pay
  one small tmpfs round-trip and skip the ``+faststart`` / fragmented-MP4
  contortions.
* **WAN_FFMPEG_BIN** env var lets users point at a specific ffmpeg even
  when multiple builds sit on PATH (the Dockerfile uses this to force
  the apt-installed ``/usr/bin/ffmpeg`` over the base image's
  conda-installed GPL-free build).
* **Settings**: ``yuv420p`` + default CRF (23) at the YAML-configured
  fps. VBench has no opinion about CRF, only about decode-ability.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


_log = logging.getLogger(__name__)

__all__ = [
    "encode_frames_to_mp4",
    "ffmpeg_available",
    "list_ffmpeg_encoders",
    "resolve_ffmpeg_path",
    "VideoEncoderError",
    "DEFAULT_H264_CODEC",
    "H264_FALLBACK_CHAIN",
]


#: Default codec we hand to ffmpeg's ``-c:v``. The fallback chain below
#: is consulted iff this exact encoder is missing from the live ffmpeg.
DEFAULT_H264_CODEC = "libx264"


#: Ordered set of H.264 encoders the autodetect path will accept, best
#: first. ``libopenh264`` is BSD-licensed and ships in many "GPL-free"
#: ffmpeg builds; everything beyond it is a hardware-specific encoder
#: that we don't want to fall into accidentally.
H264_FALLBACK_CHAIN: tuple[str, ...] = ("libx264", "libopenh264")


_FFMPEG_ENV_OVERRIDE = "WAN_FFMPEG_BIN"

# ``ffmpeg -encoders`` is the only subprocess call we cache: the binary
# doesn't change at runtime, so paying its cost (~50ms) once per process
# is fine. Keyed by absolute path so two different ffmpeg binaries on the
# same machine still get independent lookups.
_ENCODERS_CACHE: dict[str, frozenset[str]] = {}


class VideoEncoderError(RuntimeError):
    """Raised when ffmpeg cannot encode the given frames."""


def resolve_ffmpeg_path() -> str | None:
    """Return the ffmpeg binary the encoder will use, or None.

    Lookup precedence:

    1. ``$WAN_FFMPEG_BIN`` if set and the file exists. Lets the Docker
       image pin the apt-installed binary even if a different ffmpeg
       (e.g. conda's GPL-free build) is earlier on ``PATH``.
    2. ``shutil.which("ffmpeg")`` – standard ``PATH`` lookup.
    """
    override = os.environ.get(_FFMPEG_ENV_OVERRIDE)
    if override:
        if Path(override).is_file():
            return override
        _log.warning(
            "%s=%s does not exist; falling back to PATH lookup",
            _FFMPEG_ENV_OVERRIDE, override,
        )
    return shutil.which("ffmpeg")


def ffmpeg_available() -> bool:
    """Return True iff some ffmpeg binary is reachable.

    Used by the WanBackend to fail loudly at setup() rather than mid-run
    when accuracy mode is requested and ffmpeg is missing.
    """
    return resolve_ffmpeg_path() is not None


def list_ffmpeg_encoders(ffmpeg_path: str) -> frozenset[str]:
    """Return the set of encoder names this ffmpeg build supports.

    Parsed from ``ffmpeg -encoders``. Cached per binary path – the result
    can't change at runtime.
    """
    cached = _ENCODERS_CACHE.get(ffmpeg_path)
    if cached is not None:
        return cached
    proc = subprocess.run(  # noqa: S603 (intentional subprocess)
        [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-encoders"],
        capture_output=True,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        raise VideoEncoderError(
            f"`{ffmpeg_path} -encoders` exited {proc.returncode}: "
            f"{proc.stderr.strip()[-500:]}"
        )
    encoders: set[str] = set()
    in_table = False
    for line in proc.stdout.splitlines():
        # The output begins with a multi-line legend; the encoder rows
        # start after a separator line of dashes. Be liberal about the
        # exact format.
        stripped = line.strip()
        if not in_table:
            if stripped.startswith("------"):
                in_table = True
            continue
        # Encoder rows look like:
        #   "V....D libx264              libx264 H.264 / AVC / ..."
        # so the encoder name is the second whitespace-separated token.
        parts = stripped.split(None, 2)
        if len(parts) >= 2 and len(parts[0]) >= 1:
            encoders.add(parts[1])
    result = frozenset(encoders)
    _ENCODERS_CACHE[ffmpeg_path] = result
    return result


def _pick_codec(
    ffmpeg_path: str,
    requested: str,
) -> str:
    """Return a codec that this ffmpeg can actually use.

    For the default ``libx264`` we silently fall back along
    :data:`H264_FALLBACK_CHAIN` with a warning. For any other explicit
    codec we hard-fail if it's not available – an explicit choice means
    the caller has a reason.
    """
    available = list_ffmpeg_encoders(ffmpeg_path)
    if requested in available:
        return requested

    if requested == DEFAULT_H264_CODEC:
        for alt in H264_FALLBACK_CHAIN:
            if alt == requested:
                continue
            if alt in available:
                _log.warning(
                    "video_encoder: %r not in this ffmpeg build "
                    "(%s); falling back to %r. For byte-stable behaviour "
                    "match the MLPerf reference, install ffmpeg with "
                    "libx264 support (apt-get install -y ffmpeg) and set "
                    "%s=/usr/bin/ffmpeg.",
                    requested, ffmpeg_path, alt, _FFMPEG_ENV_OVERRIDE,
                )
                return alt

    h264_like = sorted(
        e for e in available if "264" in e.lower() or e == "h264"
    )
    raise VideoEncoderError(
        f"ffmpeg encoder {requested!r} is not available in "
        f"{ffmpeg_path!r}. H.264-family encoders this build can do: "
        f"{h264_like if h264_like else '<none>'}. "
        f"Install a full ffmpeg (apt-get install -y ffmpeg) or set "
        f"{_FFMPEG_ENV_OVERRIDE} to a binary that has libx264."
    )


def encode_frames_to_mp4(
    frames: "np.ndarray",
    *,
    fps: int,
    codec: str = DEFAULT_H264_CODEC,
    pix_fmt: str = "yuv420p",
    crf: int = 23,
    ffmpeg_path: str | None = None,
) -> bytes:
    """Encode an RGB frame stack into an MP4 container.

    Args:
        frames: ``uint8`` array of shape ``(num_frames, height, width, 3)``,
            channels in RGB order, values in ``[0, 255]``. Floats are not
            accepted because the conversion choice (clip vs renormalize)
            is a backend-side decision; do it in the caller.
        fps: Frames per second written into the MP4 header.
        codec: ``-c:v`` argument. Defaults to ``libx264`` (the MLPerf
            reference's choice). If this exact encoder is missing from
            the live ffmpeg, we silently fall back along
            :data:`H264_FALLBACK_CHAIN` with a warning. Pass any other
            encoder name to opt out of fallback – an explicit choice is
            treated as a hard requirement.
        pix_fmt: ``-pix_fmt`` argument; ``yuv420p`` is the most-widely-
            compatible chroma subsampling and is what VBench-style
            downstream tools expect.
        crf: H.264 quality knob (lower = better). 18 = visually
            lossless-ish, 23 = ffmpeg default, 28 = noticeably degraded.
            VBench doesn't score on file size; we stay at default.
        ffmpeg_path: Override for the ffmpeg binary. ``None`` (the
            default) resolves via :func:`resolve_ffmpeg_path` which
            honours the ``WAN_FFMPEG_BIN`` env var first and then falls
            back to ``shutil.which("ffmpeg")``.

    Returns:
        The complete MP4 byte payload.

    Raises:
        VideoEncoderError: ffmpeg is not installed, the requested codec
            is unavailable and has no fallback, the frame array has the
            wrong shape / dtype, or the subprocess returned a non-zero
            exit code.
    """
    import numpy as np  # noqa: WPS433 (intentional local import)

    arr = np.ascontiguousarray(np.asarray(frames))
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise VideoEncoderError(
            f"frames must have shape (T, H, W, 3); got {arr.shape}"
        )
    if arr.dtype != np.uint8:
        raise VideoEncoderError(
            f"frames must be dtype uint8 (after caller-side renormalisation); "
            f"got {arr.dtype}"
        )

    num_frames, height, width, _ = arr.shape

    # libx264 requires even spatial dims for yuv420p. Wan's 720x1280 (and
    # its 480x720 smoke variant) satisfy this, but a wrong YAML override
    # would fail deep inside ffmpeg with a less obvious error, so check
    # up front.
    if (height % 2) or (width % 2):
        raise VideoEncoderError(
            f"frame dimensions must be even for {pix_fmt} encoding; "
            f"got {height}x{width}"
        )

    if fps <= 0:
        raise VideoEncoderError(f"fps must be positive; got {fps}")

    # Input is well-formed – now we can require the binary.
    resolved_path = ffmpeg_path if ffmpeg_path is not None else resolve_ffmpeg_path()
    if resolved_path is None:
        raise VideoEncoderError(
            "ffmpeg binary not found on PATH; accuracy mode requires ffmpeg. "
            "On Ubuntu: apt-get install -y ffmpeg."
        )

    chosen_codec = _pick_codec(resolved_path, codec)

    with tempfile.TemporaryDirectory(prefix="wan_mp4_") as td:
        out_path = Path(td) / "out.mp4"
        cmd = [
            resolved_path, "-y",
            # Input: raw rgb24 frames on stdin.
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "pipe:0",
            # Output.
            "-c:v", chosen_codec,
            "-pix_fmt", pix_fmt,
            "-crf", str(crf),
            # Silence ffmpeg's banner / per-frame progress; surface only errors.
            "-hide_banner",
            "-loglevel", "error",
            str(out_path),
        ]

        _log.debug(
            "encode: %dx%d %d frames @ %d fps codec=%s -> %s",
            height, width, num_frames, fps, chosen_codec, out_path,
        )
        try:
            proc = subprocess.run(  # noqa: S603 (intentional subprocess)
                cmd,
                input=arr.tobytes(),
                check=False,
                capture_output=True,
            )
        except FileNotFoundError as exc:  # ffmpeg disappeared between the
            # shutil.which check and the actual invocation.
            raise VideoEncoderError(f"ffmpeg invocation failed: {exc}") from exc

        if proc.returncode != 0:
            stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-2000:]
            raise VideoEncoderError(
                f"ffmpeg exited {proc.returncode}; stderr tail:\n{stderr_tail}"
            )

        return out_path.read_bytes()
