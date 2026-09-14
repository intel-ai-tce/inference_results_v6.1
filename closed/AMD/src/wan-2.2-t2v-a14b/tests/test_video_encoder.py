"""Tests for :mod:`wan_harness.video_encoder`.

The pure validation path is tested unconditionally; the actual ffmpeg
encode is gated on ffmpeg being installed so unit tests still pass in
environments that don't have it (e.g. some CI runners). The
command-construction path is exercised via a monkey-patched
``subprocess.run`` so it runs everywhere.
"""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

# numpy is a runtime dep provided by the base image; outside the
# container we degrade gracefully so the rest of the test suite can
# still run.
np = pytest.importorskip("numpy", reason="numpy not installed in this environment")

from wan_harness import video_encoder  # noqa: E402
from wan_harness.video_encoder import (  # noqa: E402
    DEFAULT_H264_CODEC,
    H264_FALLBACK_CHAIN,
    VideoEncoderError,
    encode_frames_to_mp4,
    ffmpeg_available,
    list_ffmpeg_encoders,
    resolve_ffmpeg_path,
)


@pytest.fixture(autouse=True)
def _reset_encoder_cache():
    """``list_ffmpeg_encoders`` caches per-binary, but tests fake the
    binary path and switch return values mid-test – flush between tests
    so each one starts clean."""
    video_encoder._ENCODERS_CACHE.clear()
    yield
    video_encoder._ENCODERS_CACHE.clear()


def _stub_encoders(monkeypatch: pytest.MonkeyPatch, names: set[str]) -> None:
    """Pretend the ffmpeg at ``/usr/bin/ffmpeg`` supports exactly ``names``."""
    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        video_encoder, "list_ffmpeg_encoders", lambda _p: frozenset(names)
    )


# ----------------------------------------------------------------------
# Validation – no subprocess needed.
# ----------------------------------------------------------------------


def _ones(shape: tuple[int, ...], dtype=np.uint8) -> np.ndarray:
    return np.ones(shape, dtype=dtype)


def test_rejects_wrong_rank() -> None:
    with pytest.raises(VideoEncoderError, match="shape"):
        encode_frames_to_mp4(_ones((4, 32, 32)), fps=16)


def test_rejects_wrong_channel_count() -> None:
    with pytest.raises(VideoEncoderError, match="shape"):
        encode_frames_to_mp4(_ones((4, 32, 32, 4)), fps=16)


def test_rejects_float_dtype() -> None:
    arr = np.zeros((4, 32, 32, 3), dtype=np.float32)
    with pytest.raises(VideoEncoderError, match="uint8"):
        encode_frames_to_mp4(arr, fps=16)


def test_rejects_odd_dimensions_for_yuv420p() -> None:
    """yuv420p requires even H, W. Catch this up front rather than letting
    ffmpeg produce a confusing scaler error."""
    with pytest.raises(VideoEncoderError, match="even"):
        encode_frames_to_mp4(_ones((4, 31, 32, 3)), fps=16)
    with pytest.raises(VideoEncoderError, match="even"):
        encode_frames_to_mp4(_ones((4, 32, 31, 3)), fps=16)


def test_rejects_zero_fps() -> None:
    with pytest.raises(VideoEncoderError, match="fps"):
        encode_frames_to_mp4(_ones((4, 32, 32, 3)), fps=0)


def test_rejects_when_ffmpeg_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``ffmpeg`` is not on PATH the encoder must complain immediately
    rather than blowing up inside ``subprocess.run`` with a less clear
    ``FileNotFoundError``."""
    monkeypatch.delenv("WAN_FFMPEG_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(VideoEncoderError, match="ffmpeg binary not found"):
        encode_frames_to_mp4(_ones((4, 32, 32, 3)), fps=16)


# ----------------------------------------------------------------------
# Binary resolution: WAN_FFMPEG_BIN override and PATH fallback.
# ----------------------------------------------------------------------


def test_resolve_ffmpeg_respects_env_override(monkeypatch: pytest.MonkeyPatch,
                                              tmp_path) -> None:
    """The Dockerfile pins WAN_FFMPEG_BIN to the apt-installed binary so
    a conda-installed GPL-free build earlier on PATH cannot shadow it."""
    fake = tmp_path / "ffmpeg"
    fake.write_text("")  # only needs to be a real file
    monkeypatch.setenv("WAN_FFMPEG_BIN", str(fake))
    # Make PATH lookup return something different so we can tell which one won.
    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/ffmpeg")
    assert resolve_ffmpeg_path() == str(fake)


def test_resolve_ffmpeg_warns_when_override_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog,
) -> None:
    monkeypatch.setenv("WAN_FFMPEG_BIN", str(tmp_path / "does-not-exist"))
    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/ffmpeg")
    with caplog.at_level("WARNING", logger="wan_harness.video_encoder"):
        path = resolve_ffmpeg_path()
    assert path == "/usr/bin/ffmpeg"
    assert any("does not exist" in m for m in caplog.messages)


def test_resolve_ffmpeg_falls_back_to_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WAN_FFMPEG_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/ffmpeg")
    assert resolve_ffmpeg_path() == "/usr/bin/ffmpeg"


# ----------------------------------------------------------------------
# Encoder discovery + fallback.
# ----------------------------------------------------------------------


def test_list_ffmpeg_encoders_parses_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``ffmpeg -encoders`` table has a legend followed by a dashes
    separator and then rows like 'V....D libx264 ...'. We must extract
    just the encoder names, not the legend."""
    fake_stdout = (
        "Encoders:\n"
        " V..... = Video\n"
        " A..... = Audio\n"
        " ------\n"
        " V....D libx264              libx264 H.264 / AVC / MPEG-4 part 10\n"
        " V....D libopenh264          OpenH264 H.264/MPEG-4 AVC encoder\n"
        " V....D mpeg4                MPEG-4 part 2\n"
    )

    def fake_run(cmd, *, capture_output, check, text):  # noqa: ARG001
        assert "-encoders" in cmd
        return SimpleNamespace(returncode=0, stdout=fake_stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    encoders = list_ffmpeg_encoders("/usr/bin/ffmpeg")
    assert "libx264" in encoders
    assert "libopenh264" in encoders
    assert "mpeg4" in encoders
    # The legend lines must not leak in.
    assert "V....." not in encoders
    assert "=" not in encoders


def test_list_ffmpeg_encoders_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_run(cmd, *, capture_output, check, text):  # noqa: ARG001
        calls["n"] += 1
        return SimpleNamespace(
            returncode=0,
            stdout=" ------\n V..... libx264 desc\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    a = list_ffmpeg_encoders("/usr/bin/ffmpeg")
    b = list_ffmpeg_encoders("/usr/bin/ffmpeg")
    assert a == b == frozenset({"libx264"})
    assert calls["n"] == 1


def test_libx264_falls_back_to_libopenh264(monkeypatch: pytest.MonkeyPatch,
                                           caplog,
                                           tmp_path) -> None:
    """The user's container case: ffmpeg is present but GPL-free, so
    libx264 is missing. We must accept libopenh264 and emit a warning
    pointing at the fix."""
    _stub_encoders(monkeypatch, {"libopenh264", "mpeg4"})
    seen: dict[str, Any] = {}

    def fake_run(cmd, *, input, check, capture_output, **_):  # noqa: ARG001
        seen["cmd"] = cmd
        out_path = cmd[-1]
        with open(out_path, "wb") as fh:
            fh.write(b"FAKEMP4")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level("WARNING", logger="wan_harness.video_encoder"):
        out = encode_frames_to_mp4(_ones((4, 32, 32, 3)), fps=16)
    assert out == b"FAKEMP4"
    # The actual ffmpeg invocation must have used the fallback codec, not
    # the unavailable default.
    cmd = seen["cmd"]
    assert cmd[cmd.index("-c:v") + 1] == "libopenh264"
    # The warning must mention the user-actionable fix.
    assert any("libx264" in m and "libopenh264" in m for m in caplog.messages)


def test_explicit_codec_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the caller explicitly asks for a codec, missing-codec is a
    hard error – we do NOT silently substitute. This preserves the
    'user said exactly what they wanted' contract."""
    _stub_encoders(monkeypatch, {"libopenh264"})
    with pytest.raises(VideoEncoderError, match="not available"):
        encode_frames_to_mp4(_ones((4, 32, 32, 3)), fps=16, codec="libx265")


def test_no_h264_at_all_raises_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_encoders(monkeypatch, {"mpeg4"})  # no H.264 anywhere
    with pytest.raises(VideoEncoderError, match="apt-get install"):
        encode_frames_to_mp4(_ones((4, 32, 32, 3)), fps=16)


def test_h264_fallback_chain_constant_starts_with_libx264() -> None:
    """The first element of the fallback chain is what we pass to ffmpeg
    by default; if someone reorders the tuple to put a non-MLPerf-aligned
    encoder first we'd silently change scores."""
    assert H264_FALLBACK_CHAIN[0] == DEFAULT_H264_CODEC == "libx264"


# ----------------------------------------------------------------------
# Command construction – monkey-patch subprocess.run so we can verify
# the encoder is asking ffmpeg for the right settings (codec, pix_fmt,
# rgb24-on-stdin, fps, etc.) without actually invoking ffmpeg.
# ----------------------------------------------------------------------


def test_invokes_ffmpeg_with_expected_args(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path) -> None:
    _stub_encoders(monkeypatch, {"libx264", "libopenh264"})
    captured: dict[str, Any] = {}

    def fake_run(cmd, *, input, check, capture_output, **_):  # noqa: ARG001
        captured["cmd"] = cmd
        captured["input"] = input
        # ffmpeg's last positional arg is the output file.
        out_path = cmd[-1]
        # Minimal-but-recognisable mp4: just a stub the encoder will
        # read back as bytes. The integrity check happens elsewhere.
        with open(out_path, "wb") as fh:
            fh.write(b"FAKEMP4")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    frames = _ones((5, 480, 720, 3))
    out = encode_frames_to_mp4(frames, fps=16)
    assert out == b"FAKEMP4"

    cmd = captured["cmd"]
    # Stdin should carry exactly the raw bytes we asked ffmpeg to consume.
    assert captured["input"] == frames.tobytes()
    # The encoder must tell ffmpeg the geometry we passed in.
    assert "-video_size" in cmd
    assert cmd[cmd.index("-video_size") + 1] == "720x480"  # WxH
    # rgb24 on stdin, libx264/yuv420p on output.
    assert "-pixel_format" in cmd
    assert cmd[cmd.index("-pixel_format") + 1] == "rgb24"
    assert "-c:v" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert "-pix_fmt" in cmd
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert "-framerate" in cmd
    assert cmd[cmd.index("-framerate") + 1] == "16"
    # stdin sourcing.
    assert "-i" in cmd
    assert cmd[cmd.index("-i") + 1] == "pipe:0"


def test_propagates_ffmpeg_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_encoders(monkeypatch, {"libx264"})

    def fake_run(cmd, *, input, check, capture_output, **_):  # noqa: ARG001
        return SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"Error: simulated muxer failure\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(VideoEncoderError, match="simulated muxer failure"):
        encode_frames_to_mp4(_ones((4, 32, 32, 3)), fps=16)


def test_honours_codec_and_crf_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_encoders(monkeypatch, {"libx264", "libx265"})
    seen: dict[str, Any] = {}

    def fake_run(cmd, *, input, check, capture_output, **_):  # noqa: ARG001
        seen["cmd"] = cmd
        out_path = cmd[-1]
        with open(out_path, "wb") as fh:
            fh.write(b"X")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    encode_frames_to_mp4(_ones((4, 32, 32, 3)), fps=16, codec="libx265", crf=18)
    cmd = seen["cmd"]
    assert cmd[cmd.index("-c:v") + 1] == "libx265"
    assert cmd[cmd.index("-crf") + 1] == "18"


# ----------------------------------------------------------------------
# Integration – exercise a real ffmpeg. Skipped when the binary is
# unavailable so this test file can pass in arbitrary environments.
# ----------------------------------------------------------------------


_HAS_FFMPEG = ffmpeg_available()


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
def test_real_ffmpeg_produces_decodable_mp4(tmp_path) -> None:
    """End-to-end: generate a tiny gradient video, encode it, write the
    bytes to a file, ask ffprobe what it sees. We don't compare pixels –
    yuv420p chroma subsampling makes that fragile – we only assert that
    the byte payload is a real MP4 with the right framerate and frame
    count."""
    T, H, W = 6, 64, 64
    # Visible gradient so ffmpeg actually has data to compress (and so
    # the resulting file is non-trivial).
    frames = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        frames[t, :, :, 0] = (t * 40) % 256
        frames[t, :, :, 1] = ((t + 1) * 60) % 256
        frames[t, :, :, 2] = ((t + 2) * 90) % 256

    payload = encode_frames_to_mp4(frames, fps=16)

    # MP4 magic: bytes 4..8 are the "ftyp" box type.
    assert payload[4:8] == b"ftyp", f"first 16 bytes: {payload[:16]!r}"

    out = tmp_path / "out.mp4"
    out.write_bytes(payload)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets,r_frame_rate,width,height,codec_name",
            # nokey=1 strips the "key=" half of each line, but does NOT
            # suppress the [STREAM]/[/STREAM] section wrappers – without
            # noprint_wrappers=1 the first field we read back would be
            # the literal string "[STREAM]" rather than the codec name.
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(out),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    fields = [line for line in probe.stdout.strip().splitlines() if line]
    # Field order from the -show_entries above.
    codec, width, height, frame_rate, nb_packets = fields[:5]
    assert codec == "h264"
    assert int(width) == W
    assert int(height) == H
    # frame_rate is a rational like "16/1"; pytest will give us a nice
    # diff if we just compare strings.
    assert frame_rate == "16/1"
    assert int(nb_packets) == T
