"""Download the official MLPerf wan-2.2-t2v-a14b reference inputs.

The MLCommons reference at ``mlcommons/inference`` ships four small
files in its ``text_to_video/wan-2.2-t2v-a14b/data/`` directory:

  * ``vbench_prompts.txt``        – 248 evaluation prompts (the QSL).
  * ``fixed_latent.pt``           – 9.6 MB BF16 initial-noise tensor
    ``[1, 16, 21, 90, 160]`` that makes generation bit-identical.
  * ``calibration_prompts.txt``   – used by FP8/FP4 calibration only.
  * ``samples_filename_ids.txt``  – sample-id ↔ output-filename map
    consumed by the accuracy evaluator.

By default this script fetches the first two (everything you need to
run an Offline / SingleStream performance test against the real
backend). ``--with-calibration`` and ``--with-samples-list`` pull the
remaining files in for accuracy mode.

The downloads are pinned to a specific upstream commit so two runs of
``python -m tools.fetch_data`` produce byte-identical files. Verification
uses git's own blob SHA-1 (``sha1("blob <size>\\0" + content)``), the
same identifier GitHub exposes in its tree listing, so we don't have to
maintain a parallel checksum manifest.

Pass ``--commit master`` to opt out of the pin and grab whatever
``mlcommons/inference`` master currently has (no integrity check beyond
content-length in that mode).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger("fetch_data")

__all__ = [
    "DataFile",
    "MANIFEST",
    "DEFAULT_COMMIT",
    "git_blob_sha1",
    "raw_url",
    "fetch_one",
    "main",
]


# ----------------------------------------------------------------------
# Manifest. Update DEFAULT_COMMIT (and the per-file blob_sha1 / size) by
# re-running the GitHub contents API for the data folder; see this
# module's docstring for how the values were derived.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DataFile:
    key: str
    """CLI-friendly short name (``--what vbench_prompts``)."""

    repo_path: str
    """Path inside ``mlcommons/inference`` (forward-slash separated)."""

    local_name: str
    """Filename written into ``--data-dir``."""

    blob_sha1: str
    """Expected git blob SHA-1 at :data:`DEFAULT_COMMIT`."""

    size: int
    """Expected file size in bytes (sanity check, cheap pre-hash signal)."""

    is_optional: bool = False
    """When True, the file is only fetched if the user opts in via a flag."""


DEFAULT_COMMIT = "175ea7ba24a826946458f72ec4a2221215f52802"
"""Pinned ``mlcommons/inference`` commit. The blob SHAs below were taken
from this commit; bumping this value without updating the SHAs will
make every download fail integrity verification, which is the point."""


MANIFEST: tuple[DataFile, ...] = (
    DataFile(
        key="vbench_prompts",
        repo_path="text_to_video/wan-2.2-t2v-a14b/data/vbench_prompts.txt",
        local_name="vbench_prompts.txt",
        blob_sha1="cffe1bdad7bb9ee2aa63b243e2660004a9c0fa24",
        size=10460,
    ),
    DataFile(
        key="fixed_latent",
        repo_path="text_to_video/wan-2.2-t2v-a14b/data/fixed_latent.pt",
        local_name="fixed_latent.pt",
        blob_sha1="7c88cf35aa26a98e81a432a8cea22ba5f6b8ef04",
        size=9678476,
    ),
    DataFile(
        key="calibration_prompts",
        repo_path="text_to_video/wan-2.2-t2v-a14b/data/calibration_prompts.txt",
        local_name="calibration_prompts.txt",
        blob_sha1="5acaf78e8ca13fa09aaa9e6c20bbff266c0f1b6f",
        size=1510,
        is_optional=True,
    ),
    DataFile(
        key="samples_filename_ids",
        repo_path="text_to_video/wan-2.2-t2v-a14b/data/samples_filename_ids.txt",
        local_name="samples_filename_ids.txt",
        blob_sha1="8d9f8117a65cda4c0f3969a5bb972c19b2c36485",
        size=608,
        is_optional=True,
    ),
)


_RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/mlcommons/inference/{ref}/{repo_path}"
)


# ----------------------------------------------------------------------
# Pure helpers (no I/O) – tested in isolation.
# ----------------------------------------------------------------------


def git_blob_sha1(content: bytes) -> str:
    """Compute git's own object SHA-1 for a file blob.

    This is the value shown in GitHub tree / contents API listings, and
    is computed as ``sha1(b"blob <length>\\0" + content)``. Using it lets
    us verify downloads against numbers we can grab straight from the
    GitHub UI without maintaining a separate checksum file.
    """
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def raw_url(repo_path: str, *, ref: str = DEFAULT_COMMIT) -> str:
    """Build a ``raw.githubusercontent.com`` URL for ``repo_path`` at ``ref``."""
    return _RAW_URL_TEMPLATE.format(ref=ref, repo_path=repo_path)


def _find(key: str) -> DataFile:
    for f in MANIFEST:
        if f.key == key:
            return f
    raise KeyError(
        f"unknown data file {key!r}; valid keys: {[f.key for f in MANIFEST]!r}"
    )


# ----------------------------------------------------------------------
# I/O. Network calls live here so the helpers above can be tested
# without monkey-patching ``urlopen``.
# ----------------------------------------------------------------------


def _download(url: str, dest: Path, *, log_every_bytes: int = 1 << 20) -> bytes:
    """Stream ``url`` into ``dest`` and return the bytes we wrote.

    Streaming keeps the 9.6 MB ``fixed_latent.pt`` off the heap as a
    single string, and the periodic log line gives slow connections
    something to look at.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[bytes] = []
    total = 0
    next_log = log_every_bytes
    req = urllib.request.Request(url, headers={"User-Agent": "wan-harness/fetch_data"})
    with urllib.request.urlopen(req) as resp:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total >= next_log:
                _log.info("  %s: %d KB", dest.name, total // 1024)
                next_log += log_every_bytes
    content = b"".join(chunks)
    dest.write_bytes(content)
    return content


def fetch_one(
    data_file: DataFile,
    *,
    data_dir: Path,
    commit: str = DEFAULT_COMMIT,
    force: bool = False,
    verify: bool = True,
) -> tuple[Path, bool]:
    """Fetch one entry from :data:`MANIFEST` into ``data_dir``.

    Returns ``(local_path, downloaded)`` where ``downloaded`` is False
    when the file was already present at the expected SHA. ``verify``
    is only meaningful at ``commit == DEFAULT_COMMIT``; for other refs
    the recorded blob SHA does not apply, so we silently fall back to a
    size check.
    """
    local_path = data_dir / data_file.local_name
    expect_sha = data_file.blob_sha1 if commit == DEFAULT_COMMIT else None

    if local_path.exists() and not force:
        existing = local_path.read_bytes()
        if expect_sha is None:
            if len(existing) == data_file.size:
                _log.info("  %s: already present (size match), skipping", local_path)
                return local_path, False
        else:
            if git_blob_sha1(existing) == expect_sha:
                _log.info("  %s: already present (SHA match), skipping", local_path)
                return local_path, False
        _log.info(
            "  %s: present but %s mismatch, re-downloading",
            local_path, "SHA" if expect_sha else "size",
        )

    url = raw_url(data_file.repo_path, ref=commit)
    _log.info("  %s -> %s", url, local_path)
    content = _download(url, local_path)

    if verify and expect_sha is not None:
        got = git_blob_sha1(content)
        if got != expect_sha:
            local_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{local_path}: downloaded blob SHA {got!r} does not match "
                f"expected {expect_sha!r} (manifest pin {DEFAULT_COMMIT}). "
                f"If you intended to fetch a different revision, pass "
                f"--commit <SHA>."
            )
    elif len(content) != data_file.size and commit == DEFAULT_COMMIT:
        _log.warning(
            "  %s: unexpected size %d (manifest expected %d)",
            local_path, len(content), data_file.size,
        )
    return local_path, True


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------


def _default_data_dir() -> Path:
    """Return ``<repo>/data`` so the harness's default config paths
    (``HarnessConfig.prompts_path`` / ``fixed_latent_path``) Just Work.
    """
    return Path(__file__).resolve().parent.parent / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.fetch_data",
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Destination directory (default: <repo>/data).",
    )
    parser.add_argument(
        "--commit",
        default=DEFAULT_COMMIT,
        help=(
            "mlcommons/inference ref to fetch from. Defaults to a pinned "
            "commit (%(default)s) for reproducibility; pass 'master' to "
            "grab the current tip."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the file is already present at the expected SHA.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip blob-SHA verification (size check still applies).",
    )
    parser.add_argument(
        "--with-calibration",
        action="store_true",
        help="Also fetch calibration_prompts.txt (FP8/FP4 calibration only).",
    )
    parser.add_argument(
        "--with-samples-list",
        action="store_true",
        help="Also fetch samples_filename_ids.txt (accuracy mode).",
    )
    parser.add_argument(
        "--what",
        action="append",
        default=[],
        choices=[f.key for f in MANIFEST],
        help=(
            "Fetch exactly the given file (repeatable). Overrides the "
            "default set; ignores --with-* flags."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if args.what:
        selected = [_find(k) for k in args.what]
    else:
        selected = [f for f in MANIFEST if not f.is_optional]
        if args.with_calibration:
            selected.append(_find("calibration_prompts"))
        if args.with_samples_list:
            selected.append(_find("samples_filename_ids"))

    _log.info(
        "Fetching %d file(s) from mlcommons/inference@%s into %s",
        len(selected), args.commit, args.data_dir,
    )
    downloaded = 0
    for f in selected:
        try:
            _, did = fetch_one(
                f,
                data_dir=args.data_dir,
                commit=args.commit,
                force=args.force,
                verify=not args.no_verify,
            )
        except (urllib.error.URLError, RuntimeError) as exc:
            _log.error("FAILED %s: %s", f.local_name, exc)
            return 2
        if did:
            downloaded += 1
    _log.info("Done. %d/%d files downloaded (rest already up to date).",
              downloaded, len(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
