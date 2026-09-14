"""Tests for :mod:`tools.fetch_data`.

The downloader runs the real network only when invoked from the CLI.
These tests cover the pure-function building blocks (SHA, URL, manifest
shape) plus the cache-hit/skip path in :func:`fetch_one`, which we
exercise by pre-populating ``data_dir`` with the bytes a download would
produce. No outbound HTTP is performed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools.fetch_data import (
    DEFAULT_COMMIT,
    MANIFEST,
    DataFile,
    fetch_one,
    git_blob_sha1,
    raw_url,
)


# ----------------------------------------------------------------------
# git_blob_sha1.
# ----------------------------------------------------------------------


def test_git_blob_sha1_matches_known_value() -> None:
    """``git hash-object`` reference: ``echo "hello" | git hash-object --stdin``
    -> ``ce013625030ba8dba906f756967f9e9ca394464a``.
    """
    assert git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_git_blob_sha1_for_empty_blob() -> None:
    # The empty-blob OID is one of git's most-cited constants.
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


# ----------------------------------------------------------------------
# raw_url.
# ----------------------------------------------------------------------


def test_raw_url_uses_default_commit() -> None:
    url = raw_url("text_to_video/wan-2.2-t2v-a14b/data/vbench_prompts.txt")
    assert url == (
        f"https://raw.githubusercontent.com/mlcommons/inference/"
        f"{DEFAULT_COMMIT}/text_to_video/wan-2.2-t2v-a14b/data/vbench_prompts.txt"
    )


def test_raw_url_accepts_branch_ref() -> None:
    url = raw_url("a/b/c.txt", ref="master")
    assert url == (
        "https://raw.githubusercontent.com/mlcommons/inference/master/a/b/c.txt"
    )


# ----------------------------------------------------------------------
# Manifest hygiene.
# ----------------------------------------------------------------------


_HEX40 = re.compile(r"^[0-9a-f]{40}$")


@pytest.mark.parametrize("entry", MANIFEST, ids=lambda e: e.key)
def test_manifest_entry_well_formed(entry: DataFile) -> None:
    assert entry.key
    assert entry.local_name
    assert entry.repo_path.startswith("text_to_video/wan-2.2-t2v-a14b/data/")
    assert _HEX40.match(entry.blob_sha1), entry.blob_sha1
    assert entry.size > 0


def test_manifest_keys_are_unique() -> None:
    keys = [e.key for e in MANIFEST]
    assert len(keys) == len(set(keys))


def test_manifest_contains_required_files() -> None:
    """The harness's default ``HarnessConfig`` points at these two
    filenames; if they disappear from the manifest the fetcher silently
    stops doing what its users expect.
    """
    names = {e.local_name for e in MANIFEST if not e.is_optional}
    assert {"vbench_prompts.txt", "fixed_latent.pt"} <= names


def test_default_commit_is_pinned_sha() -> None:
    assert _HEX40.match(DEFAULT_COMMIT), DEFAULT_COMMIT


# ----------------------------------------------------------------------
# fetch_one cache-hit path. We pre-populate the destination so the
# function exits before touching the network.
# ----------------------------------------------------------------------


_FAKE = DataFile(
    key="fake",
    repo_path="text_to_video/wan-2.2-t2v-a14b/data/_pytest_fake.bin",
    local_name="_pytest_fake.bin",
    blob_sha1=git_blob_sha1(b"hello\n"),  # 6 bytes
    size=6,
)


def test_fetch_one_skips_when_existing_matches(tmp_path: Path) -> None:
    """Pinned commit + correct blob SHA on disk -> no network."""
    (tmp_path / _FAKE.local_name).write_bytes(b"hello\n")
    local_path, downloaded = fetch_one(_FAKE, data_dir=tmp_path, commit=DEFAULT_COMMIT)
    assert local_path == tmp_path / _FAKE.local_name
    assert downloaded is False


def test_fetch_one_skips_on_size_when_unpinned(tmp_path: Path) -> None:
    """When the user opts out of the pin (commit=master), we can't verify
    the blob SHA, so the cache-hit path falls back to size."""
    (tmp_path / _FAKE.local_name).write_bytes(b"123456")  # 6 bytes, wrong content
    local_path, downloaded = fetch_one(_FAKE, data_dir=tmp_path, commit="master")
    assert downloaded is False
    assert local_path.read_bytes() == b"123456"


def test_fetch_one_force_bypasses_cache(monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Path) -> None:
    """--force must skip the cache check and actually call _download."""
    (tmp_path / _FAKE.local_name).write_bytes(b"hello\n")
    called: dict[str, object] = {}

    def fake_download(url, dest, **kwargs):
        called["url"] = url
        called["dest"] = dest
        dest.write_bytes(b"hello\n")
        return b"hello\n"

    monkeypatch.setattr("tools.fetch_data._download", fake_download)
    _, downloaded = fetch_one(
        _FAKE, data_dir=tmp_path, commit=DEFAULT_COMMIT, force=True
    )
    assert downloaded is True
    assert called["dest"] == tmp_path / _FAKE.local_name


def test_fetch_one_raises_on_sha_mismatch(monkeypatch: pytest.MonkeyPatch,
                                          tmp_path: Path) -> None:
    """Corrupted download must abort, not silently overwrite cleanly."""

    def fake_download(url, dest, **kwargs):
        dest.write_bytes(b"corrupted!")
        return b"corrupted!"

    monkeypatch.setattr("tools.fetch_data._download", fake_download)
    with pytest.raises(RuntimeError, match="does not match expected"):
        fetch_one(_FAKE, data_dir=tmp_path, commit=DEFAULT_COMMIT, force=True)
    # The mismatched payload should not be left on disk.
    assert not (tmp_path / _FAKE.local_name).exists()
