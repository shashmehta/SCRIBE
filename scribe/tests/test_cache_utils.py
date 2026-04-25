"""Unit tests for cache fingerprinting and two-tier routing in scribe/cache.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe import cache, paths


# ── Fingerprint ───────────────────────────────────────────────────────────────

def test_fingerprint_stable(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world" * 100)
    assert cache._file_fingerprint(f) == cache._file_fingerprint(f)


def test_fingerprint_changes_on_content(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"content_a" * 100)
    b.write_bytes(b"content_b" * 100)
    assert cache._file_fingerprint(a) != cache._file_fingerprint(b)


def test_fingerprint_large_file(tmp_path):
    # File > 16 KB exercises the tail-read branch
    f = tmp_path / "large.bin"
    f.write_bytes(b"x" * 40000)
    fp = cache._file_fingerprint(f)
    assert isinstance(fp, str) and len(fp) == 64  # sha256 hex


# ── Cache path routing ────────────────────────────────────────────────────────

def test_cache_path_expr_is_drive(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_OUTPUT_DIR", str(tmp_path))
    p = cache._cache_path("uncorrected_expr.parquet")
    assert p == paths.get_drive_cache_dir() / "uncorrected_expr.parquet"


def test_cache_path_umap_is_local(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_OUTPUT_DIR", str(tmp_path))
    p = cache._cache_path("uncorrected_umap.parquet")
    assert p == paths.get_local_cache_dir() / "uncorrected_umap.parquet"


def test_cache_path_manifest_is_local(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_OUTPUT_DIR", str(tmp_path))
    p = cache._cache_path("cache_manifest.json")
    assert p == paths.get_local_cache_dir() / "cache_manifest.json"


def test_all_drive_files_route_to_drive(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_OUTPUT_DIR", str(tmp_path))
    for name in cache._DRIVE_FILES:
        assert cache._cache_path(name).parent == paths.get_drive_cache_dir()


# ── is_cache_stale ────────────────────────────────────────────────────────────

def test_is_stale_no_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_OUTPUT_DIR", str(tmp_path))
    # No manifest file exists → stale
    assert cache.is_cache_stale() is True


def _write_manifest(manifest_path: Path, content: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(content, f)


def _make_required_cache_files(monkeypatch, tmp_path) -> None:
    """Create empty stand-ins for all required cache files."""
    monkeypatch.setenv("SCRIBE_OUTPUT_DIR", str(tmp_path))
    paths.get_local_cache_dir().mkdir(parents=True, exist_ok=True)
    paths.get_drive_cache_dir().mkdir(parents=True, exist_ok=True)
    for name in cache._REQUIRED_CACHE_FILES:
        cache._cache_path(name).write_bytes(b"placeholder")


def test_is_stale_matching_fingerprints(monkeypatch, tmp_path):
    _make_required_cache_files(monkeypatch, tmp_path)

    uncorr_path, combat_path, _ = cache.get_h5ad_paths()
    uncorr_path.parent.mkdir(parents=True, exist_ok=True)
    uncorr_path.write_bytes(b"uncorrected_data" * 100)
    combat_path.write_bytes(b"combat_data" * 100)

    manifest = {
        "uncorrected_fingerprint": cache._file_fingerprint(uncorr_path),
        "combat_fingerprint": cache._file_fingerprint(combat_path),
    }
    _write_manifest(cache._cache_path("cache_manifest.json"), manifest)

    assert cache.is_cache_stale() is False


def test_is_stale_fingerprint_mismatch(monkeypatch, tmp_path):
    _make_required_cache_files(monkeypatch, tmp_path)

    uncorr_path, combat_path, _ = cache.get_h5ad_paths()
    uncorr_path.parent.mkdir(parents=True, exist_ok=True)
    uncorr_path.write_bytes(b"uncorrected_data" * 100)
    combat_path.write_bytes(b"combat_data" * 100)

    # Write a manifest with a wrong fingerprint for uncorrected
    manifest = {
        "uncorrected_fingerprint": "deadbeef" * 8,  # wrong
        "combat_fingerprint": cache._file_fingerprint(combat_path),
    }
    _write_manifest(cache._cache_path("cache_manifest.json"), manifest)

    assert cache.is_cache_stale() is True
