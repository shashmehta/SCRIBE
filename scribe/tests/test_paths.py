"""Unit tests for scribe/paths.py — central path resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scribe import paths


def test_default_output_dir(monkeypatch):
    monkeypatch.delenv("SCRIBE_OUTPUT_DIR", raising=False)
    assert paths.get_output_dir() == Path("./output")


def test_env_var_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_OUTPUT_DIR", str(tmp_path))
    assert paths.get_output_dir() == tmp_path


def test_derived_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_OUTPUT_DIR", str(tmp_path))
    assert paths.get_processed_dir() == tmp_path / "processed"
    assert paths.get_plots_dir() == tmp_path / "plots"
    assert paths.get_drive_cache_dir() == tmp_path / "processed" / "app_cache"


def test_local_cache_dir_is_under_home():
    local = paths.get_local_cache_dir()
    assert local == Path.home() / ".scribe" / "cache"


def test_h5ad_paths_filenames(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_OUTPUT_DIR", str(tmp_path))
    uncorr, combat, harmony = paths.get_h5ad_paths()
    assert uncorr.name == "combined_processed.h5ad"
    assert combat.name == "combined_processed_corrected.h5ad"
    assert harmony.name == "combined_processed_harmony.h5ad"


def test_h5ad_paths_under_processed(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_OUTPUT_DIR", str(tmp_path))
    for p in paths.get_h5ad_paths():
        assert p.parent == tmp_path / "processed"


def test_env_var_not_set_gives_relative_path(monkeypatch):
    monkeypatch.delenv("SCRIBE_OUTPUT_DIR", raising=False)
    # Should be relative, not absolute
    assert not paths.get_output_dir().is_absolute()
