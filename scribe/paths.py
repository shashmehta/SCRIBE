"""Centralized path resolution for SCRIBE.

All paths derive from OUTPUT_DIR, which is resolved as:
  1. SCRIBE_OUTPUT_DIR environment variable (if set)
  2. ./output (default, typically a symlink to Google Drive)
"""

from __future__ import annotations

import os
from pathlib import Path


def get_output_dir() -> Path:
    """Base output directory. Override with SCRIBE_OUTPUT_DIR env var."""
    return Path(os.environ.get("SCRIBE_OUTPUT_DIR", "./output"))


def get_processed_dir() -> Path:
    """Processed h5ad files, model artifacts."""
    return get_output_dir() / "processed"


def get_plots_dir() -> Path:
    """Generated plot PNGs."""
    return get_output_dir() / "plots"


def get_drive_cache_dir() -> Path:
    """Large cache files (expression parquets) on Drive."""
    return get_processed_dir() / "app_cache"


def get_local_cache_dir() -> Path:
    """Small cache files (metadata, UMAP, HK) on local disk."""
    return Path.home() / ".scribe" / "cache"


def get_h5ad_paths() -> tuple[Path, Path, Path]:
    """Return (uncorrected, combat, harmony) h5ad paths."""
    d = get_processed_dir()
    return (
        d / "combined_processed.h5ad",
        d / "combined_processed_corrected.h5ad",
        d / "combined_processed_harmony.h5ad",
    )
