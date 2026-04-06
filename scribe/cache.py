"""Parquet cache for the interactive Marimo app.

Converts h5ad expression matrices into columnar Parquet files so the app
can load only the genes the user selects, keeping memory low on 8 GB machines.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CACHE_DIR = Path("output/processed/app_cache")
MANIFEST_FILE = CACHE_DIR / "cache_manifest.json"

UNCORRECTED_H5AD = Path("output/processed/combined_processed.h5ad")
CORRECTED_H5AD = Path("output/processed/combined_processed_corrected.h5ad")


def get_h5ad_paths() -> tuple[Path, Path]:
    """Return (uncorrected, corrected) h5ad paths."""
    return UNCORRECTED_H5AD, CORRECTED_H5AD


def is_cache_stale() -> bool:
    """Check whether the Parquet cache is missing or older than the source h5ad files."""
    if not MANIFEST_FILE.exists():
        return True

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    uncorr_path, corr_path = get_h5ad_paths()

    for key, path in [("uncorrected_mtime", uncorr_path), ("corrected_mtime", corr_path)]:
        if key not in manifest:
            return True
        if not path.exists():
            return True
        if os.path.getmtime(path) > manifest[key]:
            return True

    # Also verify the parquet files themselves exist
    for name in ["uncorrected_expr.parquet", "corrected_expr.parquet", "obs_metadata.parquet"]:
        if not (CACHE_DIR / name).exists():
            return True

    return False


def build_cache(force: bool = False) -> None:
    """Build Parquet cache from h5ad files, loading one at a time for memory safety."""
    import scanpy as sc

    if not force and not is_cache_stale():
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    uncorr_path, corr_path = get_h5ad_paths()

    # --- Uncorrected ---
    print("Loading uncorrected h5ad...")
    adata = sc.read_h5ad(str(uncorr_path))
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    gene_names = list(adata.var_names)

    expr_df = pd.DataFrame(X, columns=gene_names, dtype=np.float32)
    expr_df.to_parquet(CACHE_DIR / "uncorrected_expr.parquet", engine="pyarrow")

    obs_df = adata.obs[["dataset", "condition"]].copy().reset_index(drop=True)
    obs_df.to_parquet(CACHE_DIR / "obs_metadata.parquet", engine="pyarrow")

    uncorr_mtime = os.path.getmtime(uncorr_path)
    del adata, X, expr_df, obs_df
    gc.collect()

    # --- Corrected ---
    print("Loading corrected h5ad...")
    adata = sc.read_h5ad(str(corr_path))
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    expr_df = pd.DataFrame(X, columns=list(adata.var_names), dtype=np.float32)
    expr_df.to_parquet(CACHE_DIR / "corrected_expr.parquet", engine="pyarrow")

    corr_mtime = os.path.getmtime(corr_path)
    del adata, X, expr_df
    gc.collect()

    # --- Manifest ---
    manifest = {
        "uncorrected_mtime": uncorr_mtime,
        "corrected_mtime": corr_mtime,
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Cache built successfully.")


def load_gene_expression(genes: list[str], corrected: bool = False) -> pd.DataFrame:
    """Load only the requested gene columns from the Parquet cache."""
    fname = "corrected_expr.parquet" if corrected else "uncorrected_expr.parquet"
    return pd.read_parquet(CACHE_DIR / fname, columns=genes, engine="pyarrow")


def load_obs_metadata() -> pd.DataFrame:
    """Load cell metadata (dataset, condition)."""
    return pd.read_parquet(CACHE_DIR / "obs_metadata.parquet", engine="pyarrow")


def get_gene_list() -> list[str]:
    """Read gene names from the Parquet schema without loading expression data."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(CACHE_DIR / "uncorrected_expr.parquet")
    return schema.names
