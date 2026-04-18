"""Parquet cache for the interactive Marimo app.

Converts h5ad expression matrices into columnar Parquet files so the app
can load only the genes the user selects, keeping memory low on 8 GB machines.

Supports three method variants: "uncorrected", "combat", "harmony".
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
COMBAT_H5AD = Path("output/processed/combined_processed_corrected.h5ad")
HARMONY_H5AD = Path("output/processed/combined_processed_harmony.h5ad")

# Legacy alias
CORRECTED_H5AD = COMBAT_H5AD

# Tirosh et al. 2015 cell cycle marker gene lists
# Used by sc.tl.score_genes_cell_cycle() to assign G1/S/G2M phase per cell.
S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
    "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
    "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
    "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
    "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
]

G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
    "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
    "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
    "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
    "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
    "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]

# Required parquet files (uncorrected + combat). Harmony files are optional.
_REQUIRED_CACHE_FILES = [
    "uncorrected_expr.parquet",
    "combat_expr.parquet",
    "obs_metadata.parquet",
    "uncorrected_umap.parquet",
    "combat_umap.parquet",
    "uncorrected_hk_expr.parquet",
    "combat_hk_expr.parquet",
    "combat_obs_metadata.parquet",
]

_HARMONY_CACHE_FILES = [
    "harmony_expr.parquet",
    "harmony_umap.parquet",
    "harmony_hk_expr.parquet",
    "harmony_obs_metadata.parquet",
]


def get_h5ad_paths() -> tuple[Path, Path, Path]:
    """Return (uncorrected, combat, harmony) h5ad paths."""
    return UNCORRECTED_H5AD, COMBAT_H5AD, HARMONY_H5AD


def has_harmony_cache() -> bool:
    """Check whether Harmony parquet files exist in the cache."""
    return all((CACHE_DIR / name).exists() for name in _HARMONY_CACHE_FILES)


def is_cache_stale() -> bool:
    """Check whether the Parquet cache is missing or older than source h5ad files."""
    if not MANIFEST_FILE.exists():
        return True

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    uncorr_path, combat_path, harmony_path = get_h5ad_paths()

    # Uncorrected and ComBat are required
    for key, path in [
        ("uncorrected_mtime", uncorr_path),
        ("combat_mtime", combat_path),
    ]:
        if key not in manifest:
            return True
        if not path.exists():
            return True
        if os.path.getmtime(path) > manifest[key]:
            return True

    for name in _REQUIRED_CACHE_FILES:
        if not (CACHE_DIR / name).exists():
            return True

    # Harmony is optional: if the h5ad exists but cache is missing or stale,
    # we rebuild. If the h5ad doesn't exist, we skip harmony entirely.
    if harmony_path.exists():
        if manifest.get("harmony_mtime", 0) < os.path.getmtime(harmony_path):
            return True
        for name in _HARMONY_CACHE_FILES:
            if not (CACHE_DIR / name).exists():
                return True

    return False


def _score_cell_cycle(adata) -> None:
    """Run cell cycle scoring on an AnnData object in-place."""
    import scanpy as sc

    s_present = [g for g in S_GENES if g in adata.var_names]
    g2m_present = [g for g in G2M_GENES if g in adata.var_names]
    print(f"  Cell cycle scoring: {len(s_present)}/{len(S_GENES)} S-phase, "
          f"{len(g2m_present)}/{len(G2M_GENES)} G2M genes found")

    if len(s_present) < 2 or len(g2m_present) < 2:
        print("  WARNING: Too few cell cycle genes — assigning all cells to G1")
        adata.obs["phase"] = "G1"
        return

    sc.tl.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g2m_present)


def _extract_umap(adata) -> pd.DataFrame:
    """Extract UMAP coordinates as a DataFrame."""
    coords = adata.obsm["X_umap"]
    return pd.DataFrame(coords[:, :2], columns=["UMAP1", "UMAP2"], dtype=np.float32)


def _extract_hk_expression(adata) -> pd.DataFrame:
    """Extract housekeeping gene expression as a DataFrame."""
    from scribe.batch import DEFAULT_HOUSEKEEPING_GENES

    hk_present = [g for g in DEFAULT_HOUSEKEEPING_GENES if g in adata.var_names]
    X_hk = adata[:, hk_present].X
    if sp.issparse(X_hk):
        X_hk = X_hk.toarray()
    return pd.DataFrame(X_hk, columns=hk_present, dtype=np.float32)


def _dense_expr_df(adata) -> pd.DataFrame:
    """Return the full expression matrix as a float32 DataFrame."""
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return pd.DataFrame(X, columns=list(adata.var_names), dtype=np.float32)


def build_cache(force: bool = False) -> None:
    """Build Parquet cache from h5ad files, loading one at a time for memory safety."""
    import scanpy as sc

    if not force and not is_cache_stale():
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    uncorr_path, combat_path, harmony_path = get_h5ad_paths()

    # --- Uncorrected ---
    print("Loading uncorrected h5ad...")
    adata = sc.read_h5ad(str(uncorr_path))
    _score_cell_cycle(adata)

    _dense_expr_df(adata).to_parquet(
        CACHE_DIR / "uncorrected_expr.parquet", engine="pyarrow"
    )

    obs_cols = ["dataset", "condition", "leiden", "phase", "sample"]
    adata.obs[obs_cols].copy().reset_index(drop=True).to_parquet(
        CACHE_DIR / "obs_metadata.parquet", engine="pyarrow"
    )

    _extract_umap(adata).to_parquet(
        CACHE_DIR / "uncorrected_umap.parquet", engine="pyarrow"
    )
    _extract_hk_expression(adata).to_parquet(
        CACHE_DIR / "uncorrected_hk_expr.parquet", engine="pyarrow"
    )

    uncorr_mtime = os.path.getmtime(uncorr_path)
    del adata
    gc.collect()

    # --- ComBat ---
    print("Loading ComBat-corrected h5ad...")
    adata = sc.read_h5ad(str(combat_path))

    # zarr_to_h5ad copies obsm verbatim from uncorrected store, so embeddings
    # reflect *uncorrected* geometry. Recompute on the corrected data so the
    # "before vs after" view shows two different embeddings. PCA runs on the
    # z-scored layer (X itself is log1p under the new contract).
    print("  Recomputing PCA / neighbors / UMAP / leiden on ComBat data...")
    pca_layer = "X_norm" if "X_norm" in adata.layers else None
    sc.pp.pca(adata, layer=pca_layer, n_comps=50)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
    _score_cell_cycle(adata)

    _dense_expr_df(adata).to_parquet(
        CACHE_DIR / "combat_expr.parquet", engine="pyarrow"
    )
    _extract_umap(adata).to_parquet(
        CACHE_DIR / "combat_umap.parquet", engine="pyarrow"
    )
    _extract_hk_expression(adata).to_parquet(
        CACHE_DIR / "combat_hk_expr.parquet", engine="pyarrow"
    )
    adata.obs[["leiden", "phase"]].copy().reset_index(drop=True).to_parquet(
        CACHE_DIR / "combat_obs_metadata.parquet", engine="pyarrow"
    )

    combat_mtime = os.path.getmtime(combat_path)
    del adata
    gc.collect()

    # --- Harmony (optional) ---
    harmony_mtime = None
    if harmony_path.exists():
        print("Loading Harmony-corrected h5ad...")
        adata = sc.read_h5ad(str(harmony_path))

        # Harmony pipeline writes X_pca_harmony to obsm. Use it for
        # neighbors/UMAP/leiden, since Harmony's native output is the
        # corrected PC embedding (not gene expression).
        print("  Recomputing neighbors / UMAP / leiden on X_pca_harmony...")
        if "X_pca_harmony" in adata.obsm:
            sc.pp.neighbors(adata, use_rep="X_pca_harmony")
        else:
            print("  WARNING: X_pca_harmony not in obsm; falling back to X_pca")
            pca_layer = "X_norm" if "X_norm" in adata.layers else None
            sc.pp.pca(adata, layer=pca_layer, n_comps=50)
            sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
        _score_cell_cycle(adata)

        _dense_expr_df(adata).to_parquet(
            CACHE_DIR / "harmony_expr.parquet", engine="pyarrow"
        )
        _extract_umap(adata).to_parquet(
            CACHE_DIR / "harmony_umap.parquet", engine="pyarrow"
        )
        _extract_hk_expression(adata).to_parquet(
            CACHE_DIR / "harmony_hk_expr.parquet", engine="pyarrow"
        )
        adata.obs[["leiden", "phase"]].copy().reset_index(drop=True).to_parquet(
            CACHE_DIR / "harmony_obs_metadata.parquet", engine="pyarrow"
        )

        harmony_mtime = os.path.getmtime(harmony_path)
        del adata
        gc.collect()
    else:
        print(f"Harmony h5ad not found at {harmony_path} — skipping Harmony cache.")
        print("  (Run `scribe correct-zarr --method harmony` to enable.)")

    # --- Manifest ---
    manifest = {
        "uncorrected_mtime": uncorr_mtime,
        "combat_mtime": combat_mtime,
    }
    if harmony_mtime is not None:
        manifest["harmony_mtime"] = harmony_mtime
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Cache built successfully.")


# ── Loaders ──────────────────────────────────────────────────────────────────
# All loaders take a `method` argument: "uncorrected" | "combat" | "harmony".


def _method_prefix(method: str) -> str:
    """Validate method and return its parquet filename prefix."""
    if method not in ("uncorrected", "combat", "harmony"):
        raise ValueError(
            f"Invalid method '{method}'. Expected one of: uncorrected, combat, harmony."
        )
    return method


def load_gene_expression(genes: list[str], method: str = "uncorrected") -> pd.DataFrame:
    """Load only the requested gene columns from the Parquet cache."""
    prefix = _method_prefix(method)
    return pd.read_parquet(
        CACHE_DIR / f"{prefix}_expr.parquet", columns=genes, engine="pyarrow"
    )


def load_obs_metadata() -> pd.DataFrame:
    """Load cell metadata (dataset, condition, leiden, phase, sample)."""
    return pd.read_parquet(CACHE_DIR / "obs_metadata.parquet", engine="pyarrow")


def load_method_obs(method: str) -> pd.DataFrame:
    """Load per-method obs (leiden + phase recomputed on that method's embedding).

    For 'uncorrected', returns the full obs metadata table.
    """
    if method == "uncorrected":
        return load_obs_metadata()
    prefix = _method_prefix(method)
    return pd.read_parquet(
        CACHE_DIR / f"{prefix}_obs_metadata.parquet", engine="pyarrow"
    )


def load_umap_coords(method: str = "uncorrected") -> pd.DataFrame:
    """Load 2D UMAP coordinates. Returns DataFrame with UMAP1, UMAP2 columns."""
    prefix = _method_prefix(method)
    return pd.read_parquet(
        CACHE_DIR / f"{prefix}_umap.parquet", engine="pyarrow"
    )


def load_hk_expression(method: str = "uncorrected") -> pd.DataFrame:
    """Load housekeeping gene expression for live PCA computation."""
    prefix = _method_prefix(method)
    return pd.read_parquet(
        CACHE_DIR / f"{prefix}_hk_expr.parquet", engine="pyarrow"
    )


def get_gene_list() -> list[str]:
    """Read gene names from the Parquet schema without loading expression data."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(CACHE_DIR / "uncorrected_expr.parquet")
    return schema.names


# ── Legacy wrappers (kept for backwards compat; prefer the method-based API) ─


def load_corrected_obs_metadata() -> pd.DataFrame:
    """Legacy: load ComBat obs. Prefer load_method_obs('combat')."""
    return load_method_obs("combat")
