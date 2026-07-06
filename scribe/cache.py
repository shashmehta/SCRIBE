"""Parquet cache for the interactive Marimo app.

Converts h5ad expression matrices into columnar Parquet files so the app
can load only the genes the user selects, keeping memory low on 8 GB machines.

Supports three method variants: "uncorrected", "combat", "harmony".
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scribe import paths

# Files that live on Drive (large expression matrices, loaded on-demand per gene).
_DRIVE_FILES = frozenset({
    "uncorrected_expr.parquet",
    "combat_expr.parquet",
    "harmony_expr.parquet",
})


def _cache_path(filename: str) -> Path:
    """Route a cache filename to local disk (small) or Drive (large)."""
    if filename in _DRIVE_FILES:
        return paths.get_drive_cache_dir() / filename
    return paths.get_local_cache_dir() / filename

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

# Subset checked to decide if Harmony panels can be shown in the app.
# harmony_expr.parquet (large Drive-only file) is excluded — the UMAP/HK/PCA
# panels only need the local small files, which are bundled in the deployment.
_HARMONY_LOCAL_FILES = [
    "harmony_umap.parquet",
    "harmony_hk_expr.parquet",
    "harmony_obs_metadata.parquet",
    "harmony_hk_pca.parquet",
]


def get_h5ad_paths() -> tuple[Path, Path, Path]:
    """Return (uncorrected, combat, harmony) h5ad paths."""
    return paths.get_h5ad_paths()


def has_harmony_cache() -> bool:
    """Check whether the Harmony local (small) parquet files exist in the cache."""
    return all(_cache_path(name).exists() for name in _HARMONY_LOCAL_FILES)


def has_full_expression_cache() -> bool:
    """Whether the large per-gene expression matrices are present.

    These are Drive-only files, absent from the HF Spaces deployment, so the
    app restricts the gene distribution viewer to bundled housekeeping genes
    when this returns False.
    """
    return _cache_path("uncorrected_expr.parquet").exists()


def _file_fingerprint(path: Path) -> str:
    """Fast content fingerprint: size + hash of first/last 16 KB.

    Stable across machines (depends on content, not filesystem metadata),
    so Google Drive sync won't cause spurious cache invalidation.
    """
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(16384))
        if size > 16384:
            f.seek(max(0, size - 16384))
            h.update(f.read(16384))
    return h.hexdigest()


def is_cache_stale() -> bool:
    """Check whether the Parquet cache is missing or differs from source h5ad files."""
    manifest_path = _cache_path("cache_manifest.json")
    if not manifest_path.exists():
        return True

    with open(manifest_path) as f:
        manifest = json.load(f)

    uncorr_path, combat_path, harmony_path = get_h5ad_paths()

    # Uncorrected and ComBat are required
    for key, path in [
        ("uncorrected_fingerprint", uncorr_path),
        ("combat_fingerprint", combat_path),
    ]:
        if key not in manifest:
            return True
        if not path.exists():
            return True
        if _file_fingerprint(path) != manifest[key]:
            return True

    for name in _REQUIRED_CACHE_FILES:
        if not _cache_path(name).exists():
            return True

    # Harmony is optional: if the h5ad exists but cache is missing or stale,
    # we rebuild. If the h5ad doesn't exist, we skip harmony entirely.
    if harmony_path.exists():
        fp = _file_fingerprint(harmony_path)
        if manifest.get("harmony_fingerprint") != fp:
            return True
        for name in _HARMONY_CACHE_FILES:
            if not _cache_path(name).exists():
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
    """Build Parquet cache from h5ad files, loading one at a time for memory safety.

    Small files (metadata, UMAP, HK) go to local disk for fast reads.
    Large expression parquets go to Drive (loaded on-demand per gene).
    """
    import scanpy as sc

    if not force and not is_cache_stale():
        return

    local_dir = paths.get_local_cache_dir()
    drive_dir = paths.get_drive_cache_dir()
    local_dir.mkdir(parents=True, exist_ok=True)
    drive_dir.mkdir(parents=True, exist_ok=True)

    uncorr_path, combat_path, harmony_path = get_h5ad_paths()

    # --- Uncorrected ---
    print("Loading uncorrected h5ad...")
    adata = sc.read_h5ad(str(uncorr_path))
    _score_cell_cycle(adata)

    expr_df = _dense_expr_df(adata)
    expr_df.to_parquet(
        _cache_path("uncorrected_expr.parquet"), engine="pyarrow"
    )
    # Small bundled gene-name list so get_gene_list() works without the parquet.
    (local_dir / "gene_list.json").write_text(json.dumps(expr_df.columns.tolist()))
    del expr_df

    obs_cols = ["dataset", "condition", "leiden", "phase", "sample"]
    adata.obs[obs_cols].copy().reset_index(drop=True).to_parquet(
        _cache_path("obs_metadata.parquet"), engine="pyarrow"
    )

    _extract_umap(adata).to_parquet(
        _cache_path("uncorrected_umap.parquet"), engine="pyarrow"
    )
    _extract_hk_expression(adata).to_parquet(
        _cache_path("uncorrected_hk_expr.parquet"), engine="pyarrow"
    )

    uncorr_fp = _file_fingerprint(uncorr_path)
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
        _cache_path("combat_expr.parquet"), engine="pyarrow"
    )
    _extract_umap(adata).to_parquet(
        _cache_path("combat_umap.parquet"), engine="pyarrow"
    )
    _extract_hk_expression(adata).to_parquet(
        _cache_path("combat_hk_expr.parquet"), engine="pyarrow"
    )
    adata.obs[["leiden", "phase"]].copy().reset_index(drop=True).to_parquet(
        _cache_path("combat_obs_metadata.parquet"), engine="pyarrow"
    )

    combat_fp = _file_fingerprint(combat_path)
    del adata
    gc.collect()

    # --- Harmony (optional) ---
    harmony_fp = None
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
            _cache_path("harmony_expr.parquet"), engine="pyarrow"
        )
        _extract_umap(adata).to_parquet(
            _cache_path("harmony_umap.parquet"), engine="pyarrow"
        )
        _extract_hk_expression(adata).to_parquet(
            _cache_path("harmony_hk_expr.parquet"), engine="pyarrow"
        )
        adata.obs[["leiden", "phase"]].copy().reset_index(drop=True).to_parquet(
            _cache_path("harmony_obs_metadata.parquet"), engine="pyarrow"
        )

        harmony_fp = _file_fingerprint(harmony_path)
        del adata
        gc.collect()
    else:
        print(f"Harmony h5ad not found at {harmony_path} — skipping Harmony cache.")
        print("  (Run `scribe correct-zarr --method harmony` to enable.)")

    # --- Manifest ---
    manifest = {
        "uncorrected_fingerprint": uncorr_fp,
        "combat_fingerprint": combat_fp,
    }
    if harmony_fp is not None:
        manifest["harmony_fingerprint"] = harmony_fp
    manifest_path = _cache_path("cache_manifest.json")
    with open(manifest_path, "w") as f:
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
        _cache_path(f"{prefix}_expr.parquet"), columns=genes, engine="pyarrow"
    )


def load_obs_metadata() -> pd.DataFrame:
    """Load cell metadata (dataset, condition, leiden, phase, sample)."""
    return pd.read_parquet(_cache_path("obs_metadata.parquet"), engine="pyarrow")


def load_method_obs(method: str) -> pd.DataFrame:
    """Load per-method obs (leiden + phase recomputed on that method's embedding).

    For 'uncorrected', returns the full obs metadata table.
    """
    if method == "uncorrected":
        return load_obs_metadata()
    prefix = _method_prefix(method)
    return pd.read_parquet(
        _cache_path(f"{prefix}_obs_metadata.parquet"), engine="pyarrow"
    )


def load_umap_coords(method: str = "uncorrected") -> pd.DataFrame:
    """Load 2D UMAP coordinates. Returns DataFrame with UMAP1, UMAP2 columns."""
    prefix = _method_prefix(method)
    return pd.read_parquet(
        _cache_path(f"{prefix}_umap.parquet"), engine="pyarrow"
    )


def load_hk_expression(method: str = "uncorrected") -> pd.DataFrame:
    """Load housekeeping gene expression for live PCA computation."""
    prefix = _method_prefix(method)
    return pd.read_parquet(
        _cache_path(f"{prefix}_hk_expr.parquet"), engine="pyarrow"
    )


def get_gene_list() -> list[str]:
    """Return gene names, reading from a small bundled JSON when available.

    Falls back to the large expression parquet's schema only if the JSON is
    absent — the JSON lets the HF Spaces deployment work without the Drive-only
    expression matrices.
    """
    gene_list_path = paths.get_local_cache_dir() / "gene_list.json"
    if gene_list_path.exists():
        return json.loads(gene_list_path.read_text())

    import pyarrow.parquet as pq

    schema = pq.read_schema(_cache_path("uncorrected_expr.parquet"))
    return schema.names


def load_hk_pca(method: str = "uncorrected") -> tuple[pd.DataFrame, tuple[float, float]]:
    """Load pre-computed 2-component HK gene PCA coordinates and variance ratios."""
    prefix = _method_prefix(method)
    df = pd.read_parquet(_cache_path(f"{prefix}_hk_pca.parquet"), engine="pyarrow")
    with open(_cache_path("hk_pca_meta.json")) as f:
        meta = json.load(f)
    return df, tuple(meta[method])


def get_hk_genes_from_cache() -> list[str]:
    """Return HK gene names used in PCA without loading expression data."""
    with open(_cache_path("hk_pca_meta.json")) as f:
        return json.load(f)["genes"]


def generate_default_umap_plot(plots_dir: Path | None = None) -> Path:
    """Pre-render the grey (no-annotation) UMAP panels as a PNG.

    Called at Docker build time so the app skips matplotlib rendering on startup.
    Reads from the Parquet cache — does NOT need h5ad files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if plots_dir is None:
        plots_dir = paths.get_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "umap_default.png"

    umap_uncorr = load_umap_coords("uncorrected")
    umap_combat = load_umap_coords("combat")
    harmony_ok = has_harmony_cache()
    umap_harmony = load_umap_coords("harmony") if harmony_ok else None

    n_panels = 3 if harmony_ok else 2
    titles = ["Uncorrected", "ComBat"] + (["Harmony"] if harmony_ok else [])
    coords_list = [umap_uncorr, umap_combat] + ([umap_harmony] if harmony_ok else [])

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(umap_uncorr))

    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    axes = list(axes) if n_panels > 1 else [axes]
    for ax, title, coords in zip(axes, titles, coords_list):
        ax.scatter(
            coords["UMAP1"].values[idx], coords["UMAP2"].values[idx],
            c="#cccccc", s=1, alpha=0.3, rasterized=True,
        )
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    fig.suptitle("UMAP", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Legacy wrappers (kept for backwards compat; prefer the method-based API) ─


def load_corrected_obs_metadata() -> pd.DataFrame:
    """Legacy: load ComBat obs. Prefer load_method_obs('combat')."""
    return load_method_obs("combat")
