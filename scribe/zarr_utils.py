"""Zarr-backed AnnData utilities for memory-efficient processing.

Provides functions to:
- Convert h5ad files to Zarr stores with configurable chunking
- Load AnnData from Zarr in backed (on-disk) mode
- Process large datasets in cell-chunks to avoid OOM
- Run memory-efficient batch correction (chunked ComBat alternative)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import anndata
import numpy as np
import scipy.sparse


def h5ad_to_zarr(
    h5ad_path: str,
    zarr_path: str | None = None,
    chunk_size: int = 5000,
    overwrite: bool = False,
) -> str:
    """Convert an h5ad file to a Zarr store for memory-efficient access.

    The Zarr format stores data in chunked arrays on disk. Each chunk can
    be loaded independently, so you never need the full matrix in memory.

    Args:
        h5ad_path: Path to input .h5ad file.
        zarr_path: Output path for the .zarr store. Defaults to replacing
            the .h5ad extension with .zarr.
        chunk_size: Number of cells per chunk along the obs (row) axis.
            Smaller chunks use less memory but have more I/O overhead.
        overwrite: If True, delete existing zarr store before writing.

    Returns:
        Path to the created Zarr store.
    """
    import zarr

    if zarr_path is None:
        zarr_path = str(Path(h5ad_path).with_suffix(".zarr"))

    if os.path.exists(zarr_path):
        if overwrite:
            shutil.rmtree(zarr_path)
        else:
            print(f"Zarr store already exists: {zarr_path}")
            print("  Pass --overwrite to replace it.")
            return zarr_path

    print(f"Converting {h5ad_path} -> {zarr_path}")
    print(f"  Chunk size: {chunk_size} cells")

    # Load the h5ad — this reads it fully into memory once.
    # For very large files, we read in backed mode first.
    import scanpy as sc
    adata = sc.read_h5ad(h5ad_path)
    n_cells, n_genes = adata.shape
    print(f"  Shape: {n_cells} cells × {n_genes} genes")

    # Densify sparse matrix if needed (Zarr stores dense arrays).
    # We do this chunk-by-chunk to avoid a full dense copy in memory.
    store = zarr.open(zarr_path, mode="w")

    # Write X matrix in chunks
    z_X = store.create_dataset(
        "X",
        shape=(n_cells, n_genes),
        chunks=(min(chunk_size, n_cells), n_genes),
        dtype="float32",
    )

    print("  Writing expression matrix in chunks...")
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = adata.X[start:end]
        if scipy.sparse.issparse(chunk):
            chunk = chunk.toarray()
        z_X[start:end] = np.asarray(chunk, dtype="float32")
        pct = end / n_cells * 100
        print(f"    {end}/{n_cells} cells ({pct:.0f}%)", end="\r")
    print()

    # Zarr 2.x needs an object_codec for variable-length string arrays.
    import numcodecs
    str_codec = numcodecs.VLenUTF8()

    def _write_string_array(group, name, data):
        """Write a string array with the correct codec for zarr 2.x."""
        arr = np.asarray(data, dtype=object)
        group.create_dataset(name, data=arr, object_codec=str_codec)

    # Write obs (cell metadata) as individual arrays
    obs_group = store.create_group("obs")
    obs_group.attrs["_index"] = adata.obs.index.name or "_index"
    _write_string_array(obs_group, "_index", adata.obs.index.values.astype(str))
    for col in adata.obs.columns:
        values = adata.obs[col].values
        if hasattr(values, "codes"):
            # Categorical — store codes and categories separately
            cat_grp = obs_group.create_group(col)
            cat_grp.create_dataset("codes", data=values.codes)
            _write_string_array(cat_grp, "categories", values.categories.astype(str))
            cat_grp.attrs["dtype"] = "categorical"
        else:
            try:
                _write_string_array(obs_group, col, values.astype(str))
            except Exception:
                pass  # Skip columns that can't be serialized

    # Write var (gene metadata)
    var_group = store.create_group("var")
    var_group.attrs["_index"] = adata.var.index.name or "_index"
    _write_string_array(var_group, "_index", adata.var.index.values.astype(str))
    for col in adata.var.columns:
        try:
            values = adata.var[col].values
            if hasattr(values, "codes"):
                cat_grp = var_group.create_group(col)
                cat_grp.create_dataset("codes", data=values.codes)
                _write_string_array(cat_grp, "categories", values.categories.astype(str))
                cat_grp.attrs["dtype"] = "categorical"
            else:
                var_group.create_dataset(col, data=np.asarray(values))
        except Exception:
            pass

    # Write obsm embeddings (PCA, UMAP, etc.)
    if adata.obsm:
        obsm_group = store.create_group("obsm")
        for key, val in adata.obsm.items():
            obsm_group.create_dataset(
                key, data=np.asarray(val),
                chunks=(min(chunk_size, n_cells), val.shape[1] if val.ndim > 1 else 1),
            )

    # Store uns metadata as attributes where possible
    store.attrs["n_obs"] = n_cells
    store.attrs["n_vars"] = n_genes

    zarr.consolidate_metadata(zarr_path)

    size_mb = sum(
        f.stat().st_size for f in Path(zarr_path).rglob("*") if f.is_file()
    ) / 1e6
    print(f"  Zarr store saved: {zarr_path} ({size_mb:.1f} MB)")
    print(f"  Chunks: {len(range(0, n_cells, chunk_size))} × {n_genes} genes each")

    return zarr_path


def load_zarr_backed(zarr_path: str) -> dict:
    """Load a Zarr store in read-only backed mode (data stays on disk).

    Returns a dict with lazy references to the data — nothing is loaded
    into memory until you slice into a specific chunk.

    Args:
        zarr_path: Path to a .zarr store created by h5ad_to_zarr().

    Returns:
        Dict with keys: 'X' (zarr array), 'obs_names', 'var_names',
        'obs' (dict of arrays), 'obsm' (dict of arrays), 'n_obs', 'n_vars',
        'zarr_path'.
    """
    import zarr

    store = zarr.open(zarr_path, mode="r")

    result = {
        "X": store["X"],
        "n_obs": store.attrs["n_obs"],
        "n_vars": store.attrs["n_vars"],
        "zarr_path": zarr_path,
    }

    # Read obs metadata
    obs = {}
    if "obs" in store:
        obs_group = store["obs"]
        result["obs_names"] = np.array(obs_group["_index"])
        for key in obs_group:
            if key == "_index":
                continue
            item = obs_group[key]
            if hasattr(item, "attrs") and item.attrs.get("dtype") == "categorical":
                codes = np.array(item["codes"])
                cats = np.array(item["categories"])
                obs[key] = cats[codes]
            elif hasattr(item, "shape"):
                obs[key] = np.array(item)
    result["obs"] = obs

    # Read var metadata
    if "var" in store:
        result["var_names"] = np.array(store["var"]["_index"])

    # Read obsm (lazy)
    obsm = {}
    if "obsm" in store:
        for key in store["obsm"]:
            obsm[key] = store["obsm"][key]
    result["obsm"] = obsm

    return result


def chunk_iterator(zarr_data: dict, chunk_size: int = 5000):
    """Iterate over cells in chunks from a Zarr-backed dataset.

    Yields (start_idx, end_idx, X_chunk) tuples where X_chunk is a
    dense numpy array of shape (chunk_cells, n_genes).

    Args:
        zarr_data: Dict from load_zarr_backed().
        chunk_size: Number of cells per chunk.

    Yields:
        Tuples of (start, end, X_chunk_array).
    """
    n_obs = zarr_data["n_obs"]
    X = zarr_data["X"]

    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        chunk = np.array(X[start:end])
        yield start, end, chunk


def zarr_to_adata_chunk(
    zarr_data: dict,
    start: int,
    end: int,
) -> anndata.AnnData:
    """Load a single chunk from Zarr as a proper AnnData object.

    Useful when you need scanpy functions that require AnnData input
    but only want to process a subset of cells.

    Args:
        zarr_data: Dict from load_zarr_backed().
        start: Start cell index.
        end: End cell index.

    Returns:
        AnnData for the specified cell range.
    """
    import pandas as pd

    X = np.array(zarr_data["X"][start:end])

    obs_dict = {}
    for key, values in zarr_data["obs"].items():
        obs_dict[key] = values[start:end]
    obs = pd.DataFrame(obs_dict)

    if "obs_names" in zarr_data:
        obs.index = zarr_data["obs_names"][start:end]

    var = pd.DataFrame(index=zarr_data.get("var_names", np.arange(X.shape[1])))

    return anndata.AnnData(X=X, obs=obs, var=var)


def chunked_combat(
    zarr_path: str,
    batch_key: str = "dataset",
    chunk_size: int = 5000,
    output_zarr_path: str | None = None,
) -> str:
    """Memory-efficient batch correction using per-batch standardization.

    Standard ComBat requires the full expression matrix in dense memory,
    which OOMs on 8 GB machines for large datasets. This function implements
    a chunked approach:

    1. Compute per-batch means and variances in a single streaming pass
       (never loads more than one chunk at a time).
    2. Apply per-batch z-score normalization to shift each batch to the
       global mean/variance, writing corrected chunks to a new Zarr store.

    This is a location-scale adjustment similar to ComBat's core operation,
    but computed in constant memory.

    Args:
        zarr_path: Input Zarr store path.
        batch_key: Obs column identifying the batch.
        chunk_size: Cells per processing chunk.
        output_zarr_path: Output path. Defaults to input with '_corrected' suffix.

    Returns:
        Path to the corrected Zarr store.
    """
    import zarr

    if output_zarr_path is None:
        output_zarr_path = zarr_path.replace(".zarr", "_corrected.zarr")

    print(f"Chunked batch correction: {zarr_path}")
    zdata = load_zarr_backed(zarr_path)
    n_obs = zdata["n_obs"]
    n_vars = zdata["n_vars"]
    batch_labels = zdata["obs"].get(batch_key)

    if batch_labels is None:
        raise ValueError(f"Batch key '{batch_key}' not found in obs columns")

    batches = np.unique(batch_labels)
    print(f"  {n_obs} cells × {n_vars} genes, {len(batches)} batches")

    # ── Pass 1: Compute per-batch running mean and variance ──────────────
    print("  Pass 1: Computing per-batch statistics...")
    batch_stats = {}
    for b in batches:
        batch_stats[b] = {"sum": np.zeros(n_vars, dtype=np.float64),
                          "sum_sq": np.zeros(n_vars, dtype=np.float64),
                          "count": 0}

    # Also compute global stats
    global_sum = np.zeros(n_vars, dtype=np.float64)
    global_sum_sq = np.zeros(n_vars, dtype=np.float64)
    global_count = 0

    for start, end, X_chunk in chunk_iterator(zdata, chunk_size):
        chunk_batches = batch_labels[start:end]
        for b in batches:
            mask = chunk_batches == b
            if not mask.any():
                continue
            x_b = X_chunk[mask].astype(np.float64)
            batch_stats[b]["sum"] += x_b.sum(axis=0)
            batch_stats[b]["sum_sq"] += (x_b ** 2).sum(axis=0)
            batch_stats[b]["count"] += mask.sum()

        global_sum += X_chunk.astype(np.float64).sum(axis=0)
        global_sum_sq += (X_chunk.astype(np.float64) ** 2).sum(axis=0)
        global_count += (end - start)

    # Compute means and stds
    global_mean = global_sum / global_count
    global_var = (global_sum_sq / global_count) - (global_mean ** 2)
    global_std = np.sqrt(np.maximum(global_var, 1e-8))

    for b in batches:
        s = batch_stats[b]
        n = s["count"]
        if n == 0:
            continue
        mean = s["sum"] / n
        var = (s["sum_sq"] / n) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 1e-8))
        batch_stats[b]["mean"] = mean
        batch_stats[b]["std"] = std
        batch_stats[b]["n"] = n
        print(f"    {b}: {n} cells, mean RSS {mean.mean():.3f}")

    # ── Pass 2: Apply location-scale correction and write to output ──────
    print("  Pass 2: Applying correction and writing output...")

    if os.path.exists(output_zarr_path):
        shutil.rmtree(output_zarr_path)
    out_store = zarr.open(output_zarr_path, mode="w")
    z_X_out = out_store.create_dataset(
        "X", shape=(n_obs, n_vars),
        chunks=(min(chunk_size, n_obs), n_vars),
        dtype="float32",
    )

    for start, end, X_chunk in chunk_iterator(zdata, chunk_size):
        chunk_batches = batch_labels[start:end]
        corrected_chunk = np.empty_like(X_chunk, dtype=np.float32)

        for b in batches:
            mask = chunk_batches == b
            if not mask.any():
                continue
            x_b = X_chunk[mask].astype(np.float64)
            # Z-score within batch, then rescale to global distribution
            z_scored = (x_b - batch_stats[b]["mean"]) / batch_stats[b]["std"]
            corrected = z_scored * global_std + global_mean
            corrected_chunk[mask] = corrected.astype(np.float32)

        z_X_out[start:end] = corrected_chunk
        pct = end / n_obs * 100
        print(f"    {end}/{n_obs} cells ({pct:.0f}%)", end="\r")
    print()

    # Copy obs, var, obsm from input store
    in_store = zarr.open(zarr_path, mode="r")
    for group_name in ["obs", "var", "obsm"]:
        if group_name in in_store:
            zarr.copy(in_store[group_name], out_store, name=group_name)

    out_store.attrs.update(dict(in_store.attrs))
    zarr.consolidate_metadata(output_zarr_path)

    size_mb = sum(
        f.stat().st_size for f in Path(output_zarr_path).rglob("*") if f.is_file()
    ) / 1e6
    print(f"  Corrected store saved: {output_zarr_path} ({size_mb:.1f} MB)")

    return output_zarr_path


def chunked_harmony(
    zarr_path: str,
    batch_key: str = "dataset",
    n_pcs: int = 50,
    source_h5ad: str | None = None,
    output_zarr_path: str | None = None,
) -> str:
    """Memory-efficient Harmony batch correction on a Zarr store.

    Harmony corrects the PCA embedding rather than the gene expression
    matrix. The step itself is cheap — only the (n_cells × n_pcs) PCA
    matrix is loaded. We keep the zarr-backed pattern to avoid loading
    the full X matrix alongside downstream inverse-PCA reconstruction.

    Pipeline:
    1. Read obsm['X_pca'] from the input zarr (small, ~9 MB).
    2. Run harmonypy on (X_pca, obs[batch_key]).
    3. Copy X, obs, var, obsm from input -> output zarr.
    4. Write obsm['X_pca_harmony'] (corrected PC coordinates).
    5. Copy varm['PCs'] (gene loadings) from source_h5ad if provided —
       needed by chunked_inverse_pca() to project PCs back to gene space.

    Args:
        zarr_path: Input zarr store (from h5ad_to_zarr).
        batch_key: Obs column identifying the batch.
        n_pcs: Number of PCs to feed into Harmony.
        source_h5ad: Path to the original h5ad (supplies varm['PCs']).
            If None, the output zarr will lack varm — downstream
            inverse-PCA will need to refit loadings.
        output_zarr_path: Output path. Defaults to <input>_harmony.zarr.

    Returns:
        Path to the Harmony-corrected Zarr store.
    """
    import zarr
    try:
        import harmonypy
    except ImportError:
        raise ImportError(
            "harmonypy is required for Harmony batch correction. "
            "Install it with: pip install harmonypy"
        )

    if output_zarr_path is None:
        output_zarr_path = zarr_path.replace(".zarr", "_harmony.zarr")

    print(f"Chunked Harmony correction: {zarr_path}")
    zdata = load_zarr_backed(zarr_path)
    n_obs = zdata["n_obs"]
    n_vars = zdata["n_vars"]

    batch_labels = zdata["obs"].get(batch_key)
    if batch_labels is None:
        raise ValueError(f"Batch key '{batch_key}' not found in obs columns")

    if "X_pca" not in zdata["obsm"]:
        raise ValueError(
            "obsm['X_pca'] is missing from the zarr store. "
            "Run the standard preprocessing pipeline to generate it before Harmony."
        )

    X_pca = np.asarray(zdata["obsm"]["X_pca"])[:, :n_pcs]
    print(f"  {n_obs} cells × {n_vars} genes, PCA shape {X_pca.shape}")

    # Build a minimal pandas DataFrame for harmony (it wants a DataFrame
    # column named batch_key).
    import pandas as pd
    meta = pd.DataFrame({batch_key: batch_labels})
    print(f"  Running Harmony on {n_pcs} PCs, batch_key='{batch_key}'...")
    # Force CPU device: MPS (Apple GPU) has OpenMP conflicts with sklearn.KMeans
    # and segfaults on macOS. CPU is slower but reliable.
    try:
        harmony_out = harmonypy.run_harmony(X_pca, meta, batch_key, device="cpu")
    except TypeError:
        # Older harmonypy without `device` kwarg
        harmony_out = harmonypy.run_harmony(X_pca, meta, batch_key)

    Z = harmony_out.Z_corr
    if hasattr(Z, "numpy"):
        Z = Z.numpy()
    Z = np.asarray(Z, dtype=np.float32)
    if Z.shape[0] == n_pcs and Z.shape[1] == n_obs:
        Z = Z.T
    assert Z.shape == (n_obs, n_pcs), f"Unexpected Harmony output shape: {Z.shape}"
    print(f"  Harmony output shape: {Z.shape}")

    # ── Write output zarr ────────────────────────────────────────────────
    if os.path.exists(output_zarr_path):
        shutil.rmtree(output_zarr_path)
    out_store = zarr.open(output_zarr_path, mode="w")
    in_store = zarr.open(zarr_path, mode="r")

    # Copy X, obs, var, obsm unchanged
    for group_name in ["X", "obs", "var", "obsm"]:
        if group_name in in_store:
            zarr.copy(in_store[group_name], out_store, name=group_name)

    # Add X_pca_harmony
    obsm_out = out_store["obsm"]
    obsm_out.create_dataset(
        "X_pca_harmony", data=Z,
        chunks=(min(5000, n_obs), n_pcs),
    )

    # Copy varm['PCs'] and gene means from the source h5ad if available.
    # varm is needed by chunked_inverse_pca() to project back to gene space.
    if source_h5ad is not None and os.path.exists(source_h5ad):
        print(f"  Copying varm['PCs'] from {source_h5ad}...")
        import scanpy as sc
        src = sc.read_h5ad(source_h5ad)
        if "PCs" in src.varm:
            varm_grp = out_store.create_group("varm")
            varm_grp.create_dataset(
                "PCs", data=np.asarray(src.varm["PCs"], dtype=np.float32),
            )
            print(f"    PCs shape: {src.varm['PCs'].shape}")
        else:
            print("    WARNING: source h5ad has no varm['PCs']; inverse PCA will need to refit.")
        del src

    out_store.attrs.update(dict(in_store.attrs))
    zarr.consolidate_metadata(output_zarr_path)

    size_mb = sum(
        f.stat().st_size for f in Path(output_zarr_path).rglob("*") if f.is_file()
    ) / 1e6
    print(f"  Harmony store saved: {output_zarr_path} ({size_mb:.1f} MB)")

    return output_zarr_path


def chunked_inverse_pca(
    zarr_path: str,
    output_zarr_path: str | None = None,
    rep: str = "X_pca_harmony",
    chunk_size: int = 5000,
) -> str:
    """Reconstruct an approximate gene-expression matrix from a PC embedding.

    After Harmony corrects PC coordinates, we project them back to gene
    space so the KDE and HK-PCA viewers can treat Harmony as a peer of
    ComBat. This is a **lossy truncated reconstruction** — only variance
    captured by the top-n_pcs PCs is recovered.

    Formula (X already centered/scaled by sc.pp.scale):
        X_hat = X_pca_harmony @ PCs.T

    Args:
        zarr_path: Input zarr (output of chunked_harmony). Must have
            obsm[rep] and varm['PCs'].
        output_zarr_path: Output zarr path. Defaults to stripping any
            '_harmony' suffix and appending '_harmony_reconstructed.zarr'.
            In practice we overwrite the input zarr's X so downstream
            commands see a standard (X, obs, var, obsm) layout.
        rep: obsm key holding the corrected PC matrix.
        chunk_size: Cells per write chunk.

    Returns:
        Path to the zarr with reconstructed X.
    """
    import zarr

    if output_zarr_path is None:
        output_zarr_path = zarr_path.replace(".zarr", "_reconstructed.zarr")

    print(f"Inverse-PCA reconstruction: {zarr_path} -> {output_zarr_path}")
    in_store = zarr.open(zarr_path, mode="r")

    if "obsm" not in in_store or rep not in in_store["obsm"]:
        raise ValueError(f"obsm['{rep}'] missing from {zarr_path}")
    if "varm" not in in_store or "PCs" not in in_store["varm"]:
        raise ValueError(
            f"varm['PCs'] missing from {zarr_path}. "
            "chunked_harmony() must be called with source_h5ad pointing to "
            "an h5ad that has PCA loadings."
        )

    pcs_harmony = np.asarray(in_store["obsm"][rep])   # (n_obs, n_pcs)
    PCs = np.asarray(in_store["varm"]["PCs"])         # (n_vars, n_pcs)
    n_obs, n_pcs = pcs_harmony.shape
    n_vars = PCs.shape[0]

    # Ensure compatible n_pcs
    if PCs.shape[1] != n_pcs:
        k = min(PCs.shape[1], n_pcs)
        print(f"  Truncating to {k} PCs (PCs: {PCs.shape[1]}, harmony: {n_pcs})")
        pcs_harmony = pcs_harmony[:, :k]
        PCs = PCs[:, :k]

    print(f"  Reconstructing {n_obs} cells × {n_vars} genes from {PCs.shape[1]} PCs")

    if os.path.exists(output_zarr_path):
        shutil.rmtree(output_zarr_path)
    out_store = zarr.open(output_zarr_path, mode="w")

    # Copy everything except X
    for group_name in ["obs", "var", "obsm", "varm"]:
        if group_name in in_store:
            zarr.copy(in_store[group_name], out_store, name=group_name)
    out_store.attrs.update(dict(in_store.attrs))

    # Create new X and fill it chunk-by-chunk
    z_X = out_store.create_dataset(
        "X", shape=(n_obs, n_vars),
        chunks=(min(chunk_size, n_obs), n_vars),
        dtype="float32",
    )

    PCs_T = PCs.T.astype(np.float32)  # (n_pcs, n_vars)
    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        chunk_pcs = pcs_harmony[start:end].astype(np.float32)
        z_X[start:end] = chunk_pcs @ PCs_T
        pct = end / n_obs * 100
        print(f"    {end}/{n_obs} cells ({pct:.0f}%)", end="\r")
    print()

    zarr.consolidate_metadata(output_zarr_path)

    size_mb = sum(
        f.stat().st_size for f in Path(output_zarr_path).rglob("*") if f.is_file()
    ) / 1e6
    print(f"  Reconstructed store saved: {output_zarr_path} ({size_mb:.1f} MB)")

    return output_zarr_path


def zarr_to_h5ad(zarr_path: str, h5ad_path: str | None = None, chunk_size: int = 5000) -> str:
    """Convert a Zarr store back to h5ad format.

    Reads the Zarr store chunk by chunk and assembles an AnnData object,
    then writes it as h5ad.

    Args:
        zarr_path: Path to the .zarr store.
        h5ad_path: Output .h5ad path. Defaults to replacing .zarr with .h5ad.
        chunk_size: Cells per read chunk.

    Returns:
        Path to the written .h5ad file.
    """
    import pandas as pd

    if h5ad_path is None:
        h5ad_path = zarr_path.replace(".zarr", ".h5ad")

    print(f"Converting {zarr_path} -> {h5ad_path}")
    zdata = load_zarr_backed(zarr_path)
    n_obs = zdata["n_obs"]
    n_vars = zdata["n_vars"]

    # Read X in chunks and build a sparse matrix to save memory
    from scipy.sparse import vstack, csr_matrix

    chunks = []
    for start, end, X_chunk in chunk_iterator(zdata, chunk_size):
        chunks.append(csr_matrix(X_chunk))
        pct = end / n_obs * 100
        print(f"  Reading: {end}/{n_obs} ({pct:.0f}%)", end="\r")
    print()

    X_sparse = vstack(chunks, format="csr")

    obs = pd.DataFrame(zdata["obs"])
    if "obs_names" in zdata:
        obs.index = zdata["obs_names"]

    var = pd.DataFrame(index=zdata.get("var_names", np.arange(n_vars)))

    adata = anndata.AnnData(X=X_sparse, obs=obs, var=var)

    # Restore obsm embeddings
    for key, arr in zdata["obsm"].items():
        adata.obsm[key] = np.array(arr)

    adata.write_h5ad(h5ad_path)
    size_mb = os.path.getsize(h5ad_path) / 1e6
    print(f"  Saved: {h5ad_path} ({size_mb:.1f} MB)")

    return h5ad_path
