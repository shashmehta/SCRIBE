"""Batch effect detection and correction for multi-dataset scRNA-seq analysis.

Provides functions to:
- Quantify batch effects via housekeeping gene expression
- Compute pairwise batch distances
- Correct batch effects using ComBat (default), Harmony, or Scanorama
- Measure batch mixing quality
- Compare correction methods
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import anndata

# Default housekeeping genes — stably expressed across cell types and conditions.
# Used as a baseline to detect systematic batch-driven expression shifts.
DEFAULT_HOUSEKEEPING_GENES = ["ACTB", "GAPDH", "B2M", "RPL13A", "RPLP0", "PPIA"]


def compute_housekeeping_expression(
    adata: anndata.AnnData,
    batch_key: str = "dataset",
    genes: list[str] | None = None,
) -> pd.DataFrame:
    """Mean expression of housekeeping genes per batch.

    These genes should have similar expression across all batches if there
    are no technical artifacts. Large differences indicate batch effects.

    Args:
        adata: AnnData with expression data and batch annotations.
        batch_key: Obs column identifying the batch.
        genes: Housekeeping gene names. Defaults to DEFAULT_HOUSEKEEPING_GENES.

    Returns:
        DataFrame with batches as rows, housekeeping genes as columns.
    """
    if genes is None:
        genes = DEFAULT_HOUSEKEEPING_GENES

    # Only use genes that exist in the dataset
    available = [g for g in genes if g in adata.var_names]
    if not available:
        raise ValueError(
            f"None of the housekeeping genes {genes} found in adata.var_names. "
            f"Available genes (first 10): {adata.var_names[:10].tolist()}"
        )

    # Subset to housekeeping genes
    subset = adata[:, available]

    # Get dense expression matrix
    import scipy.sparse
    X = subset.X.toarray() if scipy.sparse.issparse(subset.X) else np.array(subset.X)

    batches = adata.obs[batch_key]
    result = {}
    for batch in batches.unique():
        mask = (batches == batch).values
        result[batch] = X[mask].mean(axis=0)

    return pd.DataFrame(result, index=available).T


def compute_batch_distances(
    adata: anndata.AnnData,
    batch_key: str = "dataset",
    genes: list[str] | None = None,
) -> pd.DataFrame:
    """Pairwise Euclidean distances between batches in housekeeping gene space.

    Large distances between batches suggest systematic technical differences
    that should be corrected before downstream analysis.

    Args:
        adata: AnnData with expression data and batch annotations.
        batch_key: Obs column identifying the batch.
        genes: Housekeeping gene names. Defaults to DEFAULT_HOUSEKEEPING_GENES.

    Returns:
        Symmetric DataFrame of pairwise distances (diagonal = 0).
    """
    hk_expr = compute_housekeeping_expression(adata, batch_key, genes)
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(hk_expr.values, metric="euclidean"))
    return pd.DataFrame(dist_matrix, index=hk_expr.index, columns=hk_expr.index)


def correct_batch_combat(
    adata: anndata.AnnData,
    batch_key: str = "dataset",
) -> anndata.AnnData:
    """Apply ComBat batch correction (built into scanpy, zero extra dependencies).

    ComBat models batch effects as additive and multiplicative shifts in
    gene expression and removes them while preserving biological variation.

    Args:
        adata: AnnData with expression data and batch annotations.
        batch_key: Obs column identifying the batch.

    Returns:
        Copy of adata with batch-corrected expression in X.
    """
    corrected = adata.copy()
    sc.pp.combat(corrected, key=batch_key)
    # Recompute embeddings on corrected data
    sc.tl.pca(corrected)
    sc.pp.neighbors(corrected, n_pcs=30)
    sc.tl.umap(corrected)
    sc.tl.leiden(corrected, resolution=0.5)
    return corrected


def correct_batch_harmony(
    adata: anndata.AnnData,
    batch_key: str = "dataset",
    n_pcs: int = 30,
) -> anndata.AnnData:
    """Apply Harmony batch correction on PCA embeddings.

    Harmony iteratively adjusts PCA coordinates to remove batch effects
    while preserving biological variation. Operates in PCA space rather
    than on raw expression, so it's faster and often more robust.

    Requires the `harmonypy` package (pip install harmonypy).

    Args:
        adata: AnnData with pre-computed PCA.
        batch_key: Obs column identifying the batch.
        n_pcs: Number of PCs to use.

    Returns:
        Copy of adata with corrected PCA in obsm['X_pca_harmony']
        and recomputed neighbors/UMAP/Leiden.
    """
    try:
        import harmonypy
    except ImportError:
        raise ImportError(
            "harmonypy is required for Harmony batch correction. "
            "Install it with: pip install harmonypy"
        )

    corrected = adata.copy()

    # Ensure PCA is computed
    if "X_pca" not in corrected.obsm:
        sc.tl.pca(corrected, n_comps=n_pcs)

    # Run Harmony on the PCA embedding
    harmony_out = harmonypy.run_harmony(
        corrected.obsm["X_pca"][:, :n_pcs],
        corrected.obs,
        batch_key,
    )
    # Z_corr may be (n_pcs, n_cells) or (n_cells, n_pcs) depending on version
    Z = harmony_out.Z_corr
    if hasattr(Z, 'numpy'):
        Z = Z.numpy()  # convert from torch tensor if needed
    Z = np.array(Z)
    if Z.shape[0] == n_pcs and Z.shape[1] == corrected.n_obs:
        Z = Z.T  # transpose to (n_cells, n_pcs)
    corrected.obsm["X_pca_harmony"] = Z

    # Recompute neighbors/UMAP/Leiden using the corrected PCA
    sc.pp.neighbors(corrected, use_rep="X_pca_harmony")
    sc.tl.umap(corrected)
    sc.tl.leiden(corrected, resolution=0.5)
    return corrected


def correct_batch_scanorama(
    adata: anndata.AnnData,
    batch_key: str = "dataset",
) -> anndata.AnnData:
    """Apply Scanorama batch correction.

    Scanorama finds mutual nearest neighbors across batches to align
    datasets. Works well when batches share cell types but may have
    different proportions.

    Requires the `scanorama` package (pip install scanorama).

    Args:
        adata: AnnData with expression data and batch annotations.
        batch_key: Obs column identifying the batch.

    Returns:
        Copy of adata with corrected embedding in obsm['X_scanorama']
        and recomputed neighbors/UMAP/Leiden.
    """
    try:
        import scanorama
    except ImportError:
        raise ImportError(
            "scanorama is required for Scanorama batch correction. "
            "Install it with: pip install scanorama"
        )

    corrected = adata.copy()

    # Split by batch for Scanorama input
    batches = corrected.obs[batch_key].unique()
    adatas_list = [corrected[corrected.obs[batch_key] == b].copy() for b in batches]

    # Run Scanorama integration
    scanorama.integrate_scanpy(adatas_list)

    # Reassemble the corrected embedding
    corrected.obsm["X_scanorama"] = np.zeros((corrected.n_obs, adatas_list[0].obsm["X_scanorama"].shape[1]))
    for batch_adata, batch_name in zip(adatas_list, batches):
        mask = (corrected.obs[batch_key] == batch_name).values
        corrected.obsm["X_scanorama"][mask] = batch_adata.obsm["X_scanorama"]

    # Recompute neighbors/UMAP/Leiden using the corrected embedding
    sc.pp.neighbors(corrected, use_rep="X_scanorama")
    sc.tl.umap(corrected)
    sc.tl.leiden(corrected, resolution=0.5)
    return corrected


def compute_batch_mixing_score(
    adata: anndata.AnnData,
    batch_key: str = "dataset",
    n_neighbors: int = 50,
) -> float:
    """Fraction of k-nearest neighbors from a different batch.

    A score of 0 means batches are completely segregated (all neighbors
    are from the same batch). A score of 1 means perfect mixing (neighbors
    are equally likely from any batch). Higher is better after correction.

    Args:
        adata: AnnData with pre-computed neighbors graph.
        batch_key: Obs column identifying the batch.
        n_neighbors: Number of neighbors to consider.

    Returns:
        Float between 0 and 1 (higher = better mixing).
    """
    # Recompute neighbors with the requested k if needed
    if "neighbors" not in adata.uns or adata.uns["neighbors"]["params"]["n_neighbors"] != n_neighbors:
        adata_copy = adata.copy()
        sc.pp.neighbors(adata_copy, n_neighbors=n_neighbors)
    else:
        adata_copy = adata

    # Get the connectivities graph (sparse matrix of neighbor connections)
    conn = adata_copy.obsp["connectivities"]
    batch_labels = adata_copy.obs[batch_key].values

    # For each cell, count what fraction of its neighbors are from a different batch
    mixing_scores = []
    for i in range(conn.shape[0]):
        neighbors_idx = conn[i].nonzero()[1]
        if len(neighbors_idx) == 0:
            continue
        different_batch = sum(batch_labels[j] != batch_labels[i] for j in neighbors_idx)
        mixing_scores.append(different_batch / len(neighbors_idx))

    return float(np.mean(mixing_scores))


def compute_condition_distribution_distances(
    adata: anndata.AnnData,
    condition_col: str = "condition",
    batch_key: str = "dataset",
    conditions: list[str] | None = None,
    n_pcs: int = 30,
) -> dict[str, pd.DataFrame]:
    """Pairwise distribution distances between datasets for each condition.

    For each condition (e.g. malignant, normal), subsets cells to that
    condition and computes pairwise distances between datasets using
    Wasserstein, energy distance, and maximum mean discrepancy (MMD).
    Large distances for the same cell state across datasets suggest
    batch effects rather than biological variation.

    Args:
        adata: AnnData with PCA computed.
        condition_col: Obs column holding condition labels.
        batch_key: Obs column identifying the dataset/batch.
        conditions: Which conditions to evaluate. Defaults to all conditions
            present in at least two datasets.
        n_pcs: Number of PCs to use for distance computation.

    Returns:
        Dict mapping condition name to a DataFrame with columns
        [dataset_1, dataset_2, wasserstein, energy, mmd].
    """
    from scipy.stats import wasserstein_distance
    from itertools import combinations

    # Ensure PCA exists
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=n_pcs)

    pca = adata.obsm["X_pca"][:, :n_pcs]
    obs = adata.obs

    # Determine which conditions appear in at least 2 datasets
    if conditions is None:
        cond_datasets = obs.groupby(condition_col)[batch_key].nunique()
        conditions = cond_datasets[cond_datasets >= 2].index.tolist()

    results = {}
    for cond in conditions:
        cond_mask = (obs[condition_col] == cond).values
        datasets_in_cond = obs.loc[cond_mask, batch_key].unique()
        if len(datasets_in_cond) < 2:
            continue

        rows = []
        for ds_a, ds_b in combinations(sorted(datasets_in_cond), 2):
            mask_a = (cond_mask & (obs[batch_key] == ds_a).values)
            mask_b = (cond_mask & (obs[batch_key] == ds_b).values)
            pca_a = pca[mask_a]
            pca_b = pca[mask_b]

            # Wasserstein distance: average 1D Wasserstein across PCs
            w_dists = [
                wasserstein_distance(pca_a[:, pc], pca_b[:, pc])
                for pc in range(n_pcs)
            ]
            w_avg = float(np.mean(w_dists))

            # Energy distance
            e_dist = _energy_distance(pca_a, pca_b)

            # Maximum Mean Discrepancy (MMD) with RBF kernel
            mmd = _mmd_rbf(pca_a, pca_b)

            rows.append({
                "dataset_1": ds_a,
                "dataset_2": ds_b,
                "n_cells_1": int(mask_a.sum()),
                "n_cells_2": int(mask_b.sum()),
                "wasserstein_avg": round(w_avg, 4),
                "energy_distance": round(e_dist, 4),
                "mmd_rbf": round(mmd, 4),
            })

        results[cond] = pd.DataFrame(rows)

    return results


def _energy_distance(X: np.ndarray, Y: np.ndarray, max_samples: int = 5000) -> float:
    """Energy distance between two multivariate samples.

    Energy distance = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
    Subsample if datasets are large to keep computation tractable.
    """
    from scipy.spatial.distance import cdist

    rng = np.random.RandomState(42)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    xy = cdist(X, Y).mean()
    xx = cdist(X, X).mean()
    yy = cdist(Y, Y).mean()
    return float(2 * xy - xx - yy)


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, max_samples: int = 5000) -> float:
    """Maximum Mean Discrepancy with RBF kernel (median heuristic for bandwidth).

    MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
    """
    from scipy.spatial.distance import cdist

    rng = np.random.RandomState(42)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    XY = np.vstack([X, Y])
    dists_all = cdist(XY, XY)
    # Median heuristic for RBF bandwidth
    median_dist = np.median(dists_all[dists_all > 0])
    gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0

    K_xx = np.exp(-gamma * cdist(X, X) ** 2).mean()
    K_yy = np.exp(-gamma * cdist(Y, Y) ** 2).mean()
    K_xy = np.exp(-gamma * cdist(X, Y) ** 2).mean()

    mmd_sq = K_xx + K_yy - 2 * K_xy
    return float(np.sqrt(max(mmd_sq, 0)))


def compare_corrections(
    uncorrected: anndata.AnnData,
    corrections: dict[str, anndata.AnnData],
    batch_key: str = "dataset",
    n_neighbors: int = 50,
) -> pd.DataFrame:
    """Compare batch correction methods using mixing score and silhouette.

    Args:
        uncorrected: Original AnnData before correction.
        corrections: Dict mapping method name to corrected AnnData.
        batch_key: Obs column identifying the batch.
        n_neighbors: Number of neighbors for mixing score.

    Returns:
        DataFrame with methods as rows and metrics as columns.
    """
    from sklearn.metrics import silhouette_score

    results = {}

    # Score uncorrected
    mixing = compute_batch_mixing_score(uncorrected, batch_key, n_neighbors)
    if "X_pca" in uncorrected.obsm:
        sil = silhouette_score(
            uncorrected.obsm["X_pca"][:, :30],
            uncorrected.obs[batch_key],
            sample_size=min(5000, uncorrected.n_obs),
            random_state=42,
        )
    else:
        sil = float("nan")
    results["uncorrected"] = {"mixing_score": mixing, "silhouette": sil}

    # Score each correction method
    for name, corrected in corrections.items():
        mixing = compute_batch_mixing_score(corrected, batch_key, n_neighbors)
        # Use the appropriate embedding for silhouette
        if name == "harmony" and "X_pca_harmony" in corrected.obsm:
            rep = corrected.obsm["X_pca_harmony"]
        elif name == "scanorama" and "X_scanorama" in corrected.obsm:
            rep = corrected.obsm["X_scanorama"]
        elif "X_pca" in corrected.obsm:
            rep = corrected.obsm["X_pca"][:, :30]
        else:
            rep = None

        if rep is not None:
            sil = silhouette_score(
                rep,
                corrected.obs[batch_key],
                sample_size=min(5000, corrected.n_obs),
                random_state=42,
            )
        else:
            sil = float("nan")

        results[name] = {"mixing_score": mixing, "silhouette": sil}

    return pd.DataFrame(results).T
