"""Visualization functions for UMAP plots and feature importance charts."""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots (no screen needed)
import matplotlib.pyplot as plt
import seaborn as sns


def plot_umap(
    adata: anndata.AnnData,
    color: str,
    title: str | None = None,
    cmap: str = "magma",
    point_size: int = 10,
    save_path: str | None = None,
) -> None:
    """Generate a UMAP plot colored by an obs column or gene.

    Uses pre-computed UMAP embeddings from adata.obsm['X_umap'].

    Args:
        adata: AnnData object with pre-computed UMAP.
        color: Column name in obs or gene name to color by.
        title: Plot title (defaults to 'UMAP colored by {color}').
        cmap: Colormap for continuous values.
        point_size: Size of scatter points.
        save_path: If provided, save the figure to this path. Otherwise display.
    """
    # Set a default title if the caller did not provide one
    if title is None:
        title = f"UMAP colored by {color}"

    # Shuffle cell order so no single category visually dominates
    rng = np.random.RandomState(42)
    idx = rng.permutation(adata.n_obs)
    adata = adata[idx]

    # Draw the 2D UMAP scatter plot with scanpy; show=False keeps it in memory
    sc.pl.umap(adata, color=color, title=title, cmap=cmap, s=point_size, show=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create folder if needed
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # Save at high resolution
        plt.close()  # Free memory after saving
    else:
        plt.show()  # Display interactively if no save path is given


def plot_feature_importances(
    importances: pd.Series,
    title: str = "Top Gene Feature Importances",
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of gene feature importances.

    Args:
        importances: pd.Series (gene name -> importance), sorted descending.
        title: Plot title.
        save_path: If provided, save the figure. Otherwise display.
    """
    plt.figure(figsize=(12, 10))  # Wide figure so gene names have room
    plt.barh(importances.index, importances.values)  # Horizontal bars — one per gene
    plt.xlabel("Feature Importance", fontsize=16)  # How much each gene helped the model
    plt.ylabel("Gene", fontsize=16)
    plt.title(title, fontsize=18, pad=15)
    plt.gca().invert_yaxis()  # Most important gene at top
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()  # Prevent labels from being cut off

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create folder if needed
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # Save at high resolution
        plt.close()  # Free memory after saving
    else:
        plt.show()


def plot_dendrogram(
    adata: anndata.AnnData,
    groupby: str,
    save_path: str | None = None,
) -> None:
    """Generate a dendrogram showing hierarchical clustering of cell groups.

    Uses scanpy's dendrogram tool which clusters groups based on PCA
    coordinates, showing which cell populations are most similar.

    Args:
        adata: AnnData object with pre-computed PCA.
        groupby: Obs column to group cells by (e.g. 'condition', 'dataset').
        save_path: If provided, save the figure. Otherwise display.
    """
    # Compute the dendrogram grouping (stored in adata.uns)
    sc.tl.dendrogram(adata, groupby=groupby)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc.pl.dendrogram(adata, groupby=groupby, ax=ax, show=False)
    ax.set_title(f"Dendrogram grouped by {groupby}", fontsize=14, pad=15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_heatmap_dendrogram(
    adata: anndata.AnnData,
    groupby: str,
    n_genes: int = 10,
    save_path: str | None = None,
) -> None:
    """Generate a heatmap with dendrogram showing top marker genes per group.

    Combines hierarchical clustering (dendrogram) with a gene expression
    heatmap, so you can see both which groups are similar and which genes
    drive the differences.

    Args:
        adata: AnnData object with pre-computed PCA.
        groupby: Obs column to group cells by (e.g. 'condition', 'dataset').
        n_genes: Number of top marker genes per group to display.
        save_path: If provided, save the figure. Otherwise display.
    """
    # Compute dendrogram if not already present
    dendrogram_key = f"dendrogram_{groupby}"
    if dendrogram_key not in adata.uns:
        sc.tl.dendrogram(adata, groupby=groupby)

    # Rank genes per group to find markers
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon")

    sc.pl.rank_genes_groups_heatmap(
        adata,
        groupby=groupby,
        n_genes=n_genes,
        dendrogram=True,
        show=False,
        figsize=(14, 8),
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_batch_umap(
    adata: anndata.AnnData,
    batch_key: str = "dataset",
    condition_col: str = "condition",
    save_dir: str | None = None,
) -> None:
    """Side-by-side UMAPs colored by batch vs condition.

    Helps visually assess whether cells cluster by batch (bad — technical
    artifact) or by condition (good — biological signal).

    Args:
        adata: AnnData with pre-computed UMAP.
        batch_key: Obs column for batch identity.
        condition_col: Obs column for biological condition.
        save_dir: Directory to save the figure. If None, display interactively.
    """
    # Shuffle cell order so no single batch/condition is drawn on top
    rng = np.random.RandomState(42)
    idx = rng.permutation(adata.n_obs)
    adata_shuffled = adata[idx].copy()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left panel: colored by batch
    sc.pl.umap(adata_shuffled, color=batch_key, ax=axes[0], show=False, title=f"Colored by {batch_key}")

    # Right panel: colored by condition
    if condition_col in adata_shuffled.obs.columns:
        sc.pl.umap(adata_shuffled, color=condition_col, ax=axes[1], show=False, title=f"Colored by {condition_col}")
    else:
        axes[1].set_title(f"{condition_col} not found")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "batch_vs_condition_umap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved batch UMAP -> {path}")
    else:
        plt.show()


def plot_housekeeping_heatmap(
    hk_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Heatmap of housekeeping gene expression across batches.

    Housekeeping genes should be uniformly expressed. Large differences
    between rows (batches) indicate technical batch effects.

    Args:
        hk_df: DataFrame with batches as rows, genes as columns.
        save_path: Path to save the figure. If None, display interactively.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(hk_df.columns) * 1.5), max(3, len(hk_df) * 0.8)))
    sns.heatmap(
        hk_df, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
        annot_kws={"size": 9}, linewidths=0.5,
    )
    ax.set_title("Housekeeping Gene Expression by Batch", fontsize=14, pad=15)
    ax.set_ylabel("Batch")
    ax.set_xlabel("Gene")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_batch_distance_heatmap(
    distances: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Heatmap of pairwise Euclidean distances between batches.

    Args:
        distances: Symmetric DataFrame of pairwise distances.
        save_path: Path to save the figure. If None, display interactively.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(distances, annot=True, fmt=".2f", cmap="Blues", ax=ax, square=True)
    ax.set_title("Pairwise Batch Distances (Housekeeping Genes)", fontsize=14, pad=15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_condition_subset_umap(
    adata: anndata.AnnData,
    condition: str,
    condition_col: str = "condition",
    batch_key: str = "dataset",
    point_size: int = 10,
    save_path: str | None = None,
) -> None:
    """UMAP of cells from a single condition, colored by dataset.

    Subsets the AnnData to cells matching the given condition, recomputes
    PCA/neighbors/UMAP on the subset, and plots colored by dataset.
    If cells from different datasets cluster separately despite being
    the same cell state, this indicates batch effects.

    Args:
        adata: AnnData with expression data and obs annotations.
        condition: The condition value to subset to (e.g. 'malignant').
        condition_col: Obs column holding condition labels.
        batch_key: Obs column identifying the dataset/batch.
        point_size: Size of scatter points.
        save_path: If provided, save the figure. Otherwise display.
    """
    # Subset to the requested condition
    mask = adata.obs[condition_col] == condition
    subset = adata[mask].copy()

    n_datasets = subset.obs[batch_key].nunique()
    dataset_counts = subset.obs[batch_key].value_counts().to_dict()
    print(f"  {condition}: {subset.n_obs} cells across {n_datasets} datasets {dataset_counts}")

    # Recompute embeddings on the subset for a clean UMAP
    sc.tl.pca(subset)
    sc.pp.neighbors(subset, n_pcs=30)
    sc.tl.umap(subset)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(
        subset, color=batch_key, ax=ax, show=False, s=point_size,
        title=f"UMAP — {condition} cells colored by {batch_key}",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {save_path}")
    else:
        plt.show()


def plot_batch_correction_comparison(
    umaps_dict: dict[str, anndata.AnnData],
    batch_key: str = "dataset",
    save_dir: str | None = None,
) -> None:
    """Grid of UMAPs comparing uncorrected and corrected datasets.

    Args:
        umaps_dict: Dict mapping method name (e.g. "uncorrected", "combat")
            to AnnData objects with pre-computed UMAP.
        batch_key: Obs column for batch identity.
        save_dir: Directory to save the figure. If None, display interactively.
    """
    n = len(umaps_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, adata) in zip(axes, umaps_dict.items()):
        sc.pl.umap(adata, color=batch_key, ax=ax, show=False, title=name)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "batch_correction_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved correction comparison -> {path}")
    else:
        plt.show()


def plot_housekeeping_pca(
    adata_hk: anndata.AnnData,
    batch_key: str = "dataset",
    condition_col: str = "condition",
    save_dir: str | None = None,
) -> None:
    """PCA scatter plots of housekeeping genes, colored by dataset and condition.

    Since housekeeping genes should be uniformly expressed regardless of
    cell state, any separation in this PCA space is evidence of technical
    batch effects (different normalization, sequencing depth, lab protocols).

    Two panels are generated side-by-side:
    - Left: colored by dataset — shows whether batches separate
    - Right: colored by condition — shows whether biology leaks through
      (it shouldn't for well-chosen housekeeping genes)

    Args:
        adata_hk: AnnData subset to housekeeping genes with PCA computed
            (output of batch.run_housekeeping_pca).
        batch_key: Obs column for dataset/batch identity.
        condition_col: Obs column for biological condition.
        save_dir: Directory to save the figure. If None, display interactively.
    """
    # Need at least 2 PCs to render a 2D scatter. This happens when the
    # filtered HK gene set collapses to <=2 genes, yielding only 1 component.
    n_pcs = adata_hk.obsm["X_pca"].shape[1] if "X_pca" in adata_hk.obsm else 0
    if n_pcs < 2:
        print(
            f"  Skipping HK PCA plot: only {n_pcs} principal component(s) "
            f"available (need >=2). This usually means the filtered HK gene "
            f"set has fewer than 2 genes."
        )
        return

    # Shuffle cell draw order so no single batch dominates the visual
    rng = np.random.RandomState(42)
    idx = rng.permutation(adata_hk.n_obs)
    adata_shuffled = adata_hk[idx].copy()

    # Extract variance explained for axis labels
    var_ratio = adata_hk.uns.get("pca", {}).get("variance_ratio", None)
    xlabel = f"PC1 ({var_ratio[0]*100:.1f}%)" if var_ratio is not None else "PC1"
    ylabel = f"PC2 ({var_ratio[1]*100:.1f}%)" if var_ratio is not None else "PC2"

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left panel: colored by dataset (batch)
    sc.pl.pca(
        adata_shuffled, color=batch_key, ax=axes[0], show=False,
        title=f"HK Gene PCA — colored by {batch_key}",
    )
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    # Right panel: colored by condition
    if condition_col in adata_shuffled.obs.columns:
        sc.pl.pca(
            adata_shuffled, color=condition_col, ax=axes[1], show=False,
            title=f"HK Gene PCA — colored by {condition_col}",
        )
    else:
        axes[1].set_title(f"{condition_col} not found in obs")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "housekeeping_pca.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved HK PCA plot -> {path}")
    else:
        plt.show()


def plot_housekeeping_violin(
    adata: anndata.AnnData,
    hk_genes: list[str],
    batch_key: str = "dataset",
    save_dir: str | None = None,
    max_genes_per_figure: int = 12,
) -> None:
    """Violin plots of housekeeping gene expression distributions per dataset.

    Each subplot shows one gene's expression distribution across datasets.
    If housekeeping genes are truly stable, the violins should overlap.
    Systematic shifts (e.g., one dataset always higher) indicate batch effects.

    Args:
        adata: Combined AnnData with all datasets.
        hk_genes: List of housekeeping gene names to plot.
        batch_key: Obs column identifying the source dataset.
        save_dir: Directory to save figures. If None, display interactively.
        max_genes_per_figure: Maximum genes per figure (avoids overcrowding).
    """
    import scipy.sparse

    # Filter to genes present in the dataset
    available = [g for g in hk_genes if g in adata.var_names]
    if not available:
        print("  No housekeeping genes found in adata.var_names")
        return

    # Build a tidy DataFrame for plotting: one row per cell-gene observation
    # Columns: gene expression value, gene name, dataset label
    X = adata[:, available].X
    if scipy.sparse.issparse(X):
        X = X.toarray()

    records = []
    for i, gene in enumerate(available):
        for ds in adata.obs[batch_key].unique():
            mask = (adata.obs[batch_key] == ds).values
            vals = X[mask, i]
            for v in vals:
                records.append({"gene": gene, batch_key: ds, "expression": float(v)})

    df = pd.DataFrame(records)

    # Split into pages if there are many genes
    for page_start in range(0, len(available), max_genes_per_figure):
        page_genes = available[page_start:page_start + max_genes_per_figure]
        n_genes = len(page_genes)
        n_cols = min(3, n_genes)
        n_rows = (n_genes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, gene in enumerate(page_genes):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            gene_df = df[df["gene"] == gene]
            sns.violinplot(
                data=gene_df, x=batch_key, y="expression",
                ax=ax, inner="quartile", cut=0,
            )
            ax.set_title(gene, fontsize=12, fontweight="bold")
            ax.set_ylabel("Log-normalized expression")
            ax.set_xlabel("")

        # Hide unused subplots
        for idx in range(n_genes, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        fig.suptitle(
            "Housekeeping Gene Expression by Dataset",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            suffix = f"_page{page_start // max_genes_per_figure + 1}" if len(available) > max_genes_per_figure else ""
            path = os.path.join(save_dir, f"housekeeping_violin{suffix}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved HK violin plot -> {path}")
        else:
            plt.show()


def plot_hk_gene_relationship(
    uncorr_expr: pd.DataFrame,
    corr_expr: pd.DataFrame,
    obs: pd.DataFrame,
    hk_genes: list[str],
    target_genes: list[str],
    batch_key: str = "dataset",
    save_path: str | None = None,
):
    """Compare HK↔target-gene correlation structure before vs after batch correction.

    For each HK gene × target gene pair, computes the Pearson correlation
    across cells *within each batch separately*, then averages across batches.
    Doing it per-batch avoids letting batch effects inflate the uncorrected
    correlations and artificially make correction look destructive.

    Produces a 3-panel figure:
      1. Uncorrected correlation heatmap (HK rows × target cols)
      2. Corrected correlation heatmap (same shape, shared color scale)
      3. Preservation scatter: one point per (HK, target) pair, plotted as
         (uncorrected r, corrected r) with a y=x reference line. Points near
         the diagonal mean the relationship survived correction. An R² score
         between the two correlation vectors is annotated as a single-glance
         quality metric.

    Args:
        uncorr_expr: DataFrame of uncorrected expression, cells × genes.
            Must contain all `hk_genes` and `target_genes` as columns.
        corr_expr: DataFrame of corrected expression, same shape/row order.
        obs: DataFrame with per-cell metadata, same row order. Must contain
            the `batch_key` column.
        hk_genes: Housekeeping gene names to use as rows.
        target_genes: "Other" gene names to use as columns (e.g. HVGs).
        batch_key: Obs column identifying the batch (default "dataset").
        save_path: If provided, save figure to this path and close. If None,
            return the figure (useful for live rendering in Marimo).

    Returns:
        matplotlib.figure.Figure if save_path is None, otherwise None.
    """
    # Filter inputs to genes actually present in both frames (defensive)
    hk_genes = [g for g in hk_genes if g in uncorr_expr.columns and g in corr_expr.columns]
    target_genes = [
        g for g in target_genes
        if g in uncorr_expr.columns and g in corr_expr.columns and g not in hk_genes
    ]
    if not hk_genes or not target_genes:
        raise ValueError(
            "Need at least one HK gene and one target gene present in both "
            f"expression frames (got {len(hk_genes)} HK, {len(target_genes)} target)."
        )

    batches = list(obs[batch_key].unique())

    def _within_batch_mean_corr(expr: pd.DataFrame) -> np.ndarray:
        """Return a (n_hk, n_target) matrix of batch-averaged Pearson r."""
        per_batch = np.full((len(batches), len(hk_genes), len(target_genes)), np.nan)
        for bi, batch in enumerate(batches):
            mask = (obs[batch_key] == batch).values
            if mask.sum() < 3:  # need >=3 cells for a meaningful correlation
                continue
            hk_block = expr.loc[mask, hk_genes].to_numpy()
            tg_block = expr.loc[mask, target_genes].to_numpy()

            # Vectorised Pearson r: standardize each column, then dot product / n
            hk_std = hk_block.std(axis=0, ddof=0)
            tg_std = tg_block.std(axis=0, ddof=0)
            # Any gene with zero variance in this batch yields NaN — leave it NaN
            with np.errstate(invalid="ignore", divide="ignore"):
                hk_z = (hk_block - hk_block.mean(axis=0)) / hk_std
                tg_z = (tg_block - tg_block.mean(axis=0)) / tg_std
                corr = (hk_z.T @ tg_z) / hk_block.shape[0]
            # Replace infinities from zero-variance columns with NaN
            corr[~np.isfinite(corr)] = np.nan
            per_batch[bi] = corr
        # Average across batches, ignoring NaN slices
        return np.nanmean(per_batch, axis=0)

    uncorr_mat = _within_batch_mean_corr(uncorr_expr)
    corr_mat = _within_batch_mean_corr(corr_expr)

    # Shared color scale across the two heatmaps
    finite_vals = np.concatenate([
        uncorr_mat[np.isfinite(uncorr_mat)],
        corr_mat[np.isfinite(corr_mat)],
    ])
    vmax = float(np.max(np.abs(finite_vals))) if finite_vals.size else 1.0
    vmax = max(vmax, 0.1)  # avoid a degenerate color scale

    uncorr_df = pd.DataFrame(uncorr_mat, index=hk_genes, columns=target_genes)
    corr_df = pd.DataFrame(corr_mat, index=hk_genes, columns=target_genes)

    # Wider layout when there are many targets so x-labels stay readable
    heatmap_width = max(6.0, 0.25 * len(target_genes) + 3.0)
    fig = plt.figure(figsize=(heatmap_width * 2 + 6, max(4.5, 0.35 * len(hk_genes) + 3)))
    gs = fig.add_gridspec(1, 3, width_ratios=[heatmap_width, heatmap_width, 6])
    ax_u = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[0, 2])

    sns.heatmap(
        uncorr_df, ax=ax_u, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        cbar_kws={"label": "Pearson r"}, linewidths=0.25, linecolor="#eeeeee",
    )
    ax_u.set_title("Uncorrected\n(within-batch mean Pearson r)", fontsize=12, pad=10)
    ax_u.set_xlabel("Target gene")
    ax_u.set_ylabel("Housekeeping gene")
    ax_u.set_xticklabels(ax_u.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax_u.set_yticklabels(ax_u.get_yticklabels(), rotation=0, fontsize=9)

    sns.heatmap(
        corr_df, ax=ax_c, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        cbar_kws={"label": "Pearson r"}, linewidths=0.25, linecolor="#eeeeee",
    )
    ax_c.set_title("Corrected\n(within-batch mean Pearson r)", fontsize=12, pad=10)
    ax_c.set_xlabel("Target gene")
    ax_c.set_ylabel("")
    ax_c.set_xticklabels(ax_c.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax_c.set_yticklabels(ax_c.get_yticklabels(), rotation=0, fontsize=9)

    # Preservation scatter: colour by HK gene (rows)
    u_flat = uncorr_mat.flatten()
    c_flat = corr_mat.flatten()
    hk_label_flat = np.repeat(np.array(hk_genes), len(target_genes))
    keep = np.isfinite(u_flat) & np.isfinite(c_flat)
    u_flat = u_flat[keep]
    c_flat = c_flat[keep]
    hk_label_flat = hk_label_flat[keep]

    palette = sns.color_palette("tab10", n_colors=len(hk_genes))
    color_map = {g: palette[i] for i, g in enumerate(hk_genes)}
    for g in hk_genes:
        m = hk_label_flat == g
        if not m.any():
            continue
        ax_s.scatter(u_flat[m], c_flat[m], s=28, color=color_map[g], label=g,
                     edgecolors="white", linewidths=0.4, alpha=0.85)

    # y=x reference line
    lim_lo = min(float(np.min(u_flat)), float(np.min(c_flat)), -0.05)
    lim_hi = max(float(np.max(u_flat)), float(np.max(c_flat)), 0.05)
    pad = 0.05 * (lim_hi - lim_lo + 1e-9)
    lim = (lim_lo - pad, lim_hi + pad)
    ax_s.plot(lim, lim, linestyle="--", color="gray", linewidth=1, zorder=0)
    ax_s.set_xlim(lim)
    ax_s.set_ylim(lim)
    ax_s.set_aspect("equal", adjustable="box")

    # Annotate R² between the two correlation vectors (preservation score)
    if u_flat.size >= 2 and np.std(u_flat) > 0 and np.std(c_flat) > 0:
        r = float(np.corrcoef(u_flat, c_flat)[0, 1])
        r2 = r * r
        ax_s.text(
            0.03, 0.97, f"R² = {r2:.3f}\nn = {u_flat.size} pairs",
            transform=ax_s.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
        )

    ax_s.set_xlabel("Uncorrected Pearson r")
    ax_s.set_ylabel("Corrected Pearson r")
    ax_s.set_title("Preservation of HK ↔ target\ncorrelation after correction",
                   fontsize=12, pad=10)
    ax_s.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax_s.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax_s.legend(fontsize=8, loc="lower right", frameon=True, title="HK gene")

    fig.suptitle(
        "HK ↔ Gene Relationship Preservation (before vs after batch correction)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved HK relationship plot -> {save_path}")
        return None
    return fig


def generate_all_plots(
    adata: anndata.AnnData,
    importances: pd.Series,
    output_dir: str,
    umap_color_columns: list[str] | None = None,
    umap_genes: list[str] | None = None,
) -> None:
    """Generate all standard plots and save to output_dir.

    Creates:
        - UMAP plots by cell type, condition, and selected genes
        - Feature importance bar chart

    Args:
        adata: AnnData object with pre-computed UMAP.
        importances: Top gene importances from the model.
        output_dir: Directory to save all plots.
        umap_color_columns: Obs columns for UMAP plots.
            Defaults to ['condition', 'dataset', 'celltype3', 'Cell_type'].
        umap_genes: Genes to plot on UMAP. Defaults to top 2 important genes.
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)  # Create the plots subfolder if it doesn't exist

    # Default UMAP color columns — only use columns that exist in the data
    if umap_color_columns is None:
        umap_color_columns = ["condition", "dataset", "celltype3", "Cell_type"]
    umap_color_columns = [c for c in umap_color_columns if c in adata.obs.columns]  # Drop missing columns

    # Default gene list for UMAP — top 2 from feature importances
    if umap_genes is None:
        umap_genes = importances.index[:2].tolist()
    # Only plot genes that exist in the data
    umap_genes = [g for g in umap_genes if g in adata.var_names]

    # Generate UMAP plots for obs columns (e.g., color by cell type or condition)
    for col in umap_color_columns:
        path = os.path.join(plots_dir, f"umap_{col}.png")
        print(f"  Saving UMAP by {col} -> {path}")
        plot_umap(adata, color=col, save_path=path)

    # Generate UMAP plots showing where each top gene is expressed
    for gene in umap_genes:
        path = os.path.join(plots_dir, f"umap_gene_{gene}.png")
        print(f"  Saving UMAP by {gene} expression -> {path}")
        plot_umap(adata, color=gene, title=f"UMAP colored by {gene} Expression", save_path=path)

    # Feature importance bar chart showing which genes the model relied on most
    fi_path = os.path.join(plots_dir, "feature_importances.png")
    print(f"  Saving feature importances -> {fi_path}")
    plot_feature_importances(importances, save_path=fi_path)

    print(f"All plots saved to {plots_dir}/")
