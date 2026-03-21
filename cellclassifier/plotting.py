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
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: colored by batch
    sc.pl.umap(adata, color=batch_key, ax=axes[0], show=False, title=f"Colored by {batch_key}")

    # Right panel: colored by condition
    if condition_col in adata.obs.columns:
        sc.pl.umap(adata, color=condition_col, ax=axes[1], show=False, title=f"Colored by {condition_col}")
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
    fig, ax = plt.subplots(figsize=(10, max(4, len(hk_df) * 0.8)))
    sns.heatmap(hk_df, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Housekeeping Gene Expression by Batch", fontsize=14, pad=15)
    ax.set_ylabel("Batch")
    ax.set_xlabel("Gene")
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
