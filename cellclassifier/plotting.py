"""Visualization functions for UMAP plots and feature importance charts."""

import os
import pandas as pd
import scanpy as sc
import anndata
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots (no screen needed)
import matplotlib.pyplot as plt


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
            Defaults to ['celltype3', 'Cell_type', 'CONDITION'].
        umap_genes: Genes to plot on UMAP. Defaults to top 2 important genes.
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)  # Create the plots subfolder if it doesn't exist

    # Default UMAP color columns — only use columns that exist in the data
    if umap_color_columns is None:
        umap_color_columns = ["celltype3", "Cell_type", "CONDITION"]
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
