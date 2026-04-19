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

    # legend_loc="best" places the categorical legend inside the axes so the
    # figure stays compact. Continuous colormaps still get a right-margin colorbar.
    sc.pl.umap(
        adata, color=color, title=title, cmap=cmap, s=point_size,
        legend_loc="best", show=False,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create folder if needed
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # Save at high resolution
        plt.close()  # Free memory after saving
    else:
        plt.show()  # Display interactively if no save path is given


def plot_feature_importances(
    importances: pd.Series,
    top_n: int = 10,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of the top gene feature importances.

    Args:
        importances: pd.Series (gene name -> importance), sorted descending.
        top_n: Number of top genes to plot.
        title: Plot title (defaults to "Top {top_n} Gene Feature Importances
            for Condition Classification").
        save_path: If provided, save the figure. Otherwise display.
    """
    top = importances.head(top_n)
    if title is None:
        title = f"Top {top_n} Gene Feature Importances for Condition Classification"

    plt.figure(figsize=(8, 6))
    plt.barh(top.index, top.values, color="steelblue")
    plt.xlabel("Feature importance", fontsize=14)
    plt.ylabel("Gene", fontsize=14)
    plt.title(title, fontsize=14, pad=10)
    plt.gca().invert_yaxis()  # Most important gene at top
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_log_fold_change(
    lfc: pd.Series,
    top_n: int = 10,
    title: str | None = None,
    numerator: str = "Tumor",
    denominator: str = "Normal",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> None:
    """Vertical bar chart of top differentially expressed genes by |log fold change|.

    Args:
        lfc: pd.Series (gene -> signed log fold change, numerator vs denominator).
        top_n: Number of top genes by |LFC| to plot.
        title: Plot title (defaults to "Top {top_n} Differentially Expressed Genes
            in {numerator} vs {denominator}").
        numerator / denominator: Condition labels for the default title.
        save_path: If provided, save the figure. Otherwise display.
        ax: Optional matplotlib axis for subplot rendering. If provided,
            save_path is ignored and the caller manages the figure.
    """
    top = lfc.reindex(lfc.abs().sort_values(ascending=False).index).head(top_n)
    if title is None:
        title = f"Top {top_n} Differentially Expressed Genes in {numerator} vs {denominator}"

    # Viridis gradient across bars to match the reference styling
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(top)))

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(top.index, top.values, color=colors)
    ax.set_xlabel("Gene", fontsize=14)
    ax.set_ylabel("Log Fold Change", fontsize=14)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=90)

    if standalone:
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_log_fold_change_grid(
    panels: dict[str, pd.Series | str],
    save_path: str | None = None,
    suptitle: str | None = None,
    numerator: str = "malignant",
    denominator: str = "normal",
    top_n: int = 10,
    n_cols: int = 2,
) -> None:
    """Grid of LFC bar charts; entries may be Series (plot) or str (note).

    Each dict entry becomes one panel in reading order. If the value is a
    pd.Series it is rendered as a top-N LFC bar chart; if it is a string it
    is rendered as an empty panel with the string as an explanation.

    Args:
        panels: Ordered dict mapping panel title to either a signed LFC
            Series (gene -> LFC) or a note string for a blank panel.
        save_path: If provided, save the figure. Otherwise display.
        suptitle: Optional figure-level title.
        numerator / denominator: Passed through to ``plot_log_fold_change``.
        top_n: Top-N genes by |LFC| per panel.
        n_cols: Grid columns (default 2, producing a 2x2 for four panels).
    """
    titles = list(panels.keys())
    n = len(titles)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, title in zip(axes, titles):
        value = panels[title]
        if isinstance(value, pd.Series):
            plot_log_fold_change(
                value, top_n=top_n, title=title,
                numerator=numerator, denominator=denominator, ax=ax,
            )
        else:
            # Blank panel with note
            ax.text(
                0.5, 0.5, value,
                ha="center", va="center", fontsize=11, wrap=True,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5",
                          edgecolor="#cccccc"),
            )
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved LFC grid -> {save_path}")
    else:
        plt.show()


def lfc_grid_from_adata(
    adata: anndata.AnnData,
    condition_col: str = "condition",
    batch_key: str = "dataset",
    numerator: str = "malignant",
    denominator: str = "normal",
    layer: str | None = None,
    top_n: int = 10,
    save_path: str | None = None,
    suptitle: str | None = None,
) -> None:
    """Compute per-dataset LFC panels from an AnnData and render the grid.

    Subsets adata by each unique value in obs[batch_key], computes signed
    log fold change (numerator - denominator in log1p space) for each subset
    and for the combined dataset, then calls plot_log_fold_change_grid().

    Datasets missing one of the two conditions are rendered as blank panels
    with an explanatory note.

    Args:
        adata: Combined AnnData with log1p expression in X (or in layer).
        condition_col: obs column holding condition labels.
        batch_key: obs column holding dataset/batch labels.
        numerator: Condition label treated as numerator (positive LFC side).
        denominator: Condition label treated as denominator.
        layer: If provided, read expression from adata.layers[layer] instead of X.
        top_n: Top-N genes by |LFC| per panel.
        save_path: Output PNG path. If None, displays interactively.
        suptitle: Figure-level title override.
    """
    from scribe.analysis import avg_expression_by_condition, compute_log_fold_change

    datasets = list(adata.obs[batch_key].unique())
    panel_order = datasets + ["Combined (all datasets)"]
    panels: dict[str, pd.Series | str] = {}

    for panel_name in panel_order:
        if panel_name == "Combined (all datasets)":
            sub = adata
        else:
            sub = adata[adata.obs[batch_key] == panel_name]

        if layer is not None:
            import scipy.sparse
            x = sub.layers[layer]
            if scipy.sparse.issparse(x):
                x = x.toarray()
            import numpy as _np
            sub = anndata.AnnData(
                X=_np.asarray(x),
                obs=sub.obs.copy(),
                var=sub.var.copy(),
            )

        conds = set(sub.obs[condition_col].unique())
        missing = {numerator, denominator} - conds
        if missing:
            m = next(iter(missing))
            present_n = int((sub.obs[condition_col] == numerator).sum())
            present_d = int((sub.obs[condition_col] == denominator).sum())
            note = (
                f"No '{m}' cells in {panel_name}.\n"
                f"(found {numerator}={present_n}, {denominator}={present_d})\n\n"
                f"Cannot compute {numerator} vs {denominator}\n"
                f"log fold change for this dataset."
            )
            panels[panel_name] = note
        else:
            avg = avg_expression_by_condition(sub, condition_col=condition_col)
            lfc = compute_log_fold_change(avg, numerator=numerator, denominator=denominator)
            panels[panel_name] = lfc

    if suptitle is None:
        suptitle = f"Top {top_n} Differentially Expressed Genes: {numerator} vs {denominator}"

    plot_log_fold_change_grid(
        panels,
        save_path=save_path,
        suptitle=suptitle,
        numerator=numerator,
        denominator=denominator,
        top_n=top_n,
        n_cols=2,
    )


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



def plot_single_dataset_umap(
    adata: anndata.AnnData,
    dataset_name: str,
    batch_key: str = "dataset",
    condition_col: str = "condition",
    celltype_col: str | None = None,
    point_size: int = 10,
    save_path: str | None = None,
) -> None:
    """Side-by-side UMAPs for a single dataset: leiden/cell-type and condition.

    Subsets the combined AnnData to one dataset, recomputes PCA/neighbors/UMAP,
    then plots two panels:
      - Left: colored by cell type (if present) or leiden clusters
      - Right: colored by condition (malignant vs normal)

    Args:
        adata: Combined AnnData with expression data.
        dataset_name: Value in batch_key to subset to.
        batch_key: Obs column identifying the dataset.
        condition_col: Obs column for biological condition.
        celltype_col: Obs column for cell type. If None, auto-detects common
            names or falls back to leiden.
        point_size: Size of scatter points.
        save_path: If provided, save the figure. Otherwise display.
    """
    # Subset to the requested dataset
    mask = adata.obs[batch_key] == dataset_name
    subset = adata[mask].copy()
    print(f"  {dataset_name}: {subset.n_obs} cells")

    # Recompute embeddings on the subset
    sc.tl.pca(subset)
    sc.pp.neighbors(subset, n_pcs=30)
    sc.tl.umap(subset)

    # Compute leiden if not already present
    if "leiden" not in subset.obs.columns:
        sc.tl.leiden(subset)

    # Determine the left-panel coloring: cell type if available, else leiden
    if celltype_col and celltype_col in subset.obs.columns:
        left_col = celltype_col
    else:
        # Auto-detect common cell type column names
        candidates = ["cell_type", "celltype", "celltype3", "Cell_type",
                       "cell_type_ontology_term_id"]
        left_col = "leiden"
        for c in candidates:
            if c in subset.obs.columns and subset.obs[c].nunique() > 1:
                left_col = c
                break

    left_label = "Cell type" if left_col != "leiden" else "Leiden cluster"

    # Shuffle to avoid visual dominance of one category
    rng = np.random.RandomState(42)
    idx = rng.permutation(subset.n_obs)
    subset = subset[idx]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: cell type or leiden
    sc.pl.umap(
        subset, color=left_col, ax=axes[0], show=False, s=point_size,
        title=f"{dataset_name} — {left_label} ({left_col})",
    )

    # Right: condition
    if condition_col in subset.obs.columns and subset.obs[condition_col].nunique() > 1:
        sc.pl.umap(
            subset, color=condition_col, ax=axes[1], show=False, s=point_size,
            title=f"{dataset_name} — Condition ({condition_col})",
        )
    elif condition_col in subset.obs.columns:
        sc.pl.umap(
            subset, color=condition_col, ax=axes[1], show=False, s=point_size,
            title=f"{dataset_name} — Condition (only: {subset.obs[condition_col].unique()[0]})",
        )
    else:
        axes[1].set_title(f"{condition_col} not found")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {save_path}")
    else:
        plt.show()


def plot_volcano(
    de_df: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    lfc_threshold: float = 1.0,
    padj_threshold: float = 0.05,
    n_labels: int = 10,
    numerator: str = "malignant",
    denominator: str = "normal",
    subtitle: str | None = None,
) -> None:
    """Render a volcano plot on a given axis.

    x = logfoldchange (positive = up in ``numerator``).
    y = -log10(adjusted p-value).
    Genes past both thresholds are colored; the top ``n_labels`` by combined
    significance (|lfc| * -log10(padj)) are text-labeled with adjustText for
    non-overlapping placement.

    Args:
        de_df: Output of ``analysis.compute_differential_expression``.
        ax: Matplotlib axis to draw on.
        title: Axis title.
        lfc_threshold: Minimum |LFC| to be considered significant.
        padj_threshold: Maximum adjusted p-value to be considered significant.
        n_labels: Number of top genes to annotate with gene names.
        numerator / denominator: Labels for the x-axis direction.
        subtitle: Secondary subtitle shown below the title (e.g. comparison description).
    """
    x = de_df["logfoldchange"].values
    y = de_df["neg_log10_pvalue_adj"].values
    padj = de_df["pvalue_adj"].values
    lfc = de_df["logfoldchange"].values

    sig_up = np.asarray((padj < padj_threshold) & (lfc >= lfc_threshold))
    sig_dn = np.asarray((padj < padj_threshold) & (lfc <= -lfc_threshold))
    ns = ~(sig_up | sig_dn)

    ax.scatter(x[ns], y[ns], s=12, c="lightgray", alpha=0.5, edgecolors="none", label="not significant")
    ax.scatter(x[sig_up], y[sig_up], s=20, c="#d62728", alpha=0.75, edgecolors="none",
               label=f"up in {numerator}")
    ax.scatter(x[sig_dn], y[sig_dn], s=20, c="#1f77b4", alpha=0.75, edgecolors="none",
               label=f"up in {denominator}")

    ax.axvline(lfc_threshold, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(-lfc_threshold, color="black", linestyle="--", linewidth=0.5)
    ax.axhline(-np.log10(padj_threshold), color="black", linestyle="--", linewidth=0.5)

    # Label the top N most significant genes; use adjustText to prevent overlaps.
    # Split evenly between upregulated and downregulated so neither side monopolises.
    sig_df = de_df.loc[sig_up | sig_dn].copy()
    if not sig_df.empty:
        sig_df = sig_df.copy()
        half = max(1, n_labels // 2)
        # For each direction pick genes spread vertically: the most extreme
        # by |LFC| (sits near the top) and the least significant among
        # significant genes (sits near the horizontal threshold line).
        # This prevents all labels clustering in one corner.
        def _spread_select(df: pd.DataFrame, up: bool, n: int) -> pd.DataFrame:
            if len(df) == 0:
                return df
            extreme = df.nlargest(1, "logfoldchange") if up else df.nsmallest(1, "logfoldchange")
            least_sig = df.nsmallest(1, "neg_log10_pvalue_adj")
            combined = pd.concat([extreme, least_sig]).drop_duplicates()
            if n <= 2 or len(combined) >= n:
                return combined.head(n)
            # fill remaining slots by |LFC|
            used = combined.index.tolist()
            rest = df.drop(index=used, errors="ignore")
            filler = rest.nlargest(n - len(combined), "logfoldchange") if up else rest.nsmallest(n - len(combined), "logfoldchange")
            return pd.concat([combined, filler])

        up_labels = _spread_select(sig_df[sig_df["logfoldchange"] > 2], up=True, n=half)
        dn_labels = _spread_select(sig_df[sig_df["logfoldchange"] < -2], up=False, n=half)
        top_labels = pd.concat([up_labels, dn_labels])
        texts = []
        for _, row in top_labels.iterrows():
            txt = ax.text(
                row["logfoldchange"], row["neg_log10_pvalue_adj"],
                row["gene"], fontsize=7.5,
            )
            texts.append(txt)
        try:
            from adjustText import adjust_text
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.7),
                expand=(3.0, 4.0),
                force_text=(2.0, 3.0),
                force_points=(1.0, 1.5),
                lim=800,
            )
        except ImportError:
            pass  # adjustText not installed; labels may overlap

    full_title = title
    if subtitle:
        full_title = f"{title}\n{subtitle}"
    ax.set_xlabel(f"log2 fold change ({numerator} / {denominator})", fontsize=10)
    ax.set_ylabel("-log10(adj. p-value)", fontsize=10)
    ax.set_title(full_title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="best", frameon=False)


def plot_volcano_grid(
    de_dfs: dict[str, pd.DataFrame],
    save_path: str | None = None,
    suptitle: str | None = None,
    numerator: str = "malignant",
    denominator: str = "normal",
    lfc_threshold: float = 1.0,
    padj_threshold: float = 0.05,
    n_labels: int = 10,
    n_cols: int | None = None,
    panel_numerators: dict[str, str] | None = None,
    panel_denominators: dict[str, str] | None = None,
    panel_subtitles: dict[str, str] | None = None,
) -> None:
    """Render a row/grid of volcano plots, one panel per entry in de_dfs.

    Args:
        de_dfs: Ordered dict mapping panel title to DE DataFrame.
        save_path: If provided, save the figure. Otherwise display.
        suptitle: Optional figure-level title.
        numerator / denominator: Default comparison labels (overridden per panel
            by ``panel_numerators`` / ``panel_denominators``).
        lfc_threshold, padj_threshold, n_labels: Passed through.
        n_cols: Columns in the grid. Defaults to len(de_dfs) (single row).
        panel_numerators: Per-panel override for the numerator label.
        panel_denominators: Per-panel override for the denominator label.
        panel_subtitles: Per-panel secondary subtitle (comparison description).
    """
    titles = list(de_dfs.keys())
    n = len(titles)
    if n == 0:
        raise ValueError("plot_volcano_grid requires at least one panel")

    n_cols = n_cols or n
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, title in zip(axes, titles):
        panel_num = (panel_numerators or {}).get(title, numerator)
        panel_den = (panel_denominators or {}).get(title, denominator)
        panel_sub = (panel_subtitles or {}).get(title, None)
        plot_volcano(
            de_dfs[title], ax=ax, title=title,
            lfc_threshold=lfc_threshold, padj_threshold=padj_threshold,
            n_labels=n_labels, numerator=panel_num, denominator=panel_den,
            subtitle=panel_sub,
        )
    for ax in axes[n:]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved volcano grid -> {save_path}")
    else:
        plt.show()


def volcano_grid_from_adata(
    adata: anndata.AnnData,
    comparisons: list[dict] | None = None,
    condition_col: str = "condition",
    batch_key: str = "dataset",
    numerator: str = "malignant",
    denominator: str = "normal",
    lfc_threshold: float = 1.0,
    padj_threshold: float = 0.05,
    n_labels: int = 8,
    n_cols: int = 2,
    suptitle: str | None = None,
    save_path: str | None = None,
) -> None:
    """Compute per-comparison DE from an AnnData and render a volcano grid.

    If ``comparisons`` is None, builds one comparison per dataset that has both
    conditions present, plus a combined panel. Custom comparisons (e.g.
    GSE154778 metastatic vs primary) can be supplied as a list of dicts:

        {
          "dataset_filter": "GSE154778",   # subset obs[batch_key] == this
          "obs_key":        "condition",    # column to group by for DE
          "derive_from":    "sample",       # optional: derive obs_key from this col
          "prefix_map": {"metastatic": "metastatic", "primary": "primary"},
          "group_a":        "metastatic",   # numerator (positive LFC side)
          "group_b":        "primary",      # denominator
          "label":          "GSE154778 — PDAC",
        }

    When ``derive_from`` is set, a temporary column ``obs_key`` is created in
    the subset by mapping ``obs[derive_from].str.lower()`` through ``prefix_map``
    (prefix match: first key whose value starts with the mapped prefix wins;
    unmatched cells are labeled "other" and excluded from DE).

    Args:
        adata: Combined AnnData with log1p expression in X.
        comparisons: List of comparison dicts (see above). None = auto-detect.
        condition_col: Default obs column for DE when obs_key is not specified.
        batch_key: obs column holding dataset labels.
        numerator / denominator: Default group labels used for auto-detect.
        lfc_threshold / padj_threshold: Volcano significance thresholds.
        n_labels: Number of gene labels per volcano panel.
        n_cols: Grid columns.
        suptitle: Figure-level title override.
        save_path: Output PNG path. If None, displays interactively.
    """
    from scribe.analysis import compute_differential_expression

    if comparisons is None:
        datasets = list(adata.obs[batch_key].unique())
        comparisons = []
        for ds in datasets:
            sub = adata[adata.obs[batch_key] == ds]
            conds = set(sub.obs[condition_col].unique())
            if {numerator, denominator} <= conds:
                comparisons.append({
                    "dataset_filter": ds,
                    "obs_key": condition_col,
                    "group_a": numerator,
                    "group_b": denominator,
                    "label": ds,
                })
        combined_conds = set(adata.obs[condition_col].unique())
        if {numerator, denominator} <= combined_conds:
            comparisons.append({
                "dataset_filter": None,
                "obs_key": condition_col,
                "group_a": numerator,
                "group_b": denominator,
                "label": "Combined (all datasets)",
            })

    de_dfs: dict[str, pd.DataFrame] = {}
    panel_numerators: dict[str, str] = {}
    panel_denominators: dict[str, str] = {}
    panel_subtitles: dict[str, str] = {}

    for comp in comparisons:
        ds_filter = comp.get("dataset_filter")
        obs_key = comp.get("obs_key", condition_col)
        group_a = comp.get("group_a", numerator)
        group_b = comp.get("group_b", denominator)
        label = comp.get("label", f"{ds_filter or 'Combined'}: {group_a} vs {group_b}")

        sub = adata[adata.obs[batch_key] == ds_filter].copy() if ds_filter else adata.copy()

        derive_from = comp.get("derive_from")
        if derive_from:
            prefix_map: dict[str, str] = comp.get("prefix_map", {})
            def _map_prefix(s: str) -> str:
                s_low = s.lower()
                for pfx, lbl in prefix_map.items():
                    if s_low.startswith(pfx.lower()):
                        return lbl
                return "other"
            sub.obs[obs_key] = sub.obs[derive_from].map(_map_prefix).astype(str)
            sub = sub[sub.obs[obs_key] != "other"].copy()

        df = compute_differential_expression(
            sub, group_key=obs_key, group_a=group_a, group_b=group_b,
        )
        de_dfs[label] = df
        panel_numerators[label] = group_a
        panel_denominators[label] = group_b
        panel_subtitles[label] = f"{group_a} vs {group_b}"

    plot_volcano_grid(
        de_dfs,
        save_path=save_path,
        suptitle=suptitle,
        lfc_threshold=lfc_threshold,
        padj_threshold=padj_threshold,
        n_labels=n_labels,
        n_cols=n_cols,
        panel_numerators=panel_numerators,
        panel_denominators=panel_denominators,
        panel_subtitles=panel_subtitles,
    )


def plot_hk_pca_comparison(
    adatas: dict[str, anndata.AnnData],
    hk_genes: list[str] | None = None,
    batch_key: str = "dataset",
    save_path: str | None = None,
) -> None:
    """3-panel HK Gene PCA (one panel per correction method).

    Subsets each AnnData to housekeeping genes, runs 2-component PCA, and
    plots scatter panels side-by-side colored by batch_key.

    Args:
        adatas: OrderedDict mapping method label → AnnData.
                E.g. {"Uncorrected": adata, "ComBat": adata_c, "Harmony": adata_h}
        hk_genes: Housekeeping genes to use. Defaults to DEFAULT_HOUSEKEEPING_GENES.
        batch_key: obs column for coloring points (default: "dataset").
        save_path: Output PNG path. If None, displays interactively.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scribe.batch import DEFAULT_HOUSEKEEPING_GENES
    import scipy.sparse

    if hk_genes is None:
        hk_genes = DEFAULT_HOUSEKEEPING_GENES

    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, adata in adatas.items():
        present = [g for g in hk_genes if g in adata.var_names]
        if not present:
            print(f"  WARNING: no HK genes found in {label} — skipping panel")
            continue
        X_hk = adata[:, present].X
        if scipy.sparse.issparse(X_hk):
            X_hk = X_hk.toarray()
        X_scaled = StandardScaler().fit_transform(np.asarray(X_hk))
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        panels.append((label, coords, pca.explained_variance_ratio_))

    if not panels:
        raise ValueError("No panels could be computed — no HK genes found in any AnnData.")

    all_labels = None
    first_adata = next(iter(adatas.values()))
    if batch_key in first_adata.obs.columns:
        all_labels = first_adata.obs[batch_key].astype(str).values

    categories = sorted(pd.unique(all_labels)) if all_labels is not None else []
    cmap = plt.get_cmap("tab10")
    palette = {cat: cmap(i / max(len(categories), 1)) for i, cat in enumerate(categories)}

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(first_adata))

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 7))
    axes = list(np.atleast_1d(axes))

    for ax, (title, coords, var_ratio) in zip(axes, panels):
        if all_labels is not None and categories:
            for cat in categories:
                m = all_labels[idx] == cat
                ax.scatter(
                    coords[idx, 0][m], coords[idx, 1][m],
                    c=[palette[cat]], s=2, alpha=0.5, label=cat, rasterized=True,
                )
            ax.legend(fontsize=9, markerscale=4, loc="best", frameon=True)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.4, rasterized=True)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)", fontsize=12)

    fig.suptitle(f"HK Gene PCA — colored by {batch_key}", fontsize=15, fontweight="bold")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved HK PCA comparison -> {save_path}")
    else:
        plt.show()


def plot_feature_importance_grid(
    adata: anndata.AnnData,
    condition_col: str = "condition",
    batch_key: str = "dataset",
    numerator: str = "malignant",
    denominator: str = "normal",
    n_top_genes: int = 10,
    n_estimators: int = 200,
    layer: str | None = "X_norm",
    n_cols: int = 2,
    save_path: str | None = None,
    suptitle: str | None = None,
) -> None:
    """Train RF per dataset subset and plot top-N feature importances as a grid.

    Subsets adata by each unique value in obs[batch_key], trains a balanced RF
    classifier (numerator vs denominator), and plots horizontal bar charts.
    Adds a combined panel using the full adata. Datasets missing one condition
    are rendered as blank panels.

    Expression matrix: reads from adata.layers[layer] if layer is set and
    exists, else falls back to adata.X.

    Args:
        adata: Combined corrected AnnData — panels are subset from this.
        condition_col: obs column holding condition labels.
        batch_key: obs column holding dataset labels.
        numerator / denominator: Condition labels to classify.
        n_top_genes: Number of top genes per panel.
        n_estimators: RandomForest n_estimators.
        layer: Layer to read features from (default "X_norm"). Falls back to X.
        n_cols: Grid columns.
        save_path: Output PNG path. If None, displays interactively.
        suptitle: Figure-level title override.
    """
    import scipy.sparse
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    datasets = list(adata.obs[batch_key].unique())
    panel_order = datasets + ["Combined (all datasets)"]

    n = len(panel_order)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, panel_name in zip(axes, panel_order):
        sub = adata if panel_name == "Combined (all datasets)" else adata[adata.obs[batch_key] == panel_name]

        conds = set(sub.obs[condition_col].unique())
        if not {numerator, denominator} <= conds:
            missing = {numerator, denominator} - conds
            note = f"Missing condition(s): {missing}\nCannot classify {numerator} vs {denominator}."
            ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=10,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="#cccccc"))
            ax.set_title(panel_name, fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        mask = sub.obs[condition_col].isin([numerator, denominator])
        sub = sub[mask]

        if layer and layer in sub.layers:
            X = sub.layers[layer]
        else:
            X = sub.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        le = LabelEncoder()
        y = le.fit_transform(sub.obs[condition_col].values)

        clf = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        clf.fit(X, y)

        imp = pd.Series(clf.feature_importances_, index=sub.var_names).nlargest(n_top_genes)
        genes = imp.index[::-1]
        values = imp.values[::-1]

        ax.barh(genes, values, color="steelblue", edgecolor="none")
        ax.set_xlabel("Feature Importance", fontsize=10)
        ax.set_title(f"{panel_name}\n{numerator} vs {denominator}",
                     fontsize=12, fontweight="bold", pad=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    if suptitle is None:
        suptitle = (
            f"Top {n_top_genes} Gene Feature Importances by Dataset\n"
            f"(Random Forest, n_estimators={n_estimators})"
        )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved feature importance grid -> {save_path}")
    else:
        plt.show()


def generate_all_plots(
    adata: anndata.AnnData,
    importances: pd.Series,
    plots_dir: str,
    umap_color_columns: list[str] | None = None,
    umap_genes: list[str] | None = None,
    condition_col: str = "condition",
    lfc_numerator: str = "malignant",
    lfc_denominator: str = "normal",
    top_n: int = 10,
) -> None:
    """Generate all standard plots and save them under plots_dir.

    Creates:
        - UMAP plots by cell type, condition, and selected genes
        - Top-N feature importance bar chart
        - Top-N log fold change bar chart (if both conditions are present)

    Args:
        adata: AnnData object with pre-computed UMAP.
        importances: Top gene importances from the model.
        plots_dir: Directory the plots are written into (created if missing).
        umap_color_columns: Obs columns for UMAP plots.
            Defaults to ['condition', 'dataset', 'celltype3', 'Cell_type'].
        umap_genes: Genes to plot on UMAP. Defaults to top 2 important genes.
        condition_col: Obs column holding condition labels (for LFC plot).
        lfc_numerator / lfc_denominator: Condition labels compared by the LFC plot.
        top_n: Number of genes for the feature-importance and LFC plots.
    """
    os.makedirs(plots_dir, exist_ok=True)

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
    print(f"  Saving top-{top_n} feature importances -> {fi_path}")
    plot_feature_importances(importances, top_n=top_n, save_path=fi_path)

    # Log fold change bar chart for the top differentially expressed genes
    if condition_col in adata.obs.columns:
        present = set(adata.obs[condition_col].unique())
        if {lfc_numerator, lfc_denominator} <= present:
            from scribe.analysis import avg_expression_by_condition, compute_log_fold_change

            avg = avg_expression_by_condition(adata, condition_col=condition_col)
            lfc = compute_log_fold_change(avg, lfc_numerator, lfc_denominator)
            lfc_path = os.path.join(plots_dir, "log_fold_change.png")
            print(f"  Saving top-{top_n} log fold change -> {lfc_path}")
            plot_log_fold_change(
                lfc,
                top_n=top_n,
                numerator=lfc_numerator,
                denominator=lfc_denominator,
                save_path=lfc_path,
            )
        else:
            missing = {lfc_numerator, lfc_denominator} - present
            print(f"  Skipping log fold change plot — conditions not present: {missing}")

    print(f"All plots saved to {plots_dir}")
