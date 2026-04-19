"""Differential expression analysis between cell conditions."""

import numpy as np
import pandas as pd
import scipy.sparse
import scanpy as sc
import anndata


def avg_expression_by_condition(
    adata: anndata.AnnData,
    condition_col: str = "CONDITION",
) -> dict[str, pd.Series]:
    """Compute mean gene expression for each condition.

    Args:
        adata: The loaded AnnData object.
        condition_col: Name of the obs column containing condition labels.

    Returns:
        Dict mapping condition label (e.g., 'N', 'T') to a pd.Series
        of mean expression indexed by gene name.
    """
    avg_by_condition = {}  # Will hold one average-expression profile per condition
    for condition in adata.obs[condition_col].unique():  # Loop over each group (e.g., Normal, Tumor)
        subset = adata[adata.obs[condition_col] == condition]  # Grab only cells from this group

        # Convert sparse to dense if needed
        if scipy.sparse.issparse(subset.X):
            expr = subset.X.toarray()
        else:
            expr = np.array(subset.X)

        # Average expression across all cells in this group, one value per gene
        avg = pd.Series(expr.mean(axis=0), index=adata.var_names)
        avg_by_condition[condition] = avg
        print(f"  Mean expression for '{condition}': {len(subset)} cells")

    return avg_by_condition


def compute_expression_ratio(
    avg_by_condition: dict[str, pd.Series],
    numerator: str = "N",
    denominator: str = "T",
    epsilon: float = 1e-6,
) -> pd.Series:
    """Compute the expression ratio between two conditions for each gene.

    Args:
        avg_by_condition: Dict from avg_expression_by_condition().
        numerator: Condition label for the numerator (default: 'N' for normal).
        denominator: Condition label for the denominator (default: 'T' for tumor).
        epsilon: Small value to replace zeros and avoid division-by-zero.

    Returns:
        pd.Series of expression ratios indexed by gene name.
    """
    # Replace exact zeros with a tiny number so we never divide by zero
    num = avg_by_condition[numerator].replace(0, epsilon)
    denom = avg_by_condition[denominator].replace(0, epsilon)
    return num / denom  # Ratio > 1 means higher in normal; ratio < 1 means higher in tumor


def compute_log_fold_change(
    avg_by_condition: dict[str, pd.Series],
    numerator: str,
    denominator: str,
    is_log_space: bool = True,
    epsilon: float = 1e-9,
) -> pd.Series:
    """Compute signed log fold change between two conditions for each gene.

    If ``is_log_space`` is True (data already log1p-normalized), LFC is the
    difference of means. Otherwise it's log2 of the ratio of means with a
    small pseudocount.

    Args:
        avg_by_condition: Dict from avg_expression_by_condition().
        numerator: Condition label treated as the numerator (e.g. 'malignant').
        denominator: Condition label treated as the denominator (e.g. 'normal').
        is_log_space: Whether mean values are already in log space.
        epsilon: Pseudocount for non-log inputs.

    Returns:
        pd.Series of signed log fold change values, indexed by gene.
    """
    num = avg_by_condition[numerator]
    denom = avg_by_condition[denominator]
    if is_log_space:
        return num - denom
    return np.log2((num + epsilon) / (denom + epsilon))


def top_differential_genes(
    ratio: pd.Series,
    top_n: int = 10,
) -> tuple[pd.Series, pd.Series]:
    """Find genes most differentially expressed toward each condition.

    Args:
        ratio: Expression ratio series from compute_expression_ratio().
        top_n: Number of top genes to return for each direction.

    Returns:
        Tuple of (top_numerator, top_denominator):
        - top_numerator: Genes with highest absolute ratio (enriched in numerator).
        - top_denominator: Genes with lowest absolute ratio (enriched in denominator).
    """
    abs_ratio = ratio.abs()  # Use absolute value so extreme ratios in both directions rank high
    sorted_ratios = abs_ratio.sort_values(ascending=False)  # Biggest differences at the top

    top_numerator = sorted_ratios.head(top_n)  # Genes most active in the normal condition
    top_denominator = sorted_ratios.tail(top_n).sort_values(ascending=True)  # Genes most active in the tumor condition

    print(f"\nTop {top_n} genes enriched in numerator condition (highest ratios):")
    print(top_numerator.to_string())
    print(f"\nTop {top_n} genes enriched in denominator condition (lowest ratios):")
    print(top_denominator.to_string())

    return top_numerator, top_denominator


def compute_differential_expression(
    adata: anndata.AnnData,
    group_key: str,
    group_a: str,
    group_b: str,
    method: str = "wilcoxon",
) -> pd.DataFrame:
    """Per-gene DE between two groups via scanpy's rank_genes_groups.

    Runs ``sc.tl.rank_genes_groups`` with ``group_a`` as the tested group and
    ``group_b`` as the single reference. LFC sign convention: positive means
    up in ``group_a``.

    Args:
        adata: AnnData with log1p-normalized expression in ``X``.
        group_key: Column in ``obs`` holding group labels.
        group_a: Label used as the tested group (volcano x>0 side).
        group_b: Label used as the single reference group.
        method: DE method passed through to scanpy (default: 'wilcoxon').

    Returns:
        DataFrame with columns: gene, logfoldchange, pvalue, pvalue_adj,
        neg_log10_pvalue_adj. Rows = all genes in adata.var_names.
    """
    mask = adata.obs[group_key].isin([group_a, group_b])
    sub = adata[mask].copy()
    sub.obs[group_key] = sub.obs[group_key].astype(str)

    n_a = int((sub.obs[group_key] == group_a).sum())
    n_b = int((sub.obs[group_key] == group_b).sum())
    print(f"  DE {group_a} (n={n_a}) vs {group_b} (n={n_b}) via {method}")

    sc.tl.rank_genes_groups(
        sub, groupby=group_key, groups=[group_a], reference=group_b,
        method=method, use_raw=False,
    )
    res = sub.uns["rank_genes_groups"]

    df = pd.DataFrame({
        "gene": [g[0] for g in res["names"]],
        "logfoldchange": [v[0] for v in res["logfoldchanges"]],
        "pvalue": [v[0] for v in res["pvals"]],
        "pvalue_adj": [v[0] for v in res["pvals_adj"]],
    })
    df["neg_log10_pvalue_adj"] = -np.log10(df["pvalue_adj"].clip(lower=1e-300))
    return df
