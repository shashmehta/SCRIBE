"""Differential expression analysis between cell conditions."""

import numpy as np
import pandas as pd
import scipy.sparse
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
