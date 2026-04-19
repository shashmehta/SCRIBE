"""Unit tests for scribe/batch.py."""

from __future__ import annotations

import pandas as pd
import pytest

from scribe.batch import compute_housekeeping_expression, DEFAULT_HOUSEKEEPING_GENES


def test_compute_housekeeping_expression_returns_dataframe(tiny_adata):
    # Use gene names that are actually in the fixture (GENE000, etc.)
    # Override with genes that exist in the tiny dataset
    present = list(tiny_adata.var_names[:3])
    result = compute_housekeeping_expression(tiny_adata, batch_key="dataset", genes=present)
    assert isinstance(result, pd.DataFrame)


def test_compute_housekeeping_expression_index_is_batches(tiny_adata):
    present = list(tiny_adata.var_names[:3])
    result = compute_housekeeping_expression(tiny_adata, batch_key="dataset", genes=present)
    expected_batches = set(tiny_adata.obs["dataset"].unique())
    assert set(result.index) == expected_batches


def test_compute_housekeeping_expression_columns_are_genes(tiny_adata):
    present = list(tiny_adata.var_names[:3])
    result = compute_housekeeping_expression(tiny_adata, batch_key="dataset", genes=present)
    assert set(result.columns) == set(present)


def test_compute_housekeeping_expression_missing_genes_raises(tiny_adata):
    with pytest.raises(ValueError, match="None of the housekeeping genes"):
        compute_housekeeping_expression(tiny_adata, genes=["NONEXISTENT_GENE_XYZ"])


def test_default_housekeeping_genes_is_list():
    assert isinstance(DEFAULT_HOUSEKEEPING_GENES, list)
    assert len(DEFAULT_HOUSEKEEPING_GENES) > 0
