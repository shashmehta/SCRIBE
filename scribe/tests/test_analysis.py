"""Unit tests for scribe/analysis.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scribe.analysis import (
    avg_expression_by_condition,
    compute_log_fold_change,
    compute_differential_expression,
    top_differential_genes,
)


def test_avg_expression_by_condition_keys(tiny_adata):
    avg = avg_expression_by_condition(tiny_adata, condition_col="condition")
    assert set(avg.keys()) == {"malignant", "normal"}


def test_avg_expression_by_condition_shape(tiny_adata):
    avg = avg_expression_by_condition(tiny_adata, condition_col="condition")
    assert avg["malignant"].shape == (tiny_adata.n_vars,)
    assert avg["normal"].shape == (tiny_adata.n_vars,)


def test_compute_log_fold_change_sign(tiny_adata):
    """Genes 0-4 were bumped in malignant cells, so LFC(malignant/normal) > 0."""
    avg = avg_expression_by_condition(tiny_adata, condition_col="condition")
    lfc = compute_log_fold_change(avg, numerator="malignant", denominator="normal")

    top5 = [f"GENE{i:03d}" for i in range(5)]
    assert all(lfc[g] > 0 for g in top5), f"Expected positive LFC for {top5}, got {lfc[top5].to_dict()}"


def test_compute_log_fold_change_returns_series(tiny_adata):
    avg = avg_expression_by_condition(tiny_adata, condition_col="condition")
    lfc = compute_log_fold_change(avg, numerator="malignant", denominator="normal")
    assert isinstance(lfc, pd.Series)
    assert len(lfc) == tiny_adata.n_vars


def test_compute_differential_expression_columns(tiny_adata):
    df = compute_differential_expression(
        tiny_adata, group_key="condition",
        group_a="malignant", group_b="normal",
    )
    required = {"gene", "logfoldchange", "pvalue", "pvalue_adj", "neg_log10_pvalue_adj"}
    assert required <= set(df.columns), f"Missing columns: {required - set(df.columns)}"


def test_compute_differential_expression_row_count(tiny_adata):
    df = compute_differential_expression(
        tiny_adata, group_key="condition",
        group_a="malignant", group_b="normal",
    )
    assert len(df) == tiny_adata.n_vars


def test_top_differential_genes_length(tiny_adata):
    avg = avg_expression_by_condition(tiny_adata, condition_col="condition")
    lfc = compute_log_fold_change(avg, numerator="malignant", denominator="normal")
    ratio = avg["malignant"] / (avg["normal"] + 1e-6)
    top_num, top_den = top_differential_genes(ratio, top_n=5)
    assert len(top_num) == 5
    assert len(top_den) == 5
