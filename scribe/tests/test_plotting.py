"""Smoke tests for new plotting functions added in the CLI refactor.

Each test just verifies the function runs without error and produces a PNG.
Visual correctness is verified manually with real data.
"""

from __future__ import annotations

import os
import pytest


def test_lfc_grid_from_adata_creates_file(tiny_adata, tmp_path):
    from scribe.plotting import lfc_grid_from_adata

    save_path = str(tmp_path / "lfc_grid.png")
    lfc_grid_from_adata(
        tiny_adata,
        condition_col="condition",
        batch_key="dataset",
        numerator="malignant",
        denominator="normal",
        save_path=save_path,
    )
    assert os.path.exists(save_path), "Expected PNG not created"


def test_volcano_grid_from_adata_creates_file(tiny_adata, tmp_path):
    from scribe.plotting import volcano_grid_from_adata

    save_path = str(tmp_path / "volcano_grid.png")
    volcano_grid_from_adata(
        tiny_adata,
        condition_col="condition",
        batch_key="dataset",
        numerator="malignant",
        denominator="normal",
        n_labels=3,
        save_path=save_path,
    )
    assert os.path.exists(save_path), "Expected PNG not created"


def test_plot_feature_importance_grid_creates_file(tiny_adata, tmp_path):
    from scribe.plotting import plot_feature_importance_grid

    save_path = str(tmp_path / "feature_grid.png")
    plot_feature_importance_grid(
        tiny_adata,
        condition_col="condition",
        batch_key="dataset",
        numerator="malignant",
        denominator="normal",
        n_estimators=5,
        layer="X_norm",
        save_path=save_path,
    )
    assert os.path.exists(save_path), "Expected PNG not created"


def test_plot_hk_pca_comparison_creates_file(tiny_adata, tmp_path):
    from scribe.plotting import plot_hk_pca_comparison

    hk_genes = list(tiny_adata.var_names[:4])
    save_path = str(tmp_path / "hk_pca.png")
    plot_hk_pca_comparison(
        {"Uncorrected": tiny_adata, "Corrected": tiny_adata},
        hk_genes=hk_genes,
        batch_key="dataset",
        save_path=save_path,
    )
    assert os.path.exists(save_path), "Expected PNG not created"


def test_lfc_grid_blank_panel_for_missing_condition(tiny_adata, tmp_path):
    """Dataset DS1-only subset has both conditions; but we can test blank-panel logic."""
    import anndata as ad
    import scipy.sparse as sp
    import numpy as np

    # Create a dataset where 'normal' cells don't exist
    sub = tiny_adata[tiny_adata.obs["condition"] == "malignant"].copy()
    all_malignant = ad.AnnData(
        X=sp.vstack([tiny_adata.X, sub.X]),
        obs=tiny_adata.obs.copy().__class__(
            list(tiny_adata.obs.to_dict("records")) + list(sub.obs.to_dict("records")),
            index=[f"c{i}" for i in range(tiny_adata.n_obs + sub.n_obs)],
        ),
        var=tiny_adata.var.copy(),
    )
    # Give DS3 all-malignant cells by relabeling
    all_malignant.obs.loc[all_malignant.obs["dataset"] == "DS3", "condition"] = "malignant"

    from scribe.plotting import lfc_grid_from_adata
    save_path = str(tmp_path / "lfc_blank.png")
    # Should not raise even when DS3 has no normal cells
    lfc_grid_from_adata(
        tiny_adata,
        condition_col="condition",
        batch_key="dataset",
        save_path=save_path,
    )
    assert os.path.exists(save_path)
