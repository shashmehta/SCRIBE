"""Tests for cellclassifier/batch.py — batch effect detection and correction.

Synthetic data: 2 batches of 60 cells each. Batch 2 has a systematic
expression shift on all genes, simulating a real technical batch effect.
Housekeeping genes are included so batch detection functions can find them.
"""

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scanpy as sc
import scipy.sparse as sp

from cellclassifier import batch as cellbatch

# ── Synthetic data ───────────────────────────────────────────────────────────

N_CELLS_PER_BATCH = 60
N_GENES = 30
RNG = np.random.default_rng(42)

# Include some housekeeping gene names so detection functions work
HK_GENES = ["ACTB", "GAPDH", "B2M"]
OTHER_GENES = [f"GENE{i:03d}" for i in range(N_GENES - len(HK_GENES))]
ALL_GENES = HK_GENES + OTHER_GENES


def _make_batched_adata(shift: float = 2.0) -> ad.AnnData:
    """Create a 2-batch AnnData where batch 2 has a systematic expression shift.

    Args:
        shift: How much to shift batch 2 expression (higher = more batch effect).

    Returns:
        AnnData with 120 cells, obs columns 'dataset' and 'condition',
        and pre-computed PCA/neighbors/UMAP.
    """
    # Batch 1: baseline expression
    X1 = RNG.standard_normal((N_CELLS_PER_BATCH, N_GENES))
    # Batch 2: shifted expression (simulating batch effect)
    X2 = RNG.standard_normal((N_CELLS_PER_BATCH, N_GENES)) + shift

    X = np.vstack([X1, X2])
    obs = pd.DataFrame({
        "dataset": ["batch1"] * N_CELLS_PER_BATCH + ["batch2"] * N_CELLS_PER_BATCH,
        "condition": (["normal"] * (N_CELLS_PER_BATCH // 2) +
                      ["malignant"] * (N_CELLS_PER_BATCH // 2)) * 2,
    }, index=[f"cell_{i}" for i in range(N_CELLS_PER_BATCH * 2)])
    var = pd.DataFrame(index=ALL_GENES)

    adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=min(20, N_GENES - 1))
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    return adata


def _make_mixed_adata() -> ad.AnnData:
    """Create an AnnData where batches are well-mixed (no batch effect).

    Both batches drawn from the same distribution — mixing score should be high.
    """
    X = RNG.standard_normal((N_CELLS_PER_BATCH * 2, N_GENES))
    obs = pd.DataFrame({
        "dataset": ["batch1"] * N_CELLS_PER_BATCH + ["batch2"] * N_CELLS_PER_BATCH,
        "condition": ["normal"] * (N_CELLS_PER_BATCH * 2),
    }, index=[f"cell_{i}" for i in range(N_CELLS_PER_BATCH * 2)])
    var = pd.DataFrame(index=ALL_GENES)

    adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=min(20, N_GENES - 1))
    sc.tl.umap(adata)
    return adata


# ── Module-scoped fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def batched_adata():
    """AnnData with a clear batch effect (shift=2.0)."""
    return _make_batched_adata(shift=2.0)


@pytest.fixture(scope="module")
def mixed_adata():
    """AnnData with no batch effect (both batches from same distribution)."""
    return _make_mixed_adata()


# ── TestHousekeepingExpression ───────────────────────────────────────────────

class TestHousekeepingExpression:
    """Tests for compute_housekeeping_expression()."""

    def test_returns_correct_shape(self, batched_adata):
        """Output should have one row per batch, one column per housekeeping gene."""
        result = cellbatch.compute_housekeeping_expression(
            batched_adata, batch_key="dataset", genes=HK_GENES
        )
        assert result.shape == (2, len(HK_GENES))

    def test_values_differ_between_batches(self, batched_adata):
        """With a systematic shift, mean expression should differ between batches."""
        result = cellbatch.compute_housekeeping_expression(
            batched_adata, batch_key="dataset", genes=HK_GENES
        )
        # Batch 2 was shifted up — its means should be higher than batch 1
        batch1_mean = result.loc["batch1"].mean()
        batch2_mean = result.loc["batch2"].mean()
        assert batch2_mean > batch1_mean

    def test_raises_on_missing_genes(self, batched_adata):
        """Should raise ValueError if none of the requested genes exist."""
        with pytest.raises(ValueError, match="None of the housekeeping genes"):
            cellbatch.compute_housekeeping_expression(
                batched_adata, batch_key="dataset", genes=["NONEXISTENT_GENE"]
            )


# ── TestBatchDistances ───────────────────────────────────────────────────────

class TestBatchDistances:
    """Tests for compute_batch_distances()."""

    def test_symmetric_matrix(self, batched_adata):
        """Distance matrix should be symmetric."""
        result = cellbatch.compute_batch_distances(
            batched_adata, batch_key="dataset", genes=HK_GENES
        )
        np.testing.assert_array_almost_equal(result.values, result.values.T)

    def test_diagonal_is_zero(self, batched_adata):
        """Distance from a batch to itself should be zero."""
        result = cellbatch.compute_batch_distances(
            batched_adata, batch_key="dataset", genes=HK_GENES
        )
        np.testing.assert_array_almost_equal(np.diag(result.values), 0)

    def test_detects_shift(self, batched_adata):
        """Distance between shifted batches should be positive."""
        result = cellbatch.compute_batch_distances(
            batched_adata, batch_key="dataset", genes=HK_GENES
        )
        off_diag = result.values[0, 1]
        assert off_diag > 0.5, f"Expected significant distance, got {off_diag}"


# ── TestCorrectBatchCombat ───────────────────────────────────────────────────

class TestCorrectBatchCombat:
    """Tests for correct_batch_combat()."""

    @pytest.fixture(scope="class")
    def corrected(self):
        """Run ComBat once and share across all tests in this class."""
        adata = _make_batched_adata(shift=2.0)
        return cellbatch.correct_batch_combat(adata, batch_key="dataset")

    def test_same_cell_count(self, corrected, batched_adata):
        """ComBat should not add or remove cells."""
        assert corrected.n_obs == batched_adata.n_obs

    def test_obs_preserved(self, corrected):
        """Batch and condition columns should survive correction."""
        assert "dataset" in corrected.obs.columns
        assert "condition" in corrected.obs.columns

    def test_has_embeddings(self, corrected):
        """Corrected data should have recomputed PCA and UMAP."""
        assert "X_pca" in corrected.obsm
        assert "X_umap" in corrected.obsm


# ── TestBatchMixingScore ─────────────────────────────────────────────────────

class TestBatchMixingScore:
    """Tests for compute_batch_mixing_score()."""

    def test_separated_batches_low_score(self, batched_adata):
        """Clearly separated batches should have a low mixing score."""
        score = cellbatch.compute_batch_mixing_score(
            batched_adata, batch_key="dataset", n_neighbors=20
        )
        assert 0 <= score <= 1
        assert score < 0.5, f"Expected low mixing for separated batches, got {score}"

    def test_mixed_batches_higher_score(self, mixed_adata):
        """Well-mixed batches should have a higher mixing score."""
        score = cellbatch.compute_batch_mixing_score(
            mixed_adata, batch_key="dataset", n_neighbors=20
        )
        assert 0 <= score <= 1
        assert score > 0.3, f"Expected higher mixing for mixed batches, got {score}"

    def test_mixed_higher_than_separated(self, batched_adata, mixed_adata):
        """Mixed batches should score higher than separated batches."""
        sep_score = cellbatch.compute_batch_mixing_score(
            batched_adata, batch_key="dataset", n_neighbors=20
        )
        mix_score = cellbatch.compute_batch_mixing_score(
            mixed_adata, batch_key="dataset", n_neighbors=20
        )
        assert mix_score > sep_score


# ── TestCompareCorrections ───────────────────────────────────────────────────

class TestCompareCorrections:
    """Tests for compare_corrections()."""

    def test_returns_dataframe_with_metrics(self):
        """Should return a DataFrame with mixing_score and silhouette columns."""
        adata = _make_batched_adata(shift=2.0)
        corrected = cellbatch.correct_batch_combat(adata, batch_key="dataset")
        result = cellbatch.compare_corrections(
            adata, {"combat": corrected}, batch_key="dataset"
        )
        assert isinstance(result, pd.DataFrame)
        assert "mixing_score" in result.columns
        assert "silhouette" in result.columns
        assert "uncorrected" in result.index
        assert "combat" in result.index
