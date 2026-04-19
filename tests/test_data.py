"""Tests for scribe/data.py — specifically load_condition_map and merge_datasets.

Key efficiency strategy: merge_datasets runs PCA, UMAP, and Leiden clustering,
which is expensive. The merged result is computed ONCE via a module-scoped fixture
and shared across all tests that only need to read from it. Tests that modify the
AnnData call .copy() first so the shared fixture is never changed.
"""

import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from scribe.data import load_condition_map, merge_datasets

# ── Synthetic data constants ──────────────────────────────────────────────────
# We need at least 31 common genes and 32 cells so sc.pp.neighbors(n_pcs=30)
# can use 30 PCA components (PCA gives at most min(n_cells, n_genes) - 1 components).
# 60 cells × 50 genes with a 40-gene overlap safely satisfies that requirement.

N_CELLS = 60          # cells per synthetic dataset; 60 × 2 datasets = 120 total
N_COMMON_GENES = 40   # genes present in BOTH datasets (the intersection)
N_UNIQUE_GENES = 10   # genes only in one dataset; they are dropped after merging
RNG = np.random.default_rng(7)


def _gene_names_dataset1() -> list[str]:
    """Dataset 1 has genes GENE_000 through GENE_049 (50 genes)."""
    return [f"GENE_{i:03d}" for i in range(50)]


def _gene_names_dataset2() -> list[str]:
    """Dataset 2 has genes GENE_010 through GENE_059 (50 genes).

    The overlap with dataset 1 is GENE_010..049 = 40 genes (N_COMMON_GENES).
    Genes GENE_000..009 are unique to dataset 1; GENE_050..059 are unique to dataset 2.
    """
    return [f"GENE_{i:03d}" for i in range(10, 60)]


def _make_processed_h5ad(path: str, gene_names: list[str], condition: str) -> None:
    """Write a synthetic 'already-processed' h5ad file.

    We use standard-normal values (mean ~0, std ~1) to mimic data that has
    already been normalised and scaled by the convert pipeline. merge_datasets
    skips normalization on such data and only runs PCA/UMAP/Leiden.
    """
    X = RNG.standard_normal((N_CELLS, len(gene_names)))
    obs = pd.DataFrame(
        {"condition": [condition] * N_CELLS},
        index=[f"{condition}_cell_{i}" for i in range(N_CELLS)],
    )
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.write_h5ad(path)


# ── Module-scoped fixtures ─────────────────────────────────────────────────────
# merge_datasets calls sc.tl.pca → sc.pp.neighbors → sc.tl.umap → sc.tl.leiden,
# which takes several seconds. We run it ONCE and reuse the result across all
# TestMergeDatasets tests.

@pytest.fixture(scope="module")
def shared_tmp(tmp_path_factory):
    """One shared temp directory for all tests in this module."""
    return tmp_path_factory.mktemp("data_tests")


@pytest.fixture(scope="module")
def two_h5ad_paths(shared_tmp):
    """Write two synthetic processed h5ad files with partially overlapping gene sets.

    Dataset 1: 60 cells × genes GENE_000..049, condition='primary'
    Dataset 2: 60 cells × genes GENE_010..059, condition='normal'
    Common genes (intersection): GENE_010..049 = 40 genes
    """
    path1 = str(shared_tmp / "dataset1.h5ad")
    path2 = str(shared_tmp / "dataset2.h5ad")
    _make_processed_h5ad(path1, _gene_names_dataset1(), condition="primary")
    _make_processed_h5ad(path2, _gene_names_dataset2(), condition="normal")
    return path1, path2


@pytest.fixture(scope="module")
def condition_map_dict():
    """A Python dict mapping per-dataset condition labels to unified labels."""
    return {"primary": "malignant", "normal": "normal"}


@pytest.fixture(scope="module")
def condition_map_yaml(shared_tmp, condition_map_dict):
    """Write the condition_map dict to a YAML file and return its path."""
    path = shared_tmp / "condition_map.yaml"
    path.write_text(yaml.dump(condition_map_dict))
    return str(path)


@pytest.fixture(scope="module")
def merged_adata(two_h5ad_paths, condition_map_dict):
    """Run merge_datasets ONCE and share the AnnData result across all merge tests.

    This is the expensive fixture — it runs PCA, UMAP, and Leiden.
    All TestMergeDatasets tests that only READ from the result share this fixture.
    Tests that need to modify the result call .copy() first.
    """
    path1, path2 = two_h5ad_paths
    return merge_datasets([path1, path2], condition_map_dict)


# ── TestLoadConditionMap ───────────────────────────────────────────────────────

class TestLoadConditionMap:
    """Tests for load_condition_map().

    load_condition_map reads a YAML file and returns a plain Python dict.
    It is the first step before merging — it tells merge_datasets how to
    translate per-dataset labels (like 'primary') into unified labels (like 'malignant').
    """

    def test_returns_dict(self, condition_map_yaml):
        """load_condition_map must return a Python dict, not a list or None."""
        result = load_condition_map(condition_map_yaml)
        assert isinstance(result, dict)

    def test_keys_match_yaml_content(self, condition_map_yaml):
        """The dict must contain exactly the keys written in the YAML file.

        We wrote {'primary': 'malignant', 'normal': 'normal'} — both keys
        must be present in the loaded result.
        """
        result = load_condition_map(condition_map_yaml)
        assert "primary" in result
        assert "normal" in result

    def test_values_match_yaml_content(self, condition_map_yaml):
        """The values must match what was written in the YAML.

        'primary' should map to 'malignant' and 'normal' should map to 'normal'.
        If the values are wrong, every cell in the merged dataset gets the wrong label.
        """
        result = load_condition_map(condition_map_yaml)
        assert result["primary"] == "malignant"
        assert result["normal"] == "normal"

    def test_nonexistent_file_raises(self):
        """Trying to load a YAML that does not exist should raise FileNotFoundError.

        This gives a clear error message when the --condition-map path is wrong,
        rather than a confusing KeyError or a silent empty dict.
        """
        with pytest.raises(FileNotFoundError):
            load_condition_map("/tmp/this_file_does_not_exist_pascal_test_12345.yaml")

    def test_larger_map_loads_all_entries(self, shared_tmp):
        """A YAML with many entries must load all of them, not just the first few.

        The real SCRIBE condition_map.yaml has 7 entries covering normal,
        primary, metastatic, IPMN, PASC, and others. We verify the loader
        handles a realistic multi-entry map without silently dropping entries.
        """
        big_map = {
            "normal": "normal",
            "primary": "malignant",
            "metastatic": "malignant",
            "IPMN": "precancerous",
            "PASC": "malignant",
        }
        path = shared_tmp / "big_condition_map.yaml"
        path.write_text(yaml.dump(big_map))
        result = load_condition_map(str(path))
        assert len(result) == 5
        assert result["IPMN"] == "precancerous"
        assert result["metastatic"] == "malignant"


# ── TestMergeDatasets ─────────────────────────────────────────────────────────

class TestMergeDatasets:
    """Tests for merge_datasets().

    merge_datasets combines multiple processed h5ad files into one. It:
    1. Remaps condition labels using a condition_map dict
    2. Keeps only genes present in ALL datasets (the intersection)
    3. Concatenates all cells into one AnnData
    4. Runs PCA → UMAP → Leiden to produce shared embeddings

    The merged_adata fixture runs the expensive PCA/UMAP/Leiden step ONCE.
    All tests here that only read from the result share that fixture.
    """

    def test_returns_anndata(self, merged_adata):
        """merge_datasets must return an AnnData object.

        AnnData is the standard container for single-cell data. All downstream
        steps (training, plotting, saving) expect this type.
        """
        assert isinstance(merged_adata, ad.AnnData)

    def test_cell_count_is_sum_of_inputs(self, merged_adata):
        """The merged dataset must contain all cells from both input datasets.

        We put 60 cells in each of 2 datasets (120 total). If any cells were
        silently dropped during the merge, this count would be wrong.
        """
        assert merged_adata.n_obs == N_CELLS * 2

    def test_gene_count_is_intersection(self, merged_adata):
        """After merging, only genes shared by ALL datasets are kept.

        Dataset 1 has GENE_000..049 and dataset 2 has GENE_010..059.
        The overlap is GENE_010..049 = 40 genes (N_COMMON_GENES).
        Genes unique to one dataset are dropped because we cannot compare
        expression across datasets for genes that were never measured in the other.
        """
        assert merged_adata.n_vars == N_COMMON_GENES

    def test_conditions_are_remapped(self, merged_adata):
        """Condition labels must be translated through the condition_map.

        'primary' → 'malignant', 'normal' → 'normal'. After merging,
        obs['condition'] should contain the UNIFIED labels only.
        The original label 'primary' should not appear in the result.
        """
        conditions = set(merged_adata.obs["condition"].unique())
        assert conditions == {"malignant", "normal"}
        assert "primary" not in conditions  # original label should be gone

    def test_dataset_column_exists(self, merged_adata):
        """Each cell must track which dataset it came from.

        The 'dataset' column in obs is added by merge_datasets so we can later
        color UMAP plots by dataset source and spot batch effects (when cells
        from different labs cluster together just because of lab differences).
        """
        assert "dataset" in merged_adata.obs.columns

    def test_umap_embedding_computed(self, merged_adata):
        """The merged AnnData must have a UMAP embedding in obsm['X_umap'].

        UMAP is required for the plotting step. merge_datasets runs PCA →
        neighbors → UMAP → Leiden so the combined dataset has fresh joint
        embeddings that place cells from all datasets in the same coordinate space.
        """
        assert "X_umap" in merged_adata.obsm

    def test_leiden_clustering_computed(self, merged_adata):
        """Leiden cluster assignments must be present in obs['leiden'].

        Leiden clustering groups cells with similar gene expression. Running it
        on the merged data groups cells from all datasets together, which helps
        identify shared cell types across studies rather than per-dataset clusters.
        """
        assert "leiden" in merged_adata.obs.columns

    def test_unmapped_condition_kept_as_original(self, two_h5ad_paths):
        """Cells whose condition is NOT in the condition_map keep their original label.

        If a dataset has a condition not covered by the YAML (e.g. 'rare_subtype'),
        merge_datasets must not silently drop or mislabel those cells — it should
        preserve the original label and print a warning. Here, 'primary' is not in
        the partial map, so it should appear unchanged in the result.
        """
        partial_map = {"normal": "normal"}  # 'primary' is intentionally missing
        path1, path2 = two_h5ad_paths
        result = merge_datasets([path1, path2], partial_map)
        conditions = set(result.obs["condition"].unique())
        # 'primary' was unmapped so it should be kept as-is
        assert "primary" in conditions
