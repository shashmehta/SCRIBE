"""Shared fixtures for scribe unit tests.

tiny_adata: 120 cells × 20 genes, 3 datasets, 2 conditions (malignant / normal).
The malignant class has artificially elevated expression in the first 5 genes
so that LFC / DE tests can make directional assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

RNG = np.random.default_rng(0)

N_CELLS = 120   # 40 per dataset
N_GENES = 20
DATASETS = ["DS1", "DS2", "DS3"]
CONDITIONS = ["malignant", "normal"]


def _make_tiny_adata() -> ad.AnnData:
    # Base counts drawn from a negative binomial
    X = RNG.negative_binomial(3, 0.5, (N_CELLS, N_GENES)).astype(np.float32)

    # Genes 0-4: bump malignant cells by +3 so LFC is clearly positive
    cells_per_ds = N_CELLS // len(DATASETS)
    obs_rows = []
    for ds in DATASETS:
        for i in range(cells_per_ds):
            cond = "malignant" if i < cells_per_ds // 2 else "normal"
            obs_rows.append({"dataset": ds, "condition": cond, "sample": f"{ds}_s{i % 3}"})

    obs = pd.DataFrame(obs_rows, index=[f"cell_{i}" for i in range(N_CELLS)])

    malignant_mask = (obs["condition"] == "malignant").values
    X[malignant_mask, :5] += 3.0

    # log1p-normalize (simulate post-preprocessing X)
    X = np.log1p(X)

    # Add X_norm layer (z-scored)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_norm = ((X - mean) / std).astype(np.float32)

    var = pd.DataFrame(
        index=[f"GENE{i:03d}" for i in range(N_GENES)]
    )
    var["gene_mean"] = mean
    var["gene_std"] = std

    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=obs,
        var=var,
    )
    adata.layers["X_norm"] = X_norm
    adata.obsm["X_umap"] = RNG.standard_normal((N_CELLS, 2)).astype(np.float32)
    return adata


@pytest.fixture(scope="session")
def tiny_adata() -> ad.AnnData:
    return _make_tiny_adata()


@pytest.fixture(scope="session")
def tiny_combined_h5ad(tmp_path_factory, tiny_adata) -> str:
    path = str(tmp_path_factory.mktemp("data") / "tiny_combined.h5ad")
    tiny_adata.write_h5ad(path)
    return path
