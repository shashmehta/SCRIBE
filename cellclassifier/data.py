"""Data loading, downloading, and preprocessing for PDAC classification."""

from __future__ import annotations

import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def download_from_gdrive(file_id: str, dest_path: str) -> str:
    """Download a file from Google Drive using a share link file ID.

    Args:
        file_id: The Google Drive file ID (from a share link URL).
        dest_path: Local path to save the downloaded file.

    Returns:
        The local path to the downloaded file.
    """
    # Skip downloading if the file is already on disk
    if os.path.exists(dest_path):
        print(f"File already exists at {dest_path}, skipping download.")
        return dest_path

    import gdown

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Build the direct-download URL from the Google Drive file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive (file ID: {file_id})...")
    gdown.download(url, dest_path, quiet=False)  # quiet=False shows a progress bar
    print(f"Downloaded to {dest_path}")
    return dest_path


def load_adata(h5ad_path: str, condition_col: str = "CONDITION") -> anndata.AnnData:
    """Load an H5AD file and validate required columns exist.

    Args:
        h5ad_path: Path to the .h5ad file.
        condition_col: Name of the obs column containing condition labels.

    Returns:
        The loaded AnnData object.
    """
    print(f"Loading data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)  # Read the single-cell dataset from disk

    # Validate required column exists
    if condition_col not in adata.obs.columns:
        available = ", ".join(adata.obs.columns.tolist())
        raise ValueError(
            f"Column '{condition_col}' not found in obs. Available: {available}"
        )

    # Print summary
    print(f"  Shape: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Conditions ({condition_col}):")
    # Show how many cells belong to each condition (e.g., Normal vs Tumor)
    for label, count in adata.obs[condition_col].value_counts().items():
        print(f"    {label}: {count}")

    return adata


def extract_features_and_labels(
    adata: anndata.AnnData,
    condition_col: str = "CONDITION",
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    """Extract dense expression matrix and encoded condition labels.

    Args:
        adata: The loaded AnnData object.
        condition_col: Name of the obs column containing condition labels.

    Returns:
        Tuple of (X, y_encoded, label_encoder, gene_names):
        - X: Dense expression matrix (n_cells x n_genes).
        - y_encoded: Numerically encoded condition labels.
        - label_encoder: Fitted LabelEncoder for decoding predictions.
        - gene_names: List of gene names matching X columns.
    """
    # Convert sparse matrix to dense if needed — ML models need plain arrays
    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)

    # Encode condition labels as numbers (e.g., 'Normal'->0, 'Tumor'->1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(adata.obs[condition_col])

    gene_names = adata.var_names.tolist()  # Save gene names to identify features later

    print(f"  Features shape: {X.shape}")
    print(f"  Label mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

    return X, y_encoded, le, gene_names


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split preserving class proportions.

    Args:
        X: Feature matrix.
        y: Encoded labels.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # Split into training and testing sets; stratify keeps class ratios equal in both
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def load_condition_map(yaml_path: str) -> dict[str, str]:
    """Load a condition mapping YAML file.

    The YAML maps per-dataset condition labels (e.g. "primary", "IPMN")
    to a unified classification scheme (e.g. "malignant", "precancerous").

    Args:
        yaml_path: Path to the condition mapping YAML file.

    Returns:
        Dict mapping original condition strings to unified labels.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    print(f"Loaded condition map ({len(raw)} entries) from {yaml_path}")
    for orig, unified in raw.items():
        print(f"  {orig} -> {unified}")
    return raw


def merge_datasets(
    h5ad_paths: list[str],
    condition_map: dict[str, str],
    condition_col: str = "condition",
) -> anndata.AnnData:
    """Combine multiple processed h5ad files for joint training.

    Loads each dataset, remaps condition labels to a unified scheme,
    intersects gene sets, concatenates, and re-runs preprocessing so
    embeddings are comparable across datasets.

    Args:
        h5ad_paths: Paths to processed .h5ad files.
        condition_map: Maps per-dataset condition labels to unified labels
            (e.g. {"primary": "malignant", "IPMN": "precancerous"}).
        condition_col: Name of the obs column holding condition labels.

    Returns:
        Combined AnnData with unified condition labels and fresh embeddings.
    """
    adatas = []
    for path in h5ad_paths:
        print(f"\nLoading {path}...")
        adata = sc.read_h5ad(path)
        # Extract dataset ID from filename (e.g. "GSE154778_processed.h5ad" -> "GSE154778")
        dataset_id = os.path.basename(path).replace("_processed.h5ad", "")
        adata.obs["dataset"] = dataset_id

        # Remap conditions to the unified scheme
        if condition_col in adata.obs.columns:
            original = adata.obs[condition_col].astype(str)
            unified = original.map(condition_map)
            # Cells with conditions not in the map keep their original label
            unmapped = unified.isna()
            if unmapped.any():
                unmapped_vals = original[unmapped].unique().tolist()
                print(f"  WARNING: unmapped conditions {unmapped_vals} — keeping original labels")
                unified = unified.fillna(original)
            adata.obs[condition_col] = unified.values
        else:
            print(f"  WARNING: column '{condition_col}' not found — skipping remap")

        print(f"  {adata.n_obs} cells × {adata.n_vars} genes, dataset={dataset_id}")
        print(f"  Conditions: {adata.obs[condition_col].value_counts().to_dict()}")
        adatas.append(adata)

    # Find the intersection of gene names across all datasets
    common_genes = set(adatas[0].var_names)
    for adata in adatas[1:]:
        common_genes &= set(adata.var_names)
    common_genes = sorted(common_genes)
    print(f"\nCommon genes across {len(adatas)} datasets: {len(common_genes)}")

    # Subset each AnnData to the common gene set
    for i, adata in enumerate(adatas):
        adatas[i] = adata[:, common_genes].copy()

    # Concatenate all datasets into one AnnData
    combined = anndata.concat(adatas, label="dataset_key", join="inner")
    # Preserve the per-cell dataset column (concat may overwrite it)
    if "dataset" not in combined.obs.columns:
        combined.obs["dataset"] = combined.obs["dataset_key"]
    print(f"\nCombined: {combined.n_obs} cells × {combined.n_vars} genes")
    print(f"Datasets: {combined.obs['dataset'].value_counts().to_dict()}")
    print(f"Conditions: {combined.obs[condition_col].value_counts().to_dict()}")

    # The individual datasets are already normalized+log-transformed+scaled.
    # We cannot re-run filter/normalize/HVG steps on scaled data — filter_cells
    # would see all values as near-zero (the mean is 0 after scaling) and drop
    # every cell. Instead, we just compute fresh joint embeddings (PCA → UMAP →
    # Leiden) so cells from all three datasets are in the same coordinate space.
    print("\nComputing joint embeddings (PCA → neighbors → UMAP → Leiden)...")
    combined.var_names_make_unique()
    combined.obs_names_make_unique()
    sc.tl.pca(combined)
    sc.pp.neighbors(combined, n_pcs=30)
    sc.tl.umap(combined)
    sc.tl.leiden(combined, resolution=0.5)
    print(
        f"  Done: {combined.n_obs} cells × {combined.n_vars} genes, "
        f"{combined.obs['leiden'].nunique()} Leiden clusters"
    )

    return combined
