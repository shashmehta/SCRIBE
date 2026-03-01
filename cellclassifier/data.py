"""Data loading, downloading, and preprocessing for PDAC classification."""

import os
import numpy as np
import pandas as pd
import scipy.sparse
import anndata
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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
