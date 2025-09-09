"""
Pancreatic cell-type classifier (Train on human, test on mouse)
using CellxGene Census API, AnnData, and scikit-learn.

This script:
- Samples specific pancreatic cell types from human and mouse datasets.
- Loads expression data for key marker genes only, to save memory.
- Converts sparse data to dense format safely for ML processing.
- Trains a Random Forest classifier on human cells.
- Evaluates the trained classifier on mouse cells.
- Saves the trained model for later use.
"""

from __future__ import annotations  # Allows postponed evaluation of annotations for better type hinting

import argparse  # For parsing command-line arguments (inputs to the script)
import os        # For directory/file operations like creating folders
import sys       # To handle script exit on errors
import warnings  # To manage and silence warnings from external libraries
from typing import Dict, List, Sequence, Tuple  # For precise type annotations

import numpy as np      # Numerical computations and array handling
import pandas as pd     # Data manipulation with tables (DataFrames)
import anndata as ad    # Data structure specialized for single-cell data (rows=cells, cols=genes)
import matplotlib.pyplot as plt
import seaborn as sns

# High-level API to query the CellxGene Census dataset without downloading all data
import cellxgene_census

# scikit-learn: popular ML library for model training and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# joblib: efficient way to save and load Python objects like trained ML models
import joblib


# =========================
# Configuration and constants
# =========================

# These constants define species names as expected by the Census API
HUMAN = "Homo sapiens"
MOUSE = "Mus musculus"

# We only want pancreatic tissue samples
TISSUE = "pancreas"

# Cell types we're interested in classifying: pancreatic alpha, beta, and delta cells
TARGET_CELL_TYPES = [
    "pancreatic A cell",     # Alpha cells (produce glucagon)
    "type B pancreatic cell",# Beta cells (produce insulin)
    "pancreatic D cell",     # Delta cells (produce somatostatin)
]

# Canonical gene names we focus on; these genes are markers for the cell types
CANONICAL_GENES = ["GCG", "INS", "SST"]

# Gene aliases map canonical names to species-specific gene symbols used in the dataset
GENE_ALIASES: Dict[str, Dict[str, str]] = {
    HUMAN: {"GCG": "GCG", "INS": "INS", "SST": "SST"},
    MOUSE: {"GCG": "Gcg", "INS": "Ins1", "SST": "Sst"},
}


# =========================
# Helper functions
# =========================


def _list_to_filter_list_str(values: Sequence[str]) -> str:
    """
    Convert a list of strings into a string that looks like a list literal,
    e.g., ["A", "B"], to use in Census API filters.

    Parameters:
    - values: list of strings to convert

    Returns:
    - str: string formatted as a list literal suitable for API queries
    """
    # Wrap each string in double quotes and join with commas
    quoted_values = [f'"{v}"' for v in values]

    # Join all quoted strings with comma and spaces, then add square brackets
    filter_str = "[" + ", ".join(quoted_values) + "]"
    return filter_str


def safe_to_dense(X, max_bytes: int = 1_000_000_000, dtype=np.float32) -> np.ndarray:
    """
    Convert a sparse matrix to a dense NumPy array safely without running out of memory.

    Parameters:
    - X: input matrix (sparse or dense)
    - max_bytes: maximum allowed memory usage for the dense matrix (default 1GB)
    - dtype: desired data type of output array (default float32 to save memory)

    Returns:
    - np.ndarray: dense array representation of input X

    Raises:
    - MemoryError: if converting to dense would exceed max_bytes
    """
    # Get number of rows and columns from the input matrix shape
    n_rows, n_cols = X.shape

    # Calculate estimated bytes needed to store dense array:
    # number of elements * size of each element
    bytes_needed = n_rows * n_cols * np.dtype(dtype).itemsize

    # If estimated size is greater than allowed max_bytes, raise error to avoid crash
    if bytes_needed > max_bytes:
        raise MemoryError(
            f"Densifying requires ~{bytes_needed / 1e6:.1f} MB, "
            f"which exceeds the limit of {max_bytes / 1e6:.1f} MB. "
            "Try reducing the number of cells or genes."
        )

    # If the matrix is sparse (has 'toarray' method), convert it to dense format
    if hasattr(X, "toarray"):
        dense_X = X.toarray().astype(dtype, copy=False)  # copy=False avoids extra memory if not needed
    else:
        # If already dense, just ensure correct dtype
        dense_X = np.asarray(X, dtype=dtype)

    return dense_X


def sample_obs_ids(
    census,
    organism: str,
    tissue: str,
    cell_types: Sequence[str],
    n_per_type: int | None = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample a subset of cell observation IDs for specified organism, tissue, and cell types.

    This avoids loading the entire dataset, enabling manageable training/testing sizes.

    Parameters:
    - census: open Census connection object
    - organism: species name string (e.g., "Homo sapiens")
    - tissue: tissue to filter by (e.g., "pancreas")
    - cell_types: list of cell type strings to filter by
    - n_per_type: maximum number of cells to sample per cell type (None = all)
    - random_state: random seed for reproducible sampling

    Returns:
    - pd.DataFrame with columns:
        - soma_joinid: unique cell IDs in Census
        - cell_type: cell type label
    """
    # Construct a filter string for the API that selects the tissue and the list of cell types
    filter_str = (
        f'tissue == "{tissue}" and cell_type in {_list_to_filter_list_str(cell_types)}'
    )

    # Query Census for cells matching the filter, retrieving only 'soma_joinid' and 'cell_type'
    obs_df = cellxgene_census.get_obs(
        census,
        organism=organism,
        value_filter=filter_str,
        column_names=["soma_joinid", "cell_type"],
    )

    print(f"[{organism}] Found {len(obs_df)} cells in tissue '{tissue}' after filtering.")

    # If no cells found, stop with an error
    if len(obs_df) == 0:
        raise ValueError(
            f"No cells found for organism={organism}, tissue='{tissue}', cell_types={cell_types}"
        )

    # If n_per_type is None, use all matching cells; else sample up to n_per_type per cell type
    if n_per_type is None:
        sampled_df = obs_df
    else:
        # Create a random number generator with fixed seed for reproducibility
        rng = np.random.default_rng(random_state)

        sampled_parts = []  # will hold DataFrames for each cell type sample

        # For each cell type, sample randomly without replacement up to n_per_type cells
        for cell_type in cell_types:
            cells_of_type = obs_df.loc[obs_df["cell_type"] == cell_type]

            # Warn if no cells for a cell type found (still continue)
            if cells_of_type.empty:
                print(f"[WARNING] No cells found for cell_type='{cell_type}' in {organism}")
                continue

            # Sample size is min between desired n_per_type and available cells
            sample_size = min(n_per_type, len(cells_of_type))

            # Sample random indices without replacement (no repeats)
            sampled_indices = rng.choice(cells_of_type.index.values, size=sample_size, replace=False)

            # Add sampled cells to the list
            sampled_parts.append(cells_of_type.loc[sampled_indices])

        # If no cells sampled at all, raise error
        if not sampled_parts:
            raise ValueError(f"No cells were sampled for organism {organism} — all requested cell types missing.")

        # Combine sampled parts into one DataFrame and reset the index
        sampled_df = pd.concat(sampled_parts, axis=0).reset_index(drop=True)

    # Ensure columns are proper data types (int64 for IDs, str for labels)
    sampled_df["soma_joinid"] = sampled_df["soma_joinid"].astype("int64")
    sampled_df["cell_type"] = sampled_df["cell_type"].astype(str)

    # Print how many cells were sampled for each cell type
    counts = sampled_df["cell_type"].value_counts().to_dict()
    print(f"[{organism}] Sampled {sum(counts.values())} cells distributed as: {counts}")

    return sampled_df


def get_adata_for_obs_ids(
    census,
    organism: str,
    obs_ids: Sequence[int],
    var_value_filter: str,
    x_name: str = "raw",
) -> ad.AnnData:
    """
    Fetch gene expression data for selected cells and genes from Census as an AnnData object.

    AnnData stores single-cell gene expression data in a structured way.

    Parameters:
    - census: open Census connection object
    - organism: species name (e.g., "Homo sapiens")
    - obs_ids: list of cell IDs (soma_joinid) to include
    - var_value_filter: filter string to select genes by name
    - x_name: which data layer to retrieve, usually "raw" counts

    Returns:
    - AnnData object with expression matrix, cell metadata (obs), and gene metadata (var)
    """
    if not obs_ids:
        raise ValueError("No observation IDs provided to load.")

    # Request the AnnData object for specified cells and genes, selecting only RNA measurement
    adata = cellxgene_census.get_anndata(
        census,
        organism=organism,
        measurement_name="RNA",
        X_name=x_name,
        obs_coords=list(map(int, obs_ids)),
        var_value_filter=var_value_filter,
    )

    # Check required metadata columns exist in obs and var dataframes
    if "cell_type" not in adata.obs:
        raise RuntimeError("AnnData object missing 'cell_type' in obs.")
    if "feature_name" not in adata.var:
        raise RuntimeError("AnnData object missing 'feature_name' in var (gene symbols).")

    print(f"[{organism}] Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes.")

    return adata


def _build_var_filter_from_alias_map(alias_map: Dict[str, str]) -> str:
    """
    Create a filter string to select species-specific gene symbols for Census queries.

    Parameters:
    - alias_map: dict mapping canonical gene names to species gene symbols

    Returns:
    - filter string like: 'feature_name in ["Gcg","Ins1","Sst"]'
    """
    # Extract unique gene names for the species
    species_genes = sorted(set(alias_map.values()))

    # Convert gene list to Census filter list string format
    filter_str = f'feature_name in {_list_to_filter_list_str(species_genes)}'
    return filter_str


def adata_to_design_matrix(
    adata: ad.AnnData,
    canonical_order: Sequence[str],
    alias_map: Dict[str, str],
    max_dense_bytes: int = 1_000_000_000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """
    Convert AnnData gene expression data into a dense feature matrix and label array
    that scikit-learn can use for training/testing.

    Parameters:
    - adata: AnnData object with gene expression and metadata
    - canonical_order: list of canonical gene names (columns order in output)
    - alias_map: map of canonical gene names to species-specific gene symbols
    - max_dense_bytes: maximum allowed memory size when converting sparse to dense

    Returns:
    - X: 2D NumPy array (cells × genes) with gene expression features
    - y: 1D NumPy array with cell type labels as strings
    - info: dictionary with lists of present and missing canonical genes
    """
    # Extract labels (cell types) from the observation metadata
    y = adata.obs["cell_type"].astype(str).to_numpy()

    # List of gene symbols in the dataset
    gene_symbols = adata.var["feature_name"].astype(str).tolist()

    # Create a mapping from gene symbol to its column index in adata.X
    gene_to_col = {gene: idx for idx, gene in enumerate(gene_symbols)}

    # For each canonical gene, find the corresponding column index in the dataset
    col_indices = []
    present_genes = []
    missing_genes = []

    for canon_gene in canonical_order:
        # Get species-specific gene symbol for the canonical gene
        species_gene = alias_map.get(canon_gene, None)

        if species_gene in gene_to_col:
            col_indices.append(gene_to_col[species_gene])
            present_genes.append(canon_gene)
        else:
            # If gene is missing, mark None and it will be zero-filled later
            col_indices.append(None)
            missing_genes.append(canon_gene)

    # For efficient column slicing, convert matrix to compressed sparse column format if possible
    X_matrix = adata.X
    if hasattr(X_matrix, "tocsc"):
        X_matrix = X_matrix.tocsc()

    # Initialize output feature matrix with zeros: shape = (num_cells, num_canonical_genes)
    X = np.zeros((adata.n_obs, len(canonical_order)), dtype=np.float32)

    # Extract expression columns for genes that are present
    present_cols = [idx for idx in col_indices if idx is not None]

    if present_cols:
        # Convert sparse to dense safely for the selected gene columns
        X_present = safe_to_dense(
            X_matrix[:, present_cols], max_bytes=max_dense_bytes, dtype=np.float32
        )

        # Place each gene's expression column into the correct canonical column in X
        present_col_pos = 0
        for output_col_idx, dataset_col_idx in enumerate(col_indices):
            if dataset_col_idx is not None:
                X[:, output_col_idx] = X_present[:, present_col_pos]
                present_col_pos += 1

    # Warn if no canonical genes were found in the data
    if len(present_genes) == 0:
        print("[WARNING] None of the canonical genes were found in the dataset. Features are all zeros.")

    return X, y, {"present": present_genes, "missing": missing_genes}


def train_random_forest(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the provided features and labels.

    Parameters:
    - X: 2D NumPy array of features (cells × genes)
    - y: 1D NumPy array of labels (cell types)
    - random_state: seed for reproducibility

    Returns:
    - trained RandomForestClassifier model
    """
    # Random Forest is an ensemble method: many decision trees voting on classification

    # Initialize the classifier with parameters:
    clf = RandomForestClassifier(
        n_estimators=300,               # Number of trees (more trees can improve accuracy)
        class_weight="balanced_subsample",  # Adjust weights to handle imbalanced classes
        n_jobs=-1,                     # Use all CPU cores to speed up training
        random_state=random_state,     # Fix random seed for reproducible results
    )

    # Train the model: build decision trees to predict y from X
    clf.fit(X, y)

    return clf

def plot_top_pairs(
    model,
    X,
    y_true,
    feature_names,
    save_file_name,
    top_n=5,
    sample=1000,
    save_dir=None,
    prefix="",
    log_transform=False,
):
    """
    Create pairwise scatter plots of the top_n most important genes to visualize clustering.

    Parameters:
        model: Trained RandomForestClassifier (must have feature_importances_)
        X: Feature matrix (numpy array or DataFrame)
        y_true: Labels (list/array of cell types)
        feature_names: List of gene names corresponding to X columns
        top_n: Number of most important features to plot
        sample: Downsample number of points for readability
        save_dir: Directory to save plots (if None, just show them)
        prefix: Filename prefix if saving plots
        log_transform: If True, apply log scaling to features before plotting
                       (safe with zeros using log1p).
    """

    # Ensure X is DataFrame for easier handling
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)

    # Feature importances from the model
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_idx]

    # Subset to the most important features
    df = X[top_features].copy()
    df["true_label"] = y_true

    # Apply log transform if requested
    if log_transform:
        for f in top_features:
            df[f] = np.log1p(df[f])  # log1p = log(1+x), safe for zero
        print("[INFO] Applied log1p transform to feature values.")

    # Downsample if dataset is very large
    if len(df) > sample:
        df = df.sample(sample, random_state=42)

    # Fixed palette for consistency across plots
    fixed_palette = {"pancreatic A cell": "green", "type B pancreatic cell": "orange", "pancreatic D cell": "blue"}

    # Plot using Seaborn pairplot
    g = sns.pairplot(
        df,
        vars=top_features,
        hue="true_label",
        diag_kind="kde",
        plot_kws=dict(alpha=0.6, s=30),  # s=point size
        diag_kws=dict(fill=True)
    )

    g.fig.suptitle(
        f"Pairwise plots of top {top_n} features",
        fontsize=16,
        y=1.02
    )

    # Save or show the figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_file_name)
        g.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot to {save_path}")
    else:
        plt.show()
def save_model(model: RandomForestClassifier, feature_order: Sequence[str], path: str) -> None:
    """
    Save the trained model and the expected feature order to disk.

    Parameters:
    - model: trained Random Forest model to save
    - feature_order: order of genes/features expected by the model
    - path: filepath to save the model (should end with .joblib or similar)
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Bundle model and metadata into one dictionary to save together
    payload = {
        "model": model,
        "feature_order": list(feature_order),
        "metadata": {
            "task": "pancreas A/B/D classification",
            "source": "CellxGene Census",
        },
    }

    # Save using joblib (efficient for large objects)
    joblib.dump(payload, path)

    print(f"[INFO] Model saved successfully to {path}")


def load_model(path: str) -> Tuple[RandomForestClassifier, List[str]]:
    """
    Load a saved Random Forest model and its feature order from disk.

    Parameters:
    - path: filepath where the model is saved

    Returns:
    - tuple of (model, feature_order)
    """
    # Load the saved dictionary
    payload = joblib.load(path)

    model = payload["model"]
    feature_order = payload.get("feature_order", list(CANONICAL_GENES))  # fallback if missing

    print(f"[INFO] Loaded model from {path}")

    return model, feature_order


# =========================
# Main pipeline function
# =========================


def run_pipeline(
    n_per_type: int = 500,
    n_demo: int = 5,
    random_state: int = 42,
    model_path: str = "model.joblib",
) -> None:
    """
    Run the full training and evaluation pipeline:
    - Sample cells from human and mouse datasets.
    - Load gene expression data.
    - Prepare data for ML.
    - Train Random Forest on human data.
    - Evaluate on mouse data.
    - Save and reload model.
    - Show demo predictions.

    Parameters:
    - n_per_type: number of cells per cell type to sample
    - n_demo: number of demo predictions to show from test set
    - random_state: seed for reproducibility
    - model_path: filepath to save/load the model
    """

    # Suppress irrelevant warnings from dependencies for cleaner output
    warnings.simplefilter("ignore")

    # Open Census dataset connection (automatically closed at the end of this block)
    with cellxgene_census.open_soma() as census:

        # Step 1: Sample human cells of the target types from pancreas tissue
        human_cells = sample_obs_ids(
            census, organism=HUMAN, tissue=TISSUE, cell_types=TARGET_CELL_TYPES, n_per_type=n_per_type, random_state=random_state
        )

        # Step 2: Sample mouse cells for testing
        mouse_cells = sample_obs_ids(
            census, organism=MOUSE, tissue=TISSUE, cell_types=TARGET_CELL_TYPES, n_per_type=n_per_type, random_state=random_state
        )

        # Step 3: Build gene filters for human and mouse species
        human_gene_filter = _build_var_filter_from_alias_map(GENE_ALIASES[HUMAN])
        mouse_gene_filter = _build_var_filter_from_alias_map(GENE_ALIASES[MOUSE])

        # Step 4: Load expression data for sampled human cells (training set)
        adata_human = get_adata_for_obs_ids(
            census,
            organism=HUMAN,
            obs_ids=human_cells["soma_joinid"].to_list(),
            var_value_filter=human_gene_filter,
            x_name="raw",
        )

        # Step 5: Load expression data for sampled mouse cells (test set)
        adata_mouse = get_adata_for_obs_ids(
            census,
            organism=MOUSE,
            obs_ids=mouse_cells["soma_joinid"].to_list(),
            var_value_filter=mouse_gene_filter,
            x_name="raw",
        )

        # Step 6: Convert AnnData objects to ML feature matrices and label arrays
        X_train, y_train, info_h = adata_to_design_matrix(
            adata=adata_human,
            canonical_order=CANONICAL_GENES,
            alias_map=GENE_ALIASES[HUMAN],
            max_dense_bytes=1_000_000_000,
        )

        X_test, y_test, info_m = adata_to_design_matrix(
            adata=adata_mouse,
            canonical_order=CANONICAL_GENES,
            alias_map=GENE_ALIASES[MOUSE],
            max_dense_bytes=1_000_000_000,
        )

        # Step 7: Train Random Forest classifier on human training data
        clf = train_random_forest(X_train, y_train, random_state=random_state)

        # Validate clustering visually with top genes
        plot_top_pairs(
            model=clf,
            X=X_train,
            y_true=y_train,
            feature_names=CANONICAL_GENES,
            save_file_name="train_pairplot.png",
            top_n=3,          # You only have 3 canonical genes now
            sample=1000,      # Downsample if dataset is large
            save_dir="plots/train", # Optional: save figures
            prefix="train",   # Optional: naming prefix for saved plots
            log_transform=True  # Apply log transform for better visualization
        )
        
        plot_top_pairs(
            model=clf,
            X=X_test,
            y_true=y_test,
            feature_names=CANONICAL_GENES,
            save_file_name="test_pairplot.png",
            top_n=3,
            sample=1000,
            save_dir="plots/test",
            prefix="test",
            log_transform=True  # Apply log transform for better visualization
        )

        # print("[INFO] Training complete. Evaluating on mouse test data...")

        # # Step 8: Predict cell types on mouse test data
        # y_pred = clf.predict(X_test)

        # # Step 9: Print classification report to evaluate model performance
        # print(classification_report(y_test, y_pred, digits=3))

        # # Step 10: Save the trained model to disk
        # save_model(clf, feature_order=CANONICAL_GENES, path=model_path)

        # # Step 11: Load the saved model (to test persistence and reload)
        # clf_loaded, loaded_feature_order = load_model(model_path)

        # # Step 12: Demo predictions on first few test samples
        # demo_predictions = clf_loaded.predict(X_test[:n_demo])
        # print(f"[INFO] Demo predictions for first {n_demo} mouse cells: {demo_predictions}")


# =========================
# Script entry point
# =========================

if __name__ == "__main__":
    # Run the pipeline with default parameters
    run_pipeline()
