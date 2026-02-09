"""
CellClassifier CLI — PDAC cell condition classification from scRNA-seq data.

Usage examples:
    # First run: download from Google Drive, train model, generate plots
    python run.py --gdrive-id YOUR_FILE_ID --output ./results

    # Run with local data file
    python run.py --data ./data/pdac.h5ad --output ./results

    # Load existing model (skip training), regenerate plots
    python run.py --data ./data/pdac.h5ad --model ./results/model_artifact.joblib

    # Force retrain on new data
    python run.py --data ./data/new_pdac.h5ad --output ./results_v2 --retrain
"""

import argparse
import os
from pathlib import Path

from cellclassifier.data import (
    download_from_gdrive,
    load_adata,
    extract_features_and_labels,
    split_data,
)
from cellclassifier.model import (
    train,
    evaluate,
    get_feature_importances,
    save_artifact,
    load_artifact,
)
from cellclassifier.analysis import (
    avg_expression_by_condition,
    compute_expression_ratio,
    top_differential_genes,
)
from cellclassifier.plotting import generate_all_plots


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CellClassifier: classify cell conditions from scRNA-seq data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source (one of --data or --gdrive-id required)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data", type=str, help="Local path to the H5AD data file."
    )
    data_group.add_argument(
        "--gdrive-id",
        type=str,
        help="Google Drive file ID to download the H5AD file from.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for plots, model, and results (default: ./output).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to an existing model artifact to load (skips training).",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if --model is provided.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top genes for feature importance plot (default: 20).",
    )
    parser.add_argument(
        "--condition-col",
        type=str,
        default="CONDITION",
        help="Name of the obs column for condition labels (default: CONDITION).",
    )

    return parser.parse_args()


def main() -> None:
    """Run the full CellClassifier pipeline."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # --- Step 1: Get data ---
    if args.gdrive_id:
        data_dir = os.path.join(args.output, "data")
        data_path = os.path.join(data_dir, "pdac_data.h5ad")
        download_from_gdrive(args.gdrive_id, data_path)
    else:
        data_path = args.data

    # --- Step 2: Load and preprocess ---
    print("\n=== Loading Data ===")
    adata = load_adata(data_path, condition_col=args.condition_col)

    print("\n=== Extracting Features ===")
    X, y, label_encoder, gene_names = extract_features_and_labels(
        adata, condition_col=args.condition_col
    )

    # --- Step 3: Train or load model ---
    model_path = args.model
    should_train = args.retrain or model_path is None or not Path(model_path).exists()

    if not should_train:
        print("\n=== Loading Existing Model ===")
        model, label_encoder, gene_names = load_artifact(model_path)
    else:
        print("\n=== Training Model ===")
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = train(X_train, y_train)

        print("\n=== Evaluating Model ===")
        evaluate(model, X_test, y_test, label_encoder)

        # Save artifact
        artifact_path = os.path.join(args.output, "model_artifact.joblib")
        save_artifact(artifact_path, model, label_encoder, gene_names)

    # --- Step 4: Analysis ---
    print("\n=== Feature Importances ===")
    importances = get_feature_importances(model, gene_names, top_n=args.top_n)

    print("\n=== Differential Expression Analysis ===")
    avg_expr = avg_expression_by_condition(adata, condition_col=args.condition_col)
    conditions = list(avg_expr.keys())
    if len(conditions) >= 2:
        ratio = compute_expression_ratio(
            avg_expr, numerator=conditions[0], denominator=conditions[1]
        )
        top_differential_genes(ratio)

    # --- Step 5: Visualization ---
    print("\n=== Generating Plots ===")
    generate_all_plots(adata, importances, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
