"""Click CLI entry point for CellClassifier.

Click is a Python library that lets you build command-line tools using decorators
(the @cli.command() and @click.option() lines). Instead of one big script, we have
sub-commands (convert, train, evaluate, plot, run) — just like `git commit` or
`git push` are sub-commands of `git`.
"""

import os
from pathlib import Path

import click

from cellclassifier.config import load_dataset_config, load_pipeline_config
from cellclassifier import geo
from cellclassifier import data as celldata
from cellclassifier import model as cellmodel
from cellclassifier import analysis as cellanal
from cellclassifier import plotting


# @click.group() creates a parent command that groups sub-commands together.
# Running `python run.py --help` shows the available sub-commands.
@click.group()
def cli():
    """CellClassifier: classify pancreatic cell conditions from scRNA-seq data."""


# ── convert ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--config", required=True,
    type=click.Path(exists=True, dir_okay=False),   # file must already exist on disk
    help="Dataset YAML config (e.g. configs/datasets/GSE154778.yaml).",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),   # must be a directory, not a file
    default=None,
    help="Output directory for the .h5ad file. Overrides config default.",
)
def convert(config, output):
    """Convert a GEO dataset to a cellxGene-compliant .h5ad file.

    Reads the dataset YAML config, loads raw files from the source
    (local or Google Drive), runs scanpy QC and preprocessing, annotates
    cellxGene schema 5.0.0 metadata, and writes a processed .h5ad.

    Example:

        python run.py convert --config configs/datasets/GSE154778.yaml --output ./output
    """
    # Load the dataset YAML into a typed DatasetConfig dataclass
    cfg = load_dataset_config(config)
    out_dir = output or "./output"
    # Delegate all the work to geo.convert_dataset — this does loading, preprocessing, and saving
    geo.convert_dataset(cfg, out_dir)


# ── train ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--config", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Pipeline YAML config (e.g. configs/pipeline.yaml).",
)
@click.option(
    "--data", "data_path",   # "data_path" is the Python variable name; "--data" is the CLI flag
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to processed .h5ad. Overrides config value.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory. Overrides config value.",
)
def train(config, data_path, output):
    """Train a RandomForest classifier on processed scRNA-seq data.

    Loads the .h5ad specified in the pipeline config (or --data override),
    extracts gene expression features, trains a RandomForestClassifier with
    balanced class weights, evaluates on a held-out test set, and saves a
    model artifact (model_artifact.joblib) to the output directory.

    Example:

        python run.py train --config configs/pipeline.yaml
        python run.py train --config configs/pipeline.yaml --data ./output/GSE162708_processed.h5ad
    """
    # Load the pipeline YAML into a typed PipelineConfig dataclass
    cfg = load_pipeline_config(config)
    # CLI flags override whatever is in the YAML file
    h5ad_path = data_path or cfg.data
    out_dir = output or cfg.output
    os.makedirs(out_dir, exist_ok=True)  # create the output directory if it doesn't exist

    click.echo("\n=== Loading Data ===")
    adata = celldata.load_adata(h5ad_path, condition_col=cfg.condition_col)

    click.echo("\n=== Extracting Features ===")
    # Convert the AnnData object into plain numpy arrays the ML model can use
    X, y, label_encoder, gene_names = celldata.extract_features_and_labels(
        adata, condition_col=cfg.condition_col
    )
    # Split cells into training and test sets (default: 80% train, 20% test)
    X_train, X_test, y_train, y_test = celldata.split_data(
        X, y, test_size=cfg.model.test_size, random_state=cfg.model.random_state
    )

    click.echo("\n=== Training Model ===")
    clf = cellmodel.train(
        X_train, y_train,
        n_estimators=cfg.model.n_estimators,
        class_weight=cfg.model.class_weight,
        random_state=cfg.model.random_state,
    )

    click.echo("\n=== Evaluating Model ===")
    # Print precision, recall, F1, and confusion matrix to the terminal
    cellmodel.evaluate(clf, X_test, y_test, label_encoder)

    # Save the trained model, label encoder, and gene names together as one file
    artifact_path = os.path.join(out_dir, "model_artifact.joblib")
    cellmodel.save_artifact(artifact_path, clf, label_encoder, gene_names)


# ── evaluate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--config", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Pipeline YAML config.",
)
@click.option(
    "--model", "model_path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a saved model_artifact.joblib.",
)
@click.option(
    "--data", "data_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to processed .h5ad. Overrides config value.",
)
def evaluate(config, model_path, data_path):
    """Evaluate a saved model artifact on processed scRNA-seq data.

    Loads the model artifact and runs it against the full dataset,
    printing a classification report and confusion matrix to stdout.

    Example:

        python run.py evaluate --config configs/pipeline.yaml \\
            --model ./output/model_artifact.joblib
    """
    cfg = load_pipeline_config(config)
    h5ad_path = data_path or cfg.data

    click.echo("\n=== Loading Data ===")
    adata = celldata.load_adata(h5ad_path, condition_col=cfg.condition_col)

    click.echo("\n=== Extracting Features ===")
    X, y, label_encoder, gene_names = celldata.extract_features_and_labels(
        adata, condition_col=cfg.condition_col
    )

    click.echo("\n=== Loading Model ===")
    # Load the previously saved .joblib file back into memory
    clf, label_encoder, gene_names = cellmodel.load_artifact(model_path)

    click.echo("\n=== Evaluation Results ===")
    # Run the loaded model on the full dataset and print the results
    cellmodel.evaluate(clf, X, y, label_encoder)


# ── plot ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--config", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Pipeline YAML config.",
)
@click.option(
    "--model", "model_path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a saved model_artifact.joblib.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory. Overrides config value.",
)
def plot(config, model_path, output):
    """Generate UMAP and feature-importance plots.

    Reads UMAP embedding from the .h5ad and feature importances from the
    model artifact, then writes PNG files to <output>/plots/.

    Example:

        python run.py plot --config configs/pipeline.yaml \\
            --model ./output/model_artifact.joblib
    """
    cfg = load_pipeline_config(config)
    out_dir = output or cfg.output

    click.echo("\n=== Loading Data ===")
    adata = celldata.load_adata(cfg.data, condition_col=cfg.condition_col)

    click.echo("\n=== Loading Model ===")
    clf, label_encoder, gene_names = cellmodel.load_artifact(model_path)

    click.echo("\n=== Feature Importances ===")
    # Ask the Random Forest how much each gene contributed to its decisions
    importances = cellmodel.get_feature_importances(
        clf, gene_names, top_n=cfg.analysis.top_n_genes
    )

    click.echo("\n=== Generating Plots ===")
    # Save UMAP scatter plots and a feature-importance bar chart to <output>/plots/
    plotting.generate_all_plots(
        adata, importances, out_dir,
        umap_color_columns=cfg.plots.umap_columns,
        umap_genes=cfg.plots.umap_genes,
    )


# ── run (full pipeline) ───────────────────────────────────────────────────────

@cli.command("run")
@click.option(
    "--config", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Pipeline YAML config.",
)
@click.option(
    "--retrain", is_flag=True, default=False,
    # is_flag=True means --retrain is a boolean switch; present = True, absent = False
    help="Force retraining even if a model artifact already exists.",
)
@click.option(
    "--data", "data_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to processed .h5ad. Overrides config value.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory. Overrides config value.",
)
def run_pipeline(config, retrain, data_path, output):
    """Full pipeline: train → evaluate → plot.

    Combines the train, evaluate, and plot subcommands into one step.
    Skips training if model_artifact.joblib already exists in the output
    directory, unless --retrain is passed.

    Example:

        python run.py run --config configs/pipeline.yaml
        python run.py run --config configs/pipeline.yaml --retrain
    """
    cfg = load_pipeline_config(config)
    h5ad_path = data_path or cfg.data
    out_dir = output or cfg.output
    os.makedirs(out_dir, exist_ok=True)

    artifact_path = os.path.join(out_dir, "model_artifact.joblib")

    # ── Step 1: Load data ────────────────────────────────────────────────────
    click.echo("\n=== Loading Data ===")
    adata = celldata.load_adata(h5ad_path, condition_col=cfg.condition_col)

    click.echo("\n=== Extracting Features ===")
    X, y, label_encoder, gene_names = celldata.extract_features_and_labels(
        adata, condition_col=cfg.condition_col
    )

    # ── Step 2: Train or load model ──────────────────────────────────────────
    # Train fresh if: (a) --retrain flag was passed, or (b) no artifact exists yet
    should_train = retrain or not Path(artifact_path).exists()

    if should_train:
        click.echo("\n=== Training Model ===")
        X_train, X_test, y_train, y_test = celldata.split_data(
            X, y, test_size=cfg.model.test_size, random_state=cfg.model.random_state
        )
        clf = cellmodel.train(
            X_train, y_train,
            n_estimators=cfg.model.n_estimators,
            class_weight=cfg.model.class_weight,
            random_state=cfg.model.random_state,
        )
        click.echo("\n=== Evaluating Model ===")
        cellmodel.evaluate(clf, X_test, y_test, label_encoder)
        cellmodel.save_artifact(artifact_path, clf, label_encoder, gene_names)
    else:
        # Reuse the saved model to skip the expensive training step
        click.echo(f"\n=== Loading Existing Model ({artifact_path}) ===")
        clf, label_encoder, gene_names = cellmodel.load_artifact(artifact_path)

    # ── Step 3: Feature importances ──────────────────────────────────────────
    click.echo("\n=== Feature Importances ===")
    importances = cellmodel.get_feature_importances(
        clf, gene_names, top_n=cfg.analysis.top_n_genes
    )

    # ── Step 4: Differential expression ─────────────────────────────────────
    # Compare average gene expression between conditions (e.g. normal vs tumor)
    click.echo("\n=== Differential Expression Analysis ===")
    avg_expr = cellanal.avg_expression_by_condition(adata, condition_col=cfg.condition_col)
    conditions = list(avg_expr.keys())
    if len(conditions) >= 2:
        # Compute ratio: expression in condition[0] / expression in condition[1]
        ratio = cellanal.compute_expression_ratio(
            avg_expr, numerator=conditions[0], denominator=conditions[1]
        )
        # Print the genes with the biggest differences between conditions
        cellanal.top_differential_genes(ratio, top_n=cfg.analysis.top_n_genes)

    # ── Step 5: Plots ────────────────────────────────────────────────────────
    click.echo("\n=== Generating Plots ===")
    plotting.generate_all_plots(
        adata, importances, out_dir,
        umap_color_columns=cfg.plots.umap_columns,
        umap_genes=cfg.plots.umap_genes,
    )

    click.echo("\nDone!")


def main():
    """Entry point called by run.py and the `cellclassifier` console script."""
    cli()
