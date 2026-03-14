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


# ── inspect ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--config", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Dataset YAML config (e.g. configs/datasets/GSE154778.yaml).",
)
def inspect(config):
    """Inspect barcodes in a raw dataset to determine sample demultiplexing.

    Loads the raw data from the dataset config (without preprocessing) and
    prints barcode suffix distributions, helping you fill in barcode_suffix
    values in the YAML config.

    Example:

        python run.py inspect --config configs/datasets/GSE154778.yaml
    """
    cfg = load_dataset_config(config)
    base = cfg.source.base_path
    file_cfg = cfg.files[0]
    abs_path = os.path.join(base, file_cfg.relative_path)

    click.echo(f"\n=== Inspecting {cfg.id}: {cfg.title} ===")

    # Load raw data without preprocessing
    if file_cfg.format == "csv_dge":
        adata = geo.load_csv_dge(abs_path)
    elif file_cfg.format == "10x_mtx":
        adata = geo.load_10x_mtx(abs_path, name_prefix=file_cfg.name_prefix)
    elif file_cfg.format == "tar_txt_dge":
        adata = geo.load_tar_txt_dge(abs_path)
    elif file_cfg.format == "tar_10x":
        adata = geo.load_tar_10x(abs_path)
    else:
        raise click.ClickException(f"Unknown file format: {file_cfg.format!r}")

    import pandas as _pd

    barcodes = adata.obs_names.tolist()
    click.echo(f"\nTotal cells: {len(barcodes)}")
    click.echo(f"Total genes: {adata.n_vars}")

    # Show first 20 barcodes
    click.echo(f"\nFirst 20 barcodes:")
    for b in barcodes[:20]:
        click.echo(f"  {b}")

    # Detect separator: '-' (standard 10x) or ':' (e.g. GSE154778 "SAMPLE:INDEX" format)
    has_dash  = any("-" in b for b in barcodes[:100])
    has_colon = any(":" in b for b in barcodes[:100])

    if has_colon and not has_dash:
        # Prefix scheme: "P03:1" -> prefix "P03" identifies the sample
        click.echo("\nDetected COLON-separated barcode format (prefix:index).")
        prefixes = [b.split(":")[0] for b in barcodes]
        prefix_counts = _pd.Series(prefixes).value_counts().sort_index()
        click.echo(f"\nBarcode PREFIX distribution ({len(prefix_counts)} unique prefixes):")
        click.echo("  -> Use barcode_prefix in the YAML (not barcode_suffix)")
        for prefix, count in prefix_counts.items():
            example = next(b for b in barcodes if b.startswith(f"{prefix}:"))
            click.echo(f"  prefix '{prefix}': {count} cells  (e.g. {example})")
        n_unique = len(prefix_counts)
    elif has_dash:
        # Suffix scheme: "ACGT-1" -> suffix "1" identifies the sample
        click.echo("\nDetected DASH-separated barcode format (barcode-suffix).")
        suffixes = [b.split("-")[-1] for b in barcodes]
        suffix_counts = _pd.Series(suffixes).value_counts().sort_index()
        click.echo(f"\nBarcode SUFFIX distribution ({len(suffix_counts)} unique suffixes):")
        click.echo("  -> Use barcode_suffix in the YAML")
        for suffix, count in suffix_counts.items():
            examples = [b for b in barcodes if b.endswith(f"-{suffix}")][:1]
            example = examples[0] if examples else "(none)"
            click.echo(f"  suffix '{suffix}': {count} cells  (e.g. {example})")
        n_unique = len(suffix_counts)
    else:
        click.echo("\nNo standard separator found — barcodes may already be unique per cell.")
        n_unique = 1

    # Show GSM IDs if present (from TAR loaders)
    if "gsm_id" in adata.obs.columns:
        click.echo(f"\nGSM ID distribution:")
        for gsm, count in adata.obs["gsm_id"].value_counts().items():
            click.echo(f"  {gsm}: {count} cells")

    n_samples = len(cfg.samples)
    click.echo(f"\nSummary: {n_unique} unique sample identifiers, config expects {n_samples} samples")
    if n_unique == n_samples:
        click.echo("Count matches — likely 1:1 mapping between identifiers and samples.")
    else:
        click.echo("Count does NOT match — inspect manually before filling in the YAML.")


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


@cli.command()
@click.option(
    "--data", "data_paths", required=True, multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Processed .h5ad file (repeat for each dataset).",
)
@click.option(
    "--condition-map", "condition_map_path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML file mapping per-dataset conditions to unified labels.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default="./output/combined",
    help="Output directory for the combined .h5ad file.",
)
def merge(data_paths, condition_map_path, output):
    """Merge multiple processed datasets into one combined .h5ad file.

    Loads each dataset, remaps condition labels using the condition map,
    intersects gene sets, concatenates, and re-runs preprocessing for
    fresh embeddings. The combined file can then be used with `run` for
    joint training.

    Example:

        python run.py merge \\
            --data ./output/GSE154778_processed.h5ad \\
            --data ./output/GSE162708_processed.h5ad \\
            --data ./output/GSE165399_processed.h5ad \\
            --condition-map configs/condition_map.yaml \\
            --output ./output/combined
    """
    os.makedirs(output, exist_ok=True)

    click.echo("\n=== Loading Condition Map ===")
    cond_map = celldata.load_condition_map(condition_map_path)

    click.echo("\n=== Merging Datasets ===")
    combined = celldata.merge_datasets(list(data_paths), cond_map)

    out_path = os.path.join(output, "combined_processed.h5ad")
    combined.write_h5ad(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    click.echo(f"\nSaved combined dataset -> {out_path} ({size_mb:.1f} MB)")
    click.echo(f"  {combined.n_obs} cells × {combined.n_vars} genes")
    click.echo(f"  Conditions: {combined.obs['condition'].value_counts().to_dict()}")


@cli.command()
@click.option(
    "--config", "config_paths", required=True, multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Dataset YAML config (repeat for each dataset).",
)
@click.option(
    "--condition-map", "condition_map_path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML file mapping per-dataset conditions to unified labels.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default="./output/combined",
    help="Output directory for the combined .h5ad file.",
)
@click.option(
    "--skip-convert", is_flag=True, default=False,
    help="Skip the convert step if processed .h5ad files already exist.",
)
def build(config_paths, condition_map_path, output, skip_convert):
    """Convert all datasets and merge into a single unified AnnData.

    Chains the convert and merge steps into one command: for each --config,
    runs convert_dataset() to produce a processed .h5ad, then merges all
    processed files using the condition map.

    Example:

        python run.py build \\
            --config configs/datasets/GSE154778.yaml \\
            --config configs/datasets/GSE162708.yaml \\
            --config configs/datasets/GSE165399.yaml \\
            --condition-map configs/condition_map.yaml \\
            --output ./output/combined
    """
    os.makedirs(output, exist_ok=True)

    # Step 1: Convert each dataset to a processed .h5ad
    h5ad_paths = []
    for cfg_path in config_paths:
        cfg = load_dataset_config(cfg_path)
        h5ad_name = f"{cfg.id}_processed.h5ad"
        h5ad_out = os.path.join(output, h5ad_name)

        if skip_convert and os.path.exists(h5ad_out):
            click.echo(f"\nSkipping convert for {cfg.id} — {h5ad_out} already exists")
        else:
            click.echo(f"\n=== Converting {cfg.id} ===")
            geo.convert_dataset(cfg, output)

        h5ad_paths.append(h5ad_out)

    # Step 2: Load condition map and merge
    click.echo("\n=== Loading Condition Map ===")
    cond_map = celldata.load_condition_map(condition_map_path)

    click.echo("\n=== Merging Datasets ===")
    combined = celldata.merge_datasets(h5ad_paths, cond_map)

    out_path = os.path.join(output, "combined_processed.h5ad")
    combined.write_h5ad(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    click.echo(f"\nSaved combined dataset -> {out_path} ({size_mb:.1f} MB)")
    click.echo(f"  {combined.n_obs} cells × {combined.n_vars} genes")
    click.echo(f"  Conditions: {combined.obs['condition'].value_counts().to_dict()}")


def main():
    """Entry point called by run.py and the `cellclassifier` console script."""
    cli()
