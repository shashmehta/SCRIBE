"""Click CLI entry point for SCRIBE.

Click is a Python library that lets you build command-line tools using decorators
(the @cli.command() and @click.option() lines). Instead of one big script, we have
sub-commands (convert, train, evaluate, plot, run) — just like `git commit` or
`git push` are sub-commands of `git`.
"""

import os
import shutil
from pathlib import Path

import click

from scribe.config import load_dataset_config, load_pipeline_config
from scribe import paths
from scribe import geo
from scribe import data as celldata
from scribe import model as cellmodel
from scribe import analysis as cellanal
from scribe import plotting
from scribe import batch as cellbatch
from scribe import zarr_utils
from scribe import monitor as cellmonitor


# @click.group() creates a parent command that groups sub-commands together.
# Running `python run.py --help` shows the available sub-commands.
@click.group()
def cli():
    """SCRIBE: Single-Cell RNA Interpretable Biomarker Explorer."""


# ── setup ────────────────────────────────────────────────────────────────────

@cli.command()
def setup():
    """Check SCRIBE setup and display resolved paths.

    Verifies that the output directory exists, shows where data and cache
    files are stored, and reports h5ad file sizes. Run this after cloning
    the repo on a new machine to confirm everything is wired up correctly.
    """
    output_dir = paths.get_output_dir()
    processed = paths.get_processed_dir()
    plots = paths.get_plots_dir()
    local_cache = paths.get_local_cache_dir()
    drive_cache = paths.get_drive_cache_dir()

    click.echo("SCRIBE Setup")
    click.echo(f"  Output dir:     {output_dir} {'(exists)' if output_dir.exists() else '(MISSING)'}")

    if output_dir.is_symlink():
        click.echo(f"  Symlink target: {output_dir.resolve()}")

    click.echo(f"  Processed dir:  {processed} {'(exists)' if processed.exists() else '(MISSING)'}")
    click.echo(f"  Plots dir:      {plots} {'(exists)' if plots.exists() else '(MISSING)'}")
    click.echo(f"  Local cache:    {local_cache} {'(exists)' if local_cache.exists() else '(will be created on first app run)'}")
    click.echo(f"  Drive cache:    {drive_cache} {'(exists)' if drive_cache.exists() else '(will be created on first app run)'}")

    env_val = os.environ.get("SCRIBE_OUTPUT_DIR")
    if env_val:
        click.echo(f"  SCRIBE_OUTPUT_DIR: {env_val}")
    else:
        click.echo("  SCRIBE_OUTPUT_DIR: (not set, using ./output)")

    click.echo("\nH5AD files:")
    for name, p in zip(
        ["Uncorrected", "ComBat", "Harmony"],
        paths.get_h5ad_paths(),
    ):
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            click.echo(f"  {name:12s}  {p}  ({size_mb:.0f} MB)")
        else:
            click.echo(f"  {name:12s}  {p}  (not found)")


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
    out_dir = output or str(paths.get_processed_dir())
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
    "--plots-dir", "plots_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for plot PNGs. Overrides config value.",
)
def plot(config, model_path, plots_dir):
    """Generate UMAP and feature-importance plots.

    Reads UMAP embedding from the .h5ad and feature importances from the
    model artifact, then writes PNG files to the plots directory.

    Example:

        python run.py plot --config configs/pipeline.yaml \\
            --model ./output/processed/model_artifact.joblib
    """
    cfg = load_pipeline_config(config)
    plots_out = plots_dir or cfg.plots_dir

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
    plotting.generate_all_plots(
        adata, importances, plots_out,
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
    help="Directory for the model artifact. Overrides config value.",
)
@click.option(
    "--plots-dir", "plots_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for plot PNGs. Overrides config value.",
)
def run_pipeline(config, retrain, data_path, output, plots_dir):
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
    plots_out = plots_dir or cfg.plots_dir
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
        adata, importances, plots_out,
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
    default=None,
    help="Output directory for the combined .h5ad file. Default: $SCRIBE_OUTPUT_DIR/processed",
)
@click.option(
    "--n-top-genes", default=3000, type=int,
    help="Number of HVGs for joint batch-aware selection (default: 3000).",
)
@click.option(
    "--no-harmony", is_flag=True, default=False,
    help="Skip Harmony batch correction during merge.",
)
def merge(data_paths, condition_map_path, output, n_top_genes, no_harmony):
    """Merge multiple processed datasets into one combined .h5ad file.

    Loads each dataset, remaps condition labels using the condition map,
    finds common genes, concatenates, selects batch-aware highly variable
    genes, and computes embeddings with optional Harmony batch correction.
    The combined file can then be used with `run` for joint training.

    Example:

        python run.py merge \\
            --data ./output/GSE154778_processed.h5ad \\
            --data ./output/GSE162708_processed.h5ad \\
            --data ./output/GSE165399_processed.h5ad \\
            --condition-map configs/condition_map.yaml \\
            --output ./output/processed
    """
    output = output or str(paths.get_processed_dir())
    os.makedirs(output, exist_ok=True)

    click.echo("\n=== Loading Condition Map ===")
    cond_map = celldata.load_condition_map(condition_map_path)

    click.echo("\n=== Merging Datasets ===")
    combined = celldata.merge_datasets(
        list(data_paths), cond_map,
        n_top_genes=n_top_genes,
        harmony_correct=not no_harmony,
    )

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
    default=None,
    help="Output directory for the combined .h5ad file. Default: $SCRIBE_OUTPUT_DIR/processed",
)
@click.option(
    "--skip-convert", is_flag=True, default=False,
    help="Skip the convert step if processed .h5ad files already exist.",
)
@click.option(
    "--rebuild", is_flag=True, default=False,
    help="Force rebuild even if combined_processed.h5ad already exists.",
)
@click.option(
    "--n-top-genes", default=3000, type=int,
    help="Number of HVGs for joint batch-aware selection (default: 3000).",
)
@click.option(
    "--no-harmony", is_flag=True, default=False,
    help="Skip Harmony batch correction during build.",
)
def build(config_paths, condition_map_path, output, skip_convert, rebuild,
          n_top_genes, no_harmony):
    """Convert all datasets and merge into a single unified AnnData.

    Chains the convert and merge steps into one command: for each --config,
    runs convert_dataset() to produce a processed .h5ad, then merges all
    processed files using the condition map.

    If combined_processed.h5ad already exists in the output directory, the
    command skips everything and reuses it. Pass --rebuild to force a fresh
    build.

    Example:

        python run.py build \\
            --config configs/datasets/GSE154778.yaml \\
            --config configs/datasets/GSE162708.yaml \\
            --config configs/datasets/GSE165399.yaml \\
            --condition-map configs/condition_map.yaml \\
            --output ./output/processed
    """
    output = output or str(paths.get_processed_dir())
    os.makedirs(output, exist_ok=True)
    out_path = os.path.join(output, "combined_processed.h5ad")

    # If the combined file already exists and --rebuild was not passed, reuse it
    if not rebuild and os.path.exists(out_path):
        import scanpy as sc
        size_mb = os.path.getsize(out_path) / 1e6
        click.echo(f"\nCombined dataset already exists: {out_path} ({size_mb:.1f} MB)")
        combined = sc.read_h5ad(out_path)
        click.echo(f"  {combined.n_obs} cells × {combined.n_vars} genes")
        click.echo(f"  Conditions: {combined.obs['condition'].value_counts().to_dict()}")
        click.echo("\nTo force a fresh build, pass --rebuild")
        return

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
            geo.convert_dataset(cfg, output, skip_scale=True, skip_hvg=True, skip_embeddings=True)

        h5ad_paths.append(h5ad_out)

    # Step 2: Load condition map and merge (scaling happens here, after concat)
    click.echo("\n=== Loading Condition Map ===")
    cond_map = celldata.load_condition_map(condition_map_path)

    click.echo("\n=== Merging Datasets ===")
    combined = celldata.merge_datasets(
        h5ad_paths, cond_map,
        n_top_genes=n_top_genes,
        harmony_correct=not no_harmony,
    )

    combined.write_h5ad(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    click.echo(f"\nSaved combined dataset -> {out_path} ({size_mb:.1f} MB)")
    click.echo(f"  {combined.n_obs} cells × {combined.n_vars} genes")
    click.echo(f"  Conditions: {combined.obs['condition'].value_counts().to_dict()}")


@cli.command("batch-check")
@click.option(
    "--data", "data_path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a combined/processed .h5ad file.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory for batch diagnostic plots and reports.",
)
@click.option(
    "--batch-key",
    default="dataset",
    help="Obs column identifying the batch (default: 'dataset').",
)
@click.option(
    "--condition-col",
    default="condition",
    help="Obs column holding condition labels (default: 'condition').",
)
def batch_check(data_path, output, batch_key, condition_col):
    """Diagnose batch effects in a combined dataset.

    Computes housekeeping gene expression per batch, pairwise batch
    distances, and a batch mixing score. Generates diagnostic heatmaps
    and UMAP plots colored by batch vs condition.

    Example:

        python run.py batch-check --data ./output/processed/combined_processed.h5ad \\
            --output ./output/batch
    """
    import scanpy as sc

    output = output or str(paths.get_output_dir() / "batch")
    os.makedirs(output, exist_ok=True)

    click.echo("\n=== Loading Data ===")
    adata = sc.read_h5ad(data_path)
    click.echo(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    click.echo(f"  Batches ({batch_key}): {adata.obs[batch_key].value_counts().to_dict()}")

    click.echo("\n=== Housekeeping Gene Expression ===")
    # Use unscaled data from .raw if available (scaling zero-centers genes,
    # which erases cross-dataset expression differences needed here)
    hk_adata = adata
    if adata.raw is not None:
        hk_adata = adata.raw.to_adata()
        hk_adata.obs = adata.obs  # raw doesn't carry obs
        click.echo("  Using unscaled expression from .raw for housekeeping analysis")
    try:
        hk_expr = cellbatch.compute_housekeeping_expression(hk_adata, batch_key)
        click.echo(hk_expr.to_string())

        # Save housekeeping heatmap
        hk_path = os.path.join(output, "housekeeping_heatmap.png")
        plotting.plot_housekeeping_heatmap(hk_expr, save_path=hk_path)
        click.echo(f"  Saved -> {hk_path}")

        click.echo("\n=== Pairwise Batch Distances ===")
        distances = cellbatch.compute_batch_distances(hk_adata, batch_key)
        click.echo(distances.to_string())

        # Save distance heatmap
        dist_path = os.path.join(output, "batch_distances.png")
        plotting.plot_batch_distance_heatmap(distances, save_path=dist_path)
        click.echo(f"  Saved -> {dist_path}")
    except ValueError as e:
        click.echo(f"  WARNING: {e}")
        click.echo("  Skipping housekeeping analysis (genes not found in dataset)")

    click.echo("\n=== Batch Mixing Score ===")
    # Ensure neighbors are computed
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_pcs=30)
    mixing = cellbatch.compute_batch_mixing_score(adata, batch_key)
    click.echo(f"  Mixing score: {mixing:.4f} (0=segregated, 1=perfectly mixed)")

    click.echo("\n=== Batch UMAP ===")
    # Ensure UMAP is computed
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)
    plotting.plot_batch_umap(adata, batch_key, condition_col, save_dir=output)

    click.echo(f"\nBatch diagnostics saved to {output}/")


@cli.command("batch-subset")
@click.option(
    "--data", "data_path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a combined/processed .h5ad file.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory for plots and distance reports.",
)
@click.option(
    "--batch-key",
    default="dataset",
    help="Obs column identifying the batch (default: 'dataset').",
)
@click.option(
    "--condition-col",
    default="condition",
    help="Obs column holding condition labels (default: 'condition').",
)
@click.option(
    "--conditions",
    default="malignant,normal",
    help="Comma-separated conditions to analyse (default: 'malignant,normal').",
)
def batch_subset(data_path, output, batch_key, condition_col, conditions):
    """Per-condition batch effect analysis: subset UMAPs and distribution distances.

    For each condition (e.g. malignant, normal), generates a UMAP of only
    those cells colored by dataset, and computes pairwise distribution
    distances (Wasserstein, energy distance, MMD) between datasets.

    Example:

        python run.py batch-subset --data ./output/processed/combined_processed.h5ad \\
            --output ./output/batch
    """
    import scanpy as sc

    output = output or str(paths.get_output_dir() / "batch")
    os.makedirs(output, exist_ok=True)
    cond_list = [c.strip() for c in conditions.split(",")]

    click.echo("\n=== Loading Data ===")
    adata = sc.read_h5ad(data_path)
    click.echo(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    click.echo(f"  Conditions: {adata.obs[condition_col].value_counts().to_dict()}")

    # ── Subset UMAPs ────────────────────────────────────────────────────────
    click.echo("\n=== Generating Per-Condition UMAPs ===")
    for cond in cond_list:
        n_cells = (adata.obs[condition_col] == cond).sum()
        if n_cells == 0:
            click.echo(f"  Skipping '{cond}' — no cells found")
            continue
        save_path = os.path.join(output, f"umap_{cond}_by_{batch_key}.png")
        plotting.plot_condition_subset_umap(
            adata, condition=cond, condition_col=condition_col,
            batch_key=batch_key, save_path=save_path,
        )

    # ── Distribution distances ──────────────────────────────────────────────
    click.echo("\n=== Pairwise Distribution Distances (PCA space) ===")
    dist_results = cellbatch.compute_condition_distribution_distances(
        adata, condition_col=condition_col, batch_key=batch_key,
        conditions=cond_list,
    )

    for cond, df in dist_results.items():
        click.echo(f"\n  [{cond}]")
        click.echo(df.to_string(index=False))

        csv_path = os.path.join(output, f"distances_{cond}.csv")
        df.to_csv(csv_path, index=False)
        click.echo(f"  Saved -> {csv_path}")

    if not dist_results:
        click.echo("  No conditions with cells in ≥2 datasets — cannot compute distances.")

    click.echo(f"\nBatch subset analysis saved to {output}/")


@cli.command("batch-correct")
@click.option(
    "--data", "data_path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a combined/processed .h5ad file.",
)
@click.option(
    "--method",
    type=click.Choice(["combat", "harmony", "scanorama", "all"], case_sensitive=False),
    default="combat",
    help="Batch correction method (default: combat).",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory for corrected .h5ad files and comparison plots.",
)
@click.option(
    "--batch-key",
    default="dataset",
    help="Obs column identifying the batch (default: 'dataset').",
)
def batch_correct(data_path, method, output, batch_key):
    """Apply batch correction to a combined dataset.

    Corrects batch effects using ComBat (default), Harmony, Scanorama,
    or all three. Saves corrected .h5ad files and comparison plots.

    Example:

        python run.py batch-correct --data ./output/processed/combined_processed.h5ad \\
            --method combat --output ./output/corrected
    """
    import scanpy as sc
    from scribe.monitor import ResourceMonitor

    output = output or str(paths.get_output_dir() / "corrected")
    os.makedirs(output, exist_ok=True)

    click.echo("\n=== Loading Data ===")
    adata = sc.read_h5ad(data_path)
    click.echo(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    methods = ["combat", "harmony", "scanorama"] if method == "all" else [method]
    corrections = {}

    with ResourceMonitor(interval=2.0) as mon:
        for m in methods:
            click.echo(f"\n=== Applying {m.upper()} Correction ===")
            try:
                if m == "combat":
                    corrected = cellbatch.correct_batch_combat(adata, batch_key)
                elif m == "harmony":
                    corrected = cellbatch.correct_batch_harmony(adata, batch_key)
                elif m == "scanorama":
                    corrected = cellbatch.correct_batch_scanorama(adata, batch_key)
                else:
                    continue

                # Save corrected h5ad
                out_path = os.path.join(output, f"corrected_{m}.h5ad")
                corrected.write_h5ad(out_path)
                size_mb = os.path.getsize(out_path) / 1e6
                click.echo(f"  Saved -> {out_path} ({size_mb:.1f} MB)")

                # Report mixing score
                mixing = cellbatch.compute_batch_mixing_score(corrected, batch_key)
                click.echo(f"  Mixing score: {mixing:.4f}")

                # Report peak memory for this method
                click.echo(f"  Peak RSS so far: {mon.stats.peak_rss_mb:.1f} MB")

                corrections[m] = corrected
            except ImportError as e:
                click.echo(f"  SKIPPED: {e}")
            except MemoryError:
                click.echo(f"  OUT OF MEMORY at {mon.stats.peak_rss_mb:.1f} MB RSS!")
                click.echo(f"  Try: python run.py convert-zarr --data {data_path}")
                click.echo(f"       python run.py correct-zarr --data <output>.zarr")
                break
            except Exception as e:
                click.echo(f"  ERROR: {e}")

    click.echo(f"\n--- Resource Usage ---")
    click.echo(f"Peak RSS: {mon.stats.peak_rss_mb:.1f} MB")

    # Generate comparison plots if we have results
    if corrections:
        click.echo("\n=== Generating Comparison Plots ===")
        umaps_dict = {"uncorrected": adata}
        umaps_dict.update(corrections)
        plotting.plot_batch_correction_comparison(umaps_dict, batch_key, save_dir=output)

        # If multiple methods, generate comparison table
        if len(corrections) > 1:
            click.echo("\n=== Method Comparison ===")
            comparison = cellbatch.compare_corrections(adata, corrections, batch_key)
            click.echo(comparison.to_string())

            # Save comparison CSV
            csv_path = os.path.join(output, "correction_comparison.csv")
            comparison.to_csv(csv_path)
            click.echo(f"  Saved -> {csv_path}")

    click.echo(f"\nBatch correction results saved to {output}/")


# ── hk-analysis ───────────────────────────────────────────────────────────────

@cli.command("hk-analysis")
@click.option(
    "--data", required=True,
    type=click.Path(exists=True),
    help="Path to combined .h5ad OR directory containing per-dataset .h5ad files.",
)
@click.option(
    "--output", default=None,
    type=click.Path(file_okay=False),
    help="Directory for output plots and CSV results.",
)
@click.option(
    "--batch-key", default="dataset",
    help="Obs column identifying the source dataset/batch.",
)
@click.option(
    "--condition-col", default="condition",
    help="Obs column holding condition labels (e.g. normal, malignant).",
)
@click.option(
    "--pval-threshold", default=0.05, type=float,
    help="Adjusted p-value cutoff for excluding DE genes from HK set.",
)
@click.option(
    "--log2fc-threshold", default=0.5, type=float,
    help="Absolute log2FC cutoff for excluding DE genes from HK set.",
)
def hk_analysis(data, output, batch_key, condition_col, pval_threshold, log2fc_threshold):
    """Housekeeping gene analysis to disentangle batch effects from biology.

    This command provides a focused diagnostic: if housekeeping genes (which
    should be uniformly expressed) show systematic differences across datasets,
    those differences must be technical batch effects, not biological signal.

    The analysis needs access to ALL common genes (not just HVGs), because many
    housekeeping genes are excluded during HVG selection. If --data points to a
    directory containing per-dataset .h5ad files, they are concatenated on all
    common genes. If --data is a single .h5ad file, it is used directly (but
    may have fewer candidate genes available).

    Pipeline:
      1. Load datasets (concatenate on all common genes if directory given)
      2. Select housekeeping genes via data-driven filtering:
         - Start from ~40 curated candidates
         - Exclude any that are differentially expressed between normal
           and malignant cells (using Wilcoxon test on datasets that have both)
      3. Run PCA on the filtered housekeeping genes only
      4. Run differential expression (rank_genes_groups) across datasets
      5. Generate diagnostic plots (PCA scatter + per-gene violin plots)

    Example:

        python run.py hk-analysis --data ./output/processed/ --output ./output/hk_analysis

    \b
    Scripts to read for understanding this analysis:
      - scribe/batch.py   — select_housekeeping_genes(), run_housekeeping_pca(), run_housekeeping_de()
      - scribe/plotting.py — plot_housekeeping_pca(), plot_housekeeping_violin()
      - scribe/cli.py     — this command (hk-analysis)
    """
    import scanpy as sc_lib
    import anndata as ad

    output = output or str(paths.get_output_dir() / "hk_analysis")
    os.makedirs(output, exist_ok=True)

    # ── Step 1: Load data ──
    # If --data is a directory, concatenate per-dataset H5AD files on ALL common
    # genes (not just HVGs) so we have the full ~14,899-gene space for HK analysis.
    # If it's a single file, load it directly (may have fewer HK candidates).
    if os.path.isdir(data):
        click.echo(f"\n=== Loading per-dataset H5AD files from: {data} ===")
        # Only load per-dataset files (GSE*_processed.h5ad), not combined files
        h5ad_files = sorted(Path(data).glob("GSE*_processed.h5ad"))
        if not h5ad_files:
            raise click.ClickException(f"No GSE*_processed.h5ad files found in {data}")

        adatas = []
        for f in h5ad_files:
            a = sc_lib.read_h5ad(str(f))
            # Ensure the batch_key column exists — per-dataset files may not
            # have it, so derive it from the filename (e.g. GSE154778)
            if batch_key not in a.obs.columns:
                ds_name = f.stem.replace("_processed", "")
                a.obs[batch_key] = ds_name
            click.echo(f"  {f.name}: {a.n_obs} cells × {a.n_vars} genes")
            adatas.append(a)

        # Find genes common to ALL datasets — this gives ~14,899 genes,
        # far more than the ~3,004 HVG-filtered combined file
        common_genes = set(adatas[0].var_names)
        for a in adatas[1:]:
            common_genes &= set(a.var_names)
        common_genes = sorted(common_genes)
        click.echo(f"  Common genes across {len(adatas)} datasets: {len(common_genes)}")

        # Subset each dataset to common genes and concatenate
        adatas_common = [a[:, common_genes].copy() for a in adatas]
        adata = ad.concat(adatas_common, join="inner")
        adata.obs_names_make_unique()

        # Normalize and log-transform if not already done
        # (per-dataset files are already normalized+log1p from convert step)
        click.echo(f"  Combined: {adata.n_obs} cells × {adata.n_vars} genes")
    else:
        click.echo(f"\n=== Loading combined dataset: {data} ===")
        adata = sc_lib.read_h5ad(data)

    click.echo(f"  Shape: {adata.n_obs} cells × {adata.n_vars} genes")
    click.echo(f"  Datasets: {adata.obs[batch_key].value_counts().to_dict()}")
    click.echo(f"  Conditions: {adata.obs[condition_col].value_counts().to_dict()}")

    # ── Step 2: Data-driven housekeeping gene selection ──
    click.echo("\n=== Selecting Housekeeping Genes (data-driven filtering) ===")
    hk_genes = cellbatch.select_housekeeping_genes(
        adata,
        condition_col=condition_col,
        batch_key=batch_key,
        pval_threshold=pval_threshold,
        log2fc_threshold=log2fc_threshold,
    )

    if not hk_genes:
        click.echo("ERROR: No housekeeping genes survived filtering. "
                    "Try relaxing --pval-threshold or --log2fc-threshold.")
        return

    click.echo(f"\n  Selected {len(hk_genes)} housekeeping genes:")
    click.echo(f"  {hk_genes}")

    # Save the selected gene list for reproducibility
    genes_path = os.path.join(output, "selected_hk_genes.txt")
    with open(genes_path, "w") as f:
        f.write("\n".join(hk_genes))
    click.echo(f"  Saved gene list -> {genes_path}")

    # ── Step 3: PCA on housekeeping genes ──
    click.echo("\n=== PCA on Housekeeping Genes ===")
    adata_hk = cellbatch.run_housekeeping_pca(adata, hk_genes)

    # ── Step 4: Differential expression across datasets ──
    click.echo("\n=== Differential Expression of HK Genes Across Datasets ===")
    de_results = cellbatch.run_housekeeping_de(adata, hk_genes, batch_key=batch_key)

    # Save DE results as CSV
    de_path = os.path.join(output, "hk_de_results.csv")
    de_results.to_csv(de_path, index=False)
    click.echo(f"  Saved DE results -> {de_path}")

    # Print a summary table of mean expression per dataset per gene
    click.echo("\n=== Mean Expression Per Dataset ===")
    hk_expr = cellbatch.compute_housekeeping_expression(adata, batch_key, hk_genes)
    click.echo(hk_expr.round(3).to_string())

    # ── Step 5: Generate diagnostic plots ──
    click.echo("\n=== Generating Plots ===")

    # PCA scatter plots (colored by dataset and condition). Each plot is
    # wrapped so one failure doesn't block the rest of the pipeline.
    try:
        plotting.plot_housekeeping_pca(adata_hk, batch_key, condition_col, save_dir=output)
    except Exception as _exc:
        click.echo(f"  Skipping HK PCA plot: {_exc}")

    # Per-gene violin plots showing expression distributions across datasets
    try:
        plotting.plot_housekeeping_violin(adata, hk_genes, batch_key, save_dir=output)
    except Exception as _exc:
        click.echo(f"  Skipping HK violin plot: {_exc}")

    # Also save the existing housekeeping heatmap for comparison
    try:
        hk_heatmap_path = os.path.join(output, "housekeeping_heatmap.png")
        plotting.plot_housekeeping_heatmap(hk_expr, save_path=hk_heatmap_path)
        click.echo(f"  Saved HK heatmap -> {hk_heatmap_path}")
    except Exception as _exc:
        click.echo(f"  Skipping HK heatmap: {_exc}")

    click.echo(f"\n=== Housekeeping Gene Analysis Complete ===")
    click.echo(f"Results saved to {output}/")
    click.echo(f"\nKey files to review:")
    click.echo(f"  - {genes_path}  (selected genes)")
    click.echo(f"  - {de_path}  (DE results)")
    click.echo(f"  - {output}/housekeeping_pca.png  (PCA batch visualization)")
    click.echo(f"  - {output}/housekeeping_violin*.png  (per-gene distributions)")


# ── monitor ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--pid", default=None, type=int,
    help="Process ID to monitor. Defaults to current process.",
)
@click.option(
    "--interval", default=2.0, type=float,
    help="Seconds between samples (default: 2.0).",
)
@click.option(
    "--log", "log_path", default=None,
    type=click.Path(dir_okay=False),
    help="Optional CSV file to log resource readings to.",
)
def monitor(pid, interval, log_path):
    """Real-time system resource monitor.

    Displays live RSS memory, peak memory, CPU usage, and system memory
    utilization. Useful for watching memory during long-running batch
    correction or merge operations.

    Run in a separate terminal while another SCRIBE command executes:

    \b
        # Terminal 1: start the long-running operation
        python run.py batch-correct --data ./output/processed.h5ad --method combat

    \b
        # Terminal 2: monitor the process
        python run.py monitor --pid <PID_FROM_TERMINAL_1>

    Or monitor the current shell (self-test):

        python run.py monitor --interval 1

    Press Ctrl+C to stop and print a summary.
    """
    cellmonitor.monitor_command(pid=pid, interval=interval, log_path=log_path)


# ── convert-zarr ──────────────────────────────────────────────────────────────

@cli.command("convert-zarr")
@click.option(
    "--data", "data_path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to an .h5ad file to convert.",
)
@click.option(
    "--output", "zarr_path", default=None,
    type=click.Path(),
    help="Output .zarr store path. Defaults to same name with .zarr extension.",
)
@click.option(
    "--chunk-size", default=5000, type=int,
    help="Number of cells per chunk (default: 5000). Smaller = less memory.",
)
@click.option(
    "--overwrite", is_flag=True, default=False,
    help="Overwrite existing .zarr store.",
)
def convert_zarr(data_path, zarr_path, chunk_size, overwrite):
    """Convert an h5ad file to Zarr chunked format for memory-efficient access.

    Zarr stores data in chunked arrays on disk. Each chunk can be loaded
    independently, so you never need the full expression matrix in memory.
    This is essential for 8 GB machines running batch correction.

    Example:

    \b
        python run.py convert-zarr --data ./output/processed/combined_processed.h5ad
        python run.py convert-zarr --data ./output/processed/combined_processed.h5ad \\
            --chunk-size 2000 --overwrite
    """
    result = zarr_utils.h5ad_to_zarr(
        data_path, zarr_path=zarr_path,
        chunk_size=chunk_size, overwrite=overwrite,
    )
    click.echo(f"\nDone! Zarr store: {result}")
    click.echo("Use 'correct-zarr' to run memory-efficient batch correction on it.")


# ── correct-zarr ──────────────────────────────────────────────────────────────

@cli.command("correct-zarr")
@click.option(
    "--data", "zarr_path", required=True,
    type=click.Path(exists=True),
    help="Path to a .zarr store (from convert-zarr).",
)
@click.option(
    "--output", "output_path", default=None,
    type=click.Path(),
    help="Output .zarr store for corrected data. Defaults to <input>_corrected.zarr.",
)
@click.option(
    "--batch-key", default="dataset",
    help="Obs column identifying the batch (default: 'dataset').",
)
@click.option(
    "--chunk-size", default=5000, type=int,
    help="Cells per processing chunk (default: 5000).",
)
@click.option(
    "--method", type=click.Choice(["combat", "harmony"]), default="combat",
    help="Batch correction method (default: combat). "
         "'harmony' runs Harmony on the PCA embedding, then reconstructs "
         "a gene-level X via inverse-PCA projection (lossy).",
)
@click.option(
    "--source-h5ad", default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the original h5ad (only needed for --method harmony). "
         "Supplies varm['PCs'] loadings for inverse-PCA reconstruction. "
         "Defaults to <zarr_path>.h5ad if that exists.",
)
@click.option(
    "--n-pcs", default=50, type=int,
    help="Number of PCs to feed into Harmony (default: 50).",
)
@click.option(
    "--to-h5ad", is_flag=True, default=False,
    help="Also convert the corrected Zarr store back to h5ad.",
)
def correct_zarr(
    zarr_path, output_path, batch_key, chunk_size,
    method, source_h5ad, n_pcs, to_h5ad,
):
    """Memory-efficient batch correction on a Zarr store.

    Two methods available:

    \b
      --method combat  (default): Two-pass streaming location-scale
          correction on the gene expression matrix. Equivalent to ComBat's
          core operation in constant memory (~chunk_size × n_genes).
      --method harmony: Runs Harmony on the PCA embedding (small, fits in
          memory), then reconstructs X via inverse-PCA projection. The
          reconstructed X is lossy (only captures variance in top-N PCs)
          but lets Harmony plug into the same KDE / HK-PCA viewers.

    Example:

    \b
        # ComBat
        python run.py correct-zarr --data ./output/processed/combined_processed.zarr

        # Harmony (needs original h5ad for PCA loadings)
        python run.py correct-zarr --data ./output/processed/combined_processed.zarr \\
            --method harmony \\
            --source-h5ad ./output/processed/combined_processed.h5ad \\
            --to-h5ad
    """
    from scribe.monitor import ResourceMonitor

    if method == "combat":
        click.echo("\n=== Chunked ComBat Correction (Memory-Efficient) ===")

        with ResourceMonitor(interval=2.0) as mon:
            corrected_path = zarr_utils.chunked_combat(
                zarr_path,
                batch_key=batch_key,
                chunk_size=chunk_size,
                output_zarr_path=output_path,
            )
            click.echo("\n--- Resource Usage ---")
            click.echo(f"Peak RSS: {mon.stats.peak_rss_mb:.1f} MB")

    else:  # harmony
        click.echo("\n=== Chunked Harmony Correction (Memory-Efficient) ===")

        # Infer source h5ad if not given
        if source_h5ad is None:
            candidate = zarr_path.replace(".zarr", ".h5ad")
            if os.path.exists(candidate):
                source_h5ad = candidate
                click.echo(f"  Using source h5ad: {source_h5ad}")

        # Harmony output path defaults to <input>_harmony.zarr
        harmony_path = output_path
        if harmony_path is None:
            harmony_path = zarr_path.replace(".zarr", "_harmony.zarr")

        with ResourceMonitor(interval=2.0) as mon:
            # Stage 1: run Harmony on PCA embedding
            harmony_raw_path = harmony_path.replace(".zarr", "_raw.zarr")
            zarr_utils.chunked_harmony(
                zarr_path,
                batch_key=batch_key,
                n_pcs=n_pcs,
                source_h5ad=source_h5ad,
                output_zarr_path=harmony_raw_path,
            )

            # Stage 2: inverse-PCA project corrected PCs back to gene space
            corrected_path = zarr_utils.chunked_inverse_pca(
                harmony_raw_path,
                output_zarr_path=harmony_path,
                rep="X_pca_harmony",
                chunk_size=chunk_size,
            )

            # Clean up intermediate
            if os.path.exists(harmony_raw_path):
                shutil.rmtree(harmony_raw_path)

            click.echo("\n--- Resource Usage ---")
            click.echo(f"Peak RSS: {mon.stats.peak_rss_mb:.1f} MB")

    if to_h5ad:
        click.echo(f"\n=== Converting {method.title()} Zarr -> h5ad ===")
        h5ad_path = zarr_utils.zarr_to_h5ad(corrected_path, chunk_size=chunk_size)
        click.echo(f"  h5ad: {h5ad_path}")

        # The h5ad is the canonical output — drop the zarr intermediates so
        # output/processed only contains loadable h5ad files.
        for p in (zarr_path, corrected_path):
            if os.path.isdir(p):
                shutil.rmtree(p)
                click.echo(f"  Removed intermediate zarr: {p}")

    click.echo("\nDone!")


# ── zarr-to-h5ad ──────────────────────────────────────────────────────────────

@cli.command("zarr-to-h5ad")
@click.option(
    "--data", "zarr_path", required=True,
    type=click.Path(exists=True),
    help="Path to a .zarr store.",
)
@click.option(
    "--output", "h5ad_path", default=None,
    type=click.Path(dir_okay=False),
    help="Output .h5ad file path. Defaults to same name with .h5ad extension.",
)
@click.option(
    "--chunk-size", default=5000, type=int,
    help="Cells per read chunk (default: 5000).",
)
@click.option(
    "--delete-zarr", is_flag=True, default=False,
    help="Delete the source zarr store after the h5ad is written successfully.",
)
def zarr_to_h5ad_cmd(zarr_path, h5ad_path, chunk_size, delete_zarr):
    """Convert a Zarr store back to h5ad format.

    Reads the Zarr store chunk by chunk, assembles a sparse AnnData, and
    writes it as h5ad. Useful after running correct-zarr to get a file
    compatible with other SCRIBE commands.

    Example:

        python run.py zarr-to-h5ad --data ./output/processed/combined_processed_corrected.zarr
    """
    result = zarr_utils.zarr_to_h5ad(zarr_path, h5ad_path=h5ad_path, chunk_size=chunk_size)
    click.echo(f"\nDone! h5ad: {result}")

    if delete_zarr and os.path.isdir(zarr_path):
        shutil.rmtree(zarr_path)
        click.echo(f"Removed zarr store: {zarr_path}")


@cli.command("dataset-umap")
@click.option(
    "--data", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Combined .h5ad file (e.g. output/processed/combined_processed.h5ad).",
)
@click.option(
    "--dataset", "dataset_name", default=None,
    help="Dataset ID to plot (e.g. GSE154778). If omitted, plots all datasets.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory for the PNG files.",
)
def dataset_umap(data, dataset_name, output):
    """Side-by-side UMAP for a single dataset: leiden/cell-type vs condition.

    Subsets the combined dataset to one (or each) dataset, recomputes
    embeddings on the subset, and generates a two-panel UMAP:
      - Left panel: colored by cell type (if available) or leiden cluster
      - Right panel: colored by condition (malignant vs normal)

    Example:

        scribe dataset-umap --data ./output/processed/combined_processed.h5ad
        scribe dataset-umap --data ./output/processed/combined_processed.h5ad --dataset GSE162708
    """
    import scanpy as sc

    click.echo("\n=== Loading Data ===")
    adata = sc.read_h5ad(data)
    click.echo(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    datasets = (
        [dataset_name] if dataset_name
        else adata.obs["dataset"].unique().tolist()
    )

    output = output or str(paths.get_plots_dir())
    os.makedirs(output, exist_ok=True)
    for ds in datasets:
        click.echo(f"\n=== UMAP for {ds} ===")
        save_path = os.path.join(output, f"umap_{ds}.png")
        plotting.plot_single_dataset_umap(
            adata, dataset_name=ds, save_path=save_path,
        )

    click.echo(f"\nDone! Plots saved to {output}/")


# ── hk-pca-compare ────────────────────────────────────────────────────────────

@cli.command("hk-pca-compare")
@click.option(
    "--uncorrected", "uncorrected_path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the uncorrected combined .h5ad.",
)
@click.option(
    "--combat", "combat_path", default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to ComBat-corrected .h5ad (optional).",
)
@click.option(
    "--harmony", "harmony_path", default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to Harmony-corrected .h5ad (optional).",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory. Default: $SCRIBE_OUTPUT_DIR/plots",
)
@click.option(
    "--batch-key", "batch_key", default="dataset",
    help="obs column for coloring scatter points. Default: dataset",
)
@click.option(
    "--filename", default="hk_pca_comparison.png",
    help="Output filename. Default: hk_pca_comparison.png",
)
@click.option(
    "--hk-genes", "hk_genes_str", default=None,
    help="Comma-separated housekeeping genes to use. Defaults to ACTB,GAPDH,B2M,RPL13A,RPLP0,PPIA.",
)
def hk_pca_compare(uncorrected_path, combat_path, harmony_path, output, batch_key, filename, hk_genes_str):
    """Side-by-side HK Gene PCA: uncorrected / ComBat / Harmony.

    Reads each .h5ad directly (no cache), subsets to housekeeping genes,
    runs 2-component PCA per method, and plots scatter panels colored by
    batch_key.  ComBat and Harmony panels are optional.

    Examples:

        scribe hk-pca-compare \\
            --uncorrected output/processed/combined_processed.h5ad \\
            --combat      output/processed/combined_processed_corrected.h5ad \\
            --harmony     output/processed/combined_processed_harmony.h5ad
    """
    import scanpy as sc

    output = output or str(paths.get_plots_dir())
    adatas: dict = {}

    click.echo("\n=== Loading AnnData files ===")
    click.echo(f"  Uncorrected: {uncorrected_path}")
    adatas["Uncorrected"] = sc.read_h5ad(uncorrected_path)
    click.echo(f"    {adatas['Uncorrected'].n_obs} cells × {adatas['Uncorrected'].n_vars} genes")

    if combat_path:
        if not os.path.exists(combat_path):
            raise click.BadParameter(f"ComBat file not found: {combat_path}", param_hint="--combat")
        click.echo(f"  ComBat: {combat_path}")
        adatas["ComBat"] = sc.read_h5ad(combat_path)

    if harmony_path:
        if not os.path.exists(harmony_path):
            raise click.BadParameter(f"Harmony file not found: {harmony_path}", param_hint="--harmony")
        click.echo(f"  Harmony: {harmony_path}")
        adatas["Harmony (reconstructed)"] = sc.read_h5ad(harmony_path)

    os.makedirs(output, exist_ok=True)
    save_path = os.path.join(output, filename)

    hk_genes = [g.strip() for g in hk_genes_str.split(",")] if hk_genes_str else None

    click.echo(f"\n=== Computing HK Gene PCA ===")
    plotting.plot_hk_pca_comparison(
        adatas, hk_genes=hk_genes, batch_key=batch_key, save_path=save_path,
    )
    click.echo(f"\nDone! Saved to {save_path}")


# ── volcano ───────────────────────────────────────────────────────────────────

@cli.command("volcano")
@click.option(
    "--data", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Combined .h5ad file (corrected) to subset panels from.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory. Default: $SCRIBE_OUTPUT_DIR/plots/volcano",
)
@click.option(
    "--condition-col", "condition_col", default="condition",
    help="obs column holding condition labels. Default: condition",
)
@click.option(
    "--batch-key", "batch_key", default="dataset",
    help="obs column holding dataset labels. Default: dataset",
)
@click.option(
    "--numerator", default="malignant",
    help="Default numerator condition (positive LFC side). Default: malignant",
)
@click.option(
    "--denominator", default="normal",
    help="Default denominator condition. Default: normal",
)
@click.option(
    "--comparisons", "comparisons_path", default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML file listing custom comparison panels (see docs). "
         "If omitted, auto-detects per-dataset + combined panels.",
)
@click.option(
    "--pval", "padj_threshold", default=0.05, type=float,
    help="Adjusted p-value significance threshold. Default: 0.05",
)
@click.option(
    "--lfc", "lfc_threshold", default=1.0, type=float,
    help="Log fold change significance threshold. Default: 1.0",
)
@click.option(
    "--n-labels", "n_labels", default=8, type=int,
    help="Number of gene labels per panel. Default: 8",
)
@click.option(
    "--n-cols", "n_cols", default=2, type=int,
    help="Grid columns. Default: 2",
)
@click.option(
    "--filename", default=None,
    help="Output filename (without path). Default: volcano.png",
)
def volcano(data, output, condition_col, batch_key, numerator, denominator,
            comparisons_path, padj_threshold, lfc_threshold, n_labels, n_cols, filename):
    """Volcano plot grid, one panel per comparison.

    By default, auto-detects datasets with both conditions and adds a combined
    panel. Supply --comparisons to override with a YAML list of custom panels
    (useful for datasets like GSE154778 that need a non-standard comparison):

    \b
        # comparisons.yaml
        - dataset_filter: GSE154778
          obs_key: _derived_condition
          derive_from: sample
          prefix_map:
            metastatic: metastatic
            primary: primary
          group_a: metastatic
          group_b: primary
          label: "GSE154778 — PDAC"

    Examples:

        scribe volcano --data output/processed/combined_processed_harmony.h5ad

        scribe volcano --data output/processed/combined_processed_harmony.h5ad \\
            --comparisons configs/volcano_comparisons.yaml \\
            --filename volcano_harmony_4panel.png
    """
    import scanpy as sc

    output = output or str(paths.get_plots_dir() / "volcano")
    click.echo("\n=== Loading Data ===")
    adata = sc.read_h5ad(data)
    click.echo(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    comparisons = None
    if comparisons_path:
        import yaml
        with open(comparisons_path) as f:
            comparisons = yaml.safe_load(f)
        click.echo(f"  Loaded {len(comparisons)} custom comparisons from {comparisons_path}")

    os.makedirs(output, exist_ok=True)
    fname = filename or "volcano.png"
    save_path = os.path.join(output, fname)

    click.echo(f"\n=== Running DE and plotting volcano grid ===")
    plotting.volcano_grid_from_adata(
        adata,
        comparisons=comparisons,
        condition_col=condition_col,
        batch_key=batch_key,
        numerator=numerator,
        denominator=denominator,
        lfc_threshold=lfc_threshold,
        padj_threshold=padj_threshold,
        n_labels=n_labels,
        n_cols=n_cols,
        save_path=save_path,
    )
    click.echo(f"\nDone! Saved to {save_path}")


# ── feature-grid ──────────────────────────────────────────────────────────────

@cli.command("feature-grid")
@click.option(
    "--data", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Combined corrected .h5ad file — panels are subset from this.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory. Default: $SCRIBE_OUTPUT_DIR/plots/feature_importance",
)
@click.option(
    "--condition-col", "condition_col", default="condition",
    help="obs column holding condition labels. Default: condition",
)
@click.option(
    "--batch-key", "batch_key", default="dataset",
    help="obs column holding dataset labels. Default: dataset",
)
@click.option(
    "--numerator", default="malignant",
    help="Positive class label. Default: malignant",
)
@click.option(
    "--denominator", default="normal",
    help="Negative class label. Default: normal",
)
@click.option(
    "--n-top", "n_top_genes", default=10, type=int,
    help="Top-N genes per panel. Default: 10",
)
@click.option(
    "--n-estimators", "n_estimators", default=200, type=int,
    help="RandomForest n_estimators. Default: 200",
)
@click.option(
    "--layer", default="X_norm",
    help="Layer to use as features (default X_norm; falls back to X if absent).",
)
@click.option(
    "--filename", default="feature_importance_grid.png",
    help="Output filename. Default: feature_importance_grid.png",
)
def feature_grid(data, output, condition_col, batch_key, numerator, denominator,
                 n_top_genes, n_estimators, layer, filename):
    """2×N feature importance grid: one RF panel per dataset + combined.

    Subsets the corrected combined AnnData by dataset, trains a balanced
    RandomForestClassifier (numerator vs denominator) on each subset, and
    plots the top-N gene importances as horizontal bar charts.

    Use --layer X_norm (default) to train on z-scored expression, or
    omit/empty to use X (log1p).

    Examples:

        scribe feature-grid --data output/processed/combined_processed_harmony.h5ad

        scribe feature-grid --data output/processed/combined_processed_harmony.h5ad \\
            --n-estimators 100 --n-top 15
    """
    import scanpy as sc

    output = output or str(paths.get_plots_dir() / "feature_importance")
    click.echo("\n=== Loading Data ===")
    adata = sc.read_h5ad(data)
    click.echo(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    os.makedirs(output, exist_ok=True)
    save_path = os.path.join(output, filename)
    layer_arg = layer if layer else None

    click.echo(f"\n=== Training RF per dataset and plotting importances ===")
    plotting.plot_feature_importance_grid(
        adata,
        condition_col=condition_col,
        batch_key=batch_key,
        numerator=numerator,
        denominator=denominator,
        n_top_genes=n_top_genes,
        n_estimators=n_estimators,
        layer=layer_arg,
        save_path=save_path,
    )
    click.echo(f"\nDone! Saved to {save_path}")


# ── lfc-plot ──────────────────────────────────────────────────────────────────

@cli.command("lfc-plot")
@click.option(
    "--data", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Combined .h5ad file with log1p expression in X.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory for the PNG. Default: $SCRIBE_OUTPUT_DIR/plots/log_fold_change",
)
@click.option(
    "--condition-col", "condition_col", default="condition",
    help="obs column holding condition labels. Default: condition",
)
@click.option(
    "--batch-key", "batch_key", default="dataset",
    help="obs column holding dataset/batch labels. Default: dataset",
)
@click.option(
    "--numerator", default="malignant",
    help="Condition label for the numerator (positive LFC side). Default: malignant",
)
@click.option(
    "--denominator", default="normal",
    help="Condition label for the denominator. Default: normal",
)
@click.option(
    "--layer", default=None,
    help="Layer to read expression from instead of X (e.g. X_corrected).",
)
@click.option(
    "--top-n", "top_n", default=10, type=int,
    help="Top-N genes by |LFC| per panel. Default: 10",
)
@click.option(
    "--filename", default=None,
    help="Output filename (without path). Default: lfc_<numerator>_vs_<denominator>.png",
)
def lfc_plot(data, output, condition_col, batch_key, numerator, denominator, layer, top_n, filename):
    """2×N grid of log fold change bar charts, one panel per dataset + combined.

    Reads expression from X (or --layer), computes signed log fold change
    (numerator - denominator) for each dataset subset and the combined pool,
    then saves a bar-chart grid.  Datasets missing one condition are shown as
    blank panels with an explanatory note.

    Examples:

        scribe lfc-plot --data output/processed/combined_processed.h5ad

        scribe lfc-plot --data output/processed/combined_processed_harmony.h5ad \\
            --output output/plots/log_fold_change --filename lfc_harmony.png
    """
    import scanpy as sc

    output = output or str(paths.get_plots_dir() / "log_fold_change")
    click.echo("\n=== Loading Data ===")
    adata = sc.read_h5ad(data)
    click.echo(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    click.echo(f"  Conditions: {adata.obs[condition_col].value_counts().to_dict()}")
    click.echo(f"  Datasets:   {adata.obs[batch_key].value_counts().to_dict()}")

    os.makedirs(output, exist_ok=True)
    fname = filename or f"lfc_{numerator}_vs_{denominator}.png"
    save_path = os.path.join(output, fname)

    click.echo(f"\n=== Computing LFC ({numerator} vs {denominator}) ===")
    plotting.lfc_grid_from_adata(
        adata,
        condition_col=condition_col,
        batch_key=batch_key,
        numerator=numerator,
        denominator=denominator,
        layer=layer,
        top_n=top_n,
        save_path=save_path,
    )
    click.echo(f"\nDone! Saved to {save_path}")


def main():
    """Entry point called by run.py and the `scribe` console script."""
    cli()
