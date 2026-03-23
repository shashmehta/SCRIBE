"""Click CLI entry point for SCRIBE.

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
from cellclassifier import batch as cellbatch


# @click.group() creates a parent command that groups sub-commands together.
# Running `python run.py --help` shows the available sub-commands.
@click.group()
def cli():
    """SCRIBE: Single-Cell RNA Interpretable Biomarker Explorer."""


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
            --output ./output/combined
    """
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
    default="./output/combined",
    help="Output directory for the combined .h5ad file.",
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
            --output ./output/combined
    """
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
    default="./output/batch",
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

        python run.py batch-check --data ./output/combined/combined_processed.h5ad \\
            --output ./output/batch
    """
    import scanpy as sc

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
    default="./output/batch",
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

        python run.py batch-subset --data ./output/combined/combined_processed.h5ad \\
            --output ./output/batch
    """
    import scanpy as sc

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
    default="./output/corrected",
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

        python run.py batch-correct --data ./output/combined/combined_processed.h5ad \\
            --method combat --output ./output/corrected
    """
    import scanpy as sc

    os.makedirs(output, exist_ok=True)

    click.echo("\n=== Loading Data ===")
    adata = sc.read_h5ad(data_path)
    click.echo(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    methods = ["combat", "harmony", "scanorama"] if method == "all" else [method]
    corrections = {}

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

            corrections[m] = corrected
        except ImportError as e:
            click.echo(f"  SKIPPED: {e}")
        except Exception as e:
            click.echo(f"  ERROR: {e}")

    # Generate comparison plots if we have results
    if corrections:
        click.echo("\n=== Generating Comparison Plots ===")
        umaps_dict = {"uncorrected": adata}
        umaps_dict.update(corrections)
        plotting.plot_batch_correction_comparison(umaps_dict, batch_key, save_dir=output)

        # If multiple methods, generate comparison table
        if len(corrections) > 1 or True:
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
    "--output", default="./output/hk_analysis",
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

        python run.py hk-analysis --data ./output/combined/ --output ./output/hk_analysis

    \b
    Scripts to read for understanding this analysis:
      - cellclassifier/batch.py   — select_housekeeping_genes(), run_housekeeping_pca(), run_housekeeping_de()
      - cellclassifier/plotting.py — plot_housekeeping_pca(), plot_housekeeping_violin()
      - cellclassifier/cli.py     — this command (hk-analysis)
    """
    import scanpy as sc_lib
    import anndata as ad

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

    # PCA scatter plots (colored by dataset and condition)
    plotting.plot_housekeeping_pca(adata_hk, batch_key, condition_col, save_dir=output)

    # Per-gene violin plots showing expression distributions across datasets
    plotting.plot_housekeeping_violin(adata, hk_genes, batch_key, save_dir=output)

    # Also save the existing housekeeping heatmap for comparison
    hk_heatmap_path = os.path.join(output, "housekeeping_heatmap.png")
    plotting.plot_housekeeping_heatmap(hk_expr, save_path=hk_heatmap_path)
    click.echo(f"  Saved HK heatmap -> {hk_heatmap_path}")

    click.echo(f"\n=== Housekeeping Gene Analysis Complete ===")
    click.echo(f"Results saved to {output}/")
    click.echo(f"\nKey files to review:")
    click.echo(f"  - {genes_path}  (selected genes)")
    click.echo(f"  - {de_path}  (DE results)")
    click.echo(f"  - {output}/housekeeping_pca.png  (PCA batch visualization)")
    click.echo(f"  - {output}/housekeeping_violin*.png  (per-gene distributions)")


def main():
    """Entry point called by run.py and the `cellclassifier` console script."""
    cli()
