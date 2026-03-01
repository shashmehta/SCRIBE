"""Click CLI entry point for CellClassifier."""

import click

from cellclassifier.config import load_dataset_config, load_pipeline_config


@click.group()
def cli():
    """CellClassifier: classify pancreatic cell conditions from scRNA-seq data."""


@cli.command()
@click.option(
    "--config", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Dataset YAML config (e.g. configs/datasets/GSE154778.yaml).",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory for the .h5ad file. Overrides source.base_path default.",
)
def convert(config, output):
    """Convert a GEO dataset to a cellxGene-compliant .h5ad file.

    Reads the dataset YAML config, loads raw files from the source
    (local or Google Drive), runs scanpy QC and preprocessing, annotates
    cellxGene schema 5.0.0 metadata, and writes a processed .h5ad.

    Example:

        python run.py convert --config configs/datasets/GSE154778.yaml --output ./output
    """
    cfg = load_dataset_config(config)
    out_dir = output or "./output"
    click.echo(f"Converting {cfg.id}: {cfg.title}")
    click.echo(f"  Format : {cfg.files[0].format}")
    click.echo(f"  Samples: {len(cfg.samples)}")
    click.echo(f"  Output : {out_dir}")
    raise NotImplementedError("geo.convert_dataset() will be wired in stage 3")


@cli.command()
@click.option(
    "--config", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Pipeline YAML config (e.g. configs/pipeline.yaml).",
)
@click.option(
    "--data",
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
def train(config, data, output):
    """Train a RandomForest classifier on processed scRNA-seq data.

    Loads the .h5ad specified in the pipeline config (or --data override),
    extracts gene expression features, trains a RandomForestClassifier with
    balanced class weights, evaluates on a held-out test set, and saves a
    model artifact (model_artifact.joblib) to the output directory.

    Example:

        python run.py train --config configs/pipeline.yaml
    """
    cfg = load_pipeline_config(config)
    data_path = data or cfg.data
    out_dir = output or cfg.output
    click.echo(f"Training on {data_path} → {out_dir}")
    click.echo(f"  n_estimators={cfg.model.n_estimators}, test_size={cfg.model.test_size}")
    raise NotImplementedError("model.train() will be wired in stage 4")


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
    "--data",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to processed .h5ad. Overrides config value.",
)
def evaluate(config, model_path, data):
    """Evaluate a saved model artifact on processed scRNA-seq data.

    Prints a classification report and confusion matrix to stdout.

    Example:

        python run.py evaluate --config configs/pipeline.yaml \\
            --model ./output/model_artifact.joblib
    """
    cfg = load_pipeline_config(config)
    data_path = data or cfg.data
    click.echo(f"Evaluating {model_path} on {data_path}")
    raise NotImplementedError("model.evaluate() will be wired in stage 4")


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

    Reads UMAP embedding from the .h5ad and feature importances from the model
    artifact, then writes PNG files to <output>/plots/.

    Example:

        python run.py plot --config configs/pipeline.yaml \\
            --model ./output/model_artifact.joblib
    """
    cfg = load_pipeline_config(config)
    out_dir = output or cfg.output
    click.echo(f"Plotting → {out_dir}/plots/")
    raise NotImplementedError("plotting.generate_all_plots() will be wired in stage 4")


@cli.command("run")
@click.option(
    "--config", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Pipeline YAML config.",
)
@click.option(
    "--retrain", is_flag=True, default=False,
    help="Force retraining even if a model artifact already exists.",
)
@click.option(
    "--data",
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
def run_pipeline(config, retrain, data, output):
    """Full pipeline: train → evaluate → plot.

    Combines the train, evaluate, and plot subcommands into one step.
    Equivalent to the original run.py behaviour but driven by a YAML config.

    Example:

        python run.py run --config configs/pipeline.yaml
        python run.py run --config configs/pipeline.yaml --retrain
    """
    cfg = load_pipeline_config(config)
    data_path = data or cfg.data
    out_dir = output or cfg.output
    click.echo(f"Full pipeline: {data_path} → {out_dir} (retrain={retrain})")
    raise NotImplementedError("Full pipeline will be wired in stage 4")


def main():
    cli()
