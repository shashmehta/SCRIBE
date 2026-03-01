# Plan: Config-Based CLI with GEO Data Conversion Subcommand

## Goal

Refactor the project into a unified, config-driven CLI that:
1. Migrates `run.py` from ad-hoc argparse flags to YAML-config-driven subcommands using **Click**.
2. Extracts the GEO notebook (`geo_data_processing.ipynb`) into a proper `cellclassifier/geo.py` module.
3. Adds a `convert` subcommand that uses per-dataset YAML configs to produce cellxGene-compliant `.h5ad` files.
4. Keeps all existing `data.py / model.py / analysis.py / plotting.py` modules intact (no breaking changes).

---

## Final Directory Layout

```
cellclassifier/
├── __init__.py
├── config.py       # NEW — YAML config loading & dataclasses
├── geo.py          # NEW — GEO dataset loading (extracted from notebook)
├── cli.py          # NEW — subcommand entry point (replaces run.py logic)
├── data.py         # unchanged
├── model.py        # unchanged
├── analysis.py     # unchanged
└── plotting.py     # unchanged

configs/
├── datasets/
│   ├── GSE154778.yaml   # Primary vs Metastatic PDAC
│   ├── GSE162708.yaml   # Metastatic pNET
│   └── GSE165399.yaml   # Normal / IPMN / PASC
└── pipeline.yaml        # Training, evaluation, plotting parameters

run.py              # Thin shim → delegates to cellclassifier.cli.main()
plans/
└── config_based_cli.md  # This file
```

---

## Step 1 — `cellclassifier/config.py`

Defines dataclasses for each config type and loaders that validate required fields.

### `DatasetConfig` (used by `convert` subcommand)

```yaml
# configs/datasets/GSE154778.yaml
id: GSE154778
title: "Primary vs Metastatic PDAC"
description: "10 primary tumors + 6 metastatic PDAC samples from GSE154778"

source:
  type: local            # or "gdrive"
  base_path: "/path/to/Datasets"   # for local; gdrive_folder_id for gdrive

files:
  - name: "GSE154778_dgeMtx.csv.gz"
    format: csv_dge          # choices: csv_dge | 10x_mtx | tar_txt_dge | tar_10x
    relative_path: "GSE154778/GSE154778_dgeMtx.csv.gz"

preprocessing:
  min_genes: 200
  min_cells: 3
  mt_pct_threshold: 20
  n_top_genes: 2000
  n_pcs: 30
  leiden_resolution: 0.5

cellxgene:
  assay_ontology_term_id: "EFO:0009922"      # 10x 3' v2
  disease_ontology_term_id: "MONDO:0006047"  # PDAC
  tissue_ontology_term_id: "UBERON:0001264"  # pancreas
  organism_ontology_term_id: "NCBITaxon:9606"

samples:                     # optional per-sample metadata
  - barcode_suffix: "1"
    id: "Primary_T1"
    condition: primary
    tissue_ontology_term_id: "UBERON:0001264"
  - barcode_suffix: "2"
    id: "Metastatic_T1"
    condition: metastatic
    tissue_ontology_term_id: "UBERON:0001264"
```

### `PipelineConfig` (used by `train` / `evaluate` / `plot` / `run` subcommands)

```yaml
# configs/pipeline.yaml
data: "./output/GSE154778_processed.h5ad"
output: "./output"
condition_col: "CONDITION"

model:
  n_estimators: 100
  class_weight: "balanced"
  random_state: 42
  test_size: 0.2

analysis:
  top_n_genes: 20

plots:
  umap_columns: ["celltype3", "CONDITION"]
  umap_genes: []             # empty = top 2 feature-importance genes
```

### Implementation in `config.py`

```python
from dataclasses import dataclass, field
from typing import Literal
import yaml

@dataclass
class SourceConfig: ...
@dataclass
class FileConfig: ...
@dataclass
class PreprocessingConfig: ...
@dataclass
class CellxGeneConfig: ...
@dataclass
class SampleConfig: ...
@dataclass
class DatasetConfig: ...

@dataclass
class ModelConfig: ...
@dataclass
class AnalysisConfig: ...
@dataclass
class PlotsConfig: ...
@dataclass
class PipelineConfig: ...

def load_dataset_config(path: str) -> DatasetConfig: ...
def load_pipeline_config(path: str) -> PipelineConfig: ...
```

---

## Step 2 — `cellclassifier/geo.py`

Extracts all notebook logic into reusable, tested functions.

### Functions

| Function | Source cell | Description |
|---|---|---|
| `load_csv_dge(path)` | Cell 5–6 | Read gzipped CSV DGE matrix, return AnnData (cells × genes) |
| `load_10x_mtx(src_dir, file_map)` | Cell 11 | Symlink GSE-prefixed files to temp dir, call `sc.read_10x_mtx` |
| `load_tar_txt_dge(tar_path, extract_dir)` | Cell 15–16 | Extract TAR, read each TXT DGE file, concat into AnnData |
| `load_tar_10x(tar_path, extract_dir)` | Cell 7–8 | Extract TAR of per-sample 10x dirs, load each, concat |
| `assign_sample_metadata(adata, sample_configs)` | Cell 9, 12–13 | Map barcode suffixes / GSM IDs to sample metadata |
| `preprocess_adata(adata, config)` | Cell 3 (preprocess_adata) | Full scanpy QC pipeline driven by `PreprocessingConfig` |
| `annotate_cellxgene_metadata(adata, config)` | Cell 4 | Add required cellxGene obs/var/uns fields from `CellxGeneConfig` |
| `validate_cellxgene(adata, name)` | Cell 5 (validate fn) | Check all required cellxGene schema 5.0.0 fields |
| `convert_dataset(config, output_dir)` | orchestrator | Load → preprocess → annotate → validate → write `.h5ad` |

`convert_dataset` is the main entry point called by the `convert` subcommand:

```python
def convert_dataset(config: DatasetConfig, output_dir: str) -> str:
    """Load, preprocess, annotate, and save a GEO dataset as h5ad."""
    adata = _load_by_format(config)       # dispatches to load_* functions
    adata = assign_sample_metadata(adata, config.samples)
    adata = preprocess_adata(adata, config.preprocessing)
    adata = annotate_cellxgene_metadata(adata, config.cellxgene)
    ok = validate_cellxgene(adata, config.id)
    out_path = os.path.join(output_dir, f"{config.id}_processed.h5ad")
    adata.write_h5ad(out_path)
    return out_path
```

---

## Step 3 — `cellclassifier/cli.py`

Single module implementing all subcommands via **Click** groups and commands.

Click is preferred over argparse because:
- Subcommands are defined as decorated functions — no boilerplate dispatcher
- Built-in `--help` at every level, type coercion, and error messages
- `click.Path` validates file/dir existence at parse time (before any code runs)
- Easy to test via `CliRunner`

### Subcommand interface

```
python run.py <subcommand> [options]

Subcommands:
  convert   Convert a GEO dataset to cellxGene-compliant .h5ad
  train     Train a RandomForest classifier from processed data
  evaluate  Evaluate an existing model artifact on data
  plot      Generate UMAP and feature-importance plots
  run       Full pipeline: train → evaluate → plot (replaces old run.py)
```

#### `convert`
```
python run.py convert --config configs/datasets/GSE154778.yaml \
                      --output ./output
```
- Loads `DatasetConfig` from YAML
- Calls `geo.convert_dataset(config, output_dir)`

#### `train`
```
python run.py train --config configs/pipeline.yaml
```
- Loads `PipelineConfig` from YAML
- Runs `data.load_adata` → `data.extract_features_and_labels` → `data.split_data` → `model.train` → `model.evaluate` → `model.save_artifact`

#### `evaluate`
```
python run.py evaluate --config configs/pipeline.yaml --model ./output/model_artifact.joblib
```
- Loads data from config, loads model from path, calls `model.evaluate`

#### `plot`
```
python run.py plot --config configs/pipeline.yaml --model ./output/model_artifact.joblib
```
- Loads data, loads model, calls `plotting.generate_all_plots`

#### `run` (full pipeline, replaces current run.py behavior)
```
python run.py run --config configs/pipeline.yaml [--retrain]
```
- Equivalent to old `python run.py --data ... --output ...` but config-driven

### CLI structure in `cli.py`

Every subcommand accepts `--config` for its YAML. Individual flags can **override** YAML values at the command line (e.g., `--output ./custom_out`), making configs composable.

```python
import click
from cellclassifier.config import load_dataset_config, load_pipeline_config
from cellclassifier import geo, data, model, analysis, plotting

@click.group()
def cli():
    """CellClassifier: classify cell conditions from scRNA-seq data."""

@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Dataset YAML config.")
@click.option("--output", default=None, type=click.Path(), help="Output directory (overrides config).")
def convert(config, output):
    """Convert a GEO dataset to a cellxGene-compliant .h5ad file."""
    cfg = load_dataset_config(config)
    out_dir = output or "./output"
    geo.convert_dataset(cfg, out_dir)

@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Pipeline YAML config.")
@click.option("--data", "data_path", default=None, type=click.Path(), help="Override data path.")
@click.option("--output", default=None, type=click.Path(), help="Override output directory.")
def train(config, data_path, output):
    """Train a RandomForest classifier from processed h5ad data."""
    cfg = load_pipeline_config(config)
    # data_path / output override cfg values if provided
    ...

@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
def evaluate(config, model_path):
    """Evaluate an existing model artifact on data."""
    ...

@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
def plot(config, model_path):
    """Generate UMAP and feature-importance plots."""
    ...

@cli.command("run")
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--retrain", is_flag=True, default=False, help="Force retrain even if model exists.")
def run_pipeline(config, retrain):
    """Full pipeline: train → evaluate → plot."""
    ...

def main():
    cli()
```

---

## Step 4 — Update `run.py`

Replace current `run.py` content with a thin shim:

```python
"""Entry point — delegates to cellclassifier.cli."""
from cellclassifier.cli import main

if __name__ == "__main__":
    main()
```

This preserves `python run.py` as the invocation while all logic lives in the package.

Click also supports a `scripts` entry point in `setup.py` / `pyproject.toml` if the package is installed, enabling a bare `cellclassifier` command.

---

## Step 5 — YAML config files

Create three dataset configs and one pipeline config.

### `configs/datasets/GSE154778.yaml`
- Format: `csv_dge` (primary strategy)
- Fallback: `tar_10x` if CSV lacks per-sample metadata
- Conditions: `primary` (10 samples) / `metastatic` (6 samples)
- Disease: `MONDO:0006047` (PDAC)

### `configs/datasets/GSE162708.yaml`
- Format: `10x_mtx` (3 standard 10x files with GSE prefix → symlink temp dir)
- 5 samples with different tissues: pancreas, liver, blood
- Per-sample `tissue_ontology_term_id` overrides (pancreas vs liver vs blood)
- Disease: `MONDO:0006130` (pNET)

### `configs/datasets/GSE165399.yaml`
- Format: `tar_txt_dge` (TAR of per-sample TXT DGE matrices)
- Conditions: `normal` / `IPMN` / `PASC`
- Per-sample `disease_ontology_term_id` (varies by GSM)

### `configs/pipeline.yaml`
- Default values matching current `run.py` defaults
- References output h5ad paths from dataset conversion

---

## Implementation Order

1. `cellclassifier/config.py` — dataclasses + YAML loaders (no external deps beyond PyYAML)
2. `configs/datasets/*.yaml` + `configs/pipeline.yaml` — write all four configs
3. `cellclassifier/geo.py` — extract notebook logic into module functions
4. `cellclassifier/cli.py` — wire subcommands; `convert` calls `geo`, others call existing modules
5. `run.py` — replace with thin shim

---

## Dependencies

Add `pyyaml` and `click` to `requirements.txt`. All other packages (`scanpy`, `anndata`, `scipy`, `pandas`, etc.) are already listed.

---

## What Is NOT Changed

- `cellclassifier/data.py` — no changes
- `cellclassifier/model.py` — no changes
- `cellclassifier/analysis.py` — no changes
- `cellclassifier/plotting.py` — no changes
- `geo_data_processing.ipynb` — kept as-is (notebook becomes a reference artifact)
