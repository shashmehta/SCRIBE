# SCRIBE
### Single-Cell RNA Interpretable Biomarker Explorer

A Python machine learning pipeline for identifying biomarkers and therapeutic targets from single-cell RNA sequencing (scRNA-seq) data using explainable machine learning. SCRIBE converts raw GEO datasets, corrects batch effects, trains an interpretable Random Forest classifier, and generates publication-quality plots — all from a single CLI.

## Overview

SCRIBE uses Random Forest classification on gene expression data to:
- **Convert** raw GEO datasets into a standard AnnData format
- **Merge** multiple datasets with batch-aware HVG selection
- **Correct** batch effects via ComBat or Harmony
- **Classify** cells by condition (normal / precancerous / malignant)
- **Identify** top discriminating genes through feature importance analysis
- **Analyze** differential gene expression with volcano plots and LFC charts
- **Visualize** results with UMAP plots and an interactive Marimo explorer

## Datasets

SCRIBE has been validated on three published pancreatic scRNA-seq studies from GEO:

| Dataset | Description | Conditions |
|---------|-------------|------------|
| GSE154778 | Primary vs metastatic PDAC (10+6 samples) | primary, metastatic |
| GSE162708 | Pancreatic neuroendocrine tumor (24,544 cells) | primary_tumor, metastasis, normal |
| GSE165399 | Normal pancreas, IPMN, adenosquamous carcinoma | normal, IPMN, PASC |

These are unified into a 3-class scheme: **normal / precancerous / malignant**.

---

## Installation

### Prerequisites

- **Python 3.10+**
- **Conda** (recommended for environment management)
- **Google Drive for Desktop** — required for data storage (see below)

### Setup

**1. Clone the repository:**
```bash
git clone https://github.com/shashmehta/SCRIBE.git
cd SCRIBE
```

**2. Create and activate conda environment:**
```bash
conda create -n scribe python=3.10
conda activate scribe
```

**3. Install the package:**
```bash
pip install -e .
```
> Run this from the repository root (where `pyproject.toml` is), not from inside the `scribe/` subdirectory.

**4. Set up Google Drive (see next section).**

---

## Google Drive Setup

SCRIBE stores **no data on local disk**. All raw datasets, processed files, model artifacts, and plots live on Google Drive so they sync automatically across machines. The project uses an `output/` symlink pointing to a shared Drive folder.

### Step 1 — Install Google Drive for Desktop

Download and install [Google Drive for Desktop](https://www.google.com/drive/download/). Sign in and let it mount your Drive locally.

| OS | Default mount path |
|---|---|
| macOS | `/Users/YOU/Library/CloudStorage/GoogleDrive-YOUR_EMAIL/` |
| Windows | `G:\My Drive\` |
| Linux | `~/google-drive/` (varies) |

### Step 2 — Create the SCRIBE folder on Drive

Create a folder named `SCRIBE` on Google Drive with the following structure (or get access to the shared folder):

```
SCRIBE/                                    (Google Drive folder)
├── GSE154778/                             Raw GEO files
├── GSE162708/                             Raw GEO files
├── GSE165399/                             Raw GEO files
├── processed/
│   ├── GSE154778_processed.h5ad
│   ├── GSE162708_processed.h5ad
│   ├── GSE165399_processed.h5ad
│   ├── combined_processed.h5ad            (uncorrected, after merge)
│   ├── combined_processed_corrected.h5ad  (ComBat-corrected)
│   ├── combined_processed_harmony.h5ad    (Harmony-corrected)
│   ├── model_artifact.joblib
│   └── app_cache/                         Parquet cache for Marimo app
└── plots/
    ├── log_fold_change/
    ├── volcano/
    ├── feature_importance/
    ├── hk_analysis/
    └── rf_analysis/
```

### Step 3 — Create the `output/` symlink

From the repository root, create a symlink named `output` pointing to your Drive folder:

**macOS:**
```bash
# Find the path: right-click the SCRIBE folder in Finder → Get Info → copy the "Where" path
ln -s "/Users/YOU/Library/CloudStorage/GoogleDrive-YOUR_EMAIL/.shortcut-targets-by-id/FOLDER_ID/SCRIBE" ./output
```

**Linux:**
```bash
ln -s ~/google-drive/path/to/SCRIBE ./output
```

**Windows (PowerShell as Administrator):**
```powershell
New-Item -ItemType SymbolicLink -Path .\output -Target "G:\My Drive\path\to\SCRIBE"
```

**Verify:**
```bash
ls output/
# Should show: GSE154778/  GSE162708/  GSE165399/  processed/  plots/
```

All CLI commands write to `./output/` by default, so data flows to Drive automatically.

---

## Full Pipeline Walkthrough

### Step 1 — Convert raw GEO datasets

```bash
scribe convert --config configs/datasets/GSE154778.yaml
scribe convert --config configs/datasets/GSE162708.yaml
scribe convert --config configs/datasets/GSE165399.yaml
```

Each command reads raw files from `output/GSEXXXXXX/`, normalizes and embeds the data, and writes a per-dataset `.h5ad` to `output/processed/`.

Or convert all at once with `build`:
```bash
scribe build \
    --config configs/datasets/GSE154778.yaml \
    --config configs/datasets/GSE162708.yaml \
    --config configs/datasets/GSE165399.yaml \
    --condition-map configs/condition_map.yaml \
    --output ./output/processed
```

### Step 2 — Merge datasets

```bash
scribe merge \
    --data ./output/processed/GSE154778_processed.h5ad \
    --data ./output/processed/GSE162708_processed.h5ad \
    --data ./output/processed/GSE165399_processed.h5ad \
    --condition-map configs/condition_map.yaml \
    --output ./output/processed
```

Produces `combined_processed.h5ad` with batch-aware HVG selection (3,000 genes), PCA, UMAP, and Leiden clustering.

### Step 3 — Batch correction

**ComBat** (memory-efficient, chunked via Zarr):
```bash
scribe convert-zarr --data ./output/processed/combined_processed.h5ad
scribe correct-zarr --data ./output/processed/combined_processed.zarr \
    --method combat --to-h5ad
# Produces: combined_processed_corrected.h5ad
```

**Harmony** (recommended — reconstructs gene-level expression via inverse-PCA):
```bash
scribe correct-zarr --data ./output/processed/combined_processed.zarr \
    --method harmony \
    --source-h5ad ./output/processed/combined_processed.h5ad \
    --to-h5ad
# Produces: combined_processed_harmony.h5ad
```

### Step 4 — Train and evaluate

```bash
scribe run \
    --config configs/pipeline.yaml \
    --data ./output/processed/combined_processed_harmony.h5ad \
    --output ./output/processed \
    --plots-dir ./output/plots/rf_analysis
```

Trains a balanced Random Forest classifier, evaluates on a held-out test set, computes feature importances, and generates UMAP + LFC plots.

### Step 5 — Differential expression plots

**Log fold change grid** (per-dataset + combined):
```bash
scribe lfc-plot \
    --data ./output/processed/combined_processed_harmony.h5ad \
    --output ./output/plots/log_fold_change \
    --filename lfc_harmony.png
```

**Volcano plots** (auto-detect comparisons):
```bash
scribe volcano \
    --data ./output/processed/combined_processed_harmony.h5ad \
    --output ./output/plots/volcano \
    --filename volcano_harmony.png
```

**Custom volcano comparisons** (e.g. GSE154778 metastatic vs primary):
```bash
# Create a comparisons YAML:
cat > /tmp/comparisons.yaml << 'EOF'
- dataset_filter: GSE154778
  obs_key: _derived_condition
  derive_from: sample
  prefix_map:
    metastatic: metastatic
    primary: primary
  group_a: metastatic
  group_b: primary
  label: "GSE154778 — PDAC"
- dataset_filter: GSE162708
  obs_key: condition
  group_a: malignant
  group_b: normal
  label: "GSE162708 — pNET"
- dataset_filter: GSE165399
  obs_key: condition
  group_a: malignant
  group_b: normal
  label: "GSE165399 — PASC"
- obs_key: condition
  group_a: malignant
  group_b: normal
  label: "Combined"
EOF

scribe volcano \
    --data ./output/processed/combined_processed_harmony.h5ad \
    --comparisons /tmp/comparisons.yaml \
    --output ./output/plots/volcano \
    --filename volcano_harmony_4panel.png
```

**Feature importance grid** (RF per dataset subset):
```bash
scribe feature-grid \
    --data ./output/processed/combined_processed_harmony.h5ad \
    --output ./output/plots/feature_importance
```

**HK Gene PCA comparison** (batch effect before/after correction):
```bash
scribe hk-pca-compare \
    --uncorrected ./output/processed/combined_processed.h5ad \
    --combat      ./output/processed/combined_processed_corrected.h5ad \
    --harmony     ./output/processed/combined_processed_harmony.h5ad \
    --output      ./output/plots
```

### Step 6 — Batch diagnostics (optional)

```bash
# Visualize batch mixing and HK gene stability
scribe batch-check \
    --data ./output/processed/combined_processed.h5ad \
    --output ./output/plots/batch_diagnostics

# HK gene analysis to separate batch from biology
scribe hk-analysis \
    --data ./output/processed/combined_processed.h5ad \
    --output ./output/plots/hk_analysis
```

---

## Marimo Interactive Explorer

`app.py` is a [Marimo](https://marimo.io) reactive notebook for interactively exploring batch correction effects. It compares **Uncorrected**, **ComBat**, and **Harmony** side-by-side.

### What the app shows

| Section | Description |
|---------|-------------|
| Gene Distribution Viewer | KDE plots of any gene's expression across datasets, before and after correction |
| UMAP Viewer | Interactive UMAP colored by dataset, condition, leiden cluster, or cell cycle phase |
| HK Gene PCA | PCA of housekeeping genes — tight clusters indicate effective batch correction |

### Running the app

**1. Build the Parquet cache** (first run only, or when h5ad files change):

The app detects stale cache automatically on startup and rebuilds it. You can also pre-build manually:
```bash
python -c "from scribe import cache; cache.build_cache(force=True)"
```

This reads the three h5ad files and writes columnar Parquet files to `output/processed/app_cache/`. Cache rebuild takes a few minutes but only runs when source files change.

**2. Launch the app:**
```bash
marimo run app.py
```

Opens at `http://localhost:2718` by default. The app loads gene lists and UMAP coordinates in seconds from the Parquet cache; gene expression is loaded on-demand per selection.

**3. Enable Harmony panel:**

The Harmony panel appears automatically if `combined_processed_harmony.h5ad` exists. Run `scribe correct-zarr --method harmony ...` if you haven't already.

> **Note:** The Parquet cache is only used by the Marimo app. All `scribe` CLI commands read from `.h5ad` files directly and are always up to date.

---

## CLI Reference

```
scribe --help
```

| Command | Description |
|---------|-------------|
| `convert` | Convert a raw GEO dataset to `.h5ad` |
| `inspect` | Inspect barcode structure for demultiplexing |
| `merge` | Merge per-dataset `.h5ad` files into a combined file |
| `build` | Convert + merge in one step |
| `batch-check` | Diagnose batch effects (HK expression, distances, mixing score) |
| `batch-subset` | Per-condition UMAP and distribution distances |
| `batch-correct` | In-memory ComBat / Harmony / Scanorama correction |
| `convert-zarr` | Convert `.h5ad` to Zarr for memory-efficient processing |
| `correct-zarr` | Memory-efficient batch correction via Zarr chunked pipeline |
| `zarr-to-h5ad` | Convert Zarr store back to `.h5ad` |
| `hk-analysis` | Housekeeping gene analysis to separate batch from biology |
| `hk-pca-compare` | Side-by-side HK Gene PCA: uncorrected / ComBat / Harmony |
| `train` | Train a Random Forest classifier |
| `evaluate` | Evaluate a saved model artifact |
| `plot` | Generate UMAP and feature importance plots |
| `run` | Full ML pipeline: train → evaluate → plot |
| `lfc-plot` | 2×N log fold change bar chart grid |
| `volcano` | Volcano plot grid (auto-detect or custom comparisons YAML) |
| `feature-grid` | RF feature importance grid, one panel per dataset |
| `dataset-umap` | Side-by-side UMAP for a single dataset |
| `monitor` | Real-time system resource monitoring |

Get help for any command:
```bash
scribe <command> --help
```

---

## Project Structure

```
SCRIBE/
├── pyproject.toml           # Package config and dependencies
├── run.py                   # Convenience CLI entry point
├── app.py                   # Marimo interactive batch correction explorer
├── conftest.py              # pytest root configuration
├── README.md
├── CLAUDE.md
├── scribe/                  # Main Python package
│   ├── cli.py               # All CLI sub-commands
│   ├── config.py            # YAML config dataclasses
│   ├── data.py              # Data loading, merging, preprocessing
│   ├── geo.py               # GEO raw data loaders (CSV DGE, 10x MTX, TAR)
│   ├── model.py             # RF training, evaluation, artifacts
│   ├── analysis.py          # Differential expression (LFC, Wilcoxon DE)
│   ├── plotting.py          # UMAP, LFC, volcano, feature importance, HK PCA
│   ├── batch.py             # Batch effect detection and correction
│   ├── zarr_utils.py        # Memory-efficient Zarr chunked I/O
│   ├── cache.py             # Parquet cache for Marimo app
│   ├── monitor.py           # System resource monitoring
│   └── tests/               # Unit tests (34 tests, synthetic fixtures)
├── tests/                   # Integration tests
├── configs/
│   ├── pipeline.yaml        # ML pipeline config
│   ├── condition_map.yaml   # Condition label remapping
│   └── datasets/
│       ├── GSE154778.yaml
│       ├── GSE162708.yaml
│       └── GSE165399.yaml
└── output/ → Google Drive   # Symlink — all data lives on Drive
    ├── GSE*/                # Raw GEO dataset files
    ├── processed/           # h5ad files, model artifacts, Zarr stores, app cache
    └── plots/               # All generated plots
```

---

## Pipeline Architecture

```
GEO Datasets (3 studies)
        │
        ▼  scribe convert  (or  scribe build)
  Per-dataset .h5ad
  (QC filtered, log1p normalized, UMAP)
        │
        ▼  scribe merge
  combined_processed.h5ad
  (3,000 batch-aware HVGs, PCA, UMAP, Leiden)
        │
        ▼  scribe convert-zarr + correct-zarr
  combined_processed_harmony.h5ad
  (Harmony-corrected, inverse-PCA gene reconstruction)
        │
        ├──▶  scribe run          → RF classifier + UMAP plots
        ├──▶  scribe lfc-plot     → LFC bar chart grid
        ├──▶  scribe volcano      → Volcano plot grid
        ├──▶  scribe feature-grid → Feature importance per dataset
        └──▶  scribe hk-pca-compare → HK Gene PCA before/after correction
```

### Data contract

All `.h5ad` files follow a consistent contract:
- **`adata.X`** — log1p-normalized expression (non-negative, range ~0–8.7)
- **`adata.layers["X_norm"]`** — z-scored expression (used for PCA/RF features)
- **`adata.obs`** — cell metadata: `dataset`, `condition`, `sample`, `leiden`
- **`adata.obsm["X_umap"]`** — 2D UMAP coordinates
- **`adata.var`** — gene metadata: `gene_mean`, `gene_std` (for inverse-PCA)

---

## Running Tests

```bash
# Unit tests (synthetic data, no real files required, ~25 seconds)
pytest scribe/tests/ -v

# Integration tests (requires output/ data)
pytest tests/ -v
```

---

## Scientific Background

### What is PDAC?

Pancreatic Ductal Adenocarcinoma (PDAC) is one of the most lethal cancers, with a 5-year survival rate under 15%. It is difficult to treat partly because it is hard to identify cancerous cells early and distinguish them from surrounding healthy tissue.

### What is scRNA-seq?

Single-cell RNA sequencing measures which genes are active in each individual cell. The output is a matrix where rows are cells and columns are genes. By training a classifier on this matrix, we can learn which gene patterns predict whether a cell is normal or cancerous.

### Why a Random Forest?

Random Forests handle high-dimensional data (thousands of genes) well and provide **feature importances** — a score for each gene showing how much it helped the model. Top-ranked genes are candidate **biomarkers**: genes that reliably distinguish tumor from normal cells and could serve as diagnostic or therapeutic targets.

### Batch effects and correction

Cells sequenced in different experiments (batches) show systematic technical differences unrelated to biology. SCRIBE uses two complementary approaches:

- **ComBat** — linear location-scale correction, memory-efficient via chunked Zarr pipeline
- **Harmony** — embedding-space correction; SCRIBE reconstructs gene-level expression via inverse-PCA so downstream DE analysis runs on corrected data

---

## Troubleshooting

**Out of memory during batch correction**
Use the Zarr chunked pipeline: `scribe convert-zarr` then `scribe correct-zarr`. Peak memory stays around 900 MB regardless of dataset size.

**Marimo app loads slowly**
The app uses a Parquet cache for fast loading. If the cache is missing, it rebuilds automatically on first launch (a few minutes). Subsequent loads are near-instant.

**Barcode suffixes show as `null` in the YAML**
Run `scribe inspect --config configs/datasets/GSE154778.yaml` to see the actual suffix distribution and fix the config.

**`ValueError: Column 'condition' not found`**
Run `scribe convert` first to produce a processed `.h5ad` before running `scribe run`.

**Harmony segfault on macOS**
Set `OMP_NUM_THREADS=1` before running: `OMP_NUM_THREADS=1 scribe correct-zarr --method harmony ...`

**Volcano plot raises `KeyError: group not found`**
The comparison groups must both exist in the specified `obs_key` column of the subset. Use `scribe inspect` or check `adata.obs[condition_col].value_counts()` to confirm label names.
