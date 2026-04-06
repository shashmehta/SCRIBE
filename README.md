# SCRIBE
### Single-Cell RNA Interpretable Biomarker Explorer

A Python machine learning pipeline for identifying biomarkers and therapeutic targets from single-cell RNA sequencing (scRNA-seq) data using explainable machine learning methods. SCRIBE loads raw data from public databases, cleans it, trains interpretable classifiers to distinguish cell conditions, and identifies the top genes driving each classification.

## Overview

SCRIBE uses Random Forest classification on gene expression data to:
- **Convert** raw GEO datasets into a standard format
- **Classify** cells into biological conditions (e.g. normal, precancerous, malignant)
- **Identify** top discriminating genes through feature importance analysis
- **Analyze** differential gene expression between conditions
- **Detect and correct** batch effects across datasets
- **Visualize** results with UMAP plots, feature importance charts, and an interactive explorer

## Datasets

SCRIBE has been validated on three published pancreatic scRNA-seq studies from GEO:

| Dataset | Description | Conditions |
|---------|-------------|------------|
| GSE154778 | Primary vs metastatic PDAC (10+6 samples) | primary, metastatic |
| GSE162708 | Pancreatic neuroendocrine tumor (24,544 cells) | primary_tumor, metastasis, normal |
| GSE165399 | Normal pancreas, IPMN, adenosquamous carcinoma | normal, IPMN, PASC |

These are unified into a 3-class scheme: **normal / precancerous / malignant**.

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended for environment management)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/SCRIBE.git
   cd SCRIBE
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n NAME python=3.10
   conda activate NAME
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

4. **Set up Google Drive data folder:**

   SCRIBE keeps **no data on local disk**. All raw datasets, processed files, model artifacts, and plots live on Google Drive, shared as "anyone with link" so collaborators can access them.

   The project's `output/` directory is a symlink to your Google Drive folder. To set it up:

   ```bash
   # macOS (Google Drive for Desktop):
   ln -s "/Users/YOU/Library/CloudStorage/GoogleDrive-YOUR_EMAIL/.shortcut-targets-by-id/FOLDER_ID/SCRIBE" ./output

   # Linux:
   ln -s ~/google-drive/SCRIBE ./output
   ```

   > **Google Drive folder structure:**
   > ```
   > SCRIBE/                          (shared Google Drive folder)
   > ├── GSE154778/                   Raw GEO dataset files
   > ├── GSE162708/                   Raw GEO dataset files
   > ├── GSE165399/                   Raw GEO dataset files
   > ├── processed/                   Processed h5ad files, model artifacts, zarr stores
   > │   ├── GSE154778_processed.h5ad
   > │   ├── GSE162708_processed.h5ad
   > │   ├── GSE165399_processed.h5ad
   > │   ├── combined_processed.h5ad
   > │   ├── combined_processed_corrected.h5ad
   > │   ├── combined_processed.zarr/
   > │   ├── combined_processed_corrected.zarr/
   > │   ├── model_artifact.joblib
   > │   └── app_cache/              Parquet cache for Marimo app
   > └── plots/                      All generated plots
   >     ├── malignant_uncorrected_vs_corrected.png
   >     ├── normal_uncorrected_vs_corrected.png
   >     ├── hk_pca_uncorrected_vs_corrected.png
   >     └── hk_analysis/            Housekeeping gene analysis plots
   > ```

   The pipeline reads raw data from `output/GSE*/`, writes processed data to `output/processed/`, and saves plots to `output/plots/`. Since `output/` points to Google Drive, everything is automatically synced and available to collaborators.

## Usage

After installation, use the `scribe` CLI command (or `python run.py` as an alternative).

### Step 1 — Convert GEO datasets to `.h5ad`

```bash
scribe convert --config configs/datasets/GSE154778.yaml --output ./output/processed
scribe convert --config configs/datasets/GSE162708.yaml --output ./output/processed
scribe convert --config configs/datasets/GSE165399.yaml --output ./output/processed
```

### Step 2 — Merge datasets

```bash
scribe merge \
    --data ./output/GSE154778_processed.h5ad \
    --data ./output/GSE162708_processed.h5ad \
    --data ./output/GSE165399_processed.h5ad \
    --condition-map configs/condition_map.yaml \
    --output ./output/processed
```

### Step 3 — Batch correction

```bash
# Memory-efficient ComBat via Zarr chunked pipeline:
scribe convert-zarr --input ./output/processed/combined_processed.h5ad --output ./output/processed/
scribe correct-zarr --input ./output/processed/combined_processed.zarr --output ./output/processed/ --to-h5ad
```

### Step 4 — Train and evaluate

```bash
scribe run --config configs/pipeline.yaml \
    --data ./output/processed/combined_processed.h5ad \
    --output ./output/processed
```

### Step 5 — Interactive explorer

```bash
marimo run app.py --include-code
```

Opens a browser-based app for exploring gene distributions before/after batch correction and browsing analysis plots.

### Other commands

```bash
scribe inspect --config configs/datasets/GSE154778.yaml  # Inspect barcode structure
scribe batch-check --data ./output/processed/combined_processed.h5ad  # Diagnose batch effects
scribe hk-analysis --data ./output/processed/combined_processed.h5ad --output ./output/plots/hk_analysis  # HK gene analysis
scribe monitor  # Real-time system resource monitoring
```

## Project Structure

```
SCRIBE/
├── pyproject.toml           # Package config and dependencies
├── run.py                   # Convenience CLI entry point
├── app.py                   # Marimo interactive explorer
├── conftest.py              # pytest configuration
├── README.md
├── CLAUDE.md
├── .gitignore
├── scribe/                  # Main Python package
│   ├── __init__.py
│   ├── cli.py               # CLI sub-commands
│   ├── config.py            # YAML config dataclasses
│   ├── data.py              # Data loading, merging, preprocessing
│   ├── geo.py               # GEO raw data loaders
│   ├── model.py             # RF training, evaluation, artifacts
│   ├── analysis.py          # Differential expression
│   ├── plotting.py          # UMAP, feature importance, diagnostic plots
│   ├── batch.py             # Batch effect detection and correction
│   ├── monitor.py           # System resource monitoring
│   ├── zarr_utils.py        # Memory-efficient Zarr I/O
│   └── cache.py             # Parquet cache for interactive app
├── tests/                   # Test suite
├── configs/                 # Pipeline and dataset YAML configs
│   ├── pipeline.yaml
│   ├── condition_map.yaml
│   └── datasets/
└── output/ → Google Drive   # Symlink — all data lives on Drive
    ├── GSE*/                # Raw GEO dataset files
    ├── processed/           # Processed h5ad, models, zarr stores
    └── plots/               # Analysis plots
```

## How It Works

### Pipeline Architecture

```
GEO Datasets (3 studies)
        │
        ▼  scribe convert
  Per-dataset .h5ad files
  (normalized, UMAP embedded)
        │
        ▼  scribe merge
  Combined .h5ad (3-class labels:
  normal / precancerous / malignant)
        │
        ▼  scribe correct-zarr
  Batch-corrected .h5ad
  (ComBat via memory-efficient Zarr pipeline)
        │
        ▼  scribe run
  3-class Random Forest model
        │
        ├── Feature importances (top biomarker genes)
        ├── Differential expression (normal vs malignant)
        └── UMAP plots (by condition, dataset, top genes)
```

## Scientific Background

### What is PDAC?

Pancreatic Ductal Adenocarcinoma (PDAC) is one of the most lethal cancers, with a 5-year survival rate under 15%. It is difficult to treat partly because it is hard to identify cancerous cells early and distinguish them from surrounding healthy tissue.

### What is scRNA-seq?

Single-cell RNA sequencing measures which genes are active in each individual cell. The output is a table where rows are cells and columns are genes. By training a classifier on this table, we can learn which gene patterns predict whether a cell is normal or cancerous.

### Why a Random Forest?

Random Forests handle high-dimensional data (thousands of genes) well and provide **feature importances** — scores for each gene showing how much it helped the model. These top genes are candidate **biomarkers**: genes that reliably distinguish tumor from normal cells and could serve as diagnostic targets or drug targets.

## Troubleshooting

**Out of memory during batch correction**
Use the Zarr chunked pipeline: `scribe convert-zarr` then `scribe correct-zarr`. This processes data in constant memory (~900MB peak).

**Barcode suffixes show as `null` in the YAML**
Run `scribe inspect --config configs/datasets/GSE154778.yaml` to see the actual suffix distribution.

**`ValueError: Column 'condition' not found`**
Run `scribe convert` first to produce a processed `.h5ad` before running `scribe run`.
