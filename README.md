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

- **Python 3.10+**
- **Conda** (recommended for environment management)
- **Google Drive for Desktop** — required for data storage (see Step 4)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shashmehta/CellClassifier.git
   cd CellClassifier
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n scribe python=3.10
   conda activate scribe
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```
   > **Note:** Run this from the repository root (where `pyproject.toml` is), not from inside the `scribe/` subdirectory.

4. **Set up Google Drive for Desktop (required):**

   SCRIBE stores **no data on local disk**. All raw datasets, processed files, model artifacts, and plots are stored on Google Drive so they sync automatically across machines and collaborators.

   The project uses a local `output/` symlink that points to a shared Google Drive folder. Every CLI command writes to `output/` by default, so once the symlink is configured, all data flows to Drive automatically.

   #### Step 4a — Install Google Drive for Desktop

   Download and install [Google Drive for Desktop](https://www.google.com/drive/download/). Sign in with your Google account. Once installed, Drive mounts a local folder that stays in sync with your cloud storage.

   The mount location depends on your OS:
   | OS | Default mount path |
   |---|---|
   | **macOS** | `/Users/YOU/Library/CloudStorage/GoogleDrive-YOUR_EMAIL/` |
   | **Windows** | `G:\My Drive\` (or whichever drive letter is assigned) |
   | **Linux** | `~/google-drive/` (varies by setup) |

   #### Step 4b — Locate or create the SCRIBE folder

   The shared SCRIBE data folder lives on Google Drive. If you've been given access to an existing shared folder, find it in your Google Drive and note its local path. If starting fresh, create a folder on Google Drive named `SCRIBE` with this structure:

   ```
   SCRIBE/                          (Google Drive folder)
   ├── GSE154778/                   Raw GEO dataset files
   ├── GSE162708/                   Raw GEO dataset files
   ├── GSE165399/                   Raw GEO dataset files
   ├── processed/                   Processed h5ad files, model artifacts, zarr stores
   │   ├── GSE154778_processed.h5ad
   │   ├── GSE162708_processed.h5ad
   │   ├── GSE165399_processed.h5ad
   │   ├── combined_processed.h5ad
   │   ├── combined_processed_corrected.h5ad
   │   ├── combined_processed.zarr/
   │   ├── combined_processed_corrected.zarr/
   │   ├── model_artifact.joblib
   │   └── app_cache/              Parquet cache for Marimo app
   └── plots/                      All generated plots
       ├── malignant_uncorrected_vs_corrected.png
       ├── normal_uncorrected_vs_corrected.png
       ├── hk_pca_uncorrected_vs_corrected.png
       └── hk_analysis/            Housekeeping gene analysis plots
   ```

   #### Step 4c — Create the `output/` symlink

   From the repository root, create a symlink named `output` pointing to your Google Drive SCRIBE folder:

   **macOS:**
   ```bash
   # Replace YOUR_EMAIL and FOLDER_PATH with your actual values.
   # To find the path: right-click the SCRIBE folder in Finder → "Get Info" → copy the "Where" path.
   ln -s "/Users/YOU/Library/CloudStorage/GoogleDrive-YOUR_EMAIL/.shortcut-targets-by-id/FOLDER_ID/FOLDER_PATH/SCRIBE" ./output
   ```

   **Linux:**
   ```bash
   ln -s ~/google-drive/path/to/SCRIBE ./output
   ```

   **Windows (PowerShell as admin):**
   ```powershell
   New-Item -ItemType SymbolicLink -Path .\output -Target "G:\My Drive\path\to\SCRIBE"
   ```

   #### Verify the symlink

   ```bash
   ls output/
   # Should show: GSE154778/  GSE162708/  GSE165399/  processed/  plots/
   ```

   > **How it works:** All CLI commands default to writing under `./output/` (e.g. `--output ./output/processed`). Since `output/` is a symlink to Google Drive, processed data and plots are automatically synced to the cloud. No additional configuration is needed — just run the pipeline and everything lands on Drive.

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
