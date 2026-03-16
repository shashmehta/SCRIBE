# SCRIBE
### Single-Cell RNA Interpretable Biomarker Explorer

A Python machine learning pipeline for identifying biomarkers and therapeutic targets from single-cell RNA sequencing (scRNA-seq) data using explainable machine learning methods. SCRIBE loads raw data from public databases, cleans it, trains interpretable classifiers to distinguish cell conditions, and identifies the top genes driving each classification.

## Overview

SCRIBE uses Random Forest classification on gene expression data to:
- **Convert** raw GEO datasets into a standard format
- **Classify** cells into biological conditions (e.g. normal, precancerous, malignant)
- **Identify** top discriminating genes through feature importance analysis
- **Analyze** differential gene expression between conditions
- **Visualize** results with UMAP plots and feature importance charts
- **Combine** multiple datasets for a unified cross-study model

The pipeline is designed for researchers working with single-cell RNA-seq data in `.h5ad` (AnnData) format.

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

- Python 3.12
- Conda (recommended for environment management)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/SCRIBE.git
   cd SCRIBE
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n scribe python=3.12
   conda activate scribe
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

SCRIBE is run through `run.py` using sub-commands (similar to how `git commit` and `git push` are sub-commands of `git`).

### Step 1 — Inspect raw data (optional but recommended)

Before converting a dataset, you can inspect its barcode structure to confirm sample demultiplexing will work:

```bash
python run.py inspect --config configs/datasets/GSE154778.yaml
```

This prints the barcode suffix distribution so you can fill in the `barcode_suffix` values in the YAML.

### Step 2 — Convert GEO datasets to `.h5ad`

```bash
python run.py convert --config configs/datasets/GSE154778.yaml --output ./output
python run.py convert --config configs/datasets/GSE162708.yaml --output ./output
python run.py convert --config configs/datasets/GSE165399.yaml --output ./output
```

Each produces a processed `.h5ad` file with UMAP embeddings and cellxGene-compliant metadata.

### Step 3 — Run per-dataset analysis (optional)

Train and evaluate a classifier on each dataset individually:

```bash
python run.py run --config configs/pipeline.yaml \
    --data ./output/GSE154778_processed.h5ad \
    --output ./output/GSE154778
```

### Step 4 — Merge datasets for combined training

```bash
python run.py merge \
    --data ./output/GSE154778_processed.h5ad \
    --data ./output/GSE162708_processed.h5ad \
    --data ./output/GSE165399_processed.h5ad \
    --condition-map configs/condition_map.yaml \
    --output ./output/combined
```

This remaps all condition labels to the unified 3-class scheme and produces `combined_processed.h5ad`.

### Step 5 — Train 3-class combined model

```bash
python run.py run --config configs/pipeline.yaml \
    --data ./output/combined/combined_processed.h5ad \
    --output ./output/combined
```

Produces a model that classifies cells as **normal**, **precancerous**, or **malignant** across all three datasets.

### Other commands

```bash
# Evaluate a saved model on new data
python run.py evaluate --config configs/pipeline.yaml \
    --model ./output/model_artifact.joblib \
    --data ./output/new_data.h5ad

# Generate plots from a saved model
python run.py plot --config configs/pipeline.yaml \
    --model ./output/model_artifact.joblib \
    --output ./output

# Force retrain even if a model already exists
python run.py run --config configs/pipeline.yaml --retrain
```

## Output Structure

```
output/
├── GSE154778_processed.h5ad      # Per-dataset converted file
├── GSE162708_processed.h5ad
├── GSE165399_processed.h5ad
├── combined/
│   ├── combined_processed.h5ad   # Merged 3-dataset file
│   ├── model_artifact.joblib     # Trained 3-class RF model
│   └── plots/
│       ├── umap_condition.png    # UMAP colored by condition
│       ├── umap_dataset.png      # UMAP colored by dataset source
│       └── feature_importances.png
└── GSE154778/                    # Per-dataset analysis output
    ├── model_artifact.joblib
    └── plots/
```

## Project Structure

```
SCRIBE/
├── cellclassifier/           # Main Python package
│   ├── __init__.py
│   ├── cli.py               # CLI sub-commands (convert, inspect, merge, run, ...)
│   ├── config.py            # YAML config loading into dataclasses
│   ├── data.py              # Data loading, merging, and ML feature extraction
│   ├── geo.py               # GEO raw data loading and preprocessing
│   ├── model.py             # RF training, evaluation, and artifact save/load
│   ├── analysis.py          # Differential expression analysis
│   └── plotting.py          # UMAP and feature importance plots
├── configs/
│   ├── pipeline.yaml        # ML pipeline settings (RF hyperparameters, etc.)
│   ├── condition_map.yaml   # Maps per-dataset conditions to normal/precancerous/malignant
│   └── datasets/
│       ├── GSE154778.yaml
│       ├── GSE162708.yaml
│       └── GSE165399.yaml
├── tests/                   # Test suite (50 tests)
├── run.py                   # Entry point
├── requirements.txt
└── docs/
    └── CODEBASE_GUIDE.md    # Detailed technical guide
```

## How It Works

### Pipeline Architecture

```
GEO Datasets (3 studies)
        │
        ▼  python run.py convert
  Per-dataset .h5ad files
  (normalized, UMAP embedded)
        │
        ▼  python run.py merge
  Combined .h5ad (3-class labels:
  normal / precancerous / malignant)
        │
        ▼  python run.py run
  3-class Random Forest model
        │
        ├── Feature importances (top biomarker genes)
        ├── Differential expression (normal vs malignant)
        └── UMAP plots (by condition, dataset, top genes)
```

### The 3-Class Condition Scheme

Conditions from the three datasets are unified by `configs/condition_map.yaml`:

| Original label | Dataset | Unified class |
|---------------|---------|---------------|
| normal | GSE162708, GSE165399 | normal |
| primary, primary_tumor | GSE154778, GSE162708 | malignant |
| metastatic, metastasis | GSE154778, GSE162708 | malignant |
| PASC | GSE165399 | malignant |
| IPMN | GSE165399 | precancerous |

### Key Dependencies

- **scanpy** — Single-cell analysis toolkit (preprocessing, UMAP, clustering)
- **anndata** — AnnData format for single-cell data
- **scikit-learn** — Random Forest classifier
- **matplotlib** — Plotting
- **pandas / numpy** — Data manipulation
- **click** — CLI framework
- **pyyaml** — YAML config loading

## Scientific Background

### What is PDAC?

Pancreatic Ductal Adenocarcinoma (PDAC) is one of the most lethal cancers, with a 5-year survival rate under 15%. It is difficult to treat partly because it is hard to identify cancerous cells early and distinguish them from surrounding healthy tissue.

### What is scRNA-seq?

Single-cell RNA sequencing measures which genes are active in each individual cell. The output is a table where rows are cells and columns are genes. By training a classifier on this table, we can learn which gene patterns predict whether a cell is normal or cancerous.

### Why a Random Forest?

Random Forests handle high-dimensional data (thousands of genes) well and provide **feature importances** — scores for each gene showing how much it helped the model. These top genes are candidate **biomarkers**: genes that reliably distinguish tumor from normal cells and could serve as diagnostic targets or drug targets.

## Troubleshooting

**Barcode suffixes show as `null` in the YAML**
Run `python run.py inspect --config configs/datasets/GSE154778.yaml` to see the actual suffix distribution, then fill in the values.

**`ValueError: Column 'condition' not found`**
Run `convert` first to produce a processed `.h5ad` before running `run`.

**Out of memory during processing**
Reduce `n_top_genes` in the dataset YAML (e.g., from 2000 to 1000).

**Google Drive download fails**
Ensure the file is publicly shared ("Anyone with the link can view").
