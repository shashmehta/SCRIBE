# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CellClassifier is a Python ML/bioinformatics project for classifying pancreatic cell types using single-cell RNA sequencing (scRNA-seq) gene expression data. The primary goal is distinguishing normal vs tumor cells in Pancreatic Ductal Adenocarcinoma (PDAC) and classifying pancreatic cell subtypes (alpha, beta, delta). Additionally, it will determine and rank the top genes used to classify between the health and malignant cells, and determine biomarkers and potential target genes for future therapies.

## Environment Setup

- **Python 3.12** managed via Conda (`cellclassifier` environment)
- Dependencies listed in `requirements.txt`. Install with: `pip install -r requirements.txt`
- Key dependencies: `scanpy`, `anndata`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `h5py`, `joblib`, `gdown`

## Running

**Main CLI script (recommended):**
```bash
# First run — download from Google Drive, train, evaluate, plot:
python run.py --gdrive-id <GOOGLE_DRIVE_FILE_ID> --output ./output

# Run with local H5AD file:
python run.py --data ./data/pdac.h5ad --output ./output

# Load existing model (skip training):
python run.py --data ./data/pdac.h5ad --model ./output/model_artifact.joblib

# Force retrain on new data:
python run.py --data ./data/new.h5ad --retrain --output ./results_v2
```

## Architecture

The project is a Python package (`cellclassifier/`) with a CLI entry point (`run.py`):

```
cellclassifier/
├── __init__.py      # Package marker
├── cli.py           # Click CLI — convert, inspect, merge, train, evaluate, plot, run
├── config.py        # YAML config dataclasses (Dataset, Pipeline)
├── data.py          # Data loading (local + Google Drive), preprocessing, train/test split
├── geo.py           # GEO dataset loaders (CSV DGE, 10x MTX, TAR), cellxGene annotation
├── model.py         # RF training, evaluation, feature importances, save/load artifacts
├── analysis.py      # Differential expression: avg expression, ratios, top genes
└── plotting.py      # UMAP plots, feature importance bar charts
run.py               # CLI entry point — delegates to cellclassifier.cli
```

**Pipeline flow (run.py):**
1. Load H5AD data (local path or download from Google Drive via `gdown`)
2. Extract dense expression matrix + encode condition labels
3. Train RandomForestClassifier (or load existing model artifact)
4. Evaluate on test set (classification report + confusion matrix)
5. Extract feature importances → top discriminating genes
6. Differential expression analysis (normal vs tumor expression ratios)
7. Generate UMAP and feature importance plots → saved as PNGs

**Data format:** AnnData (`.h5ad`) — `X` matrix holds gene expression, `obs` holds cell metadata (cell type, condition), `var` holds gene info.

**Model artifact:** `model_artifact.joblib` — bundles the trained RandomForestClassifier, LabelEncoder, and gene names list in a single dict. Load with `cellclassifier.model.load_artifact()`.

## Key Domain Concepts

- **cellxGene Census API** (`cellxgene_census`): CZI's API for querying single-cell datasets.
- **scanpy pipeline**: `normalize_total` → `log1p` → `highly_variable_genes` → `pca` → `neighbors` is the standard preprocessing chain used throughout.
- Pancreatic cell types of interest: A cells (alpha/glucagon), B cells (beta/insulin), D cells (delta/somatostatin).
