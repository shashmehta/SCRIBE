# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SCRIBE (Single-Cell RNA Interpretable Biomarker Explorer) is a Python ML/bioinformatics project for identifying biomarkers and therapeutic targets from single-cell RNA sequencing (scRNA-seq) data using explainable machine learning methods. It classifies cell conditions, ranks discriminating genes, and identifies potential targets for future therapies.

## Environment Setup

- **Python 3.10** managed via Conda (`mlproj` environment)
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
├── cli.py           # Click CLI — convert, inspect, merge, build, batch-check, batch-correct, train, evaluate, plot, run
├── config.py        # YAML config dataclasses (Dataset, Pipeline, BatchConfig)
├── data.py          # Data loading (local + Google Drive), preprocessing, train/test split
├── geo.py           # GEO dataset loaders (CSV DGE, 10x MTX, TAR), cellxGene annotation
├── model.py         # RF training, evaluation, feature importances, save/load artifacts
├── analysis.py      # Differential expression: avg expression, ratios, top genes
├── batch.py         # Batch effect detection (housekeeping, mixing score) and correction (ComBat, Harmony, Scanorama)
└── plotting.py      # UMAP plots, feature importance bar charts, batch diagnostic plots
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

## Current Data Status

### Batch-aware HVG pipeline (in progress — 2026-03-20)

The `build` command now uses **batch-aware HVG selection after merging** instead of per-dataset HVG then intersection. This was implemented to solve the problem where per-dataset HVG selection left only 265 genes and lost all housekeeping genes.

**New pipeline flow (`merge_datasets()` in `data.py`):**
1. Each dataset is normalized + log1p independently (with `skip_hvg=True`, `skip_scale=True`, `skip_embeddings=True`)
2. Datasets are concatenated on ALL common genes (~14,899 genes, kept sparse)
3. `sc.pp.highly_variable_genes(batch_key='dataset', n_top_genes=3000)` selects genes variable within each batch
4. Housekeeping genes (ACTB, GAPDH, B2M, RPL13A, RPLP0, PPIA) are force-included
5. Subset to ~3,004 genes, then scale → PCA → optional Harmony → UMAP → Leiden

**Current combined dataset:** 45,933 cells × 3,004 genes (built without Harmony via `--no-harmony`)

**Next steps:**
- UMAPs of the uncorrected 3,004-gene dataset show batch effects are still present (datasets cluster separately)
- Need to rebuild WITH Harmony correction (`python run.py build --rebuild` without `--no-harmony`) and compare UMAPs
- Run `batch-check` and `batch-subset` to quantify improvement
- Then proceed to Phase 3 classifier training

**CLI options added:** `--n-top-genes` (default 3000) and `--no-harmony` on both `build` and `merge` commands.

### Condition–dataset confounding (unchanged)

| | GSE154778 | GSE162708 | GSE165399 |
|---|---|---|---|
| malignant | 14,924 | 9,441 | 2,113 |
| normal | 0 | 12,497 | 939 |
| precancerous | 0 | 0 | 6,019 |

- GSE154778 contains **only** malignant cells
- Precancerous cells come **exclusively** from GSE165399
- Normal cells come only from GSE162708 and GSE165399

This means batch correction cannot fully disentangle technical from biological effects. Phase 3 should use stratified k-fold CV with per-dataset accuracy breakdowns.

### Previous batch correction results (on old 265-gene dataset — obsolete)

- ComBat and Harmony had minimal impact on the old 265-gene dataset
- Old corrected files have been deleted to save disk space
- New batch correction should be evaluated on the 3,004-gene dataset
