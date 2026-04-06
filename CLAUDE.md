# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SCRIBE (Single-Cell RNA Interpretable Biomarker Explorer) is a Python ML/bioinformatics project for identifying biomarkers and therapeutic targets from single-cell RNA sequencing (scRNA-seq) data using explainable machine learning methods. It classifies cell conditions, ranks discriminating genes, and identifies potential targets for future therapies.

## Environment Setup

- **Python 3.10** managed via Conda (`mlproj` environment)
- Install: `pip install -e .` (dependencies in `pyproject.toml`)
- Key dependencies: `scanpy`, `anndata`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `h5py`, `joblib`, `gdown`

## Running

```bash
# Install as editable package:
pip install -e .

# CLI commands (after install):
scribe --help
scribe run --data ./output/processed/combined_processed.h5ad --output ./output/processed
scribe build --rebuild
scribe hk-analysis --data ./output/processed/combined_processed.h5ad --output ./output/plots/hk_analysis

# Or via run.py:
python run.py --help
```

## Architecture

```
scribe/                  # Main package
├── __init__.py          # Package marker
├── cli.py               # Click CLI — convert, inspect, merge, build, batch-check, batch-correct, hk-analysis, train, evaluate, plot, run
├── config.py            # YAML config dataclasses (Dataset, Pipeline, BatchConfig)
├── data.py              # Data loading, preprocessing, merge, train/test split
├── geo.py               # GEO dataset loaders (CSV DGE, 10x MTX, TAR), cellxGene annotation
├── model.py             # RF training, evaluation, feature importances, save/load artifacts
├── analysis.py          # Differential expression: avg expression, ratios, top genes
├── batch.py             # Batch effect detection/correction (ComBat, Harmony, Scanorama), HK gene analysis
├── plotting.py          # UMAP, feature importance, batch diagnostic plots
├── monitor.py           # System resource monitoring (RSS, CPU, peak memory)
├── zarr_utils.py        # Memory-efficient Zarr chunked I/O and batch correction
└── cache.py             # Parquet cache for Marimo interactive app
run.py                   # Convenience CLI entry point
app.py                   # Marimo interactive batch correction explorer
```

**Data format:** AnnData (`.h5ad`) — `X` matrix holds gene expression, `obs` holds cell metadata (cell type, condition), `var` holds gene info.

**Model artifact:** `model_artifact.joblib` — bundles the trained RandomForestClassifier, LabelEncoder, and gene names list. Load with `scribe.model.load_artifact()`.

**Output directory:** `output/` is gitignored and should be symlinked to Google Drive for cross-machine access. See README for setup instructions.

## Key Domain Concepts

- **cellxGene Census API** (`cellxgene_census`): CZI's API for querying single-cell datasets.
- **scanpy pipeline**: `normalize_total` → `log1p` → `highly_variable_genes` → `pca` → `neighbors` is the standard preprocessing chain.
- Pancreatic cell types of interest: A cells (alpha/glucagon), B cells (beta/insulin), D cells (delta/somatostatin).

## Current Data Status

### Combined dataset

45,933 cells × 3,004 genes from 3 GEO datasets, built with batch-aware HVG selection.

**Pipeline flow (`merge_datasets()` in `data.py`):**
1. Each dataset normalized + log1p independently
2. Concatenated on all common genes (~14,899 genes, kept sparse)
3. `sc.pp.highly_variable_genes(batch_key='dataset', n_top_genes=3000)` + 6 force-included HK genes
4. Scale → PCA → UMAP → Leiden

**Batch correction:** ComBat via memory-efficient Zarr chunked pipeline (`zarr_utils.py`). Corrected file: `output/processed/combined_processed_corrected.h5ad`.

**Output directory:** `output/` is a symlink to a shared Google Drive folder. All data (raw, processed, plots) lives on Drive, not on local disk. See README for the Drive folder structure.

### Condition–dataset confounding

| | GSE154778 | GSE162708 | GSE165399 |
|---|---|---|---|
| malignant | 14,924 | 9,441 | 2,113 |
| normal | 0 | 12,497 | 939 |
| precancerous | 0 | 0 | 6,019 |

Batch correction cannot fully disentangle technical from biological effects due to this confounding.
