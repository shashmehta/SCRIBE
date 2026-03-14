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

## Current Data Issues (Phase 2 findings — relevant for Phase 3)

The unified dataset at `output/combined/combined_processed.h5ad` has 45,933 cells × 265 genes across 3 GEO datasets. Phase 2 batch analysis uncovered two significant issues that must inform Phase 3 classifier design:

### 1. Condition–dataset confounding

The biological conditions are not evenly distributed across datasets:

| | GSE154778 | GSE162708 | GSE165399 |
|---|---|---|---|
| malignant | 14,924 | 9,441 | 2,113 |
| normal | 0 | 12,497 | 939 |
| precancerous | 0 | 0 | 6,019 |

- GSE154778 contains **only** malignant cells
- Precancerous cells come **exclusively** from GSE165399
- Normal cells come only from GSE162708 and GSE165399

This means batch correction cannot fully disentangle technical from biological effects, and **leave-one-dataset-out cross-validation is not feasible** (test folds would have missing classes). Phase 3 should use stratified k-fold CV with per-dataset accuracy breakdowns to detect batch overfitting.

### 2. Small gene space after HVG intersection

Each dataset independently selected its top 2,000 highly variable genes during preprocessing, then the merge intersected them — leaving only **265 shared genes**. Standard housekeeping genes (ACTB, GAPDH, etc.) were filtered out by HVG selection, so housekeeping-based batch quantification is not possible on the current combined data. The data was also independently scaled (zero-centered) per dataset before merging, which limits what post-hoc batch correction can achieve.

### 3. Batch correction results

- **ComBat**: Mixing score 0.129 → 0.136 (minimal improvement). Corrected file: `output/corrected/corrected_combat.h5ad`
- **Harmony**: Mixing score 0.129 → 0.129 (no change), but silhouette improved -0.023 → 0.009. Corrected file: `output/corrected/corrected_harmony.h5ad`
- Both methods had limited impact due to the confounding described above.

### Implications for Phase 3

- The classifier may learn dataset identity rather than true biological markers. **Per-dataset accuracy breakdowns** are essential.
- Consider including `dataset` as a covariate or using domain-adversarial approaches.
- Despite the confounding, cells from GSE162708 (which has both malignant and normal) do separate by condition, suggesting real biological signal exists.
