# Cell State Classifier Pipeline — 3-Phase Plan

## Context

The CellClassifier/PASCAL project has 3 scRNA-seq datasets (GSE154778, GSE162708, GSE165399) configured in `configs/datasets/`. The existing codebase can convert each dataset to H5AD, merge them, and run a basic 2-class RF classifier. This plan extends the pipeline to: (1) build a unified AnnData for multi-dataset analysis, (2) detect and correct batch effects, and (3) train explainable classifiers for 3-class cell state classification with biomarker discovery.

---

## Phase 1: Unified AnnData Object

**Goal:** Single CLI command to convert all datasets and merge into one AnnData with batch/sample tracking.

### What exists already
- `cellclassifier/geo.py:convert_dataset()` — converts each GEO dataset to H5AD
- `cellclassifier/data.py:merge_datasets()` — concatenates H5ADs, intersects genes, remaps conditions, computes joint PCA/UMAP/Leiden
- `merge` CLI subcommand in `cli.py`

### Changes

| File | Change |
|------|--------|
| `cellclassifier/data.py` | Add `batch_key="dataset"` param to `merge_datasets()`; ensure `obs["sample"]` survives concat |
| `cellclassifier/cli.py` | Add `build` subcommand that chains `convert_dataset()` for each config → `merge_datasets()` in one step |
| `cellclassifier/config.py` | Add `batch_key: str = "dataset"` to `PipelineConfig` |
| `configs/pipeline.yaml` | Add `batch_key: "dataset"` |
| `tests/test_cli.py` | Add `TestBuildCommand` (2 synthetic datasets → verify combined H5AD has correct obs columns) |

### CLI Usage
```bash
python run.py build \
  --config configs/datasets/GSE154778.yaml \
  --config configs/datasets/GSE162708.yaml \
  --config configs/datasets/GSE165399.yaml \
  --condition-map configs/condition_map.yaml \
  --output ./output/combined
```

### Verification
- Output H5AD has `obs` columns: `dataset`, `sample`, `condition` (with values normal/precancerous/malignant)
- Cell count equals sum of per-dataset cells after QC
- `pytest tests/test_cli.py::TestBuildCommand`

---

## Phase 2: Batch Effect Detection & Correction

**Goal:** Quantify batch effects via housekeeping genes and visual inspection, then correct using ComBat (default), Harmony, or Scanorama.

### New file: `cellclassifier/batch.py`

| Function | Purpose |
|----------|---------|
| `compute_housekeeping_expression(adata, batch_key, genes)` | Mean expression of housekeeping genes (ACTB, GAPDH, B2M, RPL13A, RPLP0, PPIA) per batch → DataFrame |
| `compute_batch_distances(adata, batch_key, genes)` | Pairwise Euclidean distances between batches in housekeeping gene space → symmetric DataFrame |
| `correct_batch_combat(adata, batch_key)` | Apply `scanpy.pp.combat()` — zero extra dependencies |
| `correct_batch_harmony(adata, batch_key, n_pcs)` | Apply Harmony on PCA → `obsm["X_pca_harmony"]` → re-run neighbors/UMAP/Leiden. Requires `harmonypy` |
| `correct_batch_scanorama(adata, batch_key)` | Apply Scanorama → `obsm["X_scanorama"]` → re-run neighbors/UMAP/Leiden. Requires `scanorama` |
| `compute_batch_mixing_score(adata, batch_key, n_neighbors)` | Fraction of k-NN from different batches (0=segregated, 1=mixed) |
| `compare_corrections(uncorrected, corrections, batch_key)` | Table of mixing_score + silhouette per method |

### New plotting functions in `cellclassifier/plotting.py`

| Function | Output |
|----------|--------|
| `plot_batch_umap(adata, batch_key, condition_col, save_dir)` | Side-by-side UMAPs colored by batch vs condition |
| `plot_housekeeping_heatmap(hk_df, save_path)` | Heatmap: batches x housekeeping genes |
| `plot_batch_distance_heatmap(distances, save_path)` | Heatmap: pairwise batch distances |
| `plot_batch_correction_comparison(umaps_dict, batch_key, save_dir)` | Grid of UMAPs: uncorrected + each correction method |

### New CLI subcommands in `cellclassifier/cli.py`

- **`batch-check`** `--data <h5ad> --output <dir> --batch-key dataset` — runs housekeeping analysis, mixing score, generates diagnostic plots
- **`batch-correct`** `--data <h5ad> --method {combat,harmony,scanorama,all} --output <dir> --batch-key dataset` — applies correction, saves corrected H5AD + comparison plots

### Config changes

| File | Change |
|------|--------|
| `cellclassifier/config.py` | Add `BatchConfig` dataclass (batch_key, housekeeping_genes, correction_method, n_neighbors_mixing) |
| `configs/pipeline.yaml` | Add `batch:` section |
| `requirements.txt` | Add `harmonypy>=0.0.9`, `scanorama>=1.7` (lazy imports with clear error messages) |

### New tests: `tests/test_batch.py`

- Synthetic data: 2 batches of 60 cells each, batch 2 has systematic expression shift
- `TestHousekeepingExpression` — correct shape, values differ between batches
- `TestBatchDistances` — symmetric, diagonal zero, detects shift
- `TestCorrectBatchCombat` — output has same cell count, obs preserved
- `TestBatchMixingScore` — separated batches → low score, mixed → high score

### CLI Usage
```bash
python run.py batch-check --data ./output/combined/combined_processed.h5ad --output ./output/batch
python run.py batch-correct --data ./output/combined/combined_processed.h5ad --method combat --output ./output/corrected
```

### Verification
- Housekeeping heatmap shows expression variation before correction
- Batch mixing score increases after correction
- UMAP comparison shows better mixing post-correction
- `pytest tests/test_batch.py`

---

## Phase 3: Cell State Classification & Biomarker Discovery

**Goal:** 3-class classification (normal/precancerous/malignant) with multiple explainable models, cross-validation, pairwise biomarker discovery, and transition gene analysis.

### Extended models in `cellclassifier/model.py`

| Function | Purpose |
|----------|---------|
| `train_logistic(X, y, penalty="l1", C=1.0)` | L1 Logistic Regression — drives unimportant coefficients to zero for natural gene selection |
| `train_gbt(X, y, n_estimators=100, lr=0.1, max_depth=5)` | Gradient Boosted Trees — often more accurate than RF, still has feature importances |
| `get_logistic_coefficients(model, gene_names, top_n)` | Extract non-zero coefficients as ranked DataFrame |
| `cross_validate_model(X, y, model_type, n_folds=5)` | Stratified k-fold CV → mean accuracy, F1, per-fold reports |

Artifact format extended: adds `model_type` and `cv_results` keys. `load_artifact()` handles old format via `.get("model_type", "rf")`.

### New file: `cellclassifier/biomarkers.py`

| Function | Purpose |
|----------|---------|
| `pairwise_importances(adata, condition_col, pairs, top_n)` | Train binary RF for each condition pair → per-pair feature importances with fold changes |
| `transition_genes(pairwise_results, key, top_n)` | Rank genes driving precancerous→malignant by importance * fold_change → transition_score |
| `shared_biomarkers(pairwise_results, top_n)` | Rank aggregation across all pairs → pan-malignancy markers |
| `compute_gene_signatures(adata, gene_sets, condition_col)` | Score cells on predefined gene sets (EMT, proliferation, etc.) via `scanpy.tl.score_genes()` |

### Extended analysis in `cellclassifier/analysis.py`

| Function | Purpose |
|----------|---------|
| `pairwise_differential_expression(adata, condition_col, top_n)` | Wilcoxon rank-sum DE for all condition pairs → log2FC, p-values |
| `volcano_data(de_result, fc_threshold, pval_threshold)` | Prepare data for volcano plots with significance labels |

### New plotting functions in `cellclassifier/plotting.py`

| Function | Output |
|----------|--------|
| `plot_volcano(de_data, title, save_path)` | Volcano plot: log2FC vs -log10(pval) |
| `plot_biomarker_heatmap(adata, genes, condition_col, save_path)` | Heatmap of top biomarker genes grouped by condition |
| `plot_transition_genes(transition_df, top_n, save_path)` | Bar chart of transition genes with up/down direction |
| `plot_model_comparison(results, save_path)` | Grouped bars: accuracy/F1 across RF, LR, GBT |
| `plot_cross_validation(cv_results, save_path)` | Box plot of per-fold metrics |

### New CLI subcommands in `cellclassifier/cli.py`

- **`classify`** `--config <yaml> --data <h5ad> --model-type {rf,logistic,gbt,all} --cross-validate --output <dir>` — trains classifiers, optionally compares all three, saves artifacts + comparison plots
- **`biomarkers`** `--config <yaml> --data <h5ad> --output <dir>` — runs pairwise importances, transition analysis, DE analysis, generates volcano plots + heatmaps + CSVs

### Config changes

| File | Change |
|------|--------|
| `cellclassifier/config.py` | Add `ClassifyConfig` (model_types, cross_validate, n_folds, per-model params) and `BiomarkerConfig` (top_n, fc/pval thresholds) |
| `configs/pipeline.yaml` | Add `classify:` and `biomarker:` sections |

### New tests

**`tests/test_biomarkers.py`:**
- 3-class synthetic data (60 cells each, genes 0-9 upregulated in malignant, 10-19 in precancerous)
- `TestPairwiseImportances` — correct keys, signal genes rank high
- `TestTransitionGenes` — returns scored DataFrame with direction
- `TestSharedBiomarkers` — multi-pair genes rank higher

**`tests/test_model_extended.py`:**
- `TestLogisticRegression` — trains, coefficients extractable
- `TestGradientBoostedTrees` — trains, importances extractable
- `TestCrossValidation` — returns dict with n_folds entries

**`tests/test_cli.py`:**
- `TestClassifyCommand`, `TestBiomarkersCommand` — 3-class synthetic H5AD, module-scoped fixtures

### CLI Usage
```bash
python run.py classify --config configs/pipeline.yaml \
  --data ./output/corrected/corrected_combat.h5ad \
  --model-type all --cross-validate --output ./output/classify

python run.py biomarkers --config configs/pipeline.yaml \
  --data ./output/corrected/corrected_combat.h5ad \
  --output ./output/biomarkers
```

### Verification
- `./output/classify/` contains model artifacts + `model_comparison.png` + CV box plots
- `./output/biomarkers/` contains: `pairwise_importances.csv`, `transition_genes.csv`, `shared_biomarkers.csv`, volcano PNGs, `biomarker_heatmap.png`
- `pytest tests/`

---

## Implementation Order

```
Phase 1 (~1-2 sessions):
  1. data.py: batch_key param + sample column preservation
  2. config.py: batch_key in PipelineConfig
  3. cli.py: build subcommand
  4. tests + verification

Phase 2 (~3-5 sessions):
  1. batch.py: housekeeping analysis + distances
  2. cli.py: batch-check subcommand
  3. batch.py: combat correction (zero-dep)
  4. batch.py: harmony + scanorama (lazy imports)
  5. cli.py: batch-correct subcommand
  6. plotting.py: batch visualization functions
  7. tests + verification

Phase 3 (~4-6 sessions):
  1. model.py: logistic + GBT + cross-validation
  2. cli.py: classify subcommand
  3. biomarkers.py: pairwise importances + transition genes + shared biomarkers
  4. analysis.py: pairwise DE + volcano data
  5. plotting.py: volcano, heatmap, comparison, CV plots
  6. cli.py: biomarkers subcommand
  7. config.py: ClassifyConfig + BiomarkerConfig
  8. tests + full end-to-end verification
```

## Key Design Decisions

- **ComBat as default** batch correction — built into scanpy, no extra dependencies. Harmony/Scanorama are opt-in with lazy imports.
- **All models are explainable**: RF (feature importances), L1 Logistic (sparse coefficients = gene selection), GBT (feature importances). No black-box models.
- **Pairwise RF for biomarkers** — training a separate binary classifier per condition pair isolates which genes matter for each specific transition, rather than muddling them in a single 3-class model.
- **Transition score = importance x fold_change** — combines statistical discriminative power with biological magnitude of change.
- **Backward-compatible artifacts** — `load_artifact()` gracefully handles old 2-key format.

## Critical Files Summary

| File | Status | Role |
|------|--------|------|
| `cellclassifier/batch.py` | **New** | Batch detection + correction (Phase 2 core) |
| `cellclassifier/biomarkers.py` | **New** | Pairwise importances, transition genes (Phase 3 core) |
| `cellclassifier/cli.py` | Modify | +5 subcommands: build, batch-check, batch-correct, classify, biomarkers |
| `cellclassifier/model.py` | Modify | +Logistic, GBT, cross-validation |
| `cellclassifier/analysis.py` | Modify | +Pairwise DE, volcano data prep |
| `cellclassifier/plotting.py` | Modify | +Batch, volcano, heatmap, comparison plots |
| `cellclassifier/config.py` | Modify | +BatchConfig, ClassifyConfig, BiomarkerConfig |
| `cellclassifier/data.py` | Modify | batch_key param, sample column fix |
| `configs/pipeline.yaml` | Modify | +batch, classify, biomarker sections |
| `requirements.txt` | Modify | +harmonypy, scanorama |
| `tests/test_batch.py` | **New** | Batch module tests |
| `tests/test_biomarkers.py` | **New** | Biomarker module tests |
| `tests/test_model_extended.py` | **New** | Logistic, GBT, CV tests |
