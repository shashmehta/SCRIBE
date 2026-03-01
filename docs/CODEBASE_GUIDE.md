# CellClassifier Codebase Guide

This guide explains how the CellClassifier project works — the biology behind it, the programming concepts used, and every file in the codebase. It is written for someone who understands the big picture (what the project is trying to do) but wants to understand *how* it does it in detail.

---

## Table of Contents

1. [What the Project Does (Big Picture)](#1-what-the-project-does-big-picture)
2. [Key Biological Concepts](#2-key-biological-concepts)
3. [Key Programming Concepts](#3-key-programming-concepts)
4. [How the Data Flows Through the Project](#4-how-the-data-flows-through-the-project)
5. [File-by-File Walkthrough](#5-file-by-file-walkthrough)
6. [The Refactoring: What Changed and Why](#6-the-refactoring-what-changed-and-why)
7. [The Test Suite](#7-the-test-suite)
8. [Running the Project](#8-running-the-project)
9. [Glossary](#9-glossary)

---

## 1. What the Project Does (Big Picture)

Pancreatic Ductal Adenocarcinoma (PDAC) is one of the most lethal cancers, partly because it is so hard to distinguish cancerous cells from normal pancreatic cells. This project uses **machine learning on single-cell RNA sequencing data** to:

1. **Load** raw sequencing data from public databases (GEO)
2. **Clean and preprocess** it using a standard bioinformatics pipeline
3. **Train a classifier** (Random Forest) to tell normal vs. tumor cells apart
4. **Rank the genes** that best distinguish the two — these are candidate biomarkers
5. **Visualise** the cells in 2D to confirm the classifier is finding real biology

The end result is a ranked list of genes that drive tumor identity, which could point toward therapeutic targets for future drugs.

---

## 2. Key Biological Concepts

### Single-Cell RNA Sequencing (scRNA-seq)

Every cell in your body has the same DNA, but different cells use different genes. **RNA** is the intermediate molecule — when a cell "uses" a gene, it makes RNA copies of it. By counting how many RNA molecules a cell has for each gene, we can build a profile of that cell's activity.

**scRNA-seq** measures this for thousands of individual cells simultaneously. The output is a big table: rows are cells, columns are genes, and each number is how many RNA molecules were detected.

### Why Single-Cell? Why Not Bulk?

Old RNA-seq measured RNA from a mixture of millions of cells. The result was an average — you couldn't tell which cell type contributed what. scRNA-seq gives you one row per cell, so you can see the differences between, say, cancer cells and immune cells that are mixed together in a tumor.

### Digital Gene Expression (DGE) Matrix

The raw output of scRNA-seq is called a **DGE matrix** — a table of counts. For example:

```
         Cell_1  Cell_2  Cell_3
TP53         5       0       12
INS          0     300        0
GCG         87       0        0
```

`TP53` is detected 5 times in Cell 1, 0 times in Cell 2, and 12 times in Cell 3. Most values are zero — most genes are silent in any given cell. This is called **sparsity**.

### 10x Chromium

The most popular scRNA-seq platform. It uses oil droplets to encapsulate individual cells, then tags each cell's RNA with a unique **barcode** (a short DNA sequence like `ACGTACGT-1`). The barcode tells you which cell each RNA molecule came from. The output is three files:

- `barcodes.tsv.gz` — list of cell barcodes
- `features.tsv.gz` — list of gene names
- `matrix.mtx.gz` — the count matrix in a compressed sparse format

### GEO (Gene Expression Omnibus)

GEO is a public database run by NCBI where researchers deposit their sequencing data when they publish a paper. Every dataset has a **GSE accession** (e.g., `GSE154778`) and each individual sample has a **GSM accession** (e.g., `GSM5032701`). This project downloads three GEO datasets from published PDAC studies.

### Preprocessing Steps

Raw counts cannot be used directly for machine learning. They need to be cleaned:

1. **Filter low-quality cells**: A cell with fewer than 200 detected genes is probably an empty droplet or debris.
2. **Filter rarely-detected genes**: A gene detected in fewer than 3 cells is probably noise.
3. **Mitochondrial filtering**: Dying or damaged cells leak their cytoplasmic RNA but keep their mitochondrial RNA. So a cell where >20% of counts come from mitochondrial genes is likely dying and is removed. Human mitochondrial genes are named with the prefix `MT-`.
4. **Normalization**: Some cells were simply sequenced more than others (more RNA captured). We scale every cell to a total of 10,000 counts so we can compare cells fairly. This is called `normalize_total`.
5. **Log transformation**: After normalization, gene counts range from 0 to thousands. Taking `log(x+1)` compresses this range so highly-expressed genes don't overwhelm everything else.
6. **Highly Variable Genes (HVGs)**: Most genes don't change much between cell types — they're "housekeeping" genes. We keep only the ~2,000 genes that vary the most across cells, because these carry the most information for distinguishing cell types.
7. **PCA (Principal Component Analysis)**: Even 2,000 genes is a lot of dimensions. PCA finds the 30 directions of greatest variation in the data and projects every cell into this compressed space. Think of it like finding the 30 most important summary statistics.
8. **UMAP**: A visualization technique that projects the 30-dimensional PCA space into 2D for plotting. Cells that are biologically similar end up near each other. This is the scatter plot where you can see clusters.
9. **Leiden clustering**: An algorithm that identifies groups (clusters) of similar cells in the neighborhood graph. Like finding communities in a social network.

### Ontologies and cellxGene

An **ontology** is a standardised vocabulary — a controlled list of terms that means the same thing across all databases. For example, instead of one lab calling it "pancreas" and another "pancreatic tissue", everyone uses `UBERON:0001264` which always means the same thing.

**cellxGene** is the Chan Zuckerberg Initiative's browser for single-cell datasets. It requires datasets to have specific metadata columns with ontology IDs so that data from different labs can be compared. This project adds these fields before saving the final `.h5ad` file.

---

## 3. Key Programming Concepts

### AnnData (`.h5ad` files)

`AnnData` is the standard data container for single-cell bioinformatics. Think of it as a smart spreadsheet:

```
                    genes (var)
              gene1  gene2  gene3 ...
cells  cell1    0      5     12
(obs)  cell2   87      0      0
       cell3    3      0      0
```

It has three main parts:
- **`X`**: The gene expression matrix (cells × genes)
- **`obs`**: A table of metadata for each cell (one row per cell) — e.g., condition, sample, cell type
- **`var`**: A table of metadata for each gene (one row per gene) — e.g., gene name, whether it's highly variable
- **`obsm`**: Extra matrices attached to cells — e.g., `obsm["X_umap"]` holds the 2D UMAP coordinates
- **`uns`**: Unstructured metadata — e.g., `uns["title"]` holds the dataset title

The `.h5ad` file format is HDF5-based — it stores the data efficiently and can be read back into Python instantly.

### Sparse Matrices

scRNA-seq data is about 90% zeros — most genes are not expressed in any given cell. Storing all those zeros wastes memory. Instead, we use **sparse matrices** (specifically `scipy.sparse.csr_matrix`) which only store the non-zero values and their positions. A dataset with 10,000 cells and 20,000 genes would take 1.6 GB as a dense matrix but only ~160 MB as a sparse matrix.

### Random Forest Classifier

A **Random Forest** is a machine learning model made of many decision trees. Each tree makes predictions by asking a series of yes/no questions about gene expression levels (e.g., "Is INS > 50?"). The trees vote, and the majority wins.

Why Random Forest for scRNA-seq?
- It handles high-dimensional data (thousands of genes) well
- It gives you **feature importances** — a score for each gene saying how much it helped
- It is robust to noise and doesn't need the data to follow any particular distribution
- `class_weight="balanced"` handles the fact that there may be many more normal cells than tumor cells

### Feature Importances

After training, you can ask the Random Forest: "Which genes did you actually use to make decisions?" The answer is **feature importances** — a score for each gene between 0 and 1, where higher means the gene was more useful for classification. The top genes are candidate biomarkers — genes that reliably distinguish tumor from normal cells.

### Joblib

`joblib` is a Python library for saving and loading Python objects to disk. We use it to save the trained model:

```python
artifact = {"model": clf, "label_encoder": le, "gene_names": gene_names}
joblib.dump(artifact, "model_artifact.joblib")
```

This bundles everything needed to make predictions into one file, so you can load it later without retraining.

### Click (CLI Framework)

`click` is a Python library for building command-line interfaces. Instead of one script that does everything, Click lets you define **sub-commands** — like `git commit` or `git push`. Each sub-command is a Python function decorated with `@cli.command()`.

```python
@cli.command()
@click.option("--config", required=True, ...)
def train(config):
    ...
```

The `@click.option(...)` decorator automatically handles parsing command-line arguments, showing `--help` text, and validating inputs (e.g., checking that a file exists).

### YAML Configuration Files

YAML is a human-readable format for configuration files. Instead of hardcoding parameters like `n_estimators=100` in the Python script (which means editing code to change a parameter), we put them in a `.yaml` file:

```yaml
model:
  n_estimators: 100
  test_size: 0.2
```

This separates *parameters* from *code*, making it easy to run the same pipeline with different settings by just swapping the config file.

### Python Dataclasses

A `dataclass` is a Python class that acts as a structured container for named values:

```python
@dataclass
class ModelConfig:
    n_estimators: int = 100
    class_weight: str = "balanced"
    random_state: int = 42
    test_size: float = 0.2
```

We use dataclasses to load YAML files into Python objects with type checking, instead of working with raw dictionaries. This makes it impossible to mistype `cfg["n_estimatres"]` — Python would just tell you `ModelConfig has no attribute n_estimatres`.

### Label Encoding

Machine learning models work with numbers, not strings. `LabelEncoder` converts condition labels:
- `"normal"` → `0`
- `"tumor"` → `1`

After training, we use the same encoder in reverse to convert predictions back to readable names.

---

## 4. How the Data Flows Through the Project

```
GEO (public database)
        │
        ▼
  Raw files on disk
  (CSV, 10x MTX, TAR)
        │
        ▼ cellclassifier/geo.py
  AnnData (raw counts)
        │
        ▼ preprocess_adata()
  AnnData (normalized, HVGs, UMAP)
        │
        ▼ write_h5ad()
  *.h5ad file on disk
        │
        ▼ cellclassifier/data.py
  X (numpy matrix), y (labels)
        │
        ▼ cellclassifier/model.py
  Trained RandomForest + feature importances
        │
        ├──▶ cellclassifier/analysis.py
        │    Differential expression ratios
        │
        └──▶ cellclassifier/plotting.py
             UMAP PNGs + feature importance PNG
```

---

## 5. File-by-File Walkthrough

### `run.py`

The entry point — just three lines that hand off to the CLI:

```python
from cellclassifier.cli import main
if __name__ == "__main__":
    main()
```

**Before the refactoring**, this was a ~200-line script with `argparse` that did everything inline. Now it is a thin wrapper so the actual logic lives in the `cellclassifier/` package and is testable.

---

### `cellclassifier/config.py`

Defines Python **dataclasses** that represent each YAML config file, and functions to load and validate those files.

**Key classes:**

| Class | Purpose |
|---|---|
| `SourceConfig` | Where raw data lives (local path or Google Drive) |
| `FileConfig` | Which file to load and in what format |
| `PreprocessingConfig` | QC thresholds and HVG/PCA settings |
| `CellxGeneConfig` | Ontology IDs required by the cellxGene schema |
| `SampleConfig` | Metadata for one biological sample |
| `DatasetConfig` | Everything needed to convert one GEO dataset |
| `ModelConfig` | Random Forest hyperparameters |
| `AnalysisConfig` | How many top genes to report |
| `PlotsConfig` | Which columns and genes to plot on UMAP |
| `PipelineConfig` | Everything needed to run the ML pipeline |

**Key functions:**

- `load_dataset_config(path)` — reads a dataset YAML and returns a `DatasetConfig`
- `load_pipeline_config(path)` — reads a pipeline YAML and returns a `PipelineConfig`
- `_require_keys(d, keys, location)` — raises a helpful error if a required key is missing

---

### `cellclassifier/geo.py`

The GEO conversion module — the most complex file. It handles four different raw data formats and converts them all to a standardised AnnData.

**Loader functions (one per format):**

| Function | Format | When to use |
|---|---|---|
| `load_csv_dge()` | Gzipped CSV (genes × cells) | GSE154778 |
| `load_10x_mtx()` | 10x barcodes/features/matrix | GSE162708 |
| `load_tar_txt_dge()` | TAR of per-sample TXT files | GSE165399 |
| `load_tar_10x()` | TAR of per-sample 10x directories | Generic |

**Processing functions:**

- `assign_sample_metadata()` — figures out which cell belongs to which sample using barcode suffixes (for 10x) or GSM IDs (for TAR files), then tags each cell with its condition, tissue, and disease ontology ID. Uses positional numpy arrays instead of pandas label-based indexing to handle duplicate barcode names after concatenation.

- `preprocess_adata()` — runs the full scanpy pipeline: filter → MT QC → normalize → log1p → HVGs → scale → PCA → neighbors → UMAP → Leiden.

- `annotate_cellxgene_metadata()` — adds the required cellxGene schema 5.0.0 fields to obs, var, and uns. Per-sample overrides from `assign_sample_metadata` take precedence.

- `validate_cellxgene()` — checks that all required fields are present before saving.

- `convert_dataset()` — the top-level orchestrator; runs all steps in order and saves the `.h5ad`.

---

### `cellclassifier/data.py`

Loads `.h5ad` files and prepares data for machine learning.

- `download_from_gdrive()` — downloads a file from Google Drive using `gdown`. Skips if already on disk.
- `load_adata()` — reads an `.h5ad` file and validates that the required condition column exists.
- `extract_features_and_labels()` — converts the AnnData matrix to a dense numpy array (`X`) and encodes condition labels as integers (`y`).
- `split_data()` — stratified train/test split (keeps class proportions the same in both sets).

---

### `cellclassifier/model.py`

Trains and evaluates the Random Forest classifier.

- `train()` — builds and fits a `RandomForestClassifier`. Uses all CPU cores (`n_jobs=-1`) and balanced class weights.
- `evaluate()` — runs the model on a test set and prints a classification report (precision, recall, F1) and confusion matrix.
- `get_feature_importances()` — extracts the top N genes by feature importance score.
- `save_artifact()` — bundles model + label encoder + gene names into one `.joblib` file.
- `load_artifact()` — loads a saved `.joblib` file and validates that all three components are present.

---

### `cellclassifier/analysis.py`

Differential expression analysis — compares gene activity between conditions.

- `avg_expression_by_condition()` — computes the mean expression of each gene separately for each condition (e.g., mean in normal cells, mean in tumor cells).
- `compute_expression_ratio()` — divides the two averages to get a ratio per gene. A ratio > 1 means the gene is more active in the numerator condition. `epsilon=1e-6` prevents division by zero for genes with no expression.
- `top_differential_genes()` — sorts genes by their absolute ratio and prints the top N in each direction.

---

### `cellclassifier/plotting.py`

Generates visualizations and saves them as PNG files.

- `plot_umap()` — draws a 2D scatter plot of cells using pre-computed UMAP coordinates from `adata.obsm["X_umap"]`. Can color points by any obs column (e.g., condition) or by gene expression.
- `plot_feature_importances()` — horizontal bar chart of the top genes by importance score.
- `generate_all_plots()` — calls both plot functions for each requested column and gene, saves everything to `<output>/plots/`.

Uses `matplotlib.use("Agg")` — a non-interactive backend that saves plots to files without needing a screen (important for running on servers).

---

### `cellclassifier/cli.py`

The Click CLI — defines all sub-commands.

| Command | What it does |
|---|---|
| `convert` | Run `geo.convert_dataset()` on a dataset YAML |
| `train` | Load data, train RF, save artifact |
| `evaluate` | Load data + artifact, print metrics |
| `plot` | Load data + artifact, save PNGs |
| `run` | Full pipeline (train → evaluate → plot); skips training if artifact already exists unless `--retrain` |

Each command reads a YAML config and accepts optional `--data` and `--output` flags to override the YAML values without editing the file.

---

### `configs/datasets/*.yaml`

Three dataset config files, one per GEO study:

| File | Dataset | Format | Condition |
|---|---|---|---|
| `GSE154778.yaml` | Primary vs metastatic PDAC | CSV DGE | primary / metastatic |
| `GSE162708.yaml` | Pancreatic neuroendocrine tumor (pNET) | 10x MTX | tumor / normal |
| `GSE165399.yaml` | Normal, IPMN, PASC pancreas | TAR TXT DGE | normal / ipmn / pasc |

### `configs/pipeline.yaml`

Default settings for the ML pipeline — which `.h5ad` to use, where to write output, and hyperparameters for the Random Forest.

---

## 6. The Refactoring: What Changed and Why

Before this refactoring, the project was a mix of:
- A Jupyter notebook (`geo_data_processing.ipynb`) for data conversion — hard to run non-interactively
- A monolithic `run.py` with argparse — all logic in one file, hard to test

### What Was Done

**Stage 1 — Config system** (`cellclassifier/config.py` + YAML files):
- Created dataclasses to represent all configuration parameters
- Created YAML files for each dataset and for the pipeline
- This separates *what to run* (the YAML) from *how to run it* (the Python)

**Stage 2 — GEO module** (`cellclassifier/geo.py`):
- Extracted all the notebook code into a proper Python module
- Added support for four different raw data formats
- Fixed a bug where duplicate cell barcodes (which appear after concatenating samples) caused pandas to return a Series instead of a scalar — solved by using positional numpy arrays
- Added `filter="data"` to `tarfile.extractall()` (required by Python 3.12 for security)

**Stage 3 — CLI wiring** (`cellclassifier/cli.py` + updated `run.py`):
- Replaced argparse with Click
- Split the monolithic script into five focused sub-commands
- Each sub-command can be run and tested independently

### Why Click Instead of argparse?

- **Decorators** make it clear what each option does without boilerplate
- **Automatic `--help`** with clean formatting
- **CliRunner** — Click provides a test utility that runs CLI commands in-process, so tests run fast without spawning subprocesses
- **Type validation** — `type=click.Path(exists=True)` automatically checks that a file exists before your code runs

### Why YAML Configs?

Before: changing a parameter required editing Python code. Now: you can run the same pipeline on a different dataset by writing a new YAML file without touching any Python.

---

## 7. The Test Suite

Tests live in `tests/`. Run all tests with:

```bash
cd /path/to/CellClassifier
pytest tests/ -v
```

### `tests/test_geo.py` — 31 tests

Tests every function in `geo.py` using synthetic data created in-memory. Key techniques:
- Creates fake `.csv`, `.mtx`, and `.tar` files in a temporary directory (`tmp_path`)
- Tests each loader independently — no real GEO files needed
- Tests `assign_sample_metadata` with both barcode-suffix and GSM-ID strategies
- Tests `preprocess_adata` with small synthetic counts
- Tests `validate_cellxgene` to confirm required fields are detected

### `tests/test_cli.py` — 22 tests

Tests every CLI sub-command using Click's `CliRunner` (runs commands in-process).

Uses **module-scoped fixtures** (`scope="module"`) — the synthetic `.h5ad` file and the trained model artifact are created once and reused across all tests. This avoids re-training the Random Forest for every single test.

Key tests:
- `test_retrain_flag_retrains` — checks that `--retrain` produces a newer file (by comparing modification timestamps)
- `test_no_retrain_reuses_artifact` — checks that without `--retrain`, the artifact is not overwritten
- `test_missing_config_fails` — checks that a nonexistent config file gives an exit code ≠ 0

---

## 8. Running the Project

### Convert a GEO dataset to `.h5ad`

```bash
python run.py convert --config configs/datasets/GSE162708.yaml --output ./output
```

This reads the YAML, loads the raw files, runs preprocessing, and saves `./output/GSE162708_processed.h5ad`.

### Train the classifier

```bash
python run.py train --config configs/pipeline.yaml
```

### Run the full pipeline at once

```bash
python run.py run --config configs/pipeline.yaml --output ./output
```

Add `--retrain` to force retraining even if a model already exists.

### Evaluate a saved model

```bash
python run.py evaluate --config configs/pipeline.yaml --model ./output/model_artifact.joblib
```

### Generate plots

```bash
python run.py plot --config configs/pipeline.yaml --model ./output/model_artifact.joblib --output ./output
```

---

## 9. Glossary

| Term | Meaning |
|---|---|
| scRNA-seq | Single-cell RNA sequencing — measures gene expression in individual cells |
| PDAC | Pancreatic Ductal Adenocarcinoma — the cancer type being studied |
| DGE matrix | Digital Gene Expression matrix — the raw counts table |
| Barcode | A short DNA sequence that identifies which cell an RNA molecule came from |
| GEO | Gene Expression Omnibus — the public database where sequencing data is deposited |
| GSE | GEO Series — one complete study (e.g., GSE154778) |
| GSM | GEO Sample — one biological sample within a study (e.g., GSM5032701) |
| AnnData | Annotated Data — the Python object / `.h5ad` file used for single-cell data |
| obs | Observations — the rows of the AnnData matrix (one row per cell) |
| var | Variables — the columns of the AnnData matrix (one column per gene) |
| Sparse matrix | A matrix that only stores non-zero values to save memory |
| HVG | Highly Variable Gene — a gene that differs a lot across cells |
| PCA | Principal Component Analysis — compression of high-dimensional data |
| UMAP | 2D projection of high-dimensional data for visualisation |
| Leiden | Graph-based algorithm for clustering cells |
| Ontology | A standardised vocabulary of terms (like a controlled dictionary) |
| cellxGene | CZI's browser for single-cell datasets; requires specific metadata |
| Random Forest | An ensemble of decision trees that vote on predictions |
| Feature importance | How much each gene contributed to the Random Forest's decisions |
| LabelEncoder | Converts text labels (e.g., "tumor") to numbers (e.g., 1) |
| Joblib | Python library for saving/loading Python objects to/from disk |
| Click | Python library for building CLI tools with sub-commands |
| YAML | Human-readable configuration file format |
| Dataclass | A Python class used as a structured container for named values |
| CliRunner | Click's test utility for running CLI commands in-process |
| Stratified split | A train/test split that keeps the class proportions the same in both sets |
| MT genes | Mitochondrial genes (named `MT-...`) — used as a marker for dying cells |
