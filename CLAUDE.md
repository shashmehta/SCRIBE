# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CellClassifier is a Python ML/bioinformatics project for classifying pancreatic cell types using single-cell RNA sequencing (scRNA-seq) gene expression data. The primary goal is distinguishing normal vs tumor cells in Pancreatic Ductal Adenocarcinoma (PDAC) and classifying pancreatic cell subtypes (alpha, beta, delta). Additionally, it will determine and rank the top genes used to classify between the health and malignant cells, and determine biomarkers and potential target genes for future therapies.

## Environment Setup

- **Python 3.12** managed via Conda (`.conda/` directory)
- No `requirements.txt` exists. Key dependencies: `scanpy`, `anndata`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `cellxgene-census`, `scipy`, `h5py`, `joblib`
- The notebook (`download_pdac_data.ipynb`) installs deps inline with `pip install`

## Running

- **Main training script:** `python RFMouseHumanV2.py` — connects to cellxGene Census API, trains RandomForestClassifier on pancreatic cell types. This was originally a test file to practice working with CellxGene data and Random Forest Classifiers.
- **PDAC notebook:** `download_pdac_data.ipynb` — full pipeline from data loading through model training and visualization (designed for Google Colab). This is the currently the most up to date code.
- **Dendrogram script:** `python DendrogramPlot.py` — hierarchical clustering visualization (incomplete)

## Architecture

The project follows a standard ML pipeline, implemented across two main files:

**`RFMouseHumanV2.py`** — Fetches pancreatic cell data (alpha/beta/delta) from the CZI cellxGene Census API, preprocesses with scanpy (normalize → log-transform → HVG filtering → PCA), and trains a RandomForestClassifier. Steps 7-10 are incomplete stubs. This was a test file to practice working with CellxGene data and Random Forest Classifiers. It is not important.

**`download_pdac_data.ipynb`** — This is the most up-to-date code. Complete pipeline for PDAC classification:
1. Loads a 57,423-cell × 2,033-gene H5AD dataset from Google Drive
2. Encodes condition labels ('N' normal / 'T' tumor) → binary
3. 80/20 stratified train/test split
4. RandomForestClassifier with balanced class weights
5. Feature importance extraction → top discriminating genes
6. UMAP visualization by cell type, condition, and gene expression
7. Differential expression analysis identifying biomarkers (CFHR1, RBP2, etc.)

**Data format:** AnnData (`.h5ad`) — `X` matrix holds gene expression, `obs` holds cell metadata (cell type, condition), `var` holds gene info.

**Model artifact:** `model.joblib` — serialized trained RandomForestClassifier, loadable with `joblib.load()`.

## Key Domain Concepts

- **cellxGene Census API** (`cellxgene_census`): CZI's API for querying single-cell datasets. Used in `RFMouseHumanV2.py` to fetch specific cell types by `cell_type` filter.
- **scanpy pipeline**: `normalize_total` → `log1p` → `highly_variable_genes` → `pca` → `neighbors` is the standard preprocessing chain used throughout.
- Pancreatic cell types of interest: A cells (alpha/glucagon), B cells (beta/insulin), D cells (delta/somatostatin).
