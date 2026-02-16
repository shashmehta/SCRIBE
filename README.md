# CellClassifier

A Python machine learning pipeline for classifying pancreatic cell types using single-cell RNA sequencing (scRNA-seq) data. This tool helps distinguish normal vs tumor cells in Pancreatic Ductal Adenocarcinoma (PDAC) and identifies key biomarkers for potential therapeutic targets.

## Overview

CellClassifier uses Random Forest classification on gene expression data to:
- **Classify** normal vs malignant pancreatic cells
- **Identify** top discriminating genes through feature importance analysis
- **Analyze** differential gene expression between conditions
- **Visualize** results with UMAP plots and feature importance charts

The pipeline is designed for researchers working with single-cell RNA-seq data in `.h5ad` (AnnData) format.

## Features

- 🧬 **Random Forest Classification** - Robust cell type classification
- 📊 **Differential Expression Analysis** - Identify genes enriched in tumor vs normal cells
- 🎯 **Feature Importance Ranking** - Discover top discriminating genes
- 📈 **UMAP Visualization** - Generate publication-ready plots
- 💾 **Model Persistence** - Save and reload trained models
- ☁️ **Google Drive Integration** - Download datasets directly from shared links

## Installation

### Prerequisites

- Python 3.12
- Conda (recommended for environment management)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/CellClassifier.git
   cd CellClassifier
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n cellclassifier python=3.12
   conda activate cellclassifier
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

**Option 1: Download data from Google Drive**
```bash
python run.py --gdrive-id YOUR_GOOGLE_DRIVE_FILE_ID --output ./output
```

**Option 2: Use local H5AD file**
```bash
python run.py --data ./data/pdac.h5ad --output ./output
```

### Command Line Options

```bash
python run.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--data PATH` | Path to local H5AD file | None |
| `--gdrive-id ID` | Google Drive file ID to download | None |
| `--output DIR` | Output directory for results | `./output` |
| `--model PATH` | Load existing model (skip training) | None |
| `--retrain` | Force retrain even if model exists | False |

### Examples

**First run with Google Drive data:**
```bash
python run.py --gdrive-id 1abc123xyz --output ./results
```
This will:
- Download the dataset from Google Drive
- Train a new Random Forest model
- Generate evaluation metrics
- Create UMAP and feature importance plots
- Save model artifact for reuse

**Reuse trained model:**
```bash
python run.py --data ./results/data/pdac_data.h5ad --model ./results/model_artifact.joblib
```
This will load the existing model and skip training.

**Retrain on new data:**
```bash
python run.py --data ./new_data/pdac_updated.h5ad --retrain --output ./results_v2
```

## How It Works

### Pipeline Architecture

```
┌─────────────────┐
│  Load H5AD Data │  ← AnnData format with expression matrix
└────────┬────────┘
         │
┌────────▼────────┐
│  Preprocessing  │  ← Normalize, log transform, filter genes
└────────┬────────┘
         │
┌────────▼────────┐
│  Train/Load RF  │  ← Random Forest classifier
└────────┬────────┘
         │
┌────────▼────────┐
│   Evaluation    │  ← Metrics + confusion matrix
└────────┬────────┘
         │
┌────────▼────────┐
│    Analysis     │  ← Feature importances + diff. expression
└────────┬────────┘
         │
┌────────▼────────┐
│  Visualization  │  ← UMAP plots + bar charts
└─────────────────┘
```

### Data Format

Input data must be in **AnnData** (`.h5ad`) format with:
- **`X`**: Gene expression matrix (cells × genes)
- **`obs`**: Cell metadata including:
  - `CONDITION`: Cell condition label (e.g., 'N' for normal, 'T' for tumor)
  - `Cell_type` or `celltype3`: Cell type annotations (optional, for visualization)
- **`var`**: Gene metadata with gene names

### Processing Steps

1. **Data Loading**: Reads H5AD file and extracts expression matrix
2. **Preprocessing**:
   - Normalize total counts per cell
   - Log-transform expression values
   - Filter for highly variable genes
   - Compute PCA and UMAP embeddings
3. **Training**: Fits Random Forest classifier on 80% of data
4. **Evaluation**: Tests on held-out 20% and reports metrics
5. **Feature Analysis**:
   - Ranks genes by feature importance
   - Calculates mean expression ratios between conditions
6. **Visualization**: Generates UMAP plots and feature importance charts

## Output Structure

```
output/
├── data/
│   └── pdac_data.h5ad          # Downloaded/processed data
├── model_artifact.joblib        # Trained model + metadata
└── plots/
    ├── umap_CONDITION.png       # UMAP colored by condition
    ├── umap_Cell_type.png       # UMAP colored by cell type
    ├── umap_gene_FXYD2.png      # Gene expression overlays
    ├── umap_gene_CTRB1.png
    └── feature_importances.png  # Top discriminating genes
```

### Understanding the Results

**Feature Importances** show which genes are most important for distinguishing normal vs tumor cells:
```
FXYD2      0.026164
CTRB1      0.021886
CLPS       0.021697
...
```

**Differential Expression** shows genes enriched in each condition:
- **High ratios** (>1): Genes upregulated in tumor cells
- **Low ratios** (<1): Genes upregulated in normal cells

**UMAP plots** provide visual confirmation that:
- Cells cluster by biological characteristics
- The model captures meaningful biological variation

## Project Structure

```
CellClassifier/
├── cellclassifier/           # Main package
│   ├── __init__.py
│   ├── data.py              # Data loading & preprocessing
│   ├── model.py             # Model training & evaluation
│   ├── analysis.py          # Differential expression analysis
│   └── plotting.py          # Visualization functions
├── run.py                   # CLI entry point
├── requirements.txt         # Python dependencies
├── CLAUDE.md               # Development guidelines
└── README.md               # This file
```

## Key Dependencies

- **scanpy**: Single-cell analysis toolkit
- **anndata**: Annotated data structures
- **scikit-learn**: Machine learning (Random Forest)
- **matplotlib/seaborn**: Plotting
- **pandas/numpy**: Data manipulation
- **gdown**: Google Drive downloads

## Scientific Background

### Cell Types

The classifier focuses on pancreatic cells involved in PDAC:
- **Normal cells**: Healthy pancreatic tissue
- **Tumor cells**: Malignant PDAC cells
- **Subtypes**: Alpha (glucagon), Beta (insulin), Delta (somatostatin) cells

### Biomarker Discovery

Feature importance analysis identifies genes like:
- **FXYD2, FXYD3**: Ion transport proteins
- **CTRB1, CLPS**: Digestive enzymes
- **S100A4**: Cancer progression marker

These genes represent potential therapeutic targets or diagnostic biomarkers for PDAC.

## Troubleshooting

**Issue**: `conda: command not found` warning
- **Solution**: This is harmless if you're not using conda, or ensure conda is properly initialized in your shell

**Issue**: Out of memory during processing
- **Solution**: Reduce the dataset size or increase available RAM

**Issue**: Google Drive download fails
- **Solution**: Ensure the file is publicly shared and the ID is correct

<!-- ## Contributing

Contributions are welcome! Please ensure code follows the project structure and includes appropriate documentation. -->

<!-- ## License

[Add your license here]

## Citation

If you use CellClassifier in your research, please cite:

```bibtex
[Add citation information]
```

## Contact

[Add contact information or link to issues page] -->
