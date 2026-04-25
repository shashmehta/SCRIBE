"""GEO dataset loading and conversion to AnnData.

GEO (Gene Expression Omnibus) is the public database where researchers deposit
their sequencing data. Each study has a GSE accession (the whole dataset) and
individual samples have GSM accessions. This module downloads nothing — it
expects the files to already be on disk (or Google Drive) and converts them
into the AnnData format used by the rest of the project.
"""

from __future__ import annotations

import glob
import os
import shutil
import tarfile
import tempfile
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from scribe.config import (
    DatasetConfig,
    FileConfig,
    PreprocessingConfig,
    SampleConfig,
)

# Suppress unimportant deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", category=FutureWarning)


# ── Loaders ───────────────────────────────────────────────────────────────────
# Each loader handles one raw file format and returns an AnnData object.
# AnnData is a standard single-cell data container: rows = cells, columns = genes.

def load_csv_dge(path: str) -> ad.AnnData:
    """Load a gzipped CSV DGE matrix (genes × cells) and return AnnData (cells × genes).

    A DGE (Digital Gene Expression) matrix lists how many times each gene was
    detected in each cell. The file has genes as rows and cells as columns, but
    AnnData expects the opposite, so we transpose it.

    Args:
        path: Path to the .csv or .csv.gz file.

    Returns:
        AnnData with cells as observations and genes as variables.
    """
    print(f"Loading CSV DGE: {path}")
    # read_csv loads the file; index_col=0 makes the first column the row labels (gene names)
    dge = pd.read_csv(path, index_col=0)
    print(f"  Raw shape (genes × cells): {dge.shape}")

    adata = ad.AnnData(
        X=sp.csr_matrix(dge.values.T),        # .T transposes genes×cells → cells×genes
                                               # csr_matrix stores only non-zero values (sparse)
        obs=pd.DataFrame(index=dge.columns),   # cell barcodes become row labels
        var=pd.DataFrame(index=dge.index),     # gene names become column labels
    )
    print(f"  AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def load_10x_mtx(src_dir: str, name_prefix: str = "") -> ad.AnnData:
    """Load a 10x MTX dataset, remapping GSE-prefixed filenames if needed.

    10x Chromium sequencing produces three files: barcodes.tsv.gz (cell IDs),
    features.tsv.gz (gene names), and matrix.mtx.gz (the counts). GEO adds a
    dataset prefix to these filenames (e.g. "GSE162708_barcodes.tsv.gz"), so
    we temporarily create symlinks with the standard names before loading.

    Args:
        src_dir: Directory containing the 10x files.
        name_prefix: Prefix to strip from filenames to get standard 10x names.

    Returns:
        AnnData with cells as observations and genes as variables.
    """
    print(f"Loading 10x MTX: {src_dir}")

    if name_prefix:
        # Create a temporary directory with standardly-named symlinks
        tmpdir = tempfile.mkdtemp(prefix="10x_mtx_")
        try:
            standard_names = ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]
            # Resolve the source directory to an absolute path so symlinks
            # placed in tmpdir don't try to resolve a relative target from
            # tmpdir's location.
            abs_src_dir = os.path.abspath(src_dir)
            for std_name in standard_names:
                src = os.path.join(abs_src_dir, name_prefix + std_name)
                dst = os.path.join(tmpdir, std_name)
                if os.path.exists(src):
                    # symlink = a pointer to the original file, not a copy
                    os.symlink(src, dst)
                else:
                    # Older 10x format used "genes.tsv.gz" instead of "features.tsv.gz"
                    alt = src.replace("features.tsv.gz", "genes.tsv.gz")
                    if std_name == "features.tsv.gz" and os.path.exists(alt):
                        os.symlink(alt, dst)
                    else:
                        raise FileNotFoundError(f"Expected file not found: {src}")
            adata = sc.read_10x_mtx(tmpdir, var_names="gene_symbols", make_unique=True)
        finally:
            shutil.rmtree(tmpdir)  # always clean up the temp dir, even if loading fails
    else:
        # Files already have standard names — load directly
        adata = sc.read_10x_mtx(src_dir, var_names="gene_symbols", make_unique=True)

    print(f"  AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def load_tar_txt_dge(tar_path: str) -> ad.AnnData:
    """Extract a TAR of per-sample TXT DGE matrices and concatenate into AnnData.

    A TAR file is like a zip — it bundles multiple files together. Here each
    file inside the TAR is a tab-separated DGE matrix for one sample. We extract
    them, load each one, and merge them into a single AnnData.

    Args:
        tar_path: Path to the .tar file containing TXT DGE matrices.

    Returns:
        Concatenated AnnData across all samples, with obs['gsm_id'] set.
    """
    print(f"Loading TAR of TXT DGE matrices: {tar_path}")
    # Extract next to the TAR file (only once — skip if already done)
    extract_dir = tar_path.replace(".tar", "_extracted")

    if not os.path.isdir(extract_dir):
        print(f"  Extracting to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_dir, filter="data")  # filter="data" is a security setting

    # Find all .txt or .txt.gz files inside the extracted folder
    txt_files = sorted(
        glob.glob(os.path.join(extract_dir, "**", "*.txt*"), recursive=True)
    )
    if not txt_files:  # fall back to .tsv extension
        txt_files = sorted(
            glob.glob(os.path.join(extract_dir, "**", "*.tsv*"), recursive=True)
        )
    print(f"  Found {len(txt_files)} DGE file(s)")

    adatas = []
    for fpath in txt_files:
        fname = os.path.basename(fpath)
        # Extract the GSM accession from the filename (e.g. "GSM5032701_DGE.txt.gz" → "GSM5032701")
        gsm_id = fname.split("_")[0] if fname.startswith("GSM") else fname.split(".")[0]
        print(f"  Loading {fname} (GSM: {gsm_id})...")

        try:
            df = pd.read_csv(fpath, sep="\t", index_col=0)  # try tab-separated first
        except Exception:
            df = pd.read_csv(fpath, sep=",", index_col=0)   # fall back to comma-separated

        # Figure out if the file is genes×cells or cells×genes by checking the row names.
        # Cell barcodes are DNA sequences (only A, C, G, T, N), so we use that as a clue.
        first_name = str(df.index[0])
        is_barcode = all(c in "ACGTN-0123456789" for c in first_name.replace("_", ""))
        if is_barcode:
            # Rows are already cells — no transpose needed
            adata_s = ad.AnnData(
                X=sp.csr_matrix(df.values),
                obs=pd.DataFrame(index=df.index),
                var=pd.DataFrame(index=df.columns),
            )
        else:
            # Rows are genes — transpose so cells become rows
            adata_s = ad.AnnData(
                X=sp.csr_matrix(df.values.T),
                obs=pd.DataFrame(index=df.columns),
                var=pd.DataFrame(index=df.index),
            )

        adata_s.obs["gsm_id"] = gsm_id  # tag every cell with its sample accession
        adatas.append(adata_s)
        print(f"    {adata_s.n_obs} cells × {adata_s.n_vars} genes")

    # Stack all samples into one AnnData; label="gsm_key" adds a column recording the source
    adata = ad.concat(adatas, label="gsm_key", keys=[a.obs["gsm_id"].iloc[0] for a in adatas])
    print(f"  Concatenated: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def load_tar_10x(tar_path: str) -> ad.AnnData:
    """Extract a TAR of per-sample 10x directories and concatenate into AnnData.

    Some GEO deposits package each sample's 10x files in its own sub-folder
    inside a TAR archive. We extract them all and load each folder separately.

    Args:
        tar_path: Path to the .tar file containing per-sample 10x directories.

    Returns:
        Concatenated AnnData across all samples, with obs['sample'] set.
    """
    print(f"Loading TAR of 10x directories: {tar_path}")
    extract_dir = tar_path.replace(".tar", "_extracted")

    if not os.path.isdir(extract_dir):
        print(f"  Extracting to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_dir, filter="data")

    # Walk the extracted folder tree; any folder containing matrix.mtx is a 10x sample
    sample_dirs = []
    for root, _, files in os.walk(extract_dir):
        if any(f.endswith(("matrix.mtx", "matrix.mtx.gz")) for f in files):
            sample_dirs.append(root)

    if not sample_dirs:
        raise FileNotFoundError(f"No 10x matrix directories found inside {tar_path}")

    print(f"  Found {len(sample_dirs)} sample directory(ies)")
    adatas = []
    for sd in sorted(sample_dirs):
        sample_name = os.path.basename(sd)
        print(f"  Loading {sample_name}...")
        try:
            a = sc.read_10x_mtx(sd, var_names="gene_symbols", make_unique=True)
            a.obs["sample"] = sample_name  # tag every cell with its sample folder name
            adatas.append(a)
            print(f"    {a.n_obs} cells × {a.n_vars} genes")
        except Exception as exc:
            print(f"    Skipping {sample_name}: {exc}")

    adata = ad.concat(
        adatas,
        label="sample_key",
        keys=[a.obs["sample"].iloc[0] for a in adatas],
    )
    print(f"  Concatenated: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


# ── Sample metadata assignment ────────────────────────────────────────────────

def assign_sample_metadata(
    adata: ad.AnnData,
    samples: list[SampleConfig],
) -> ad.AnnData:
    """Map per-sample metadata onto obs columns.

    After loading, cells don't know which patient or condition they came from.
    This function tags each cell with its sample ID and condition using one of
    three strategies:
      - barcode_prefix: the part before ":" in "SAMPLE:INDEX" barcodes (e.g. "P03")
      - barcode_suffix: the number at the end of a 10x barcode (e.g. "-1")
      - gsm_id: the GSM accession already stored in obs['gsm_id']

    Args:
        adata: AnnData to annotate.
        samples: List of SampleConfig entries from the dataset YAML.

    Returns:
        The annotated AnnData.
    """
    if not samples:
        print("  No sample configs provided — skipping sample metadata assignment.")
        return adata

    n = adata.n_obs
    sample_arr    = np.full(n, "unknown", dtype=object)
    condition_arr = np.full(n, "unknown", dtype=object)

    use_suffix = any(s.barcode_suffix is not None for s in samples)
    use_prefix = any(s.barcode_prefix is not None for s in samples)
    use_gsm    = any(s.gsm_id is not None for s in samples)

    if use_prefix:
        prefix_map = {s.barcode_prefix: s for s in samples if s.barcode_prefix is not None}
        obs_prefixes = np.array([
            bc.split(":")[0] if ":" in bc else "" for bc in adata.obs_names
        ])
        for prefix, s in prefix_map.items():
            mask = obs_prefixes == prefix
            sample_arr[mask]    = s.id
            condition_arr[mask] = s.condition

    elif use_suffix:
        suffix_map = {s.barcode_suffix: s for s in samples if s.barcode_suffix is not None}
        obs_suffixes = np.array([
            bc.split("-")[-1] if "-" in bc else "" for bc in adata.obs_names
        ])
        for suffix, s in suffix_map.items():
            mask = obs_suffixes == suffix
            sample_arr[mask]    = s.id
            condition_arr[mask] = s.condition

    elif use_gsm and "gsm_id" in adata.obs.columns:
        gsm_map = {s.gsm_id: s for s in samples if s.gsm_id is not None}
        obs_gsm = adata.obs["gsm_id"].to_numpy(dtype=str)
        for gsm_id_val, s in gsm_map.items():
            mask = obs_gsm == gsm_id_val
            sample_arr[mask]    = s.id
            condition_arr[mask] = s.condition

    adata.obs["sample"]    = sample_arr
    adata.obs["condition"] = condition_arr

    print("  Sample assignment summary:")
    for val, count in adata.obs["sample"].value_counts().items():
        print(f"    {val}: {count} cells")

    return adata


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_adata(adata: ad.AnnData, config: PreprocessingConfig, skip_scale: bool = False, skip_hvg: bool = False, skip_embeddings: bool = False) -> ad.AnnData:
    """Run standard scanpy QC and preprocessing pipeline.

    Raw scRNA-seq counts need several cleaning and transformation steps before
    they can be used for machine learning or visualisation:
      1. Remove low-quality cells and rarely-detected genes
      2. Filter out dying cells (high mitochondrial gene expression)
      3. Normalise so all cells are comparable regardless of sequencing depth
      4. Log-transform to compress the dynamic range
      5. Select only the most informative genes (highly variable genes)
      6. Reduce dimensions with PCA, then build a UMAP for visualisation
      7. Cluster cells with the Leiden algorithm

    Args:
        adata: Input AnnData (cells × genes, raw counts).
        config: PreprocessingConfig with QC thresholds and HVG/PCA params.

    Returns:
        Preprocessed AnnData (HVG-subset, with UMAP and Leiden in obs/obsm).
    """
    print(f"Preprocessing: {adata.n_obs} cells × {adata.n_vars} genes")

    # Step 1 — make sure every cell and gene has a unique name
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # Step 2 — remove very sparse cells and genes (likely low-quality or noise)
    sc.pp.filter_cells(adata, min_genes=config.min_genes)  # drop cells with too few genes detected
    sc.pp.filter_genes(adata, min_cells=config.min_cells)  # drop genes seen in very few cells
    print(f"  After basic filtering: {adata.n_obs} cells × {adata.n_vars} genes")

    # Step 3 — calculate what percentage of each cell's counts are mitochondrial.
    # Dying or damaged cells leak cytoplasmic RNA, leaving only mitochondrial RNA behind.
    # Human mitochondrial genes are named "MT-..." so we flag them here.
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    print(
        f"  MT%: mean={adata.obs['pct_counts_mt'].mean():.1f}, "
        f"max={adata.obs['pct_counts_mt'].max():.1f}"
    )

    # Remove cells with too high MT% — they are likely dying
    n_before = adata.n_obs
    adata = adata[adata.obs["pct_counts_mt"] < config.mt_pct_threshold, :].copy()
    print(
        f"  After MT filter (<{config.mt_pct_threshold}%): "
        f"removed {n_before - adata.n_obs} cells, {adata.n_obs} remaining"
    )

    # Step 4 — normalise: scale every cell so its total counts sum to 10,000.
    # This removes the effect of sequencing depth (some cells were just sequenced more).
    sc.pp.normalize_total(adata, target_sum=1e4)
    # Log-transform: log(x+1) compresses the large range of counts (0 to thousands)
    # so that highly-expressed genes don't dominate the analysis.
    sc.pp.log1p(adata)

    # Step 5 — save the normalised counts before further transformation.
    # This "raw" slot is used later for differential expression analysis.
    adata.raw = adata

    # Step 6 — select the most variable genes (those that differ most across cells).
    # These carry the most biological information for distinguishing cell types.
    # When skip_hvg=True (used for multi-dataset merging), we keep ALL genes so
    # the intersection across datasets retains maximum information for batch
    # correction and downstream analysis.
    if not skip_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=config.n_top_genes)

        # Force-include housekeeping genes even though they have low variance.
        # These are needed for batch effect detection and correction downstream.
        from scribe.batch import DEFAULT_HOUSEKEEPING_GENES
        hk_in_data = [g for g in DEFAULT_HOUSEKEEPING_GENES if g in adata.var_names]
        for gene in hk_in_data:
            adata.var.loc[gene, "highly_variable"] = True
        n_hk_added = len(hk_in_data)

        print(f"  Highly variable genes: {adata.var['highly_variable'].sum()} "
              f"({n_hk_added} housekeeping genes force-included)")
        adata = adata[:, adata.var.highly_variable].copy()  # keep HVGs + housekeeping
    else:
        print(f"  Skipping HVG selection — keeping all {adata.n_vars} genes")

    # Step 7 — scale genes so they all have mean 0 and similar variance.
    # max_value=10 clips extreme outliers.
    # When skip_scale=True (used for multi-dataset merging), we defer scaling
    # to after concatenation so cross-dataset expression differences are preserved
    # for batch effect detection and correction.
    if not skip_scale:
        sc.pp.scale(adata, max_value=10)

    # When skip_embeddings=True (used for multi-dataset build pipeline), we skip
    # PCA/UMAP/Leiden since merge_datasets() recomputes them on the combined data.
    # This saves time and avoids running PCA on the full unfiltered gene set.
    if not skip_embeddings:
        # PCA (Principal Component Analysis): compress thousands of gene dimensions
        # into 30 principal components that capture most of the variation.
        sc.tl.pca(adata)

        # Build a "neighborhood graph": connect each cell to its most similar neighbours
        # in PCA space. Downstream clustering and UMAP use this graph.
        sc.pp.neighbors(adata, n_pcs=config.n_pcs)

        # UMAP: project the high-dimensional data into 2D for visualisation.
        # Cells that are similar biologically end up close together on the plot.
        sc.tl.umap(adata)

        # Leiden clustering: group cells into clusters based on the neighborhood graph.
        # Higher resolution = more, smaller clusters.
        sc.tl.leiden(adata, resolution=config.leiden_resolution)

    if not skip_embeddings:
        print(
            f"  Output: {adata.n_obs} cells × {adata.n_vars} genes, "
            f"{adata.obs['leiden'].nunique()} Leiden clusters"
        )
    else:
        print(f"  Output: {adata.n_obs} cells × {adata.n_vars} genes (embeddings skipped)")
    return adata




# ── Orchestrator ──────────────────────────────────────────────────────────────

def convert_dataset(config: DatasetConfig, output_dir: str, skip_scale: bool = False, skip_hvg: bool = False, skip_embeddings: bool = False) -> str:
    """Load, preprocess, and save a GEO dataset as .h5ad.

    Runs all steps in order and saves the result as an HDF5-backed AnnData file.

    Args:
        config: Populated DatasetConfig from a dataset YAML file.
        output_dir: Directory to write the processed .h5ad file.

    Returns:
        Path to the written .h5ad file.

    Raises:
        ValueError: If the file format is unrecognised.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = config.source.base_path  # root directory where raw files live

    print(f"\n{'='*60}")
    print(f"Converting {config.id}: {config.title}")
    print(f"{'='*60}")

    # Step 1 — load raw data using the correct loader for this file format
    file_cfg: FileConfig = config.files[0]
    abs_path = os.path.join(base, file_cfg.relative_path)  # full path to the data file

    if file_cfg.format == "csv_dge":
        adata = load_csv_dge(abs_path)
    elif file_cfg.format == "10x_mtx":
        adata = load_10x_mtx(abs_path, name_prefix=file_cfg.name_prefix)
    elif file_cfg.format == "tar_txt_dge":
        adata = load_tar_txt_dge(abs_path)
    elif file_cfg.format == "tar_10x":
        adata = load_tar_10x(abs_path)
    else:
        raise ValueError(f"Unknown file format: {file_cfg.format!r}")

    # Step 2 — label each cell with its sample and condition
    adata = assign_sample_metadata(adata, config.samples)

    # Step 3 — run QC, normalisation, HVG selection, PCA, UMAP, and clustering
    adata = preprocess_adata(adata, config.preprocessing, skip_scale=skip_scale, skip_hvg=skip_hvg, skip_embeddings=skip_embeddings)

    # Step 4 — write the processed AnnData to disk as an .h5ad file
    out_path = os.path.join(output_dir, f"{config.id}_processed.h5ad")
    adata.write_h5ad(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved → {out_path} ({size_mb:.1f} MB)")

    return out_path
