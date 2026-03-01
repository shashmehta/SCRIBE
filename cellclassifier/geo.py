"""GEO dataset loading and conversion to cellxGene-compliant AnnData."""

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

from cellclassifier.config import (
    CellxGeneConfig,
    DatasetConfig,
    FileConfig,
    PreprocessingConfig,
    SampleConfig,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_csv_dge(path: str) -> ad.AnnData:
    """Load a gzipped CSV DGE matrix (genes × cells) and return AnnData (cells × genes).

    Args:
        path: Path to the .csv or .csv.gz file.

    Returns:
        AnnData with cells as observations and genes as variables.
    """
    print(f"Loading CSV DGE: {path}")
    dge = pd.read_csv(path, index_col=0)
    print(f"  Raw shape (genes × cells): {dge.shape}")

    adata = ad.AnnData(
        X=sp.csr_matrix(dge.values.T),
        obs=pd.DataFrame(index=dge.columns),
        var=pd.DataFrame(index=dge.index),
    )
    print(f"  AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def load_10x_mtx(src_dir: str, name_prefix: str = "") -> ad.AnnData:
    """Load a 10x MTX dataset, remapping GSE-prefixed filenames if needed.

    If name_prefix is set (e.g. "GSE162708_"), the three component files
    are symlinked under standard names (barcodes.tsv.gz, features.tsv.gz,
    matrix.mtx.gz) in a temp directory before loading.

    Args:
        src_dir: Directory containing the 10x files.
        name_prefix: Prefix to strip from filenames to get standard 10x names.

    Returns:
        AnnData with cells as observations and genes as variables.
    """
    print(f"Loading 10x MTX: {src_dir}")

    if name_prefix:
        tmpdir = tempfile.mkdtemp(prefix="10x_mtx_")
        try:
            standard_names = ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]
            for std_name in standard_names:
                src = os.path.join(src_dir, name_prefix + std_name)
                dst = os.path.join(tmpdir, std_name)
                if os.path.exists(src):
                    os.symlink(src, dst)
                else:
                    # Try genes.tsv.gz as fallback for older 10x format
                    alt = src.replace("features.tsv.gz", "genes.tsv.gz")
                    if std_name == "features.tsv.gz" and os.path.exists(alt):
                        os.symlink(alt, dst)
                    else:
                        raise FileNotFoundError(f"Expected file not found: {src}")
            adata = sc.read_10x_mtx(tmpdir, var_names="gene_symbols", make_unique=True)
        finally:
            shutil.rmtree(tmpdir)
    else:
        adata = sc.read_10x_mtx(src_dir, var_names="gene_symbols", make_unique=True)

    print(f"  AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def load_tar_txt_dge(tar_path: str) -> ad.AnnData:
    """Extract a TAR of per-sample TXT DGE matrices and concatenate into AnnData.

    Each TXT file is expected to be tab-separated with genes as rows and cells
    as columns (standard DGE convention). The GSM accession is extracted from
    the filename prefix and stored in obs['gsm_id'].

    Args:
        tar_path: Path to the .tar file containing TXT DGE matrices.

    Returns:
        Concatenated AnnData across all samples, with obs['gsm_id'] set.
    """
    print(f"Loading TAR of TXT DGE matrices: {tar_path}")
    extract_dir = tar_path.replace(".tar", "_extracted")

    if not os.path.isdir(extract_dir):
        print(f"  Extracting to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_dir, filter="data")

    txt_files = sorted(
        glob.glob(os.path.join(extract_dir, "**", "*.txt*"), recursive=True)
    )
    if not txt_files:
        txt_files = sorted(
            glob.glob(os.path.join(extract_dir, "**", "*.tsv*"), recursive=True)
        )
    print(f"  Found {len(txt_files)} DGE file(s)")

    adatas = []
    for fpath in txt_files:
        fname = os.path.basename(fpath)
        gsm_id = fname.split("_")[0] if fname.startswith("GSM") else fname.split(".")[0]
        print(f"  Loading {fname} (GSM: {gsm_id})...")

        try:
            df = pd.read_csv(fpath, sep="\t", index_col=0)
        except Exception:
            df = pd.read_csv(fpath, sep=",", index_col=0)

        # Determine orientation: if first row name looks like a barcode, already cells × genes
        first_name = str(df.index[0])
        is_barcode = all(c in "ACGTN-0123456789" for c in first_name.replace("_", ""))
        if is_barcode:
            adata_s = ad.AnnData(
                X=sp.csr_matrix(df.values),
                obs=pd.DataFrame(index=df.index),
                var=pd.DataFrame(index=df.columns),
            )
        else:
            adata_s = ad.AnnData(
                X=sp.csr_matrix(df.values.T),
                obs=pd.DataFrame(index=df.columns),
                var=pd.DataFrame(index=df.index),
            )

        adata_s.obs["gsm_id"] = gsm_id
        adatas.append(adata_s)
        print(f"    {adata_s.n_obs} cells × {adata_s.n_vars} genes")

    adata = ad.concat(adatas, label="gsm_key", keys=[a.obs["gsm_id"].iloc[0] for a in adatas])
    print(f"  Concatenated: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def load_tar_10x(tar_path: str) -> ad.AnnData:
    """Extract a TAR of per-sample 10x directories and concatenate into AnnData.

    Each sub-directory inside the TAR that contains a matrix.mtx or matrix.mtx.gz
    is treated as one sample. obs['sample'] is set to the directory name.

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

    # Find all sub-directories that contain a matrix file
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
            a.obs["sample"] = sample_name
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
    dataset_tissue: str,
    dataset_disease: str,
) -> ad.AnnData:
    """Map per-sample metadata onto obs columns.

    Supports two demultiplexing strategies driven by SampleConfig:
      - barcode_suffix: cells whose barcode ends with "-<suffix>" belong to that sample
      - gsm_id: cells whose obs['gsm_id'] matches belong to that sample

    Sets obs columns: sample, condition, tissue_ontology_term_id,
    disease_ontology_term_id, donor_id.

    Args:
        adata: AnnData to annotate (modified in place).
        samples: List of SampleConfig entries from the dataset YAML.
        dataset_tissue: Fallback tissue ontology term from CellxGeneConfig.
        dataset_disease: Fallback disease ontology term from CellxGeneConfig.

    Returns:
        The annotated AnnData.
    """
    if not samples:
        print("  No sample configs provided — skipping sample metadata assignment.")
        return adata

    n = adata.n_obs

    # Use positional numpy arrays to avoid label-indexing issues when obs
    # names are non-unique (common after ad.concat across samples).
    sample_arr    = np.full(n, "unknown",       dtype=object)
    condition_arr = np.full(n, "unknown",       dtype=object)
    tissue_arr    = np.full(n, dataset_tissue,  dtype=object)
    disease_arr   = np.full(n, dataset_disease, dtype=object)

    # Determine strategy from which field is populated in SampleConfig
    use_suffix = any(s.barcode_suffix is not None for s in samples)
    use_gsm    = any(s.gsm_id is not None for s in samples)

    if use_suffix:
        suffix_map = {s.barcode_suffix: s for s in samples if s.barcode_suffix is not None}
        obs_suffixes = np.array([
            bc.split("-")[-1] if "-" in bc else "" for bc in adata.obs_names
        ])
        for suffix, s in suffix_map.items():
            mask = obs_suffixes == suffix
            sample_arr[mask]    = s.id
            condition_arr[mask] = s.condition
            if s.tissue_ontology_term_id:
                tissue_arr[mask]  = s.tissue_ontology_term_id
            if s.disease_ontology_term_id:
                disease_arr[mask] = s.disease_ontology_term_id

    elif use_gsm and "gsm_id" in adata.obs.columns:
        gsm_map = {s.gsm_id: s for s in samples if s.gsm_id is not None}
        obs_gsm = adata.obs["gsm_id"].to_numpy(dtype=str)
        for gsm_id_val, s in gsm_map.items():
            mask = obs_gsm == gsm_id_val
            sample_arr[mask]    = s.id
            condition_arr[mask] = s.condition
            if s.tissue_ontology_term_id:
                tissue_arr[mask]  = s.tissue_ontology_term_id
            if s.disease_ontology_term_id:
                disease_arr[mask] = s.disease_ontology_term_id

    adata.obs["sample"]                   = sample_arr
    adata.obs["condition"]                = condition_arr
    adata.obs["tissue_ontology_term_id"]  = tissue_arr
    adata.obs["disease_ontology_term_id"] = disease_arr
    adata.obs["donor_id"]                 = sample_arr

    print("  Sample assignment summary:")
    for val, count in adata.obs["sample"].value_counts().items():
        print(f"    {val}: {count} cells")

    return adata


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_adata(adata: ad.AnnData, config: PreprocessingConfig) -> ad.AnnData:
    """Run standard scanpy QC and preprocessing pipeline.

    Steps:
      1. Make names unique
      2. Filter cells/genes by minimum counts
      3. Mitochondrial QC and filtering
      4. Normalize total counts and log1p transform
      5. Store raw (for downstream DE)
      6. Highly variable gene selection
      7. Scale → PCA → neighbors → UMAP → Leiden clustering

    Args:
        adata: Input AnnData (cells × genes, raw counts).
        config: PreprocessingConfig with QC thresholds and HVG/PCA params.

    Returns:
        Preprocessed AnnData (HVG-subset, with UMAP and Leiden in obs/obsm).
    """
    print(f"Preprocessing: {adata.n_obs} cells × {adata.n_vars} genes")

    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    sc.pp.filter_cells(adata, min_genes=config.min_genes)
    sc.pp.filter_genes(adata, min_cells=config.min_cells)
    print(f"  After basic filtering: {adata.n_obs} cells × {adata.n_vars} genes")

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    print(
        f"  MT%: mean={adata.obs['pct_counts_mt'].mean():.1f}, "
        f"max={adata.obs['pct_counts_mt'].max():.1f}"
    )

    n_before = adata.n_obs
    adata = adata[adata.obs["pct_counts_mt"] < config.mt_pct_threshold, :].copy()
    print(
        f"  After MT filter (<{config.mt_pct_threshold}%): "
        f"removed {n_before - adata.n_obs} cells, {adata.n_obs} remaining"
    )

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(adata, n_top_genes=config.n_top_genes)
    print(f"  Highly variable genes: {adata.var['highly_variable'].sum()}")
    adata = adata[:, adata.var.highly_variable].copy()

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=config.n_pcs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=config.leiden_resolution)

    print(
        f"  Output: {adata.n_obs} cells × {adata.n_vars} genes, "
        f"{adata.obs['leiden'].nunique()} Leiden clusters"
    )
    return adata


# ── cellxGene annotation & validation ────────────────────────────────────────

def annotate_cellxgene_metadata(
    adata: ad.AnnData, config: CellxGeneConfig, title: str
) -> ad.AnnData:
    """Add required cellxGene schema 5.0.0 fields to obs, var, and uns.

    Per-cell tissue/disease overrides assigned by assign_sample_metadata()
    take precedence over the dataset-level defaults in config.

    Args:
        adata: AnnData to annotate (modified in place).
        config: CellxGeneConfig with ontology term IDs.
        title: Dataset title stored in uns['title'].

    Returns:
        The annotated AnnData.
    """
    # obs fields — only set if not already present (per-sample overrides win)
    if "organism_ontology_term_id" not in adata.obs.columns:
        adata.obs["organism_ontology_term_id"] = config.organism_ontology_term_id
    if "assay_ontology_term_id" not in adata.obs.columns:
        adata.obs["assay_ontology_term_id"] = config.assay_ontology_term_id
    if "tissue_ontology_term_id" not in adata.obs.columns:
        adata.obs["tissue_ontology_term_id"] = config.tissue_ontology_term_id
    if "disease_ontology_term_id" not in adata.obs.columns:
        adata.obs["disease_ontology_term_id"] = config.disease_ontology_term_id

    for col, default in [
        ("cell_type_ontology_term_id", "unknown"),
        ("donor_id", "unknown"),
        ("suspension_type", "cell"),
        ("sex_ontology_term_id", "unknown"),
        ("development_stage_ontology_term_id", "unknown"),
        ("self_reported_ethnicity_ontology_term_id", "unknown"),
    ]:
        if col not in adata.obs.columns:
            adata.obs[col] = default

    if "is_primary_data" not in adata.obs.columns:
        adata.obs["is_primary_data"] = True

    # var fields
    if "feature_is_filtered" not in adata.var.columns:
        adata.var["feature_is_filtered"] = False
    if "feature_name" not in adata.var.columns:
        adata.var["feature_name"] = adata.var_names
    if "feature_biotype" not in adata.var.columns:
        adata.var["feature_biotype"] = "gene"

    # uns fields
    adata.uns["schema_version"] = "5.0.0"
    adata.uns["title"] = title
    if "X_umap" in adata.obsm:
        adata.uns["default_embedding"] = "X_umap"

    return adata


def validate_cellxgene(adata: ad.AnnData, name: str = "dataset") -> bool:
    """Check that all required cellxGene schema 5.0.0 fields are present.

    Args:
        adata: AnnData to validate.
        name: Label used in printed output.

    Returns:
        True if all required fields are present, False otherwise.
    """
    issues = []

    required_obs = [
        "organism_ontology_term_id",
        "tissue_ontology_term_id",
        "assay_ontology_term_id",
        "disease_ontology_term_id",
        "cell_type_ontology_term_id",
        "donor_id",
        "suspension_type",
        "is_primary_data",
        "sex_ontology_term_id",
        "development_stage_ontology_term_id",
        "self_reported_ethnicity_ontology_term_id",
    ]
    for col in required_obs:
        if col not in adata.obs.columns:
            issues.append(f"Missing obs column: {col}")

    for col in ["feature_is_filtered", "feature_name", "feature_biotype"]:
        if col not in adata.var.columns:
            issues.append(f"Missing var column: {col}")

    for key in ["schema_version", "title"]:
        if key not in adata.uns:
            issues.append(f"Missing uns key: {key}")

    if "X_umap" not in adata.obsm:
        issues.append("Missing obsm embedding: X_umap")

    if issues:
        print(f"VALIDATION FAILED for {name}:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print(f"VALIDATION PASSED for {name}: {adata.n_obs} cells × {adata.n_vars} genes")
    return True


# ── Orchestrator ──────────────────────────────────────────────────────────────

def convert_dataset(config: DatasetConfig, output_dir: str) -> str:
    """Load, preprocess, annotate, validate, and save a GEO dataset as .h5ad.

    Args:
        config: Populated DatasetConfig from a dataset YAML file.
        output_dir: Directory to write the processed .h5ad file.

    Returns:
        Path to the written .h5ad file.

    Raises:
        ValueError: If the file format is unrecognised or validation fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = config.source.base_path

    print(f"\n{'='*60}")
    print(f"Converting {config.id}: {config.title}")
    print(f"{'='*60}")

    # Load raw data
    file_cfg: FileConfig = config.files[0]
    abs_path = os.path.join(base, file_cfg.relative_path)

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

    # Assign per-sample metadata
    adata = assign_sample_metadata(
        adata,
        config.samples,
        dataset_tissue=config.cellxgene.tissue_ontology_term_id,
        dataset_disease=config.cellxgene.disease_ontology_term_id,
    )

    # Preprocess
    adata = preprocess_adata(adata, config.preprocessing)

    # Annotate cellxGene fields
    adata = annotate_cellxgene_metadata(adata, config.cellxgene, title=config.title)

    # Validate
    ok = validate_cellxgene(adata, name=config.id)
    if not ok:
        raise ValueError(f"cellxGene validation failed for {config.id}")

    # Write
    out_path = os.path.join(output_dir, f"{config.id}_processed.h5ad")
    adata.write_h5ad(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved → {out_path} ({size_mb:.1f} MB)")

    return out_path
