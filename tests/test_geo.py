"""Tests for cellclassifier/geo.py using synthetic in-memory data.

Key efficiency strategy: preprocess_adata() is the expensive step (runs UMAP,
PCA, Leiden). It is computed ONCE via module-scoped fixtures and shared across
all tests that need it. Tests that mutate the AnnData call .copy() first so
the shared fixture is never modified.
"""

import gzip
import io
import os
import tarfile
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellclassifier.config import (
    CellxGeneConfig,
    DatasetConfig,
    FileConfig,
    PreprocessingConfig,
    SampleConfig,
    SourceConfig,
)
from cellclassifier.geo import (
    annotate_cellxgene_metadata,
    assign_sample_metadata,
    convert_dataset,
    load_csv_dge,
    load_10x_mtx,
    load_tar_txt_dge,
    load_tar_10x,
    preprocess_adata,
    validate_cellxgene,
)

# ── Synthetic data constants and helpers ───────────────────────────────────────
# We use negative binomial distribution because real scRNA-seq counts follow it.
# ~60% zeros mimics the sparsity of actual single-cell data.

N_GENES = 80   # large enough for HVG selection (n_top_genes=50) to work
N_CELLS = 120  # large enough for Leiden clustering and train/test splits
RNG = np.random.default_rng(0)


def _make_count_matrix() -> np.ndarray:
    """Return an N_CELLS × N_GENES array of sparse-ish non-negative integers."""
    raw = RNG.negative_binomial(5, 0.5, size=(N_CELLS, N_GENES)).astype(float)
    raw[RNG.random(raw.shape) < 0.6] = 0  # zero out ~60% to mimic real data
    return raw


def _gene_names() -> list[str]:
    """Return N_GENES names: 78 dummy + 2 mitochondrial (MT-).

    The MT- prefix is what scanpy uses to detect mitochondrial genes during QC.
    """
    return [f"GENE{i:04d}" for i in range(N_GENES - 2)] + ["MT-CO1", "MT-ND1"]


def _barcode_names(suffix: str | None = None) -> list[str]:
    """Return N_CELLS cell barcodes, optionally with a 10x sample suffix (e.g. '-1').

    10x datasets append '-1', '-2', etc. to barcodes to encode which sample
    each cell came from during library preparation.
    """
    barcodes = [f"ACGT{i:04d}TGCA" for i in range(N_CELLS)]
    if suffix:
        barcodes = [f"{b}-{suffix}" for b in barcodes]
    return barcodes


def _make_adata(suffix: str | None = None) -> ad.AnnData:
    """Build a minimal AnnData with synthetic counts."""
    return ad.AnnData(
        X=sp.csr_matrix(_make_count_matrix()),
        obs=pd.DataFrame(index=_barcode_names(suffix)),
        var=pd.DataFrame(index=_gene_names()),
    )


def _minimal_preprocessing_config() -> PreprocessingConfig:
    """Permissive QC thresholds so no synthetic cells are accidentally dropped.

    mt_pct_threshold=100 keeps all cells; n_top_genes=50 and n_pcs=10 keep
    HVG selection and PCA fast on the small synthetic dataset.
    """
    return PreprocessingConfig(
        min_genes=1,
        min_cells=1,
        mt_pct_threshold=100.0,
        n_top_genes=50,
        n_pcs=10,
        leiden_resolution=0.3,
    )


def _write_10x_files(directory: str, prefix: str = "") -> None:
    """Write minimal gzipped 10x MTX files: barcodes, features, matrix.

    GEO deposits often prefix these files with the study ID (e.g. 'GSE162708_').
    Passing prefix= lets us test both the prefixed and standard-name cases.
    """
    X = _make_count_matrix()
    barcodes, genes = _barcode_names(), _gene_names()

    with gzip.open(os.path.join(directory, f"{prefix}barcodes.tsv.gz"), "wt") as f:
        f.write("\n".join(barcodes) + "\n")

    with gzip.open(os.path.join(directory, f"{prefix}features.tsv.gz"), "wt") as f:
        for g in genes:
            f.write(f"{g}\t{g}\tGene Expression\n")

    coo = sp.csr_matrix(X).T.tocoo()  # MTX format stores genes × cells
    with gzip.open(os.path.join(directory, f"{prefix}matrix.mtx.gz"), "wt") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n%\n")
        f.write(f"{len(genes)} {len(barcodes)} {coo.nnz}\n")
        for r, c, v in zip(coo.row + 1, coo.col + 1, coo.data):
            f.write(f"{r} {c} {int(v)}\n")


# ── Module-scoped fixtures ─────────────────────────────────────────────────────
# These run ONCE for the entire test session. Since preprocess_adata() runs
# scanpy's full pipeline (PCA, UMAP, Leiden), reusing one result across all
# tests that need it cuts runtime significantly.

@pytest.fixture(scope="module")
def csv_dge_path(tmp_path_factory):
    """A gzipped CSV DGE file (genes × cells) written once for all CSV loader tests."""
    tmp = tmp_path_factory.mktemp("csv_dge")
    X = _make_count_matrix()
    df = pd.DataFrame(X.T, index=_gene_names(), columns=_barcode_names())
    path = tmp / "dge.csv.gz"
    df.to_csv(str(path))
    return str(path)


@pytest.fixture(scope="module")
def mtx_dir(tmp_path_factory):
    """A directory with standard (un-prefixed) 10x MTX files written once."""
    tmp = tmp_path_factory.mktemp("10x_std")
    _write_10x_files(str(tmp))
    return str(tmp)


@pytest.fixture(scope="module")
def preprocessed_adata():
    """A fully preprocessed AnnData (UMAP, PCA, Leiden) computed once.

    Shared by TestPreprocessAdata, TestAnnotateCellxgeneMetadata, and
    TestValidateCellxgene. Tests that need to mutate it must call .copy().
    """
    return preprocess_adata(_make_adata(), _minimal_preprocessing_config())


@pytest.fixture(scope="module")
def annotated_adata(preprocessed_adata):
    """Preprocessed AnnData with cellxGene schema metadata applied, computed once.

    Shared by three TestAnnotateCellxgeneMetadata tests and all four
    TestValidateCellxgene tests. Tests that delete fields must call .copy().
    """
    adata = preprocessed_adata.copy()
    annotate_cellxgene_metadata(adata, CellxGeneConfig(), title="Test")
    return adata


# ── load_csv_dge ─────────────────────────────────────────────────────────────

class TestLoadCsvDge:
    """load_csv_dge reads a gzipped CSV (genes × cells) and returns AnnData (cells × genes).

    The loader must transpose the matrix and store it as a sparse matrix.
    All three tests share a single CSV file from the csv_dge_path fixture.
    """

    def test_shape(self, csv_dge_path):
        """After transposing, AnnData should be N_CELLS rows × N_GENES columns."""
        adata = load_csv_dge(csv_dge_path)
        assert adata.n_obs == N_CELLS
        assert adata.n_vars == N_GENES

    def test_gene_and_barcode_names(self, csv_dge_path):
        """obs names = barcodes (rows), var names = genes (columns)."""
        adata = load_csv_dge(csv_dge_path)
        assert list(adata.var_names) == _gene_names()
        assert list(adata.obs_names) == _barcode_names()

    def test_x_is_sparse(self, csv_dge_path):
        """X must be stored as a sparse matrix — scRNA-seq data is ~90% zeros,
        so dense storage wastes huge amounts of memory."""
        adata = load_csv_dge(csv_dge_path)
        assert sp.issparse(adata.X)


# ── load_10x_mtx ─────────────────────────────────────────────────────────────

class TestLoad10xMtx:
    """load_10x_mtx reads 10x barcodes / features / matrix files.

    GEO deposits often add a dataset prefix to filenames. The loader creates
    temporary symlinks with standard names so scanpy can read them normally.
    """

    def test_standard_names(self, mtx_dir):
        """Files with the standard names (no prefix) should load directly."""
        adata = load_10x_mtx(mtx_dir)
        assert adata.n_obs == N_CELLS
        assert adata.n_vars == N_GENES

    def test_prefixed_names(self, tmp_path):
        """GSE-prefixed filenames should be remapped via symlinks before loading."""
        _write_10x_files(str(tmp_path), prefix="GSE999999_")
        adata = load_10x_mtx(str(tmp_path), name_prefix="GSE999999_")
        assert adata.n_obs == N_CELLS
        assert adata.n_vars == N_GENES

    def test_missing_file_raises(self, tmp_path):
        """If expected 10x files are absent, a clear FileNotFoundError should be raised
        rather than a confusing internal scanpy error."""
        with gzip.open(str(tmp_path / "GSE999999_barcodes.tsv.gz"), "wt") as f:
            f.write("BARCODE1\n")  # only barcodes — features and matrix are missing
        with pytest.raises(FileNotFoundError):
            load_10x_mtx(str(tmp_path), name_prefix="GSE999999_")


# ── load_tar_txt_dge ─────────────────────────────────────────────────────────

def _write_tar_txt_dge(tar_path: str, n_samples: int = 2) -> list[str]:
    """Create a TAR archive of n_samples gzipped TSV DGE files.

    Each file is named '<GSM_ID>_DGE.txt.gz', matching the GEO naming convention.
    Returns the list of GSM IDs used so tests can assert on obs['gsm_id'].
    """
    gsm_ids = [f"GSM{9000000 + i}" for i in range(n_samples)]
    with tarfile.open(tar_path, "w") as tar:
        for gsm in gsm_ids:
            X = _make_count_matrix()
            df = pd.DataFrame(X.T, index=_gene_names(), columns=_barcode_names())
            buf = io.BytesIO()
            with gzip.open(buf, "wt") as gz:
                df.to_csv(gz, sep="\t")
            buf.seek(0)
            info = tarfile.TarInfo(name=f"{gsm}_DGE.txt.gz")
            info.size = len(buf.getvalue())
            tar.addfile(info, buf)
    return gsm_ids


class TestLoadTarTxtDge:
    """load_tar_txt_dge extracts a TAR of per-sample TSV DGE files and concatenates them.

    Each sample is a separate gzipped TSV file named by its GSM accession.
    After loading, all samples are stacked and each cell is tagged with its
    GSM ID in obs['gsm_id'] so we know which patient it came from.
    """

    def test_shape_and_gsm_col(self, tmp_path):
        """Total cells = N_CELLS × n_samples; every cell must have a gsm_id tag."""
        tar_path = str(tmp_path / "GSE_RAW.tar")
        gsm_ids = _write_tar_txt_dge(tar_path, n_samples=2)
        adata = load_tar_txt_dge(tar_path)
        assert adata.n_obs == N_CELLS * 2
        assert adata.n_vars == N_GENES
        assert "gsm_id" in adata.obs.columns
        assert set(adata.obs["gsm_id"].unique()) == set(gsm_ids)

    def test_extraction_idempotent(self, tmp_path):
        """Calling the loader twice must not re-extract the TAR or raise an error.

        The loader caches the extracted directory next to the TAR file, so large
        datasets are only decompressed once regardless of how many times the
        function is called.
        """
        tar_path = str(tmp_path / "GSE_RAW.tar")
        _write_tar_txt_dge(tar_path, n_samples=1)
        load_tar_txt_dge(tar_path)          # first call — extracts
        adata = load_tar_txt_dge(tar_path)  # second call — reuses extracted dir
        assert adata.n_obs == N_CELLS


# ── load_tar_10x ─────────────────────────────────────────────────────────────

def _write_tar_10x(tar_path: str, n_samples: int = 2) -> list[str]:
    """Create a TAR archive of per-sample 10x directories.

    Each directory inside the TAR contains the standard barcodes/features/matrix
    files. Returns the sample directory names used so tests can assert on obs['sample'].
    """
    sample_names = [f"Sample{i + 1}" for i in range(n_samples)]
    with tempfile.TemporaryDirectory() as staging:
        for name in sample_names:
            sample_dir = os.path.join(staging, name)
            os.makedirs(sample_dir)
            _write_10x_files(sample_dir)
        with tarfile.open(tar_path, "w") as tar:
            for name in sample_names:
                tar.add(os.path.join(staging, name), arcname=name)
    return sample_names


class TestLoadTar10x:
    """load_tar_10x extracts a TAR of per-sample 10x directories.

    Useful when each sample's 10x files are bundled in their own sub-directory
    inside a single TAR archive. Each sample directory becomes one AnnData,
    then they are all concatenated.
    """

    def test_shape_and_sample_col(self, tmp_path):
        """Total cells = N_CELLS × n_samples; every cell must have a sample tag."""
        tar_path = str(tmp_path / "RAW.tar")
        sample_names = _write_tar_10x(tar_path, n_samples=2)
        adata = load_tar_10x(tar_path)
        assert adata.n_obs == N_CELLS * 2
        assert adata.n_vars == N_GENES
        assert "sample" in adata.obs.columns
        assert set(adata.obs["sample"].unique()) == set(sample_names)

    def test_no_matrix_raises(self, tmp_path):
        """An empty TAR (no 10x directories inside) should raise FileNotFoundError
        rather than returning an empty AnnData silently."""
        tar_path = str(tmp_path / "empty.tar")
        with tarfile.open(tar_path, "w"):
            pass  # create an empty archive
        with pytest.raises(FileNotFoundError):
            load_tar_10x(tar_path)


# ── assign_sample_metadata ───────────────────────────────────────────────────

class TestAssignSampleMetadata:
    """assign_sample_metadata tags each cell with its sample ID, condition,
    tissue, and disease ontology ID.

    Two demultiplexing strategies are supported:
    - barcode_suffix: reads the number after '-' in a 10x barcode (e.g. '-1' = sample 1)
    - gsm_id: reads the GSM accession stored in obs['gsm_id'] during loading
    """

    def test_barcode_suffix_strategy(self):
        """Cells with '-1' barcodes → TumorA; cells with '-2' barcodes → Normal.

        This is the 10x demultiplexing strategy: sample identity is encoded in
        the number appended to each cell barcode during library preparation.
        Per-sample disease ontology overrides must also be applied correctly.
        """
        adata = ad.concat([_make_adata(suffix="1"), _make_adata(suffix="2")])
        samples = [
            SampleConfig(id="TumorA", condition="tumor",  barcode_suffix="1",
                         tissue_ontology_term_id="UBERON:0001264",
                         disease_ontology_term_id="MONDO:0006047"),
            SampleConfig(id="Normal", condition="normal", barcode_suffix="2",
                         tissue_ontology_term_id="UBERON:0001264",
                         disease_ontology_term_id="PATO:0000461"),
        ]
        adata = assign_sample_metadata(adata, samples,
                                       dataset_tissue="UBERON:0001264",
                                       dataset_disease="unknown")

        assert set(adata.obs["sample"].unique()) == {"TumorA", "Normal"}
        assert set(adata.obs["condition"].unique()) == {"tumor", "normal"}
        # Per-sample disease overrides should replace the dataset-level default
        assert set(adata.obs["disease_ontology_term_id"].unique()) == {
            "MONDO:0006047", "PATO:0000461"
        }
        # donor_id mirrors sample — one patient per sample in these datasets
        assert (adata.obs["donor_id"] == adata.obs["sample"]).all()

    def test_gsm_id_strategy(self):
        """Cells tagged with 'GSM100' → normal; cells tagged with 'GSM200' → tumor.

        This strategy is used for TAR-based datasets where each file is named
        by its GSM accession and obs['gsm_id'] is set during loading.
        """
        adata = _make_adata()
        adata.obs["gsm_id"] = ["GSM100" if i < N_CELLS // 2 else "GSM200"
                                for i in range(N_CELLS)]
        samples = [
            SampleConfig(id="normal", condition="normal", gsm_id="GSM100"),
            SampleConfig(id="tumor",  condition="tumor",  gsm_id="GSM200"),
        ]
        adata = assign_sample_metadata(adata, samples,
                                       dataset_tissue="UBERON:0001264",
                                       dataset_disease="unknown")
        assert set(adata.obs["sample"].unique()) == {"normal", "tumor"}

    def test_no_samples_is_noop(self):
        """Passing an empty samples list should leave obs completely unchanged.

        Datasets without per-sample metadata configs skip assignment entirely
        rather than adding empty or 'unknown' columns.
        """
        adata = _make_adata()
        original_cols = set(adata.obs.columns)
        result = assign_sample_metadata(adata, [], dataset_tissue="UBERON:0001264",
                                        dataset_disease="unknown")
        assert set(result.obs.columns) == original_cols

    def test_unmatched_barcodes_get_unknown(self):
        """Barcodes that don't match any configured suffix should stay 'unknown'.

        This prevents silent data loss — unmatched cells are flagged rather than
        silently dropped or mislabelled with a wrong sample's condition.
        """
        adata = _make_adata(suffix="9")  # suffix '9' not in the samples list
        samples = [SampleConfig(id="A", condition="normal", barcode_suffix="1")]
        adata = assign_sample_metadata(adata, samples, dataset_tissue="UBERON:0001264",
                                       dataset_disease="unknown")
        assert (adata.obs["sample"] == "unknown").all()


# ── preprocess_adata ──────────────────────────────────────────────────────────

class TestPreprocessAdata:
    """preprocess_adata runs the full scanpy QC and preprocessing pipeline:
    filter → MT QC → normalize → log1p → HVG → scale → PCA → UMAP → Leiden.

    Three tests share the module-scoped preprocessed_adata fixture (same data,
    same config, computed once). The MT filter test needs its own data because
    it artificially inflates MT expression to trigger filtering.
    """

    def test_output_has_umap_and_leiden(self, preprocessed_adata):
        """After the pipeline, obsm must contain X_umap and obs must contain leiden.

        X_umap is the 2D embedding used for visualisation; leiden is the cluster
        label assigned to each cell by the graph-based clustering algorithm.
        """
        assert "X_umap"  in preprocessed_adata.obsm
        assert "X_pca"   in preprocessed_adata.obsm
        assert "leiden"  in preprocessed_adata.obs.columns

    def test_hvg_subset(self, preprocessed_adata):
        """The output should contain exactly n_top_genes gene columns.

        HVG selection keeps only the most variable genes, reducing dimensionality
        from all genes to the most informative ones for downstream ML.
        """
        assert preprocessed_adata.n_vars == _minimal_preprocessing_config().n_top_genes

    def test_raw_is_stored(self, preprocessed_adata):
        """adata.raw must hold the normalised (pre-HVG) counts.

        Storing raw enables differential expression analysis on ALL genes even
        after the var matrix has been subset to the HVG list.
        """
        assert preprocessed_adata.raw is not None

    def test_mt_filter_removes_cells(self):
        """Cells with disproportionately high mitochondrial expression are removed.

        Dying or damaged cells lose their cytoplasmic RNA but retain mitochondrial RNA,
        resulting in unusually high MT%. We force 10 cells to have near-100% MT counts
        and confirm they are dropped. This test uses its own data because it
        modifies the count matrix in a way that would corrupt the shared fixture.
        """
        adata = _make_adata()
        mt_idx = [i for i, g in enumerate(_gene_names()) if g.startswith("MT-")]
        X = adata.X.toarray()
        X[:10, mt_idx] = 9999  # make the first 10 cells look like dying/damaged cells
        adata.X = sp.csr_matrix(X)

        cfg = _minimal_preprocessing_config()
        cfg.mt_pct_threshold = 50.0  # drop cells where >50% of counts are MT

        result = preprocess_adata(adata, cfg)
        assert result.n_obs < N_CELLS  # at least some cells were filtered out


# ── annotate_cellxgene_metadata ───────────────────────────────────────────────

class TestAnnotateCellxgeneMetadata:
    """annotate_cellxgene_metadata adds metadata fields required by cellxGene schema 5.0.0.

    Three tests read from the shared annotated_adata fixture (no mutations needed).
    The override test uses a fresh copy of preprocessed_adata because it must
    set a column BEFORE annotation runs.
    """

    def test_required_obs_fields_added(self, annotated_adata):
        """All 11 required per-cell (obs) columns must be present after annotation."""
        required = [
            "organism_ontology_term_id", "tissue_ontology_term_id",
            "assay_ontology_term_id", "disease_ontology_term_id",
            "cell_type_ontology_term_id", "donor_id", "suspension_type",
            "is_primary_data", "sex_ontology_term_id",
            "development_stage_ontology_term_id",
            "self_reported_ethnicity_ontology_term_id",
        ]
        for col in required:
            assert col in annotated_adata.obs.columns, f"Missing obs column: {col}"

    def test_required_var_fields_added(self, annotated_adata):
        """Required per-gene (var) columns must be present.

        These fields tell cellxGene what type of feature each column represents
        (e.g. 'gene' biotype, whether it was manually filtered out).
        """
        for col in ["feature_is_filtered", "feature_name", "feature_biotype"]:
            assert col in annotated_adata.var.columns, f"Missing var column: {col}"

    def test_uns_fields(self, annotated_adata):
        """Dataset-level uns keys must be set to the correct values.

        schema_version tells cellxGene which validation rules to apply.
        default_embedding tells it which 2D plot to show by default.
        """
        assert annotated_adata.uns["schema_version"] == "5.0.0"
        assert annotated_adata.uns["title"] == "Test"
        assert annotated_adata.uns["default_embedding"] == "X_umap"

    def test_per_cell_overrides_not_clobbered(self, preprocessed_adata):
        """Values already set in obs (e.g. by assign_sample_metadata) must not be overwritten.

        annotate_cellxgene_metadata only fills columns that don't yet exist,
        so per-sample tissue/disease values set during loading are preserved.
        This test sets a column BEFORE calling annotate, then verifies it survives.
        """
        adata = preprocessed_adata.copy()
        adata.obs["tissue_ontology_term_id"] = "UBERON:0002107"  # liver — pre-set
        annotate_cellxgene_metadata(adata, CellxGeneConfig(), title="Test")
        # annotate must not overwrite the pre-set value with the config default (pancreas)
        assert (adata.obs["tissue_ontology_term_id"] == "UBERON:0002107").all()


# ── validate_cellxgene ────────────────────────────────────────────────────────

class TestValidateCellxgene:
    """validate_cellxgene checks that all required cellxGene schema fields are present.

    All four tests start from annotated_adata.copy() and delete one field to
    confirm the validator detects each specific missing piece. Using .copy()
    ensures deletions in one test do not affect other tests.
    """

    def test_passes_on_complete_adata(self, annotated_adata):
        """A fully annotated AnnData must pass validation without errors."""
        assert validate_cellxgene(annotated_adata.copy(), "test") is True

    def test_fails_on_missing_obs_column(self, annotated_adata):
        """Removing any required obs column should cause validation to fail.

        We remove donor_id as a representative required field.
        """
        adata = annotated_adata.copy()
        del adata.obs["donor_id"]
        assert validate_cellxgene(adata, "test") is False

    def test_fails_on_missing_umap(self, annotated_adata):
        """Without X_umap, cellxGene cannot display the scatter plot — must fail.

        The UMAP embedding is mandatory because it is the primary visualisation
        in the cellxGene browser.
        """
        adata = annotated_adata.copy()
        del adata.obsm["X_umap"]
        assert validate_cellxgene(adata, "test") is False

    def test_fails_on_missing_uns_key(self, annotated_adata):
        """Removing a required uns key such as 'title' should fail validation."""
        adata = annotated_adata.copy()
        del adata.uns["title"]
        assert validate_cellxgene(adata, "test") is False


# ── convert_dataset (end-to-end) ─────────────────────────────────────────────

class TestConvertDataset:
    """convert_dataset orchestrates the full pipeline: load → assign → preprocess → annotate → validate → save.

    Each test exercises a different input format. They write to separate tmp_path
    directories so they are fully independent and can run in any order.
    """

    def _base_config(self, source_base: str, file_cfg: FileConfig,
                     samples: list[SampleConfig] | None = None) -> DatasetConfig:
        """Minimal DatasetConfig pointing at synthetic test data."""
        return DatasetConfig(
            id="GSE_TEST",
            title="Test Dataset",
            description="Synthetic test dataset",
            source=SourceConfig(type="local", base_path=source_base),
            files=[file_cfg],
            preprocessing=_minimal_preprocessing_config(),
            cellxgene=CellxGeneConfig(),
            samples=samples or [],
        )

    def test_csv_dge_end_to_end(self, tmp_path):
        """CSV DGE format: file is converted, validated, and saved as .h5ad.

        Also verifies the output filename follows '{dataset_id}_processed.h5ad'
        — both properties are checked in one run to avoid running convert twice.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "test_dge.csv.gz"
        X = _make_count_matrix()
        pd.DataFrame(X.T, index=_gene_names(), columns=_barcode_names()).to_csv(str(csv_path))

        file_cfg = FileConfig(format="csv_dge", relative_path="data/test_dge.csv.gz")
        cfg = self._base_config(str(tmp_path), file_cfg)
        out_path = convert_dataset(cfg, str(tmp_path / "output"))

        assert os.path.exists(out_path)
        result = ad.read_h5ad(out_path)
        assert result.n_obs > 0
        assert validate_cellxgene(result, "csv_dge") is True
        # Filename must follow the standard convention so downstream scripts can find it
        assert os.path.basename(out_path) == "GSE_TEST_processed.h5ad"

    def test_10x_mtx_end_to_end(self, tmp_path):
        """10x MTX format: GSE-prefixed files are remapped and loaded correctly."""
        src_dir = tmp_path / "GSE_TEST"
        src_dir.mkdir()
        _write_10x_files(str(src_dir), prefix="GSE_TEST_")

        file_cfg = FileConfig(format="10x_mtx", relative_path="GSE_TEST",
                              name_prefix="GSE_TEST_")
        cfg = self._base_config(str(tmp_path), file_cfg)
        out_path = convert_dataset(cfg, str(tmp_path / "output"))

        result = ad.read_h5ad(out_path)
        assert result.n_obs > 0
        assert validate_cellxgene(result, "10x_mtx") is True

    def test_tar_txt_dge_end_to_end(self, tmp_path):
        """TAR TXT DGE format: sample condition labels flow through to the .h5ad.

        This also tests that sample metadata from the YAML config (normal vs tumor)
        is correctly written into obs['condition'] of the saved file.
        """
        tar_path = tmp_path / "RAW.tar"
        gsm_ids = _write_tar_txt_dge(str(tar_path), n_samples=2)
        samples = [
            SampleConfig(id="normal", condition="normal", gsm_id=gsm_ids[0]),
            SampleConfig(id="tumor",  condition="tumor",  gsm_id=gsm_ids[1]),
        ]
        file_cfg = FileConfig(format="tar_txt_dge", relative_path="RAW.tar")
        cfg = self._base_config(str(tmp_path), file_cfg, samples=samples)
        out_path = convert_dataset(cfg, str(tmp_path / "output"))

        result = ad.read_h5ad(out_path)
        assert result.n_obs > 0
        assert set(result.obs["condition"].unique()) == {"normal", "tumor"}
        assert validate_cellxgene(result, "tar_txt_dge") is True

    def test_unknown_format_raises(self, tmp_path):
        """An unrecognised file format value must raise an error immediately.

        This catches typos in the 'format:' field of a dataset YAML config
        before any expensive file loading begins.
        """
        file_cfg = FileConfig(format="csv_dge", relative_path="nonexistent.csv.gz")
        cfg = self._base_config(str(tmp_path), file_cfg)
        cfg.files[0].__dict__["format"] = "bad_format"  # bypass Literal type guard
        with pytest.raises((ValueError, FileNotFoundError)):
            convert_dataset(cfg, str(tmp_path / "output"))
