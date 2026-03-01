"""Tests for cellclassifier/geo.py using synthetic in-memory data."""

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

# ── Helpers ───────────────────────────────────────────────────────────────────

N_GENES = 80        # enough for HVG selection (n_top_genes=50 in tests)
N_CELLS = 120       # enough for train/test split and Leiden
RNG = np.random.default_rng(0)


def _make_count_matrix() -> np.ndarray:
    """Sparse-ish non-negative integer matrix (N_CELLS × N_GENES)."""
    raw = RNG.negative_binomial(5, 0.5, size=(N_CELLS, N_GENES)).astype(float)
    # Zero out ~60 % of entries to mimic scRNA-seq sparsity
    mask = RNG.random(raw.shape) < 0.6
    raw[mask] = 0
    return raw


def _gene_names() -> list[str]:
    genes = [f"GENE{i:04d}" for i in range(N_GENES - 2)]
    genes += ["MT-CO1", "MT-ND1"]          # two mitochondrial genes
    return genes


def _barcode_names(suffix: str | None = None) -> list[str]:
    barcodes = [f"ACGT{i:04d}TGCA" for i in range(N_CELLS)]
    if suffix:
        barcodes = [f"{b}-{suffix}" for b in barcodes]
    return barcodes


def _make_adata(suffix: str | None = None) -> ad.AnnData:
    X = _make_count_matrix()
    return ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(index=_barcode_names(suffix)),
        var=pd.DataFrame(index=_gene_names()),
    )


def _minimal_preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        min_genes=1,
        min_cells=1,
        mt_pct_threshold=100.0,   # keep all cells
        n_top_genes=50,
        n_pcs=10,
        leiden_resolution=0.3,
    )


def _minimal_cellxgene_config() -> CellxGeneConfig:
    return CellxGeneConfig()


# ── load_csv_dge ─────────────────────────────────────────────────────────────

class TestLoadCsvDge:
    def test_shape(self, tmp_path):
        """Loaded AnnData should be cells × genes."""
        X = _make_count_matrix()
        df = pd.DataFrame(X.T, index=_gene_names(), columns=_barcode_names())
        path = tmp_path / "dge.csv.gz"
        df.to_csv(path)

        adata = load_csv_dge(str(path))

        assert adata.n_obs == N_CELLS
        assert adata.n_vars == N_GENES

    def test_gene_and_barcode_names(self, tmp_path):
        X = _make_count_matrix()
        genes = _gene_names()
        barcodes = _barcode_names()
        df = pd.DataFrame(X.T, index=genes, columns=barcodes)
        path = tmp_path / "dge.csv.gz"
        df.to_csv(path)

        adata = load_csv_dge(str(path))

        assert list(adata.var_names) == genes
        assert list(adata.obs_names) == barcodes

    def test_x_is_sparse(self, tmp_path):
        X = _make_count_matrix()
        df = pd.DataFrame(X.T, index=_gene_names(), columns=_barcode_names())
        path = tmp_path / "dge.csv.gz"
        df.to_csv(path)

        adata = load_csv_dge(str(path))

        assert sp.issparse(adata.X)


# ── load_10x_mtx ─────────────────────────────────────────────────────────────

def _write_10x_files(directory: str, prefix: str = "") -> None:
    """Write minimal gzipped 10x MTX files (optionally with a name prefix)."""
    X = _make_count_matrix()
    barcodes = _barcode_names()
    genes = _gene_names()

    # barcodes.tsv.gz
    bc_path = os.path.join(directory, f"{prefix}barcodes.tsv.gz")
    with gzip.open(bc_path, "wt") as f:
        f.write("\n".join(barcodes) + "\n")

    # features.tsv.gz  (gene_id <tab> gene_name <tab> Gene Expression)
    feat_path = os.path.join(directory, f"{prefix}features.tsv.gz")
    with gzip.open(feat_path, "wt") as f:
        for g in genes:
            f.write(f"{g}\t{g}\tGene Expression\n")

    # matrix.mtx.gz  (COO format)
    mat_path = os.path.join(directory, f"{prefix}matrix.mtx.gz")
    coo = sp.csr_matrix(X).T.tocoo()   # MTX is genes × cells
    with gzip.open(mat_path, "wt") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n%\n")
        f.write(f"{len(genes)} {len(barcodes)} {coo.nnz}\n")
        for r, c, v in zip(coo.row + 1, coo.col + 1, coo.data):
            f.write(f"{r} {c} {int(v)}\n")


class TestLoad10xMtx:
    def test_standard_names(self, tmp_path):
        _write_10x_files(str(tmp_path))
        adata = load_10x_mtx(str(tmp_path))

        assert adata.n_obs == N_CELLS
        assert adata.n_vars == N_GENES

    def test_prefixed_names(self, tmp_path):
        _write_10x_files(str(tmp_path), prefix="GSE999999_")
        adata = load_10x_mtx(str(tmp_path), name_prefix="GSE999999_")

        assert adata.n_obs == N_CELLS
        assert adata.n_vars == N_GENES

    def test_missing_file_raises(self, tmp_path):
        # Write only barcodes — no features/matrix
        bc_path = tmp_path / "GSE999999_barcodes.tsv.gz"
        with gzip.open(str(bc_path), "wt") as f:
            f.write("BARCODE1\n")

        with pytest.raises(FileNotFoundError):
            load_10x_mtx(str(tmp_path), name_prefix="GSE999999_")


# ── load_tar_txt_dge ─────────────────────────────────────────────────────────

def _write_tar_txt_dge(tar_path: str, n_samples: int = 2) -> list[str]:
    """Write a TAR containing n_samples gzipped TSV DGE files. Returns GSM IDs."""
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
    def test_shape_and_gsm_col(self, tmp_path):
        tar_path = str(tmp_path / "GSE_RAW.tar")
        gsm_ids = _write_tar_txt_dge(tar_path, n_samples=2)

        adata = load_tar_txt_dge(tar_path)

        assert adata.n_obs == N_CELLS * 2
        assert adata.n_vars == N_GENES
        assert "gsm_id" in adata.obs.columns
        assert set(adata.obs["gsm_id"].unique()) == set(gsm_ids)

    def test_extraction_idempotent(self, tmp_path):
        """Calling twice should not re-extract or error."""
        tar_path = str(tmp_path / "GSE_RAW.tar")
        _write_tar_txt_dge(tar_path, n_samples=1)

        load_tar_txt_dge(tar_path)
        adata = load_tar_txt_dge(tar_path)   # second call

        assert adata.n_obs == N_CELLS


# ── load_tar_10x ─────────────────────────────────────────────────────────────

def _write_tar_10x(tar_path: str, n_samples: int = 2) -> list[str]:
    """Write a TAR of per-sample 10x directories."""
    sample_names = [f"Sample{i+1}" for i in range(n_samples)]
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
    def test_shape_and_sample_col(self, tmp_path):
        tar_path = str(tmp_path / "RAW.tar")
        sample_names = _write_tar_10x(tar_path, n_samples=2)

        adata = load_tar_10x(tar_path)

        assert adata.n_obs == N_CELLS * 2
        assert adata.n_vars == N_GENES
        assert "sample" in adata.obs.columns
        assert set(adata.obs["sample"].unique()) == set(sample_names)

    def test_no_matrix_raises(self, tmp_path):
        """TAR with no 10x matrices should raise FileNotFoundError."""
        tar_path = str(tmp_path / "empty.tar")
        with tarfile.open(tar_path, "w"):
            pass

        with pytest.raises(FileNotFoundError):
            load_tar_10x(tar_path)


# ── assign_sample_metadata ───────────────────────────────────────────────────

class TestAssignSampleMetadata:
    def test_barcode_suffix_strategy(self):
        adata1 = _make_adata(suffix="1")
        adata2 = _make_adata(suffix="2")
        adata = ad.concat([adata1, adata2])

        samples = [
            SampleConfig(id="TumorA", condition="tumor", barcode_suffix="1",
                         tissue_ontology_term_id="UBERON:0001264",
                         disease_ontology_term_id="MONDO:0006047"),
            SampleConfig(id="Normal", condition="normal", barcode_suffix="2",
                         tissue_ontology_term_id="UBERON:0001264",
                         disease_ontology_term_id="PATO:0000461"),
        ]

        adata = assign_sample_metadata(
            adata, samples,
            dataset_tissue="UBERON:0001264",
            dataset_disease="unknown",
        )

        assert set(adata.obs["sample"].unique()) == {"TumorA", "Normal"}
        assert set(adata.obs["condition"].unique()) == {"tumor", "normal"}
        assert set(adata.obs["disease_ontology_term_id"].unique()) == {
            "MONDO:0006047", "PATO:0000461"
        }
        assert (adata.obs["donor_id"] == adata.obs["sample"]).all()

    def test_gsm_id_strategy(self):
        adata = _make_adata()
        adata.obs["gsm_id"] = ["GSM100" if i < N_CELLS // 2 else "GSM200"
                                for i in range(N_CELLS)]

        samples = [
            SampleConfig(id="normal", condition="normal", gsm_id="GSM100"),
            SampleConfig(id="tumor",  condition="tumor",  gsm_id="GSM200"),
        ]

        adata = assign_sample_metadata(
            adata, samples,
            dataset_tissue="UBERON:0001264",
            dataset_disease="unknown",
        )

        assert set(adata.obs["sample"].unique()) == {"normal", "tumor"}

    def test_no_samples_is_noop(self):
        adata = _make_adata()
        original_cols = set(adata.obs.columns)

        result = assign_sample_metadata(
            adata, [],
            dataset_tissue="UBERON:0001264",
            dataset_disease="unknown",
        )

        assert set(result.obs.columns) == original_cols

    def test_unmatched_barcodes_get_unknown(self):
        """Barcodes that don't match any suffix should stay 'unknown'."""
        adata = _make_adata(suffix="9")   # suffix "9" not in samples list
        samples = [SampleConfig(id="A", condition="normal", barcode_suffix="1")]

        adata = assign_sample_metadata(
            adata, samples,
            dataset_tissue="UBERON:0001264",
            dataset_disease="unknown",
        )

        assert (adata.obs["sample"] == "unknown").all()


# ── preprocess_adata ──────────────────────────────────────────────────────────

class TestPreprocessAdata:
    def test_output_has_umap_and_leiden(self):
        adata = _make_adata()
        cfg = _minimal_preprocessing_config()

        result = preprocess_adata(adata, cfg)

        assert "X_umap" in result.obsm
        assert "X_pca" in result.obsm
        assert "leiden" in result.obs.columns

    def test_hvg_subset(self):
        adata = _make_adata()
        cfg = _minimal_preprocessing_config()

        result = preprocess_adata(adata, cfg)

        assert result.n_vars == cfg.n_top_genes

    def test_mt_filter_removes_cells(self):
        """Cells with high MT% should be filtered out."""
        adata = _make_adata()
        # Force first 10 cells to have very high MT expression
        mt_idx = [i for i, g in enumerate(_gene_names()) if g.startswith("MT-")]
        X = adata.X.toarray()
        X[:10, mt_idx] = 9999
        adata.X = sp.csr_matrix(X)

        cfg = _minimal_preprocessing_config()
        cfg.mt_pct_threshold = 50.0

        result = preprocess_adata(adata, cfg)

        assert result.n_obs < N_CELLS

    def test_raw_is_stored(self):
        adata = _make_adata()
        result = preprocess_adata(adata, _minimal_preprocessing_config())

        assert result.raw is not None


# ── annotate_cellxgene_metadata ───────────────────────────────────────────────

class TestAnnotateCellxgeneMetadata:
    def _preprocessed(self):
        adata = _make_adata()
        return preprocess_adata(adata, _minimal_preprocessing_config())

    def test_required_obs_fields_added(self):
        adata = self._preprocessed()
        annotate_cellxgene_metadata(adata, _minimal_cellxgene_config(), title="Test")

        required = [
            "organism_ontology_term_id", "tissue_ontology_term_id",
            "assay_ontology_term_id", "disease_ontology_term_id",
            "cell_type_ontology_term_id", "donor_id", "suspension_type",
            "is_primary_data", "sex_ontology_term_id",
            "development_stage_ontology_term_id",
            "self_reported_ethnicity_ontology_term_id",
        ]
        for col in required:
            assert col in adata.obs.columns, f"Missing obs column: {col}"

    def test_required_var_fields_added(self):
        adata = self._preprocessed()
        annotate_cellxgene_metadata(adata, _minimal_cellxgene_config(), title="Test")

        for col in ["feature_is_filtered", "feature_name", "feature_biotype"]:
            assert col in adata.var.columns, f"Missing var column: {col}"

    def test_uns_fields(self):
        adata = self._preprocessed()
        annotate_cellxgene_metadata(adata, _minimal_cellxgene_config(), title="MyDataset")

        assert adata.uns["schema_version"] == "5.0.0"
        assert adata.uns["title"] == "MyDataset"
        assert adata.uns["default_embedding"] == "X_umap"

    def test_per_cell_overrides_not_clobbered(self):
        """Values already in obs (from assign_sample_metadata) must not be overwritten."""
        adata = self._preprocessed()
        adata.obs["tissue_ontology_term_id"] = "UBERON:0002107"   # liver

        annotate_cellxgene_metadata(adata, _minimal_cellxgene_config(), title="Test")

        assert (adata.obs["tissue_ontology_term_id"] == "UBERON:0002107").all()


# ── validate_cellxgene ────────────────────────────────────────────────────────

class TestValidateCellxgene:
    def _fully_annotated(self):
        adata = _make_adata()
        adata = preprocess_adata(adata, _minimal_preprocessing_config())
        annotate_cellxgene_metadata(adata, _minimal_cellxgene_config(), title="Test")
        return adata

    def test_passes_on_complete_adata(self):
        adata = self._fully_annotated()
        assert validate_cellxgene(adata, "test") is True

    def test_fails_on_missing_obs_column(self):
        adata = self._fully_annotated()
        del adata.obs["donor_id"]
        assert validate_cellxgene(adata, "test") is False

    def test_fails_on_missing_umap(self):
        adata = self._fully_annotated()
        del adata.obsm["X_umap"]
        assert validate_cellxgene(adata, "test") is False

    def test_fails_on_missing_uns_key(self):
        adata = self._fully_annotated()
        del adata.uns["title"]
        assert validate_cellxgene(adata, "test") is False


# ── convert_dataset (end-to-end) ─────────────────────────────────────────────

class TestConvertDataset:
    def _base_config(self, source_base: str, file_cfg: FileConfig,
                     samples: list[SampleConfig] | None = None) -> DatasetConfig:
        return DatasetConfig(
            id="GSE_TEST",
            title="Test Dataset",
            description="Synthetic test dataset",
            source=SourceConfig(type="local", base_path=source_base),
            files=[file_cfg],
            preprocessing=_minimal_preprocessing_config(),
            cellxgene=_minimal_cellxgene_config(),
            samples=samples or [],
        )

    def test_csv_dge_end_to_end(self, tmp_path):
        # Write a CSV DGE file
        X = _make_count_matrix()
        df = pd.DataFrame(X.T, index=_gene_names(), columns=_barcode_names())
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "test_dge.csv.gz"
        df.to_csv(str(csv_path))

        file_cfg = FileConfig(format="csv_dge", relative_path="data/test_dge.csv.gz")
        cfg = self._base_config(str(tmp_path), file_cfg)
        out_dir = str(tmp_path / "output")

        out_path = convert_dataset(cfg, out_dir)

        assert os.path.exists(out_path)
        result = ad.read_h5ad(out_path)
        assert result.n_obs > 0
        assert validate_cellxgene(result, "csv_dge") is True

    def test_10x_mtx_end_to_end(self, tmp_path):
        src_dir = tmp_path / "GSE_TEST"
        src_dir.mkdir()
        _write_10x_files(str(src_dir), prefix="GSE_TEST_")

        file_cfg = FileConfig(
            format="10x_mtx",
            relative_path="GSE_TEST",
            name_prefix="GSE_TEST_",
        )
        cfg = self._base_config(str(tmp_path), file_cfg)
        out_path = convert_dataset(cfg, str(tmp_path / "output"))

        result = ad.read_h5ad(out_path)
        assert result.n_obs > 0
        assert validate_cellxgene(result, "10x_mtx") is True

    def test_tar_txt_dge_end_to_end(self, tmp_path):
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

    def test_output_filename_uses_dataset_id(self, tmp_path):
        X = _make_count_matrix()
        df = pd.DataFrame(X.T, index=_gene_names(), columns=_barcode_names())
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "test.csv.gz").write_bytes(b"")   # placeholder — write real file
        csv_path = data_dir / "test.csv.gz"
        df.to_csv(str(csv_path))

        file_cfg = FileConfig(format="csv_dge", relative_path="data/test.csv.gz")
        cfg = self._base_config(str(tmp_path), file_cfg)
        out_path = convert_dataset(cfg, str(tmp_path / "output"))

        assert os.path.basename(out_path) == "GSE_TEST_processed.h5ad"

    def test_unknown_format_raises(self, tmp_path):
        file_cfg = FileConfig(format="csv_dge", relative_path="nonexistent.csv.gz")
        cfg = self._base_config(str(tmp_path), file_cfg)
        # Patch format to something invalid after construction
        cfg.files[0].__dict__["format"] = "bad_format"

        with pytest.raises((ValueError, FileNotFoundError)):
            convert_dataset(cfg, str(tmp_path / "output"))
