"""YAML config dataclasses and loaders for CellClassifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import yaml


# ── Dataset config ────────────────────────────────────────────────────────────

@dataclass
class SourceConfig:
    """Where raw GEO files live."""
    type: Literal["local", "gdrive"]
    base_path: str = ""           # absolute path for type="local"
    gdrive_folder_id: str = ""    # Drive folder ID for type="gdrive"


@dataclass
class FileConfig:
    """A single raw file (or directory) that feeds one loader."""
    format: Literal["csv_dge", "10x_mtx", "tar_txt_dge", "tar_10x"]
    relative_path: str            # path relative to source.base_path
    name_prefix: str = ""         # for 10x_mtx: strip prefix to get standard names
                                  # e.g. "GSE162708_" → barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz


@dataclass
class PreprocessingConfig:
    """Scanpy QC and dimensionality-reduction parameters."""
    min_genes: int = 200
    min_cells: int = 3
    mt_pct_threshold: float = 20.0
    n_top_genes: int = 2000
    n_pcs: int = 30
    leiden_resolution: float = 0.5


@dataclass
class CellxGeneConfig:
    """Ontology term IDs for cellxGene schema 5.0.0 compliance."""
    assay_ontology_term_id: str = "EFO:0009922"       # 10x 3' v2
    disease_ontology_term_id: str = "unknown"
    tissue_ontology_term_id: str = "UBERON:0001264"   # pancreas
    organism_ontology_term_id: str = "NCBITaxon:9606" # human


@dataclass
class SampleConfig:
    """Per-sample metadata used to annotate cells after loading.

    Exactly one of barcode_suffix or gsm_id should be set depending on the
    demultiplexing strategy:
      - barcode_suffix: for 10x datasets where sample is encoded in barcode
                        suffix (e.g. "-1", "-2")
      - gsm_id: for TAR datasets where each file is named by GSM accession
    """
    id: str                                      # human-readable sample label
    condition: str                               # e.g. "primary", "metastatic", "normal"
    barcode_suffix: str | None = None
    gsm_id: str | None = None
    tissue_ontology_term_id: str | None = None   # overrides dataset-level default
    disease_ontology_term_id: str | None = None  # overrides dataset-level default


@dataclass
class DatasetConfig:
    """Full configuration for one GEO dataset conversion."""
    id: str
    title: str
    description: str
    source: SourceConfig
    files: list[FileConfig]
    preprocessing: PreprocessingConfig
    cellxgene: CellxGeneConfig
    samples: list[SampleConfig] = field(default_factory=list)


# ── Pipeline config ───────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """RandomForestClassifier hyperparameters and train/test split."""
    n_estimators: int = 100
    class_weight: str = "balanced"
    random_state: int = 42
    test_size: float = 0.2


@dataclass
class AnalysisConfig:
    """Differential expression analysis parameters."""
    top_n_genes: int = 20


@dataclass
class PlotsConfig:
    """Visualization parameters."""
    umap_columns: list[str] = field(default_factory=lambda: ["celltype3", "CONDITION"])
    umap_genes: list[str] = field(default_factory=list)  # empty = top 2 feature-importance genes


@dataclass
class PipelineConfig:
    """Full configuration for the train/evaluate/plot pipeline."""
    data: str                        # path to processed .h5ad
    output: str = "./output"
    condition_col: str = "CONDITION"
    model: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    plots: PlotsConfig = field(default_factory=PlotsConfig)


# ── Loaders ───────────────────────────────────────────────────────────────────

_VALID_SOURCE_TYPES = {"local", "gdrive"}
_VALID_FILE_FORMATS = {"csv_dge", "10x_mtx", "tar_txt_dge", "tar_10x"}


def load_dataset_config(path: str) -> DatasetConfig:
    """Load and validate a dataset YAML config file.

    Args:
        path: Path to a dataset YAML file (e.g. configs/datasets/GSE154778.yaml).

    Returns:
        Populated DatasetConfig.

    Raises:
        ValueError: If required keys are missing or values are invalid.
        FileNotFoundError: If path does not exist.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    _require_keys(raw, ["id", "title", "source", "files"], path)

    source_raw = raw["source"]
    _require_keys(source_raw, ["type"], f"{path} > source")
    if source_raw["type"] not in _VALID_SOURCE_TYPES:
        raise ValueError(
            f"source.type must be one of {_VALID_SOURCE_TYPES}, "
            f"got {source_raw['type']!r} in {path}"
        )
    source = SourceConfig(**source_raw)

    files = []
    for i, file_raw in enumerate(raw["files"]):
        _require_keys(file_raw, ["format", "relative_path"], f"{path} > files[{i}]")
        if file_raw["format"] not in _VALID_FILE_FORMATS:
            raise ValueError(
                f"files[{i}].format must be one of {_VALID_FILE_FORMATS}, "
                f"got {file_raw['format']!r} in {path}"
            )
        files.append(FileConfig(**file_raw))

    preprocessing = PreprocessingConfig(**raw.get("preprocessing", {}))
    cellxgene = CellxGeneConfig(**raw.get("cellxgene", {}))
    samples = [SampleConfig(**s) for s in raw.get("samples", [])]

    return DatasetConfig(
        id=raw["id"],
        title=raw["title"],
        description=raw.get("description", ""),
        source=source,
        files=files,
        preprocessing=preprocessing,
        cellxgene=cellxgene,
        samples=samples,
    )


def load_pipeline_config(path: str) -> PipelineConfig:
    """Load and validate a pipeline YAML config file.

    Args:
        path: Path to a pipeline YAML file (e.g. configs/pipeline.yaml).

    Returns:
        Populated PipelineConfig.

    Raises:
        ValueError: If required keys are missing.
        FileNotFoundError: If path does not exist.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    _require_keys(raw, ["data"], path)

    model = ModelConfig(**raw.get("model", {}))
    analysis = AnalysisConfig(**raw.get("analysis", {}))
    plots = PlotsConfig(**raw.get("plots", {}))

    return PipelineConfig(
        data=raw["data"],
        output=raw.get("output", "./output"),
        condition_col=raw.get("condition_col", "CONDITION"),
        model=model,
        analysis=analysis,
        plots=plots,
    )


def _require_keys(d: dict, keys: list[str], location: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing required keys {missing} in {location}")
