"""YAML config dataclasses and loaders for SCRIBE.

Each YAML file is loaded into one of these dataclass objects, which the
rest of the code uses instead of reading the YAML directly. This keeps
configuration separate from logic.
"""

from __future__ import annotations

# dataclass turns a plain class into a structured container with named fields.
# field() lets us set default values for lists and nested objects safely.
from dataclasses import dataclass, field

# Literal restricts a variable to a fixed set of allowed string values.
from typing import Literal

import yaml  # reads .yaml files into Python dicts


# ── Dataset config ────────────────────────────────────────────────────────────
# These classes describe everything needed to load and convert one GEO dataset.

@dataclass
class SourceConfig:
    """Where raw GEO files live — either on the local filesystem or Google Drive."""
    type: Literal["local", "gdrive"]  # must be one of these two strings
    base_path: str = ""               # full path to the folder (for local files)
    gdrive_folder_id: str = ""        # Google Drive folder ID (for cloud files)


@dataclass
class FileConfig:
    """Describes one raw data file and which loader should read it."""
    # format tells the loader which function to call (e.g. load_csv_dge)
    format: Literal["csv_dge", "10x_mtx", "tar_txt_dge", "tar_10x"]
    relative_path: str   # path inside source.base_path (file or directory)
    name_prefix: str = ""  # for 10x_mtx: the GSE prefix to strip off filenames
                           # e.g. "GSE162708_" → barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz


@dataclass
class PreprocessingConfig:
    """Quality-control and dimensionality-reduction settings for scanpy."""
    min_genes: int = 200          # drop cells that express fewer than this many genes
    min_cells: int = 3            # drop genes detected in fewer than this many cells
    mt_pct_threshold: float = 20.0  # drop cells where >20% of reads are mitochondrial
    n_top_genes: int = 2000       # keep only the 2000 most variable genes
    n_pcs: int = 30               # number of principal components for PCA
    leiden_resolution: float = 0.5  # higher = more, smaller clusters


@dataclass
class SampleConfig:
    """Metadata for one biological sample within a dataset.

    After loading, each cell needs to be labelled with its sample of origin.
    We support two ways to figure out which cell belongs to which sample:
      - barcode_suffix: 10x datasets append "-1", "-2", etc. to each cell barcode
      - gsm_id: TAR datasets have one file per sample named by its GEO accession
    """
    id: str           # short human-readable name, e.g. "primary_tumor_1"
    condition: str    # biological condition, e.g. "primary", "metastatic", "normal"
    barcode_suffix: str | None = None   # the number after the "-" in a 10x barcode
    barcode_prefix: str | None = None   # the part before ":" in "SAMPLE:INDEX" barcodes (e.g. "P03")
    gsm_id: str | None = None           # GEO sample accession, e.g. "GSM5032701"


@dataclass
class DatasetConfig:
    """Everything needed to convert one GEO dataset into a processed .h5ad file."""
    id: str           # GEO series accession, e.g. "GSE154778"
    title: str        # human-readable title for the dataset
    description: str  # longer description of what the dataset contains
    source: SourceConfig          # where the raw files live
    files: list[FileConfig]       # which files to load and how
    preprocessing: PreprocessingConfig  # QC and dimensionality-reduction settings
    samples: list[SampleConfig] = field(default_factory=list)  # per-sample metadata


# ── Pipeline config ───────────────────────────────────────────────────────────
# These classes describe the ML training / evaluation / plotting pipeline.

@dataclass
class ModelConfig:
    """Settings for the Random Forest classifier."""
    n_estimators: int = 100       # number of decision trees in the forest
    class_weight: str = "balanced"  # upweight minority class to handle imbalance
    random_state: int = 42        # fixed seed so results are reproducible
    test_size: float = 0.2        # hold out 20% of cells for testing


@dataclass
class AnalysisConfig:
    """Settings for differential expression analysis."""
    top_n_genes: int = 20  # how many top genes to report and plot


@dataclass
class PlotsConfig:
    """Settings for UMAP and feature-importance visualisations."""
    # obs columns to colour the UMAP by (only used if they exist in the data)
    umap_columns: list[str] = field(default_factory=lambda: ["celltype3", "CONDITION"])
    # specific genes to overlay on the UMAP; empty = top 2 from feature importances
    umap_genes: list[str] = field(default_factory=list)


@dataclass
class BatchConfig:
    """Settings for batch effect detection and correction."""
    batch_key: str = "dataset"              # obs column identifying the source dataset
    housekeeping_genes: list[str] = field(  # genes for batch effect quantification
        default_factory=lambda: ["ACTB", "GAPDH", "B2M", "RPL13A", "RPLP0", "PPIA"]
    )
    correction_method: str = "harmony"      # default correction: harmony, combat, scanorama, or all
    n_top_genes: int = 3000                 # number of batch-aware HVGs for joint selection after merge
    n_neighbors_mixing: int = 50            # k for batch mixing score computation


@dataclass
class PipelineConfig:
    """Everything needed to run the train → evaluate → plot pipeline."""
    data: str                    # path to the processed .h5ad file
    output: str = ""             # where to write the model artifact (resolved via paths module)
    plots_dir: str = ""          # where to write all plot PNGs (resolved via paths module)
    condition_col: str = "CONDITION"  # obs column that holds the class labels
    batch_key: str = "dataset"   # obs column identifying the source dataset (for batch correction)
    model: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    plots: PlotsConfig = field(default_factory=PlotsConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)


# ── Loaders ───────────────────────────────────────────────────────────────────

# These sets are used to validate that YAML values are one of the expected options.
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
    # Read the YAML file into a plain Python dict
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Check that the top-level required keys are present
    _require_keys(raw, ["id", "title", "source", "files"], path)

    # Validate and build the source sub-config
    source_raw = raw["source"]
    _require_keys(source_raw, ["type"], f"{path} > source")
    if source_raw["type"] not in _VALID_SOURCE_TYPES:
        raise ValueError(
            f"source.type must be one of {_VALID_SOURCE_TYPES}, "
            f"got {source_raw['type']!r} in {path}"
        )
    source = SourceConfig(**source_raw)  # ** unpacks the dict into keyword args

    # Validate and build each file config entry
    files = []
    for i, file_raw in enumerate(raw["files"]):
        _require_keys(file_raw, ["format", "relative_path"], f"{path} > files[{i}]")
        if file_raw["format"] not in _VALID_FILE_FORMATS:
            raise ValueError(
                f"files[{i}].format must be one of {_VALID_FILE_FORMATS}, "
                f"got {file_raw['format']!r} in {path}"
            )
        files.append(FileConfig(**file_raw))

    # Sub-configs are optional in the YAML; fall back to defaults if missing
    preprocessing = PreprocessingConfig(**raw.get("preprocessing", {}))
    samples_raw = raw.get("samples", [])
    # Drop any per-sample ontology fields that may exist in older configs
    _ontology_keys = {"tissue_ontology_term_id", "disease_ontology_term_id"}
    samples = [SampleConfig(**{k: v for k, v in s.items() if k not in _ontology_keys}) for s in samples_raw]

    return DatasetConfig(
        id=raw["id"],
        title=raw["title"],
        description=raw.get("description", ""),
        source=source,
        files=files,
        preprocessing=preprocessing,
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

    # "data" is the only mandatory key — everything else has a default
    _require_keys(raw, ["data"], path)

    model = ModelConfig(**raw.get("model", {}))
    analysis = AnalysisConfig(**raw.get("analysis", {}))
    plots = PlotsConfig(**raw.get("plots", {}))
    batch = BatchConfig(**raw.get("batch", {}))

    from scribe import paths

    return PipelineConfig(
        data=raw["data"],
        output=raw.get("output") or str(paths.get_processed_dir()),
        plots_dir=raw.get("plots_dir") or str(paths.get_plots_dir()),
        condition_col=raw.get("condition_col", "CONDITION"),
        batch_key=raw.get("batch_key", "dataset"),
        model=model,
        analysis=analysis,
        plots=plots,
        batch=batch,
    )


def _require_keys(d: dict, keys: list[str], location: str) -> None:
    """Raise ValueError listing any keys missing from dict d."""
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing required keys {missing} in {location}")
