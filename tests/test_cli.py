"""Tests for cellclassifier/cli.py subcommands using Click's CliRunner.

Key efficiency strategy: CLI commands that run the ML pipeline are expensive.
Module-scoped fixtures run each command ONCE and share the result across all
tests that check different aspects of the same invocation. Only tests that
specifically test different flags (--retrain, --data, error paths) get their
own separate invocation.

CliRunner runs commands in-process (no subprocess spawning), so it is fast
and captures stdout for assertion. scope="module" means fixtures run once per
file, not once per test.
"""

import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml
from click.testing import CliRunner

from cellclassifier.cli import cli

# ── Shared synthetic data ─────────────────────────────────────────────────────
# 200 cells × 50 genes is small enough that the Random Forest trains in seconds
# but large enough that sklearn's stratified split and evaluation work correctly.

N_CELLS = 200
N_GENES = 50
RNG = np.random.default_rng(42)


def _make_synthetic_h5ad(path: str) -> None:
    """Write a minimal .h5ad with CONDITION labels and a pre-computed UMAP.

    We include X_umap so the plot command doesn't need to recompute it,
    keeping plot tests fast.
    """
    X = RNG.negative_binomial(5, 0.5, (N_CELLS, N_GENES)).astype(float)
    X[RNG.random(X.shape) < 0.6] = 0  # ~60% zeros mimics real scRNA-seq data

    obs = pd.DataFrame(
        {"CONDITION": ["normal"] * (N_CELLS // 2) + ["tumor"] * (N_CELLS // 2)},
        index=[f"cell_{i}" for i in range(N_CELLS)],
    )
    var = pd.DataFrame(index=[f"GENE{i:03d}" for i in range(N_GENES)])
    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=obs,
        var=var,
        obsm={"X_umap": RNG.standard_normal((N_CELLS, 2))},
    )
    adata.write_h5ad(path)


# ── Module-scoped base fixtures ───────────────────────────────────────────────
# shared_tmp, h5ad_path, pipeline_yaml, and trained_artifact are all computed
# once for the entire module. trained_artifact does the actual RF training.

@pytest.fixture(scope="module")
def shared_tmp(tmp_path_factory):
    """One shared temp directory for the whole module — avoids re-creating files."""
    return tmp_path_factory.mktemp("cli_tests")


@pytest.fixture(scope="module")
def h5ad_path(shared_tmp):
    """A synthetic .h5ad file written once and reused by all CLI tests."""
    path = str(shared_tmp / "synthetic.h5ad")
    _make_synthetic_h5ad(path)
    return path


@pytest.fixture(scope="module")
def pipeline_yaml(shared_tmp, h5ad_path):
    """Pipeline YAML pointing at the synthetic h5ad with deliberately fast RF settings.

    n_estimators=10 (vs. default 100) and top_n_genes=5 keep training and
    feature importance extraction fast while still exercising the full code path.
    """
    cfg = {
        "data": h5ad_path,
        "output": str(shared_tmp / "output"),
        "condition_col": "CONDITION",
        "model": {
            "n_estimators": 10,
            "class_weight": "balanced",
            "random_state": 42,
            "test_size": 0.2,
        },
        "analysis": {"top_n_genes": 5},
        "plots": {
            "umap_columns": ["CONDITION"],
            "umap_genes": [],
        },
    }
    path = shared_tmp / "pipeline.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


@pytest.fixture(scope="module")
def trained_artifact(shared_tmp, pipeline_yaml):
    """Train the RF model ONCE and return the saved artifact path.

    This is the most expensive fixture — it runs the full train pipeline.
    All evaluate and plot tests reuse this artifact rather than retraining.
    The fixture itself asserts exit_code == 0 and artifact existence, so
    individual tests don't need to repeat those checks.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--config", pipeline_yaml])
    assert result.exit_code == 0, result.output
    artifact = Path(shared_tmp / "output" / "model_artifact.joblib")
    assert artifact.exists()
    return str(artifact)


# ── Shared command fixtures ────────────────────────────────────────────────────
# Each fixture below runs a command ONCE and shares the result across multiple
# tests. Without these, tests like test_exit_code_zero and test_output_contains_report
# would both invoke 'evaluate' separately even though they need the same output.

@pytest.fixture(scope="module")
def evaluate_result(pipeline_yaml, trained_artifact):
    """Run 'evaluate' once; share the CliRunner result across all evaluate tests.

    Both test_exit_code_zero and test_output_contains_report need the same
    evaluate invocation — this fixture prevents running it twice.
    """
    runner = CliRunner()
    return runner.invoke(cli, [
        "evaluate", "--config", pipeline_yaml, "--model", trained_artifact,
    ])


@pytest.fixture(scope="module")
def plot_dir(shared_tmp, pipeline_yaml, trained_artifact):
    """Run 'plot' once and return the output directory path.

    Shared by test_exit_code_zero, test_pngs_created, and
    test_feature_importance_png_created — all check different aspects of the
    same single plot invocation.
    """
    out_dir = str(shared_tmp / "plot_shared")
    runner = CliRunner()
    result = runner.invoke(cli, [
        "plot", "--config", pipeline_yaml, "--model", trained_artifact,
        "--output", out_dir,
    ])
    assert result.exit_code == 0, result.output
    return out_dir


@pytest.fixture(scope="module")
def run_result(shared_tmp, pipeline_yaml):
    """Run the full pipeline ONCE and return (CliRunner result, output directory).

    Shared by test_exit_code_zero, test_produces_artifact_and_plots, and
    test_done_message — all three check different aspects of the same run.
    """
    out_dir = str(shared_tmp / "run_shared")
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--config", pipeline_yaml, "--output", out_dir])
    return result, out_dir


# ── train ─────────────────────────────────────────────────────────────────────

class TestTrainCommand:
    """Tests for the 'train' subcommand.

    The trained_artifact fixture handles the primary successful train invocation
    (and already asserts exit_code == 0 and artifact existence), so tests here
    focus on flag behaviour and error handling.
    """

    def test_trains_successfully(self, trained_artifact):
        """The train command should produce a valid artifact path.

        This documents that the trained_artifact fixture succeeded — it asserted
        exit_code == 0 and artifact.exists() during setup.
        """
        assert trained_artifact  # fixture-produced path is non-empty

    def test_output_mentions_training(self, pipeline_yaml, shared_tmp):
        """The command should print a 'Training' progress message to stdout.

        Users should be able to see what the command is doing while it runs.
        """
        runner = CliRunner()
        result = runner.invoke(cli, [
            "train", "--config", pipeline_yaml,
            "--output", str(shared_tmp / "train_output_check"),
        ])
        assert result.exit_code == 0, result.output
        assert "training" in result.output.lower()

    def test_output_flag_writes_to_custom_dir(self, pipeline_yaml, shared_tmp):
        """--output should write the artifact to a user-specified directory
        instead of the one in the YAML config."""
        custom_out = str(shared_tmp / "custom_train_out")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "train", "--config", pipeline_yaml, "--output", custom_out,
        ])
        assert result.exit_code == 0, result.output
        assert Path(custom_out, "model_artifact.joblib").exists()

    def test_data_flag_overrides_yaml(self, pipeline_yaml, h5ad_path, shared_tmp):
        """--data should override the .h5ad path specified in the YAML config.

        This lets users train on a different dataset without editing the YAML.
        """
        out_dir = str(shared_tmp / "train_data_override")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "train", "--config", pipeline_yaml,
            "--data", h5ad_path,
            "--output", out_dir,
        ])
        assert result.exit_code == 0, result.output
        assert Path(out_dir, "model_artifact.joblib").exists()

    def test_nonexistent_config_fails(self):
        """A config file that doesn't exist should cause a non-zero exit with a clear message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--config", "no_such_file.yaml"])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_missing_config_flag_fails(self):
        """Omitting the required --config flag should print 'Missing option'."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train"])
        assert result.exit_code != 0
        assert "Missing option" in result.output


# ── evaluate ──────────────────────────────────────────────────────────────────

class TestEvaluateCommand:
    """Tests for the 'evaluate' subcommand.

    test_exit_code_zero and test_output_contains_report both need the same
    evaluate invocation — they share the evaluate_result fixture so the
    command runs only once.
    """

    def test_exit_code_zero(self, evaluate_result):
        """Evaluate on a valid model and dataset should succeed."""
        assert evaluate_result.exit_code == 0, evaluate_result.output

    def test_output_contains_report(self, evaluate_result):
        """stdout should include a sklearn classification report.

        The report always contains 'precision', 'recall', and 'f1-score',
        which let us verify the model was actually evaluated (not just loaded).
        """
        assert "precision" in evaluate_result.output.lower()

    def test_data_flag_overrides_yaml(self, pipeline_yaml, trained_artifact, h5ad_path):
        """--data should override the .h5ad path from the YAML."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate", "--config", pipeline_yaml,
            "--model", trained_artifact,
            "--data", h5ad_path,
        ])
        assert result.exit_code == 0, result.output

    def test_missing_model_flag_fails(self, pipeline_yaml):
        """Omitting the required --model flag should print 'Missing option'."""
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--config", pipeline_yaml])
        assert result.exit_code != 0
        assert "Missing option" in result.output

    def test_nonexistent_model_fails(self, pipeline_yaml):
        """A --model path that doesn't exist should fail with a clear error message."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate", "--config", pipeline_yaml,
            "--model", "ghost_artifact.joblib",
        ])
        assert result.exit_code != 0
        assert "does not exist" in result.output


# ── plot ──────────────────────────────────────────────────────────────────────

class TestPlotCommand:
    """Tests for the 'plot' subcommand.

    test_exit_code_zero, test_pngs_created, and test_feature_importance_png_created
    all inspect the same output directory — they share the plot_dir fixture so
    the plot command runs only once.
    """

    def test_exit_code_zero(self, plot_dir):
        """The shared plot run (via plot_dir fixture) must succeed.

        The fixture itself asserts exit_code == 0, so this test documents
        the expectation without running plot again.
        """
        assert Path(plot_dir).is_dir()

    def test_pngs_created(self, plot_dir):
        """At least one PNG file should be saved to <output>/plots/.

        Each requested UMAP column and each top gene produces a PNG, plus
        the feature importance bar chart — so there should always be >0 files.
        """
        pngs = list((Path(plot_dir) / "plots").glob("*.png"))
        assert len(pngs) > 0, f"No PNGs found in {Path(plot_dir) / 'plots'}"

    def test_feature_importance_png_created(self, plot_dir):
        """The feature importance bar chart must always be created.

        This is the main deliverable of the project — ranking which genes best
        distinguish tumor from normal cells.
        """
        fi_png = Path(plot_dir) / "plots" / "feature_importances.png"
        assert fi_png.exists()

    def test_missing_model_flag_fails(self, pipeline_yaml):
        """Omitting --model should fail — plot requires a trained artifact."""
        runner = CliRunner()
        result = runner.invoke(cli, ["plot", "--config", pipeline_yaml])
        assert result.exit_code != 0
        assert "Missing option" in result.output


# ── run (full pipeline) ───────────────────────────────────────────────────────

class TestRunCommand:
    """Tests for the 'run' subcommand (full pipeline: train → evaluate → plot).

    test_exit_code_zero, test_produces_artifact_and_plots, and test_done_message
    all inspect the same run invocation — they share the run_result fixture so
    the full pipeline runs only once for these three tests.
    """

    def test_exit_code_zero(self, run_result):
        """The full pipeline must complete without errors."""
        result, _ = run_result
        assert result.exit_code == 0, result.output

    def test_produces_artifact_and_plots(self, run_result):
        """The full pipeline must produce both a model artifact and at least one PNG."""
        _, out_dir = run_result
        assert Path(out_dir, "model_artifact.joblib").exists()
        pngs = list((Path(out_dir) / "plots").glob("*.png"))
        assert len(pngs) > 0

    def test_done_message(self, run_result):
        """stdout should end with 'Done!' confirming all steps completed."""
        result, _ = run_result
        assert "Done" in result.output

    def test_retrain_flag_overwrites_artifact(self, pipeline_yaml, shared_tmp):
        """--retrain must overwrite an existing artifact (newer modification time).

        Without --retrain, the run command reuses an existing model to save time.
        --retrain forces a fresh training run, which is needed when the dataset changes.
        We verify by comparing file modification timestamps before and after.
        """
        out_dir = str(shared_tmp / "run_retrain")
        runner = CliRunner()

        runner.invoke(cli, ["run", "--config", pipeline_yaml, "--output", out_dir])
        artifact = Path(out_dir, "model_artifact.joblib")
        mtime_first = artifact.stat().st_mtime

        time.sleep(0.05)  # ensure the filesystem clock ticks so mtime differs

        runner.invoke(cli, ["run", "--config", pipeline_yaml, "--output", out_dir,
                            "--retrain"])
        mtime_second = artifact.stat().st_mtime

        assert mtime_second > mtime_first, "Artifact was not overwritten by --retrain"

    def test_no_retrain_reuses_artifact(self, pipeline_yaml, shared_tmp):
        """Without --retrain, an existing artifact must not be overwritten.

        Reusing the saved model avoids expensive retraining when nothing has changed.
        We verify the file modification time is unchanged after the second run,
        and that the output says 'Loading Existing Model'.
        """
        out_dir = str(shared_tmp / "run_no_retrain")
        runner = CliRunner()

        runner.invoke(cli, ["run", "--config", pipeline_yaml, "--output", out_dir])
        artifact = Path(out_dir, "model_artifact.joblib")
        mtime_first = artifact.stat().st_mtime

        time.sleep(0.05)

        result = runner.invoke(cli, ["run", "--config", pipeline_yaml, "--output", out_dir])
        mtime_second = artifact.stat().st_mtime

        assert mtime_second == mtime_first, "Artifact was unexpectedly overwritten"
        assert "Loading Existing Model" in result.output


# ── inspect helpers ────────────────────────────────────────────────────────────
# The inspect command loads RAW data from the filesystem path in the YAML config,
# then prints barcode distributions. We write a minimal CSV DGE file with specific
# barcode names, then point a dataset YAML at it so inspect can load it.

def _write_csv_dge_with_barcodes(path: str, barcodes: list[str]) -> None:
    """Write a gzip-compressed CSV DGE file with specific cell barcode column names.

    CSV DGE format is genes × cells (rows = genes, columns = cells).
    We only need real-looking structure — inspect reads obs_names (the barcodes),
    not the expression values.
    """
    import gzip

    rng = np.random.default_rng(99)
    n_genes = 10
    X = rng.integers(0, 100, size=(n_genes, len(barcodes)))
    gene_names = [f"IGENE{i}" for i in range(n_genes)]
    df = pd.DataFrame(X, index=gene_names, columns=barcodes)
    with gzip.open(path, "wt") as f:
        df.to_csv(f)


def _make_inspect_yaml(tmp_path, barcodes: list[str], n_samples: int) -> str:
    """Write a minimal dataset YAML and a matching CSV DGE file; return the YAML path.

    The YAML's source.base_path and files[0].relative_path together form the
    absolute path that the inspect command opens. n_samples controls how many
    sample entries appear in the config (used for the '# samples expected' summary).
    """
    csv_path = tmp_path / "inspect_dge.csv.gz"
    _write_csv_dge_with_barcodes(str(csv_path), barcodes)

    cfg = {
        "id": "GSE_INSPECT_TEST",
        "title": "Inspect Test Dataset",
        "description": "Synthetic data for testing the inspect command",
        "source": {"type": "local", "base_path": str(tmp_path)},
        "files": [{"format": "csv_dge", "relative_path": "inspect_dge.csv.gz"}],
        "preprocessing": {
            "min_genes": 1, "min_cells": 1, "mt_pct_threshold": 100,
            "n_top_genes": 5, "n_pcs": 2, "leiden_resolution": 0.5,
        },
        "samples": [{"id": f"sample_{i}", "condition": "test"} for i in range(n_samples)],
    }
    yaml_path = tmp_path / "inspect_test.yaml"
    yaml_path.write_text(yaml.dump(cfg))
    return str(yaml_path)


# ── inspect ────────────────────────────────────────────────────────────────────

class TestInspectCommand:
    """Tests for the 'inspect' subcommand.

    The inspect command loads raw data from a dataset config and prints barcode
    distributions to help the user figure out which barcode pattern maps to
    which sample. It supports two formats:
    - Dash-separated: 'ACGTACGT-1' (standard 10x Chromium barcodes)
    - Colon-separated: 'P01:1' (used by GSE154778)

    Each test uses its own tmp_path so they are fully independent.
    """

    def test_detects_dash_suffix_format(self, tmp_path):
        """Barcodes like 'ACGTACGT-1' should trigger the dash-suffix path.

        Standard 10x Chromium barcodes end with '-N' where N identifies the
        sample (e.g. '-1' = sample 1, '-2' = sample 2). The inspect command
        should print 'DASH-separated' and show the suffix count table.
        """
        barcodes = ([f"ACGT{i:04d}-1" for i in range(10)] +
                    [f"ACGT{i:04d}-2" for i in range(10)])
        yaml_path = _make_inspect_yaml(tmp_path, barcodes, n_samples=2)

        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "--config", yaml_path])

        assert result.exit_code == 0, result.output
        assert "DASH" in result.output.upper()

    def test_detects_colon_prefix_format(self, tmp_path):
        """Barcodes like 'P01:1' should trigger the colon-prefix path.

        GSE154778 uses 'PREFIX:INDEX' barcodes where the part BEFORE the colon
        identifies the sample (e.g. 'P01' = primary tumor 1, 'MET01' = metastatic 1).
        The inspect command should print 'COLON-separated'.
        """
        barcodes = ([f"P01:{i}" for i in range(10)] +
                    [f"MET01:{i}" for i in range(10)])
        yaml_path = _make_inspect_yaml(tmp_path, barcodes, n_samples=2)

        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "--config", yaml_path])

        assert result.exit_code == 0, result.output
        assert "COLON" in result.output.upper()

    def test_shows_correct_total_cell_count(self, tmp_path):
        """The output must display the total number of cells that were loaded.

        We write exactly 20 barcodes, so '20' must appear in the output.
        Without this, the user can't confirm the file loaded correctly before
        deciding which barcode_suffix or barcode_prefix values to put in the YAML.
        """
        barcodes = [f"CELL{i:03d}-1" for i in range(20)]
        yaml_path = _make_inspect_yaml(tmp_path, barcodes, n_samples=1)

        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "--config", yaml_path])

        assert result.exit_code == 0, result.output
        assert "20" in result.output

    def test_missing_config_fails(self):
        """A config path that does not exist should cause the command to fail.

        Click validates the --config path before running any code. Failing fast
        prevents the user from waiting for a slow file load that will never work.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code != 0

    def test_prefix_names_appear_in_output(self, tmp_path):
        """The actual prefix strings from the barcodes must appear in the output.

        If barcodes start with 'P01:' and 'MET01:', the user needs to see both
        'P01' and 'MET01' listed so they know which barcode_prefix values to add
        to their YAML config's samples section.
        """
        barcodes = ([f"P01:{i}" for i in range(5)] +
                    [f"MET01:{i}" for i in range(5)])
        yaml_path = _make_inspect_yaml(tmp_path, barcodes, n_samples=2)

        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "--config", yaml_path])

        assert result.exit_code == 0, result.output
        assert "P01" in result.output
        assert "MET01" in result.output


# ── merge helpers + fixtures ───────────────────────────────────────────────────
# merge_datasets runs PCA → UMAP → Leiden, which takes several seconds.
# We use module-scoped fixtures to run the merge command ONCE and then check
# different aspects of the output file in separate individual tests.

# These constants are separate from the module-level N_CELLS/N_GENES to avoid
# confusion with the existing run/train/evaluate synthetic data above.
N_MERGE_CELLS = 60    # cells per dataset; 60 × 2 = 120 total after merging
N_MERGE_GENES = 50    # genes per dataset; 40 overlap → 40 common genes
RNG_MERGE = np.random.default_rng(13)


def _make_merge_h5ad(path: str, gene_start: int, condition: str) -> None:
    """Write a synthetic processed h5ad for merge command tests.

    gene_start shifts which genes are included so two datasets with overlapping
    ranges share a common gene subset. For example:
      gene_start=0  → MGENE_000..049 (50 genes)
      gene_start=10 → MGENE_010..059 (50 genes)
    Intersection = MGENE_010..049 = 40 common genes.

    We use standard-normal values to mimic already-scaled data, since
    merge_datasets skips the normalisation step on pre-processed input.
    """
    genes = [f"MGENE_{i:03d}" for i in range(gene_start, gene_start + N_MERGE_GENES)]
    X = RNG_MERGE.standard_normal((N_MERGE_CELLS, N_MERGE_GENES))
    obs = pd.DataFrame(
        {"condition": [condition] * N_MERGE_CELLS},
        index=[f"{condition}_cell_{i}" for i in range(N_MERGE_CELLS)],
    )
    var = pd.DataFrame(index=genes)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.write_h5ad(path)


@pytest.fixture(scope="module")
def merge_shared_tmp(tmp_path_factory):
    """Shared temp directory for all merge CLI tests."""
    return tmp_path_factory.mktemp("merge_cli_tests")


@pytest.fixture(scope="module")
def merge_inputs(merge_shared_tmp):
    """Write two processed h5ad files and a condition_map YAML; return their paths.

    Dataset 1: genes MGENE_000..049, condition='primary'
    Dataset 2: genes MGENE_010..059, condition='normal'
    Common genes: MGENE_010..049 = 40 genes
    condition_map: primary → malignant, normal → normal
    """
    path1 = str(merge_shared_tmp / "ds1.h5ad")
    path2 = str(merge_shared_tmp / "ds2.h5ad")
    _make_merge_h5ad(path1, gene_start=0,  condition="primary")
    _make_merge_h5ad(path2, gene_start=10, condition="normal")

    cmap = {"primary": "malignant", "normal": "normal"}
    cmap_path = merge_shared_tmp / "cmap.yaml"
    cmap_path.write_text(yaml.dump(cmap))

    return path1, path2, str(cmap_path)


@pytest.fixture(scope="module")
def merge_result(merge_shared_tmp, merge_inputs):
    """Run the merge command ONCE and return (CliRunner result, output directory).

    Shared by test_exit_code_zero, test_combined_h5ad_created, test_cell_count,
    test_conditions_remapped, and test_output_reports_cell_count — all check
    different aspects of the same single merge invocation.
    """
    path1, path2, cmap_path = merge_inputs
    out_dir = str(merge_shared_tmp / "merge_output")

    runner = CliRunner()
    result = runner.invoke(cli, [
        "merge",
        "--data", path1,
        "--data", path2,
        "--condition-map", cmap_path,
        "--output", out_dir,
    ])
    return result, out_dir


# ── merge ──────────────────────────────────────────────────────────────────────

class TestMergeCommand:
    """Tests for the 'merge' subcommand.

    The merge command combines multiple processed h5ad files from different
    datasets into one combined file. It remaps condition labels, keeps only
    genes present in ALL datasets, and runs PCA/UMAP/Leiden for shared embeddings.
    The output is 'combined_processed.h5ad' which can be fed directly into
    'python run.py run' to train a classifier across all datasets at once.
    """

    def test_exit_code_zero(self, merge_result):
        """The merge command must complete without errors (exit code 0)."""
        result, _ = merge_result
        assert result.exit_code == 0, result.output

    def test_combined_h5ad_created(self, merge_result):
        """The merge command must write 'combined_processed.h5ad' to the output directory.

        Downstream steps look for exactly this filename. If it's missing or named
        differently, 'python run.py run' will fail with a FileNotFoundError.
        """
        _, out_dir = merge_result
        combined_path = os.path.join(out_dir, "combined_processed.h5ad")
        assert os.path.exists(combined_path)

    def test_combined_h5ad_has_correct_cell_count(self, merge_result):
        """The written h5ad must contain cells from both input datasets.

        We put 60 cells in each of 2 datasets, so the combined file must have
        60 + 60 = 120 cells. If the merge silently dropped cells or failed to
        concatenate correctly, the cell count would be wrong.
        """
        _, out_dir = merge_result
        loaded = ad.read_h5ad(os.path.join(out_dir, "combined_processed.h5ad"))
        assert loaded.n_obs == N_MERGE_CELLS * 2

    def test_conditions_remapped_in_output(self, merge_result):
        """The combined file must use unified labels, not the original per-dataset ones.

        'primary' was mapped to 'malignant' by the condition map. After merging,
        obs['condition'] should contain 'malignant' and 'normal' — not 'primary'.
        Wrong condition labels would cause the classifier to learn the wrong classes.
        """
        _, out_dir = merge_result
        loaded = ad.read_h5ad(os.path.join(out_dir, "combined_processed.h5ad"))
        conditions = set(loaded.obs["condition"].unique())
        assert "malignant" in conditions
        assert "primary" not in conditions  # original label should be gone

    def test_output_reports_cell_count(self, merge_result):
        """The command's stdout must display the total number of cells in the merged file.

        This lets the user confirm the merge without opening the h5ad file.
        We wrote 120 cells total (60 + 60), so '120' should appear in the output.
        """
        result, _ = merge_result
        assert "120" in result.output

    def test_missing_condition_map_fails(self, merge_inputs):
        """Omitting --condition-map must cause the command to fail.

        Without a condition map, there is no way to unify labels across datasets.
        Click marks --condition-map as required, so the command should exit with
        a non-zero code rather than silently producing an incorrectly labelled file.
        """
        path1, path2, _ = merge_inputs
        runner = CliRunner()
        result = runner.invoke(cli, [
            "merge",
            "--data", path1,
            "--data", path2,
            # intentionally omit --condition-map
        ])
        assert result.exit_code != 0
