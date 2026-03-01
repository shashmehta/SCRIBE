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
