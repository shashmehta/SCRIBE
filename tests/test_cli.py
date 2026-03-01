"""Tests for cellclassifier/cli.py subcommands using Click's CliRunner."""

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
from cellclassifier.config import load_pipeline_config

# ── Shared synthetic data ─────────────────────────────────────────────────────

N_CELLS = 200
N_GENES = 50
RNG = np.random.default_rng(42)


def _make_synthetic_h5ad(path: str) -> None:
    """Write a minimal .h5ad with CONDITION labels and a UMAP embedding."""
    X = RNG.negative_binomial(5, 0.5, (N_CELLS, N_GENES)).astype(float)
    X[RNG.random(X.shape) < 0.6] = 0

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


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def shared_tmp(tmp_path_factory):
    """One shared tmp directory for the whole module (avoids re-training)."""
    return tmp_path_factory.mktemp("cli_tests")


@pytest.fixture(scope="module")
def h5ad_path(shared_tmp):
    path = str(shared_tmp / "synthetic.h5ad")
    _make_synthetic_h5ad(path)
    return path


@pytest.fixture(scope="module")
def pipeline_yaml(shared_tmp, h5ad_path):
    """Pipeline YAML pointing at the synthetic h5ad with fast RF settings."""
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
    """Run `train` once and return the artifact path for reuse."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--config", pipeline_yaml])
    assert result.exit_code == 0, result.output
    artifact = Path(shared_tmp / "output" / "model_artifact.joblib")
    assert artifact.exists()
    return str(artifact)


# ── train ─────────────────────────────────────────────────────────────────────

class TestTrainCommand:
    def test_exit_code_zero(self, pipeline_yaml, trained_artifact):
        # trained_artifact fixture already ran train; just confirm it passed
        assert trained_artifact  # non-empty path

    def test_artifact_created(self, shared_tmp, trained_artifact):
        assert Path(trained_artifact).exists()

    def test_output_mentions_training(self, pipeline_yaml, shared_tmp):
        runner = CliRunner()
        out_dir = str(shared_tmp / "train_output_check")
        result = runner.invoke(cli, [
            "train", "--config", pipeline_yaml, "--output", out_dir,
        ])
        assert result.exit_code == 0, result.output
        assert "Training" in result.output or "training" in result.output.lower()

    def test_output_override_writes_to_custom_dir(self, pipeline_yaml, shared_tmp):
        custom_out = str(shared_tmp / "custom_train_out")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "train", "--config", pipeline_yaml, "--output", custom_out,
        ])
        assert result.exit_code == 0, result.output
        assert Path(custom_out, "model_artifact.joblib").exists()

    def test_data_override(self, pipeline_yaml, h5ad_path, shared_tmp):
        """--data flag should override the path in the YAML."""
        out_dir = str(shared_tmp / "train_data_override")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "train", "--config", pipeline_yaml,
            "--data", h5ad_path,
            "--output", out_dir,
        ])
        assert result.exit_code == 0, result.output
        assert Path(out_dir, "model_artifact.joblib").exists()

    def test_missing_config_fails(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--config", "no_such_file.yaml"])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_missing_config_flag_fails(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["train"])
        assert result.exit_code != 0
        assert "Missing option" in result.output


# ── evaluate ──────────────────────────────────────────────────────────────────

class TestEvaluateCommand:
    def test_exit_code_zero(self, pipeline_yaml, trained_artifact):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate",
            "--config", pipeline_yaml,
            "--model", trained_artifact,
        ])
        assert result.exit_code == 0, result.output

    def test_output_contains_report(self, pipeline_yaml, trained_artifact):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate",
            "--config", pipeline_yaml,
            "--model", trained_artifact,
        ])
        # sklearn classification report always contains "precision"
        assert "precision" in result.output.lower()

    def test_data_override(self, pipeline_yaml, trained_artifact, h5ad_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate",
            "--config", pipeline_yaml,
            "--model", trained_artifact,
            "--data", h5ad_path,
        ])
        assert result.exit_code == 0, result.output

    def test_missing_model_flag_fails(self, pipeline_yaml):
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--config", pipeline_yaml])
        assert result.exit_code != 0
        assert "Missing option" in result.output

    def test_nonexistent_model_fails(self, pipeline_yaml):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate",
            "--config", pipeline_yaml,
            "--model", "ghost_artifact.joblib",
        ])
        assert result.exit_code != 0
        assert "does not exist" in result.output


# ── plot ──────────────────────────────────────────────────────────────────────

class TestPlotCommand:
    def test_exit_code_zero(self, pipeline_yaml, trained_artifact, shared_tmp):
        out_dir = str(shared_tmp / "plot_out")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "plot",
            "--config", pipeline_yaml,
            "--model", trained_artifact,
            "--output", out_dir,
        ])
        assert result.exit_code == 0, result.output

    def test_pngs_created(self, pipeline_yaml, trained_artifact, shared_tmp):
        out_dir = str(shared_tmp / "plot_pngs")
        runner = CliRunner()
        runner.invoke(cli, [
            "plot",
            "--config", pipeline_yaml,
            "--model", trained_artifact,
            "--output", out_dir,
        ])
        plots_dir = Path(out_dir) / "plots"
        pngs = list(plots_dir.glob("*.png"))
        assert len(pngs) > 0, f"No PNGs found in {plots_dir}"

    def test_feature_importance_png_created(self, pipeline_yaml, trained_artifact, shared_tmp):
        out_dir = str(shared_tmp / "plot_fi")
        runner = CliRunner()
        runner.invoke(cli, [
            "plot",
            "--config", pipeline_yaml,
            "--model", trained_artifact,
            "--output", out_dir,
        ])
        fi_png = Path(out_dir) / "plots" / "feature_importances.png"
        assert fi_png.exists()

    def test_output_override(self, pipeline_yaml, trained_artifact, shared_tmp):
        custom_out = str(shared_tmp / "plot_custom_out")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "plot",
            "--config", pipeline_yaml,
            "--model", trained_artifact,
            "--output", custom_out,
        ])
        assert result.exit_code == 0, result.output
        assert Path(custom_out, "plots").is_dir()


# ── run (full pipeline) ───────────────────────────────────────────────────────

class TestRunCommand:
    def test_exit_code_zero(self, pipeline_yaml, shared_tmp):
        out_dir = str(shared_tmp / "run_out")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--config", pipeline_yaml, "--output", out_dir,
        ])
        assert result.exit_code == 0, result.output

    def test_produces_artifact_and_plots(self, pipeline_yaml, shared_tmp):
        out_dir = str(shared_tmp / "run_full")
        runner = CliRunner()
        runner.invoke(cli, [
            "run", "--config", pipeline_yaml, "--output", out_dir,
        ])
        assert Path(out_dir, "model_artifact.joblib").exists()
        pngs = list((Path(out_dir) / "plots").glob("*.png"))
        assert len(pngs) > 0

    def test_done_message(self, pipeline_yaml, shared_tmp):
        out_dir = str(shared_tmp / "run_done_msg")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--config", pipeline_yaml, "--output", out_dir,
        ])
        assert "Done" in result.output

    def test_retrain_flag_retrains(self, pipeline_yaml, shared_tmp):
        """--retrain should overwrite an existing artifact (newer mtime)."""
        out_dir = str(shared_tmp / "run_retrain")
        runner = CliRunner()

        # First run
        runner.invoke(cli, ["run", "--config", pipeline_yaml, "--output", out_dir])
        artifact = Path(out_dir, "model_artifact.joblib")
        mtime_first = artifact.stat().st_mtime

        time.sleep(0.05)  # ensure filesystem mtime differs

        # Second run with --retrain
        runner.invoke(cli, [
            "run", "--config", pipeline_yaml, "--output", out_dir, "--retrain",
        ])
        mtime_second = artifact.stat().st_mtime

        assert mtime_second > mtime_first, "Artifact was not overwritten by --retrain"

    def test_no_retrain_reuses_artifact(self, pipeline_yaml, shared_tmp):
        """Without --retrain, an existing artifact must not be overwritten."""
        out_dir = str(shared_tmp / "run_no_retrain")
        runner = CliRunner()

        # First run to create artifact
        runner.invoke(cli, ["run", "--config", pipeline_yaml, "--output", out_dir])
        artifact = Path(out_dir, "model_artifact.joblib")
        mtime_first = artifact.stat().st_mtime

        time.sleep(0.05)

        # Second run without --retrain
        result = runner.invoke(cli, [
            "run", "--config", pipeline_yaml, "--output", out_dir,
        ])
        mtime_second = artifact.stat().st_mtime

        assert mtime_second == mtime_first, "Artifact was unexpectedly overwritten"
        assert "Loading Existing Model" in result.output

    def test_output_override(self, pipeline_yaml, h5ad_path, shared_tmp):
        custom_out = str(shared_tmp / "run_custom_out")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--config", pipeline_yaml, "--output", custom_out,
        ])
        assert result.exit_code == 0, result.output
        assert Path(custom_out, "model_artifact.joblib").exists()
