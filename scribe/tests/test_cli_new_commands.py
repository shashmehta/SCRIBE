"""CLI integration tests for the new commands added in the scripts refactor.

Uses Click's CliRunner for in-process execution. All tests use the tiny
synthetic .h5ad fixture — no real output/ data required.
"""

from __future__ import annotations

import os
import pytest
from click.testing import CliRunner

from scribe.cli import cli


# ── TestLfcPlotCommand ────────────────────────────────────────────────────────

class TestLfcPlotCommand:
    def test_success_creates_png(self, tiny_combined_h5ad, tmp_path):
        r = CliRunner().invoke(cli, [
            "lfc-plot",
            "--data", tiny_combined_h5ad,
            "--output", str(tmp_path),
            "--condition-col", "condition",
            "--batch-key", "dataset",
            "--numerator", "malignant",
            "--denominator", "normal",
        ])
        assert r.exit_code == 0, r.output
        assert any(f.endswith(".png") for f in os.listdir(tmp_path))

    def test_missing_data_flag_errors(self):
        r = CliRunner().invoke(cli, ["lfc-plot"])
        assert r.exit_code != 0
        assert "Missing option" in r.output or "Error" in r.output

    def test_custom_filename(self, tiny_combined_h5ad, tmp_path):
        r = CliRunner().invoke(cli, [
            "lfc-plot",
            "--data", tiny_combined_h5ad,
            "--output", str(tmp_path),
            "--filename", "my_lfc.png",
        ])
        assert r.exit_code == 0, r.output
        assert (tmp_path / "my_lfc.png").exists()


# ── TestVolcanoCommand ────────────────────────────────────────────────────────

class TestVolcanoCommand:
    def test_success_creates_png(self, tiny_combined_h5ad, tmp_path):
        r = CliRunner().invoke(cli, [
            "volcano",
            "--data", tiny_combined_h5ad,
            "--output", str(tmp_path),
            "--condition-col", "condition",
            "--batch-key", "dataset",
        ])
        assert r.exit_code == 0, r.output
        assert any(f.endswith(".png") for f in os.listdir(tmp_path))

    def test_missing_data_flag_errors(self):
        r = CliRunner().invoke(cli, ["volcano"])
        assert r.exit_code != 0

    def test_custom_thresholds(self, tiny_combined_h5ad, tmp_path):
        r = CliRunner().invoke(cli, [
            "volcano",
            "--data", tiny_combined_h5ad,
            "--output", str(tmp_path),
            "--pval", "0.1",
            "--lfc", "0.5",
            "--n-labels", "3",
            "--filename", "volcano_custom.png",
        ])
        assert r.exit_code == 0, r.output
        assert (tmp_path / "volcano_custom.png").exists()


# ── TestFeatureGridCommand ────────────────────────────────────────────────────

class TestFeatureGridCommand:
    def test_success_creates_png(self, tiny_combined_h5ad, tmp_path):
        r = CliRunner().invoke(cli, [
            "feature-grid",
            "--data", tiny_combined_h5ad,
            "--output", str(tmp_path),
            "--n-estimators", "5",
        ])
        assert r.exit_code == 0, r.output
        assert any(f.endswith(".png") for f in os.listdir(tmp_path))

    def test_missing_data_flag_errors(self):
        r = CliRunner().invoke(cli, ["feature-grid"])
        assert r.exit_code != 0

    def test_custom_n_top(self, tiny_combined_h5ad, tmp_path):
        r = CliRunner().invoke(cli, [
            "feature-grid",
            "--data", tiny_combined_h5ad,
            "--output", str(tmp_path),
            "--n-top", "3",
            "--n-estimators", "5",
            "--filename", "fi_custom.png",
        ])
        assert r.exit_code == 0, r.output
        assert (tmp_path / "fi_custom.png").exists()


# ── TestHkPcaCompareCommand ───────────────────────────────────────────────────

_HK_GENES = "GENE000,GENE001,GENE002,GENE003"  # use fixture genes (no real HK genes in synthetic data)


class TestHkPcaCompareCommand:
    def test_success_creates_png(self, tiny_combined_h5ad, tmp_path):
        r = CliRunner().invoke(cli, [
            "hk-pca-compare",
            "--uncorrected", tiny_combined_h5ad,
            "--output", str(tmp_path),
            "--hk-genes", _HK_GENES,
        ])
        assert r.exit_code == 0, r.output
        assert any(f.endswith(".png") for f in os.listdir(tmp_path))

    def test_missing_uncorrected_flag_errors(self):
        r = CliRunner().invoke(cli, ["hk-pca-compare"])
        assert r.exit_code != 0

    def test_two_panels_with_combat(self, tiny_combined_h5ad, tmp_path):
        r = CliRunner().invoke(cli, [
            "hk-pca-compare",
            "--uncorrected", tiny_combined_h5ad,
            "--combat", tiny_combined_h5ad,   # use same file as stand-in for combat
            "--output", str(tmp_path),
            "--hk-genes", _HK_GENES,
            "--filename", "hk_2panel.png",
        ])
        assert r.exit_code == 0, r.output
        assert (tmp_path / "hk_2panel.png").exists()
