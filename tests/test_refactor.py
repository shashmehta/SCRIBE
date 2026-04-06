"""Tests to verify the codebase refactor (cellclassifier -> scribe) is complete and correct."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


class TestPackageRename:
    """Verify the package rename from cellclassifier to scribe."""

    def test_scribe_package_importable(self):
        """scribe package can be imported."""
        import scribe
        assert hasattr(scribe, '__doc__')

    def test_all_modules_importable(self):
        """All scribe submodules can be imported."""
        modules = [
            'scribe.cli', 'scribe.config', 'scribe.data', 'scribe.geo',
            'scribe.model', 'scribe.analysis', 'scribe.plotting',
            'scribe.batch', 'scribe.monitor', 'scribe.zarr_utils', 'scribe.cache',
        ]
        for mod_name in modules:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Failed to import {mod_name}"

    def test_no_cellclassifier_references(self):
        """No source files still reference 'cellclassifier' (excluding this test file)."""
        root = Path(__file__).parent.parent
        this_file = Path(__file__).resolve()
        offending = []
        for pattern in ['scribe/**/*.py', 'tests/**/*.py', 'app.py', 'run.py', 'conftest.py']:
            for f in root.glob(pattern):
                if f.resolve() == this_file:
                    continue
                content = f.read_text()
                if 'cellclassifier' in content:
                    offending.append(str(f.relative_to(root)))
        assert offending == [], f"Files still referencing 'cellclassifier': {offending}"

    def test_old_package_not_importable(self):
        """cellclassifier package no longer exists."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module('cellclassifier')


class TestProjectStructure:
    """Verify cleanup removed expected files/dirs."""

    def test_no_empty_dirs(self):
        """Empty placeholder directories were removed."""
        root = Path(__file__).parent.parent
        for name in ['data', 'models', 'plots']:
            assert not (root / name).exists(), f"Empty dir '{name}/' should have been deleted"

    def test_no_dev_artifacts(self):
        """Development artifacts were removed."""
        root = Path(__file__).parent.parent
        assert not (root / 'plans').exists(), "plans/ should have been deleted"
        assert not (root / 'geo_data_processing.ipynb').exists(), "notebook should have been deleted"

    def test_pyproject_toml_exists(self):
        """pyproject.toml was created."""
        root = Path(__file__).parent.parent
        assert (root / 'pyproject.toml').exists()

    def test_no_requirements_txt(self):
        """requirements.txt was replaced by pyproject.toml."""
        root = Path(__file__).parent.parent
        assert not (root / 'requirements.txt').exists()


class TestCLIEntryPoint:
    """Verify the CLI works after refactor."""

    def test_scribe_cli_help(self):
        """scribe CLI entry point runs --help without error."""
        result = subprocess.run(
            ['scribe', '--help'],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'usage' in output.lower() or 'Usage' in output

    def test_run_py_still_works(self):
        """run.py convenience entry point still works."""
        root = Path(__file__).parent.parent
        result = subprocess.run(
            [sys.executable, str(root / 'run.py'), '--help'],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
