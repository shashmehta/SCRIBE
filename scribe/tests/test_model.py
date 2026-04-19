"""Unit tests for scribe/model.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from scribe import model as cellmodel


RNG = np.random.default_rng(7)
N = 100
N_FEATURES = 20


@pytest.fixture(scope="module")
def xy():
    X = RNG.standard_normal((N, N_FEATURES)).astype(np.float32)
    y_labels = np.array(["malignant"] * (N // 2) + ["normal"] * (N // 2))
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    return X, y, le


@pytest.fixture(scope="module")
def trained_clf(xy):
    X, y, _ = xy
    return cellmodel.train(X, y, n_estimators=5)


def test_train_returns_classifier(trained_clf):
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(trained_clf, RandomForestClassifier)
    assert hasattr(trained_clf, "feature_importances_")


def test_evaluate_returns_dict_with_expected_keys(trained_clf, xy):
    X, y, le = xy
    result = cellmodel.evaluate(trained_clf, X, y, le)
    assert isinstance(result, dict)
    assert "y_pred" in result
    assert "report_str" in result
    assert "confusion_matrix" in result


def test_evaluate_y_pred_shape(trained_clf, xy):
    X, y, le = xy
    result = cellmodel.evaluate(trained_clf, X, y, le)
    assert result["y_pred"].shape == y.shape


def test_save_load_roundtrip(tmp_path, trained_clf, xy):
    _, _, le = xy
    gene_names = [f"GENE{i:03d}" for i in range(N_FEATURES)]
    save_path = str(tmp_path / "artifact.joblib")

    cellmodel.save_artifact(save_path, trained_clf, le, gene_names)
    loaded_clf, loaded_le, loaded_genes = cellmodel.load_artifact(save_path)

    assert loaded_genes == gene_names
    assert list(loaded_le.classes_) == list(le.classes_)
    assert hasattr(loaded_clf, "feature_importances_")


def test_get_feature_importances_shape(trained_clf):
    gene_names = [f"GENE{i:03d}" for i in range(N_FEATURES)]
    imp = cellmodel.get_feature_importances(trained_clf, gene_names, top_n=5)
    assert len(imp) == 5
