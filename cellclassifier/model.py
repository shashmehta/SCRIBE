"""Model training, evaluation, and persistence for PDAC classification."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a RandomForestClassifier on the training data.

    Uses balanced class weights by default to handle tumor/normal imbalance.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        n_estimators: Number of trees in the forest.
        class_weight: Class weight strategy ('balanced' adjusts for imbalance).
        random_state: Random seed for reproducibility.

    Returns:
        The trained RandomForestClassifier.
    """
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("  Training complete.")
    return clf


def evaluate(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    """Evaluate the model on the test set.

    Args:
        model: Trained RandomForestClassifier.
        X_test: Test feature matrix.
        y_test: Test labels.
        label_encoder: Fitted LabelEncoder for decoding class names.

    Returns:
        Dict with keys: 'y_pred', 'report_str', 'confusion_matrix'.
    """
    y_pred = model.predict(X_test)
    target_names = label_encoder.classes_.tolist()

    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return {"y_pred": y_pred, "report_str": report, "confusion_matrix": cm}


def get_feature_importances(
    model: RandomForestClassifier,
    gene_names: list[str],
    top_n: int = 20,
) -> pd.Series:
    """Get the top N most important genes from the trained model.

    Args:
        model: Trained RandomForestClassifier.
        gene_names: Gene names corresponding to feature columns.
        top_n: Number of top genes to return.

    Returns:
        pd.Series of top gene importances, sorted descending.
    """
    importances = pd.Series(model.feature_importances_, index=gene_names)
    top = importances.nlargest(top_n)

    print(f"\nTop {top_n} gene feature importances:")
    print(top.to_string())

    return top


def save_artifact(
    path: str,
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    gene_names: list[str],
) -> None:
    """Save model, label encoder, and gene names as a single artifact.

    Args:
        path: File path to save the artifact (e.g., 'model_artifact.joblib').
        model: Trained RandomForestClassifier.
        label_encoder: Fitted LabelEncoder.
        gene_names: List of gene names matching the model's feature order.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "gene_names": gene_names,
    }
    joblib.dump(artifact, path)
    print(f"Model artifact saved to {path}")


def load_artifact(
    path: str,
) -> tuple[RandomForestClassifier, LabelEncoder, list[str]]:
    """Load a previously saved model artifact.

    Args:
        path: Path to the saved artifact file.

    Returns:
        Tuple of (model, label_encoder, gene_names).

    Raises:
        FileNotFoundError: If the artifact file does not exist.
        KeyError: If the artifact is missing required keys.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")

    artifact = joblib.load(path)

    # Validate required keys
    required = {"model", "label_encoder", "gene_names"}
    missing = required - set(artifact.keys())
    if missing:
        raise KeyError(f"Artifact is missing required keys: {missing}")

    print(f"Loaded model artifact from {path}")
    return artifact["model"], artifact["label_encoder"], artifact["gene_names"]
