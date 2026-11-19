"""Load the champion model from the MLflow registry (or a dummy for tests)."""

import os
from dataclasses import dataclass

import numpy as np

from fraud import config


class DummyModel:
    """Always predicts a 1% fraud probability. Used in tests and before a model exists."""

    def predict_proba(self, X):
        p = np.full(len(X), 0.01)
        return np.column_stack([1 - p, p])


@dataclass
class LoadedModel:
    model: object
    version: str
    threshold: float


def load_model() -> LoadedModel:
    """FRAUD_MODEL_SOURCE=dummy -> DummyModel; otherwise models:/<name>@<alias>."""
    if os.getenv("FRAUD_MODEL_SOURCE", "registry") == "dummy":
        return LoadedModel(DummyModel(), "dummy", config.DEFAULT_THRESHOLD)

    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mv = MlflowClient().get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}")
    threshold = float(mv.tags.get("threshold", config.DEFAULT_THRESHOLD))
    return LoadedModel(model, str(mv.version), threshold)
