"""Stage 2: train candidate models, track them in MLflow, register the best.

Owner: Akshaj. Run with `python -m fraud.models.train` (MLflow server must be up).

Contract with serving: the registered model is logged with `mlflow.sklearn`
(so it has predict_proba), takes the raw columns in config.FEATURES, gets the
alias config.MODEL_ALIAS, and carries a "threshold" tag on its model version.
"""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud import config
from fraud.models.metrics import pr_auc, recall_at_precision


def candidate_models() -> dict:
    """Models to compare. Each must be an sklearn-compatible estimator.

    TODO(Akshaj): add XGBoost (scale_pos_weight) and FLAML AutoML. For FLAML,
    log `automl.model.estimator` (the fitted sklearn-style model), not the
    AutoML object, so serving can load it with mlflow.sklearn.
    """
    return {
        "logreg_balanced": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
        ),
    }


def main() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df = pd.read_csv(config.TRAIN_PATH)
    X, y = df[config.FEATURES], df[config.TARGET]
    # Model selection uses a validation split of TRAIN only; test.csv stays untouched.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=config.VALIDATION_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    best = None  # (pr_auc, model_uri, threshold, run_name)
    for name, model in candidate_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_tr, y_tr)
            scores = model.predict_proba(X_val)[:, 1]
            ap = pr_auc(y_val, scores)
            recall, threshold = recall_at_precision(y_val, scores, config.TARGET_PRECISION)

            mlflow.log_param("model_type", name)
            mlflow.log_metrics(
                {"val_pr_auc": ap, "val_recall_at_p90": recall, "val_threshold_at_p90": threshold}
            )
            info = mlflow.sklearn.log_model(model, name="model", input_example=X_tr.head(5))
            print(f"{name}: PR-AUC={ap:.4f} recall@P90={recall:.4f}")

            if best is None or ap > best[0]:
                best = (ap, info.model_uri, threshold, name)

    ap, model_uri, threshold, name = best
    version = mlflow.register_model(model_uri, config.MODEL_NAME)
    client = MlflowClient()
    client.set_model_version_tag(config.MODEL_NAME, version.version, "threshold", str(threshold))
    # Semantic version label (registry versions are plain integers).
    # TODO(Akshaj): bump MAJOR for feature/schema changes, MINOR for new algorithms, PATCH retrains.
    client.set_model_version_tag(
        config.MODEL_NAME, version.version, "semver", os.getenv("MODEL_SEMVER", "0.1.0")
    )
    client.set_registered_model_alias(config.MODEL_NAME, config.MODEL_ALIAS, version.version)
    print(f"Registered {name} as {config.MODEL_NAME} v{version.version} @{config.MODEL_ALIAS}")


if __name__ == "__main__":
    main()
