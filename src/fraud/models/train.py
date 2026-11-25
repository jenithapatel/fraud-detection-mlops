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
from flaml import AutoML
from matplotlib.figure import Figure
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud import config
from fraud.models.metrics import pr_auc, recall_at_precision
from fraud.models.versioning import bump_semver, schema_hash

# Wall-clock budget (seconds) for the FLAML search.
FLAML_TIME_BUDGET = int(os.getenv("FLAML_TIME_BUDGET", "60"))


def candidate_models(y_train) -> dict:
    """Models to compare. Each is an sklearn-compatible estimator, or a FLAML AutoML.

    `y_train` is used to compute XGBoost's `scale_pos_weight` (negatives per
    positive), the standard way to handle the 0.17% fraud rate. FLAML is
    limited to estimators already in requirements-serving.txt (XGBoost and
    sklearn trees), so the registered model always loads in the API image.
    """
    n_pos = int((y_train == 1).sum())
    spw = (len(y_train) - n_pos) / max(n_pos, 1)
    return {
        "logreg_balanced": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
        ),
        "xgboost_spw": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "flaml_automl": AutoML(),
    }


def fit_model(model, X, y):
    """Fit a candidate and return a fitted sklearn-style estimator.

    FLAML's AutoML object is not loadable by serving, so we run the search and
    return the best fitted estimator inside it (`automl.model.estimator`).
    """
    if isinstance(model, AutoML):
        model.fit(
            X,
            y,
            task="classification",
            metric="ap",  # average precision == PR-AUC
            time_budget=FLAML_TIME_BUDGET,
            estimator_list=["xgboost", "rf", "extra_tree"],
            seed=config.RANDOM_STATE,
            verbose=0,
            # FLAML would auto-log trials with skops (breaks on XGBoost); we log the run ourselves.
            mlflow_logging=False,
        )
        return model.model.estimator
    model.fit(X, y)
    return model


def _loggable_params(model) -> dict:
    """Hyperparameters of the (final) model that MLflow can store as strings/numbers."""
    params = model.get_params() if hasattr(model, "get_params") else {}
    return {
        k: v for k, v in params.items() if isinstance(v, (str, int, float, bool)) and v is not None
    }


def _pr_curve_figure(y_true, scores, name: str, ap: float):
    """Precision-recall curve plot, logged as a run artifact."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    # Figure (not pyplot) needs no GUI backend, so it works inside Airflow/containers.
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    ax.plot(recall, precision)
    ax.axhline(config.TARGET_PRECISION, color="gray", linestyle="--", label="target precision")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"{name} (PR-AUC={ap:.3f})")
    ax.legend()
    fig.tight_layout()
    return fig


def _previous_tags(client: MlflowClient) -> dict | None:
    """Tags of the latest registered version, or None if nothing is registered yet."""
    versions = client.search_model_versions(f"name='{config.MODEL_NAME}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version)).tags


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
    for name, model in candidate_models(y_tr).items():
        with mlflow.start_run(run_name=name):
            model = fit_model(model, X_tr, y_tr)
            scores = model.predict_proba(X_val)[:, 1]
            ap = pr_auc(y_val, scores)
            recall, threshold = recall_at_precision(y_val, scores, config.TARGET_PRECISION)

            mlflow.log_param("model_type", name)
            mlflow.log_params(_loggable_params(model))
            if name == "flaml_automl":
                mlflow.log_param("flaml_time_budget_s", FLAML_TIME_BUDGET)
            mlflow.log_metrics(
                {"val_pr_auc": ap, "val_recall_at_p90": recall, "val_threshold_at_p90": threshold}
            )
            mlflow.log_figure(_pr_curve_figure(y_val, scores, name, ap), "pr_curve.png")
            # cloudpickle: MLflow 3's default skops format rejects XGBoost types, and
            # serving's mlflow.sklearn.load_model passes no trusted-types list.
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=X_tr.head(5),
                serialization_format="cloudpickle",
            )
            print(f"{name}: PR-AUC={ap:.4f} recall@P90={recall:.4f}")

            if best is None or ap > best[0]:
                best = (ap, info.model_uri, threshold, name)

    ap, model_uri, threshold, name = best
    client = MlflowClient()
    prev = _previous_tags(client)  # read before registering so we compare to the old model
    version = mlflow.register_model(model_uri, config.MODEL_NAME)
    features_hash = schema_hash(config.FEATURES)
    semver = os.getenv("MODEL_SEMVER") or bump_semver(prev, name, features_hash)
    for key, value in {
        "threshold": threshold,
        "semver": semver,  # MAJOR=schema change, MINOR=new algorithm, PATCH=retrain
        "model_type": name,
        "schema_hash": features_hash,
    }.items():
        client.set_model_version_tag(config.MODEL_NAME, version.version, key, str(value))
    client.set_registered_model_alias(config.MODEL_NAME, config.MODEL_ALIAS, version.version)
    print(
        f"Registered {name} as {config.MODEL_NAME} v{version.version} "
        f"(semver {semver}) @{config.MODEL_ALIAS}"
    )


if __name__ == "__main__":
    main()
