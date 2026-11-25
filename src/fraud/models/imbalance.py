"""Class-imbalance study: class weights vs. SMOTE oversampling.

Owner: Akshaj. Run with `python -m fraud.models.imbalance` (MLflow server must be up).

Each base learner is trained twice on the same train/validation split as
`train.py`: once with class weighting and once with SMOTE inside an imblearn
Pipeline (so synthetic rows are only created from the training fold, never the
validation data). Runs go to a separate experiment and nothing is registered,
so the serving image never needs imbalanced-learn.
"""

import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud import config
from fraud.models.metrics import pr_auc, recall_at_precision

IMBALANCE_EXPERIMENT = f"{config.EXPERIMENT_NAME}-imbalance"


def build_strategies(y_train) -> dict:
    """(learner, strategy) -> unfitted model, covering class weights and SMOTE."""
    n_pos = int((y_train == 1).sum())
    spw = (len(y_train) - n_pos) / max(n_pos, 1)

    def xgb(**kw):
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="aucpr",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            **kw,
        )

    def smote():
        return SMOTE(random_state=config.RANDOM_STATE)

    return {
        ("logreg", "class_weight"): make_imb_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
        ),
        ("logreg", "smote"): make_imb_pipeline(
            StandardScaler(), smote(), LogisticRegression(max_iter=1000)
        ),
        ("xgboost", "class_weight"): xgb(scale_pos_weight=spw),
        ("xgboost", "smote"): make_imb_pipeline(smote(), xgb()),
    }


def main() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(IMBALANCE_EXPERIMENT)

    df = pd.read_csv(config.TRAIN_PATH)  # test.csv is never read here
    X, y = df[config.FEATURES], df[config.TARGET]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=config.VALIDATION_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    rows = []
    for (learner, strategy), model in build_strategies(y_tr).items():
        with mlflow.start_run(run_name=f"{learner}_{strategy}"):
            model.fit(X_tr, y_tr)
            scores = model.predict_proba(X_val)[:, 1]
            ap = pr_auc(y_val, scores)
            recall, threshold = recall_at_precision(y_val, scores, config.TARGET_PRECISION)
            mlflow.log_params({"learner": learner, "imbalance_strategy": strategy})
            mlflow.log_metrics(
                {"val_pr_auc": ap, "val_recall_at_p90": recall, "val_threshold_at_p90": threshold}
            )
            rows.append(
                {"learner": learner, "strategy": strategy, "pr_auc": ap, "recall_at_p90": recall}
            )
            print(f"{learner}/{strategy}: PR-AUC={ap:.4f} recall@P90={recall:.4f}")

    config.REPORTS_DIR.mkdir(exist_ok=True)
    out = config.REPORTS_DIR / "imbalance_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
