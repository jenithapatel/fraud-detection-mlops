import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud import config
from fraud.models.imbalance import build_strategies
from fraud.models.metrics import pr_auc


def test_all_four_strategies_fit_and_predict():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(800, len(config.FEATURES))), columns=config.FEATURES)
    y = pd.Series((rng.random(800) < 0.1).astype(int))
    X.loc[y == 1, "V1"] += 3
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, random_state=0)

    strategies = build_strategies(y_tr)
    assert set(strategies) == {
        ("logreg", "class_weight"),
        ("logreg", "smote"),
        ("xgboost", "class_weight"),
        ("xgboost", "smote"),
    }
    for model in strategies.values():
        model.fit(X_tr, y_tr)
        assert pr_auc(y_val, model.predict_proba(X_val)[:, 1]) > 0.5
