import numpy as np
import pandas as pd

from fraud import config
from fraud.models import train


def _toy(n=600):
    """Small imbalanced frame with the real schema; fraud shifts V1."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, len(config.FEATURES))), columns=config.FEATURES)
    y = pd.Series((rng.random(n) < 0.1).astype(int), name=config.TARGET)
    X.loc[y == 1, "V1"] += 3
    return X, y


def test_candidates_include_all_algorithms():
    _, y = _toy()
    assert {"logreg_balanced", "xgboost_spw", "flaml_automl"} <= set(train.candidate_models(y))


def test_xgboost_scale_pos_weight_matches_imbalance():
    _, y = _toy()
    spw = train.candidate_models(y)["xgboost_spw"].scale_pos_weight
    assert spw == (y == 0).sum() / (y == 1).sum()


def test_flaml_returns_servable_estimator(monkeypatch):
    X, y = _toy()
    monkeypatch.setattr(train, "FLAML_TIME_BUDGET", 3)
    fitted = train.fit_model(train.candidate_models(y)["flaml_automl"], X, y)
    # Must be a plain estimator with predict_proba, not the AutoML object.
    assert not isinstance(fitted, train.AutoML)
    assert fitted.predict_proba(X.head(5)).shape == (5, 2)
