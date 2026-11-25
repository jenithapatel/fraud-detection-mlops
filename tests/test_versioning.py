from fraud import config
from fraud.models.versioning import bump_semver, schema_hash

H = schema_hash(config.FEATURES)


def tags(semver="1.2.3", model_type="xgboost_spw", h=H):
    return {"semver": semver, "model_type": model_type, "schema_hash": h}


def test_first_model():
    assert bump_semver(None, "logreg_balanced", H) == "0.1.0"


def test_retrain_bumps_patch():
    assert bump_semver(tags(), "xgboost_spw", H) == "1.2.4"


def test_new_algorithm_bumps_minor():
    assert bump_semver(tags(), "flaml_automl", H) == "1.3.0"


def test_schema_change_bumps_major():
    assert bump_semver(tags(), "xgboost_spw", schema_hash(["a", "b"])) == "2.0.0"


def test_schema_hash_is_order_sensitive():
    assert schema_hash(["a", "b"]) != schema_hash(["b", "a"])
