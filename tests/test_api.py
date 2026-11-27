import pytest
from fastapi.testclient import TestClient

from fraud import config


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUD_MODEL_SOURCE", "dummy")
    monkeypatch.setattr(config, "PREDICTION_LOG_PATH", tmp_path / "predictions.csv")
    from fraud.serving.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_and_log(client, fake_transactions):
    rows = fake_transactions[config.FEATURES].head(3).to_dict(orient="records")
    resp = client.post("/predict", json={"transactions": rows})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 3
    assert 0 <= body["predictions"][0]["fraud_probability"] <= 1
    assert config.PREDICTION_LOG_PATH.exists()


def test_missing_feature_rejected(client):
    resp = client.post("/predict", json={"transactions": [{"Time": 0.0}]})
    assert resp.status_code == 422


def test_missing_feature_error_names_the_field(client):
    resp = client.post("/predict", json={"transactions": [{"Time": 0.0}]})
    body = resp.json()
    assert "transactions.0.V1" in body["error"]
    assert body["detail"]  # FastAPI's usual error list is still there


def test_request_id_echoed(client):
    resp = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert resp.headers["X-Request-ID"] == "abc123"
    assert client.get("/health").headers["X-Request-ID"]  # generated when absent


def test_model_failure_returns_500(client, fake_transactions):
    from fraud.serving import app as app_module

    class BrokenModel:
        def predict_proba(self, X):
            raise ValueError("boom")

    app_module.STATE["loaded"].model = BrokenModel()
    rows = fake_transactions[config.FEATURES].head(2).to_dict(orient="records")
    resp = client.post("/predict", json={"transactions": rows})
    assert resp.status_code == 500
    assert "failed to score" in resp.json()["detail"]


def test_no_model_returns_503(client, fake_transactions):
    from fraud.serving import app as app_module

    app_module.STATE["loaded"] = None
    rows = fake_transactions[config.FEATURES].head(1).to_dict(orient="records")
    resp = client.post("/predict", json={"transactions": rows})
    assert resp.status_code == 503


def test_schema_violation_counted(client):
    from prometheus_client import REGISTRY

    def count() -> float:
        value = REGISTRY.get_sample_value(
            "fraud_schema_violations_total", {"path": "/predict"}
        )
        return value or 0.0

    before = count()
    client.post("/predict", json={"transactions": [{"Time": 0.0}]})
    client.post("/predict", json={"transactions": [{"Amount": "not a number"}]})
    assert count() == before + 2
    assert "fraud_schema_violations_total" in client.get("/metrics/").text


def test_startup_retries_until_model_loads(monkeypatch):
    """The first two loads fail (MLflow not up yet), the third succeeds."""
    from fraud.serving import app as app_module
    from fraud.serving.model_loader import DummyModel, LoadedModel

    calls = {"n": 0}

    def flaky_load():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("MLflow not ready")
        return LoadedModel(DummyModel(), "7", 0.5)

    monkeypatch.setattr(app_module, "load_model", flaky_load)
    monkeypatch.setenv("FRAUD_LOAD_RETRY_DELAY", "0")
    app_module._load_with_retry()
    assert calls["n"] == 3
    assert app_module.STATE["loaded"].version == "7"


def test_startup_gives_up_but_api_stays_up(monkeypatch):
    from fraud.serving import app as app_module

    def always_fail():
        raise ConnectionError("MLflow down")

    monkeypatch.setattr(app_module, "load_model", always_fail)
    monkeypatch.setenv("FRAUD_LOAD_RETRIES", "3")
    monkeypatch.setenv("FRAUD_LOAD_RETRY_DELAY", "0")
    with TestClient(app_module.app) as c:
        assert c.get("/health").json()["status"] == "no_model"


def test_health_explains_failed_load(monkeypatch):
    from fraud.serving import app as app_module

    def no_champion():
        raise LookupError("alias champion not found")

    monkeypatch.setattr(app_module, "load_model", no_champion)
    monkeypatch.setenv("FRAUD_LOAD_RETRIES", "1")
    with TestClient(app_module.app) as c:
        body = c.get("/health").json()
    assert body["status"] == "no_model"
    assert body["error"] == "LookupError: alias champion not found"


def test_health_has_no_error_when_loaded(client):
    assert "error" not in client.get("/health").json()
