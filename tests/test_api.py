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
