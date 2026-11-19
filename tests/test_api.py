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
