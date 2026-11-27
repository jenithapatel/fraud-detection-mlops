from fraud.monitoring.simulate_drift import (
    drop_column,
    inject_out_of_range,
    rename_column,
    scale_amount,
    swap_columns,
)


def test_scale_amount(fake_transactions):
    out = scale_amount(fake_transactions, 10)
    assert (out["Amount"] == fake_transactions["Amount"] * 10).all()


def test_out_of_range_changes_fraction(fake_transactions):
    out = inject_out_of_range(fake_transactions, frac=0.2)
    changed = (out["V1"] != fake_transactions["V1"]).sum()
    assert changed == 200


def test_swap_columns(fake_transactions):
    out = swap_columns(fake_transactions, "V1", "V2")
    assert (out["V1"] == fake_transactions["V2"]).all()
    assert (out["V2"] == fake_transactions["V1"]).all()


def test_drift_report_detects_scaled_amount(fake_transactions):
    from fraud import config
    from fraud.monitoring.drift_report import build_report, drift_share

    ref = fake_transactions[config.FEATURES]
    clean = drift_share(build_report(ref, ref.sample(frac=1, random_state=1)))
    drifted = drift_share(build_report(ref, scale_amount(ref, 10)))
    assert clean == 0.0 and drifted > 0.0


def test_drop_column(fake_transactions):
    out = drop_column(fake_transactions, "V14")
    assert "V14" not in out.columns
    assert len(out.columns) == len(fake_transactions.columns) - 1


def test_rename_column(fake_transactions):
    out = rename_column(fake_transactions, "Amount", "amount_usd")
    assert "Amount" not in out.columns and "amount_usd" in out.columns


def test_schema_scenarios_are_rejected_by_api(fake_transactions):
    """Dropped and renamed columns must come back as 422 from /predict."""
    from fastapi.testclient import TestClient

    from fraud.serving.app import app

    client = TestClient(app)
    for broken in (drop_column(fake_transactions), rename_column(fake_transactions)):
        rows = broken.drop(columns=["Class"]).head(3).to_dict(orient="records")
        resp = client.post("/predict", json={"transactions": rows})
        assert resp.status_code == 422
        assert "Field required" in resp.json()["error"]


def test_push_drift_share_pushes_gauge(monkeypatch):
    """The drift share is pushed as the fraud_data_drift_share gauge under its own job."""
    from fraud.monitoring import drift_report

    captured = {}

    def fake_push(url, job, registry):
        captured["url"], captured["job"] = url, job
        captured["value"] = registry.get_sample_value("fraud_data_drift_share")

    monkeypatch.setattr(drift_report, "push_to_gateway", fake_push)
    assert drift_report.push_drift_share(0.4, "http://gw:9091") is True
    assert captured == {"url": "http://gw:9091", "job": "fraud_drift_report", "value": 0.4}


def test_push_drift_share_survives_gateway_down():
    """A dead Pushgateway must not crash the drift report."""
    from fraud.monitoring.drift_report import push_drift_share

    assert push_drift_share(0.1, "http://127.0.0.1:1") is False
