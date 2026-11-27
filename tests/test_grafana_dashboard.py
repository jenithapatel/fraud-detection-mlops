"""Guards for the provisioned Grafana dashboard (monitoring/grafana/)."""

import json
import re
from pathlib import Path

DASHBOARD = Path("monitoring/grafana/dashboards/fraud-monitoring.json")
# Metrics the API and the drift report actually export; a typo in a panel query
# would otherwise only show up as an empty panel in Grafana.
KNOWN_METRICS = {
    "fraud_predictions_total",
    "fraud_predict_latency_seconds_count",
    "fraud_predict_latency_seconds_bucket",
    "fraud_probability_bucket",
    "fraud_model_loaded",
    "fraud_data_drift_share",
}


def test_dashboard_is_valid_and_has_all_panels():
    """The JSON parses, has the fixed uid, and covers the six required views."""
    dash = json.loads(DASHBOARD.read_text())
    assert dash["uid"] == "fraud-monitoring"
    titles = " ".join(p["title"].lower() for p in dash["panels"])
    for needed in ("request rate", "fraud rate", "probability", "latency", "drift", "model loaded"):
        assert needed in titles


def test_panel_queries_use_known_metrics_and_datasource():
    """Every query references an exported metric and the provisioned datasource uid."""
    dash = json.loads(DASHBOARD.read_text())
    for panel in dash["panels"]:
        assert panel["datasource"]["uid"] == "prometheus"
        for target in panel["targets"]:
            used = set(re.findall(r"fraud_[a-z_]+", target["expr"]))
            assert used and used <= KNOWN_METRICS, (panel["title"], used - KNOWN_METRICS)
