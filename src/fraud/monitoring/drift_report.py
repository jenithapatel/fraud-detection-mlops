"""Stage 4: compare recent API traffic with the training data using Evidently.

Owner: Devyani. Run with `python -m fraud.monitoring.drift_report`.
Writes reports/drift_report.html and exits with code 1 if drift is detected,
so Airflow or a cron job can alert on it.
"""

import argparse
import logging
import os
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from fraud import config

log = logging.getLogger("fraud.monitoring")

# Where the Pushgateway listens (docker-compose.yml). Override with FRAUD_PUSHGATEWAY_URL,
# e.g. http://pushgateway:9091 when this runs inside Airflow.
PUSHGATEWAY_URL = os.getenv("FRAUD_PUSHGATEWAY_URL", "http://localhost:9091")


def build_report(reference: pd.DataFrame, current: pd.DataFrame):
    report = Report([DataDriftPreset()])
    return report.run(current_data=current, reference_data=reference)


def drift_share(snapshot) -> float:
    """Share of drifted columns, read from the DriftedColumnsCount metric."""
    for metric in snapshot.dict()["metrics"]:
        if metric.get("metric_name", "").startswith("DriftedColumnsCount"):
            return float(metric["value"]["share"])
    raise KeyError("DriftedColumnsCount not found in report")


def push_drift_share(share: float, url: str = PUSHGATEWAY_URL) -> bool:
    """Push the drifted-column share to the Prometheus Pushgateway.

    The report is a batch job, so Prometheus cannot scrape it; it pushes instead and the
    Pushgateway keeps the last value until the next run. Returns False (and only logs)
    if the gateway is down, so a monitoring outage never breaks the drift report itself.
    """
    registry = CollectorRegistry()  # a private registry: push only our own metrics
    Gauge(
        "fraud_data_drift_share", "Share of feature columns Evidently flags as drifted",
        registry=registry,
    ).set(share)
    try:
        push_to_gateway(url, job="fraud_drift_report", registry=registry)
    except Exception as exc:  # network errors, gateway down, bad URL
        log.warning("Could not push drift share to %s: %s", url, exc)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=5000, help="most recent predictions to check")
    parser.add_argument("--threshold", type=float, default=0.03, help="drifted-column share alarm")
    parser.add_argument("--pushgateway", default=PUSHGATEWAY_URL, help="Pushgateway base URL")
    parser.add_argument("--no-push", action="store_true", help="skip pushing to Prometheus")
    args = parser.parse_args()

    reference = pd.read_csv(config.TRAIN_PATH)[config.FEATURES].sample(
        n=20000, random_state=config.RANDOM_STATE
    )
    current = pd.read_csv(config.PREDICTION_LOG_PATH)[config.FEATURES].tail(args.last)

    snapshot = build_report(reference, current)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    out = config.REPORTS_DIR / "drift_report.html"
    snapshot.save_html(str(out))

    share = drift_share(snapshot)
    print(f"Drifted columns: {share:.0%} (alarm at {args.threshold:.0%}) -> {out}")
    if not args.no_push:
        pushed = push_drift_share(share, args.pushgateway)
        print(f"Pushed to Pushgateway: {pushed}")
    sys.exit(1 if share >= args.threshold else 0)


if __name__ == "__main__":
    main()
