"""Stage 4: compare recent API traffic with the training data using Evidently.

Owner: Devyani. Run with `python -m fraud.monitoring.drift_report`.
Writes reports/drift_report.html and exits with code 1 if drift is detected,
so Airflow or a cron job can alert on it.
"""

import argparse
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from fraud import config


def build_report(reference: pd.DataFrame, current: pd.DataFrame):
    report = Report([DataDriftPreset()])
    return report.run(current_data=current, reference_data=reference)


def drift_share(snapshot) -> float:
    """Share of drifted columns, read from the DriftedColumnsCount metric."""
    for metric in snapshot.dict()["metrics"]:
        if metric.get("metric_name", "").startswith("DriftedColumnsCount"):
            return float(metric["value"]["share"])
    raise KeyError("DriftedColumnsCount not found in report")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=5000, help="most recent predictions to check")
    parser.add_argument("--threshold", type=float, default=0.03, help="drifted-column share alarm")
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
    # TODO(Devyani): also push `share` to Prometheus (Pushgateway) so Grafana can alert on it.
    sys.exit(1 if share >= args.threshold else 0)


if __name__ == "__main__":
    main()
