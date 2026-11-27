"""Send a CSV of transactions to the deployed API in batches.

Used for baseline validation (clean test set) and drift scenarios:
  python -m fraud.monitoring.send_to_api data/processed/test.csv
  python -m fraud.monitoring.send_to_api data/drift/amount_x10.csv
"""

import argparse
import os

import pandas as pd
import requests

from fraud import config
from fraud.models.metrics import pr_auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--url", default=os.getenv("FRAUD_API_URL", "http://localhost:8000"))
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # Send the columns exactly as they appear in the file (minus the label), instead of
    # forcing them into config.FEATURES: schema-change scenarios have dropped or renamed
    # columns, and the API, not this script, should be the one to reject them.
    send_cols = [c for c in df.columns if c != config.TARGET]
    probs: list[float] = []
    rejected_rows = 0
    first_error = ""
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start : start + args.batch_size][send_cols]
        resp = requests.post(
            f"{args.url}/predict",
            json={"transactions": batch.to_dict(orient="records")},
            timeout=60,
        )
        if resp.status_code == 422:
            # Schema violation: count it (the API counts these as a drift signal) and go on.
            rejected_rows += len(batch)
            first_error = first_error or resp.json().get("error", "")
            continue
        resp.raise_for_status()  # any other error (5xx, 503 no model) is a real failure
        probs += [p["fraud_probability"] for p in resp.json()["predictions"]]

    if rejected_rows:
        print(f"API rejected {rejected_rows:,} of {len(df):,} rows with 422 (schema violation)")
        print(f"First error: {first_error}")
    if probs:
        print(f"Sent {len(probs):,} rows; mean fraud probability {sum(probs) / len(probs):.4f}")
        if config.TARGET in df.columns and not rejected_rows:
            print(f"PR-AUC against labels: {pr_auc(df[config.TARGET], probs):.4f}")


if __name__ == "__main__":
    main()
