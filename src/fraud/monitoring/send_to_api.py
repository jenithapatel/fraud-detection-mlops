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
    probs = []
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start : start + args.batch_size][config.FEATURES]
        resp = requests.post(
            f"{args.url}/predict",
            json={"transactions": batch.to_dict(orient="records")},
            timeout=60,
        )
        resp.raise_for_status()
        probs += [p["fraud_probability"] for p in resp.json()["predictions"]]

    print(f"Sent {len(df):,} rows; mean fraud probability {sum(probs) / len(probs):.4f}")
    if config.TARGET in df.columns:
        print(f"PR-AUC against labels: {pr_auc(df[config.TARGET], probs):.4f}")


if __name__ == "__main__":
    main()
