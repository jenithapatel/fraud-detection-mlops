# Real-Time Credit Card Fraud Detection: MLOps Pipeline

ADSP 32021 (Machine Learning Operations and Inference Engineering) final project.
Team: Jenitha Patel, Akshaj Chandwani, Sameeksha Kini, Devyani Rastogi.

An automated pipeline that trains, registers, serves and monitors a fraud
classifier on the Kaggle ULB credit card dataset (284,807 transactions, 0.17% fraud).

**Primary metric:** PR-AUC. **Operational metric:** recall at 90% precision.

## Architecture

```
creditcard.csv (DVC) → Airflow DAG [ingest → preprocess → train] → MLflow Tracking
   → Model Registry (fraud-detector@champion) → FastAPI in Docker (/predict)
   → prediction log + Prometheus metrics → Evidently drift report + Grafana / alerts
```

See `docs/architecture.md` for the diagram.

## Repository layout

| Path | What | Owner |
|---|---|---|
| `src/fraud/config.py` | Shared contract: paths, features, model name | Team |
| `src/fraud/data/` | Ingestion, validation, train/test split | Jenitha |
| `dags/` | Airflow DAG | Jenitha |
| `src/fraud/models/` | Training, metrics, MLflow registry | Akshaj |
| `src/fraud/serving/`, `docker/` | FastAPI app, Docker image | Sameeksha |
| `src/fraud/monitoring/`, `monitoring/` | Drift simulation, Evidently, Prometheus | Devyani |
| `tests/` | Unit and API tests (run in CI) | Everyone |

## Reproduce locally

Requirements: Python 3.11, Docker, and access to the team's shared Google Drive
DVC folder (or a Kaggle account as a fallback).

```bash
# 1. Install and enable the commit hook
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
git config core.hooksPath .githooks

# 2. Get the data (versioned with DVC; the first pull opens a Google sign-in)
dvc pull
# Fallback without Drive access (same file, verify the md5 matches the .dvc file):
# kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

# 3. Split the data
python -m fraud.data.preprocess

# 4. Start MLflow, the API, Prometheus and Grafana
docker compose up -d --build

# 5. Train and register the model, then load it into the API
python -m fraud.models.train
curl -X POST localhost:8000/reload

# 6. Baseline validation with the clean test set
python -m fraud.monitoring.send_to_api data/processed/test.csv

# 7. Drift simulation, then the drift report
python -m fraud.monitoring.simulate_drift
python -m fraud.monitoring.send_to_api data/drift/amount_x10.csv
python -m fraud.monitoring.drift_report     # writes reports/drift_report.html
```

UIs: MLflow http://localhost:5000 · API docs http://localhost:8000/docs ·
Prometheus http://localhost:9090 (Alerts tab) · Grafana http://localhost:3000.

## Try the API

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d "{\"transactions\": [$(python -c "import pandas as p,json;print(json.dumps(p.read_csv('data/processed/test.csv').drop(columns='Class').iloc[0].to_dict()))")]}"
```

## Tests

```bash
ruff check . && pytest -q
```

CI (GitHub Actions) runs lint, tests and a Docker build on every PR.

## Contributions

See `docs/contributions.md`.
