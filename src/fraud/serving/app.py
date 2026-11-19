"""Stage 3: real-time inference API.

Owner: Sameeksha. Run locally with `uvicorn fraud.serving.app:app --reload`.
Endpoints: GET /health, POST /predict, POST /reload, GET /metrics (Prometheus).
Every prediction is appended to config.PREDICTION_LOG_PATH for drift monitoring.
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from fraud import config
from fraud.serving.model_loader import LoadedModel, load_model
from fraud.serving.schemas import Prediction, PredictRequest, PredictResponse

log = logging.getLogger("fraud.serving")
STATE: dict[str, LoadedModel | None] = {"loaded": None}

# --- Prometheus metrics (scraped by monitoring/prometheus.yml) --------------
PREDICTIONS = Counter("fraud_predictions_total", "Predictions served", ["label"])
LATENCY = Histogram("fraud_predict_latency_seconds", "Latency of /predict requests")
FRAUD_PROBABILITY = Histogram(
    "fraud_probability", "Predicted fraud probability",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
)
MODEL_LOADED = Gauge("fraud_model_loaded", "1 if a model is loaded, else 0")


def _try_load() -> None:
    try:
        STATE["loaded"] = load_model()
        MODEL_LOADED.set(1)
        log.info("Loaded model version %s", STATE["loaded"].version)
    except Exception as exc:  # keep the API up so /health explains the problem
        STATE["loaded"] = None
        MODEL_LOADED.set(0)
        log.error("Could not load model: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _try_load()
    yield


app = FastAPI(title="Fraud Detection API", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health() -> dict:
    loaded = STATE["loaded"]
    return {"status": "ok" if loaded else "no_model", "model_version": loaded and loaded.version}


@app.post("/reload")
def reload_model() -> dict:
    """Re-read the champion model, e.g. after retraining."""
    _try_load()
    return health()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    loaded = STATE["loaded"]
    if loaded is None:
        raise HTTPException(503, "No model loaded. Train and register one, then POST /reload.")

    start = time.perf_counter()
    X = pd.DataFrame([t.model_dump() for t in req.transactions])[config.FEATURES]
    proba = loaded.model.predict_proba(X)[:, 1]
    labels = proba >= loaded.threshold
    LATENCY.observe(time.perf_counter() - start)

    for p, is_fraud in zip(proba, labels, strict=True):
        FRAUD_PROBABILITY.observe(float(p))
        PREDICTIONS.labels(label="fraud" if is_fraud else "legit").inc()
    _log_predictions(X, proba, labels, loaded.version)

    return PredictResponse(
        model_version=loaded.version,
        threshold=loaded.threshold,
        predictions=[
            Prediction(fraud_probability=float(p), is_fraud=bool(f))
            for p, f in zip(proba, labels, strict=True)
        ],
    )


def _log_predictions(X: pd.DataFrame, proba, labels, version: str) -> None:
    """Append inputs + outputs to the prediction log read by the drift report."""
    out = X.copy()
    out["fraud_probability"] = proba
    out["is_fraud"] = labels.astype(int)
    out["model_version"] = version
    out["timestamp"] = datetime.now(timezone.utc).isoformat()
    path = config.PREDICTION_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
