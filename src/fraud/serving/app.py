"""Stage 3: real-time inference API.

Owner: Sameeksha. Run locally with `uvicorn fraud.serving.app:app --reload`.
Endpoints: GET /health, POST /predict, POST /reload, GET /metrics (Prometheus).
Every prediction is appended to config.PREDICTION_LOG_PATH for drift monitoring.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from fraud import config
from fraud.serving.model_loader import LoadedModel, load_model
from fraud.serving.schemas import Prediction, PredictRequest, PredictResponse

log = logging.getLogger("fraud.serving")
# uvicorn only configures its own loggers, so give ours a handler and a level.
# FRAUD_LOG_LEVEL=DEBUG gives more detail when debugging a container.
logging.basicConfig(
    level=os.getenv("FRAUD_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
STATE: dict[str, LoadedModel | None] = {"loaded": None}

# --- Prometheus metrics (scraped by monitoring/prometheus.yml) --------------
PREDICTIONS = Counter("fraud_predictions_total", "Predictions served", ["label"])
LATENCY = Histogram("fraud_predict_latency_seconds", "Latency of /predict requests")
FRAUD_PROBABILITY = Histogram(
    "fraud_probability", "Predicted fraud probability",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
)
MODEL_LOADED = Gauge("fraud_model_loaded", "1 if a model is loaded, else 0")
# Requests rejected by the schema (missing, renamed or non-numeric columns). A jump here
# means an upstream producer changed its payload: a schema-drift signal for monitoring.
SCHEMA_VIOLATIONS = Counter(
    "fraud_schema_violations_total", "Requests rejected with 422 (schema violation)", ["path"]
)


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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log one line per request (method, path, status, duration) with a request id.

    The id is taken from the caller's X-Request-ID header if present, otherwise
    generated, and echoed back in the response so a client error can be matched
    to the server log line.
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Should not happen (handlers below turn errors into responses), but if it
        # does we still want a log line tied to the request id before re-raising.
        log.exception("request_id=%s %s %s failed", request_id, request.method, request.url.path)
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    log.info(
        "request_id=%s %s %s status=%d duration_ms=%.1f",
        request_id, request.method, request.url.path, response.status_code, duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a 422 that says which fields are wrong, in plain words.

    Keeps FastAPI's usual `detail` list so existing clients still work, and adds a
    short `error` summary, e.g. "transactions.0.V3: Field required".
    """
    errors = exc.errors()
    summary = "; ".join(
        f"{'.'.join(str(part) for part in err['loc'] if part != 'body')}: {err['msg']}"
        for err in errors[:5]  # the first few are enough to fix the request
    )
    if len(errors) > 5:
        summary += f"; ... and {len(errors) - 5} more"
    log.warning("Schema violation on %s: %s", request.url.path, summary)
    SCHEMA_VIOLATIONS.labels(path=request.url.path).inc()
    return JSONResponse(
        status_code=422,
        content={"error": f"Invalid request: {summary}", "detail": jsonable_encoder(errors)},
    )


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
    try:
        proba = loaded.model.predict_proba(X)[:, 1]
    except Exception as exc:
        # The request was valid, so this is our fault (bad model artifact, version
        # mismatch, ...): log the traceback and give the client a clear 500.
        log.exception("Model %s failed on a batch of %d rows", loaded.version, len(X))
        raise HTTPException(500, f"Model {loaded.version} failed to score the batch.") from exc
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
