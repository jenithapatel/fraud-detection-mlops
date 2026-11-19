"""Request/response contract for the /predict endpoint."""

from pydantic import BaseModel, Field, create_model

from fraud import config

# One float field per feature in config.FEATURES. Values are not range-checked
# on purpose: out-of-range data must reach the model so monitoring can catch it.
Transaction = create_model("Transaction", **{f: (float, ...) for f in config.FEATURES})


class PredictRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=5000)


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool


class PredictResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]
