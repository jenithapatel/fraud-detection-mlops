"""Shared project contract.

Every stage imports paths, column names and model names from here, so the
stages stay compatible. Change this file only through a PR that the whole
team reviews: it is the interface between everyone's work.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Data locations ---------------------------------------------------------
DATA_DIR = Path(os.getenv("FRAUD_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"  # isolated until production validation
DRIFT_DIR = DATA_DIR / "drift"
PREDICTION_LOG_PATH = DATA_DIR / "predictions" / "predictions.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Schema (Kaggle "mlg-ulb/creditcardfraud", column order preserved) -----
TARGET = "Class"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

# --- Splitting --------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2  # held-out test set
VALIDATION_SIZE = 0.2  # carved out of the training set for model selection

# --- MLflow / model registry ------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "fraud-detection"
MODEL_NAME = "fraud-detector"
MODEL_ALIAS = "champion"  # serving always loads models:/fraud-detector@champion

# --- Evaluation / decision --------------------------------------------------
TARGET_PRECISION = 0.90  # operational metric: recall at 90% precision
DEFAULT_THRESHOLD = 0.5  # used only if the registered model has no "threshold" tag
