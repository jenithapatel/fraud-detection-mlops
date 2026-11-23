"""Stage 1a: load the raw Kaggle CSV, validate it and log a data summary.

Owner: Jenitha. Run with `python -m fraud.data.ingest`.
This is the first task of the Airflow DAG, so a failed validation stops the
pipeline before bad data reaches preprocessing or training.
"""

import logging

import pandas as pd

from fraud import config

logger = logging.getLogger(__name__)

DOWNLOAD_HINT = (
    "Get it with `dvc pull` (team remote, see README), or download it with:\n"
    "  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip"
)

# Columns that are physically non-negative: seconds since the first
# transaction, and the transaction amount. The V1..V28 PCA components can
# take any sign, so they have no range check.
NON_NEGATIVE_COLUMNS = ["Time", "Amount"]


def find_schema_problems(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems with the raw data (empty if valid).

    Checks, in order: required columns present, non-empty, numeric dtypes,
    no nulls, binary target with both classes, non-negative Time/Amount.
    """
    required = [*config.FEATURES, config.TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # The remaining checks index these columns, so stop here.
        return [f"missing columns: {missing}"]
    if df.empty:
        return ["dataset has no rows"]

    problems = []
    non_numeric = [c for c in required if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"non-numeric columns: {non_numeric}")

    null_counts = df[required].isna().sum()
    if null_counts.any():
        problems.append(f"null values: {null_counts[null_counts > 0].to_dict()}")

    labels = set(df[config.TARGET].dropna().unique())
    if not labels <= {0, 1}:
        problems.append(f"{config.TARGET} must be binary 0/1, found {sorted(labels)}")
    elif labels != {0, 1}:
        # A single-class dataset can't train a classifier or compute PR-AUC.
        problems.append(f"{config.TARGET} has only one class: {sorted(labels)}")

    for col in NON_NEGATIVE_COLUMNS:
        if col in non_numeric:
            continue  # comparison would fail; already reported above
        n_negative = int((df[col] < 0).sum())
        if n_negative:
            problems.append(f"{col} has {n_negative} negative values")
    return problems


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError listing every problem found by `find_schema_problems`."""
    problems = find_schema_problems(df)
    if problems:
        raise ValueError("Raw data failed validation:\n  - " + "\n  - ".join(problems))


def summarize(df: pd.DataFrame) -> dict:
    """Compute the data-quality numbers logged on every ingest (and used in the EDA)."""
    fraud_count = int(df[config.TARGET].sum())
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "fraud_count": fraud_count,
        "fraud_rate": fraud_count / len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "amount_min": float(df["Amount"].min()),
        "amount_max": float(df["Amount"].max()),
    }


def load_raw(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read and validate the raw dataset."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {DOWNLOAD_HINT}")
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def main() -> None:
    """Load, validate and log a summary of the raw data (Airflow task `ingest`)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_raw()
    s = summarize(df)
    logger.info("Loaded %s rows x %s columns from %s", f"{s['rows']:,}", s["columns"],
                config.RAW_DATA_PATH)
    logger.info("Fraud: %s rows (%.4f%%)", f"{s['fraud_count']:,}", s["fraud_rate"] * 100)
    logger.info("Amount range: %.2f to %.2f", s["amount_min"], s["amount_max"])
    if s["duplicate_rows"]:
        # Not an error: preprocess drops them. Logged so the count is traceable.
        logger.warning("%s duplicate rows (dropped in preprocess)", f"{s['duplicate_rows']:,}")
    logger.info("Validation passed")


if __name__ == "__main__":
    main()
