"""Stage 1a: load the raw Kaggle CSV and validate its schema.

Owner: Jenitha. Run with `python -m fraud.data.ingest`.
"""

import pandas as pd

from fraud import config

DOWNLOAD_HINT = (
    "Download it with:\n"
    "  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip\n"
    "or `dvc pull` if the team remote is set up."
)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing or the target is not binary."""
    required = [*config.FEATURES, config.TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing columns: {missing}")
    if not set(df[config.TARGET].unique()) <= {0, 1}:
        raise ValueError(f"{config.TARGET} must be binary 0/1")


def load_raw(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read and validate the raw dataset."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {DOWNLOAD_HINT}")
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def main() -> None:
    df = load_raw()
    print(f"Loaded {len(df):,} rows; fraud rate {df[config.TARGET].mean():.4%}")


if __name__ == "__main__":
    main()
