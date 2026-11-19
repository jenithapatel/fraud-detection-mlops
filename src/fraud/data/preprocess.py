"""Stage 1b: clean the raw data and make the stratified train/test split.

Owner: Jenitha. Run with `python -m fraud.data.preprocess`.
Scaling is NOT done here: it lives inside the model pipeline, so the API can
accept raw transaction features.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud import config
from fraud.data.ingest import load_raw


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split so both sets keep the ~0.17% fraud rate."""
    df = df[[*config.FEATURES, config.TARGET]]
    train, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[config.TARGET],
        random_state=config.RANDOM_STATE,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    df = load_raw()
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} duplicate rows")
    train, test = split(df)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.TRAIN_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)
    print(f"train: {len(train):,} rows -> {config.TRAIN_PATH}")
    print(f"test:  {len(test):,} rows -> {config.TEST_PATH}")


if __name__ == "__main__":
    main()
