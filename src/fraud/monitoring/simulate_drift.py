"""Stage 5: corrupt the clean test set to simulate drift.

Owner: Devyani. Run with `python -m fraud.monitoring.simulate_drift`.
Writes one CSV per scenario to data/drift/.
"""

import numpy as np
import pandas as pd

from fraud import config


def scale_amount(df: pd.DataFrame, factor: float = 10.0) -> pd.DataFrame:
    """Covariate drift: transaction amounts inflate (e.g. currency/pricing change)."""
    out = df.copy()
    out["Amount"] = out["Amount"] * factor
    return out


def inject_out_of_range(
    df: pd.DataFrame, frac: float = 0.2, magnitude: float = 50.0, seed: int = 0
) -> pd.DataFrame:
    """Data-quality drift: a fraction of rows get extreme values in the V features."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    v_cols = [c for c in config.FEATURES if c.startswith("V")]
    rows = rng.choice(len(out), size=int(len(out) * frac), replace=False)
    out.loc[out.index[rows], v_cols] = rng.uniform(-magnitude, magnitude, (len(rows), len(v_cols)))
    return out


def swap_columns(df: pd.DataFrame, a: str = "V1", b: str = "V2") -> pd.DataFrame:
    """Schema drift: an upstream bug swaps two feature columns."""
    out = df.copy()
    out[[a, b]] = out[[b, a]].to_numpy()
    return out


def drop_column(df: pd.DataFrame, column: str = "V14") -> pd.DataFrame:
    """Schema change: an upstream producer stops sending a feature.

    The API requires all 30 features, so every request should be rejected with a 422.
    """
    return df.drop(columns=[column])


def rename_column(df: pd.DataFrame, old: str = "Amount", new: str = "amount_usd") -> pd.DataFrame:
    """Schema change: an upstream producer renames a feature.

    The old name is now missing, so the API should reject every request with a 422.
    """
    return df.rename(columns={old: new})


SCENARIOS = {
    "amount_x10": scale_amount,
    "out_of_range": inject_out_of_range,
    "swap_v1_v2": swap_columns,
    "schema_drop_v14": drop_column,
    "schema_rename_amount": rename_column,
}


def main() -> None:
    test = pd.read_csv(config.TEST_PATH)
    config.DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in SCENARIOS.items():
        path = config.DRIFT_DIR / f"{name}.csv"
        fn(test).to_csv(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
