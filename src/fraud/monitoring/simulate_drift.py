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


SCENARIOS = {
    "amount_x10": scale_amount,
    "out_of_range": inject_out_of_range,
    "swap_v1_v2": swap_columns,
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
