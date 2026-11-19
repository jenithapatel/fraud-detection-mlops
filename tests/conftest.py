import numpy as np
import pandas as pd
import pytest

from fraud import config


@pytest.fixture
def fake_transactions() -> pd.DataFrame:
    """1,000 synthetic rows in the real schema, 5% fraud."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(1000, len(config.FEATURES))), columns=config.FEATURES)
    df["Amount"] = rng.exponential(80, 1000)
    df[config.TARGET] = (np.arange(1000) % 20 == 0).astype(int)
    return df
