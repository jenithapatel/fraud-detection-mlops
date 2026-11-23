import numpy as np
import pytest

from fraud import config
from fraud.data.ingest import find_schema_problems, summarize, validate_schema


@pytest.fixture
def valid_raw(fake_transactions):
    """The shared fixture with Time made non-negative, like the real data."""
    df = fake_transactions.copy()
    df["Time"] = df["Time"].abs()
    return df


def test_valid_data_has_no_problems(valid_raw):
    assert find_schema_problems(valid_raw) == []
    validate_schema(valid_raw)  # does not raise


def test_missing_column_is_reported(valid_raw):
    problems = find_schema_problems(valid_raw.drop(columns=["V3"]))
    assert problems == ["missing columns: ['V3']"]


def test_empty_dataset_is_rejected(valid_raw):
    assert find_schema_problems(valid_raw.iloc[0:0]) == ["dataset has no rows"]


def test_nulls_are_reported(valid_raw):
    valid_raw.loc[0, "V1"] = np.nan
    assert any("null values" in p and "V1" in p for p in find_schema_problems(valid_raw))


def test_non_numeric_column_is_reported(valid_raw):
    valid_raw["Amount"] = valid_raw["Amount"].astype(str)
    problems = find_schema_problems(valid_raw)
    assert any("non-numeric" in p and "Amount" in p for p in problems)


def test_non_binary_target_is_reported(valid_raw):
    valid_raw.loc[0, config.TARGET] = 2
    assert any("binary" in p for p in find_schema_problems(valid_raw))


def test_single_class_target_is_reported(valid_raw):
    valid_raw[config.TARGET] = 0
    assert any("only one class" in p for p in find_schema_problems(valid_raw))


def test_negative_amount_and_time_are_reported(valid_raw):
    valid_raw.loc[:2, "Amount"] = -1.0
    valid_raw.loc[0, "Time"] = -5.0
    problems = find_schema_problems(valid_raw)
    assert "Amount has 3 negative values" in problems
    assert "Time has 1 negative values" in problems


def test_validate_schema_lists_all_problems(valid_raw):
    valid_raw.loc[0, "V1"] = np.nan
    valid_raw.loc[0, "Amount"] = -1.0
    with pytest.raises(ValueError, match="null values") as exc:
        validate_schema(valid_raw)
    assert "negative" in str(exc.value)


def test_summarize_counts(valid_raw):
    s = summarize(valid_raw)
    assert s["rows"] == 1000
    assert s["fraud_count"] == 50  # fixture marks every 20th row as fraud
    assert s["fraud_rate"] == pytest.approx(0.05)
    assert s["duplicate_rows"] == 0
