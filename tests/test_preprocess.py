import pytest

from fraud import config
from fraud.data.ingest import validate_schema
from fraud.data.preprocess import split


def test_split_is_stratified_and_disjoint(fake_transactions):
    train, test = split(fake_transactions)
    assert len(train) + len(test) == len(fake_transactions)
    assert abs(train[config.TARGET].mean() - test[config.TARGET].mean()) < 0.01


def test_validate_schema_rejects_missing_column(fake_transactions):
    with pytest.raises(ValueError):
        validate_schema(fake_transactions.drop(columns=["V3"]))
