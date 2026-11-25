import json

from fraud.data.eda import hourly_profile, run_eda


def test_hourly_profile_covers_24_hours(fake_transactions):
    df = fake_transactions.copy()
    df["Time"] = range(0, len(df) * 300, 300)  # one transaction every 5 minutes
    profile = hourly_profile(df)
    assert list(profile.index) == list(range(24))
    assert profile["transactions"].sum() == len(df)


def test_run_eda_writes_charts_and_summary(fake_transactions, tmp_path):
    df = fake_transactions.copy()
    df["Time"] = df["Time"].abs() * 3600
    summary = run_eda(df, out_dir=tmp_path)
    for name in ["class_balance.png", "amount_by_class.png", "time_patterns.png"]:
        assert (tmp_path / name).stat().st_size > 0
    assert json.loads((tmp_path / "summary.json").read_text())["rows"] == summary["rows"] == 1000
