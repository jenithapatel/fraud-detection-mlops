from fraud.monitoring.simulate_drift import inject_out_of_range, scale_amount, swap_columns


def test_scale_amount(fake_transactions):
    out = scale_amount(fake_transactions, 10)
    assert (out["Amount"] == fake_transactions["Amount"] * 10).all()


def test_out_of_range_changes_fraction(fake_transactions):
    out = inject_out_of_range(fake_transactions, frac=0.2)
    changed = (out["V1"] != fake_transactions["V1"]).sum()
    assert changed == 200


def test_swap_columns(fake_transactions):
    out = swap_columns(fake_transactions, "V1", "V2")
    assert (out["V1"] == fake_transactions["V2"]).all()
    assert (out["V2"] == fake_transactions["V1"]).all()


def test_drift_report_detects_scaled_amount(fake_transactions):
    from fraud import config
    from fraud.monitoring.drift_report import build_report, drift_share

    ref = fake_transactions[config.FEATURES]
    clean = drift_share(build_report(ref, ref.sample(frac=1, random_state=1)))
    drifted = drift_share(build_report(ref, scale_amount(ref, 10)))
    assert clean == 0.0 and drifted > 0.0
