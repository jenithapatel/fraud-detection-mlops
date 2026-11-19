from fraud import config


def test_feature_contract():
    assert len(config.FEATURES) == 30
    assert config.FEATURES[0] == "Time" and config.FEATURES[-1] == "Amount"
    assert config.TARGET not in config.FEATURES
