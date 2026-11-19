from fraud.models.metrics import pr_auc, recall_at_precision


def test_perfect_scores():
    y, s = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    assert pr_auc(y, s) == 1.0
    assert recall_at_precision(y, s, 0.9)[0] == 1.0


def test_unreachable_precision():
    assert recall_at_precision([1, 0], [0.1, 0.9], 0.99) == (0.0, 1.0)
