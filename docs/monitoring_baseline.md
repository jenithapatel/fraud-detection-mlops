# Monitoring baseline (clean test set through the API)

Measured by sending `data/processed/test.csv` (56,746 rows, never used for training or
tuning) through `POST /predict` in batches of 1,000 against `fraud-detector@champion`
(v1, XGBoost with `scale_pos_weight`, semver 0.1.0).

Reproduce: `make up`, `make train`, `make baseline`, `make report`.

| Metric | Baseline value |
|---|---|
| PR-AUC against labels (production validation) | 0.8252 |
| Validation PR-AUC at training time | 0.8820 |
| Actual fraud rate in `test.csv` | 0.167% |
| Predicted fraud rate (`is_fraud`, threshold from model tag) | 0.153% |
| Precision / recall at the registered threshold | 0.874 / 0.800 |
| Mean fraud probability | 0.0017 |
| Evidently drift share (reference: 20,000 train rows) | 0% (alarm at 3%) |
| `/predict` p95 latency (1,000-row batch) | about 10 ms |

The production PR-AUC (0.825) is lower than the validation PR-AUC (0.882). That is
expected: the test set has only about 95 fraud rows, so the estimate is noisy.

## Alert thresholds derived from this baseline (`monitoring/alerts.yml`)

| Alert | Old | New | Reason |
|---|---|---|---|
| `FraudRateSpike` | > 2% | > 0.5% | About 3x the 0.15% baseline. 2% was over 13x baseline and would fire late. |
| `HighPredictLatency` | p95 > 500 ms | p95 > 100 ms | Baseline is about 10 ms, so 100 ms is 10x headroom. |
| `NoModelLoaded` | unchanged | unchanged | Binary signal, nothing to tune. |

The drift alarm in `drift_report.py` stays at 3% of columns: the clean baseline is 0%, so
any real drifted column share is a clear signal.
