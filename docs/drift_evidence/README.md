# Drift evidence

Each scenario is built from the clean `test.csv` by `python -m fraud.monitoring.simulate_drift`,
sent through the API with `send_to_api`, then checked with `drift_report` (reference: 20,000
training rows, current: the last 5,000 predictions). Reproduce with `make baseline`,
`make drift`, `make report`.

| Scenario | What changes | Drifted columns | Predicted fraud rate | Mean fraud probability | `DataDriftDetected` |
|---|---|---|---|---|---|
| Clean baseline | nothing | 0% | 0.16% | 0.0019 | quiet |
| `amount_x10` | `Amount` x 10 | 3% (1 of 30 columns) | 0.20% | 0.0023 | fires (3.3% vs 3% threshold) |
| `swap_v1_v2` | V1 and V2 swapped | 7% | 0.16% | 0.0018 | fires |
| `out_of_range` | 20% of rows get extreme V values | 93% | 0.40% | 0.0048 | fires |
| `schema_drop_v14`, `schema_rename_amount` | column dropped or renamed | n/a: API returns 422 for every row | n/a | n/a | not applicable |

## What the numbers say

- Evidently catches every scenario that reaches the model, and the drifted-column share
  scales with the severity of the corruption (3% to 93%).
- The fraud-rate alert is a weaker signal. Even `out_of_range` only raises the predicted fraud
  rate from 0.16% to 0.40%, which stays under the `FraudRateSpike` threshold of 0.5%.
  `amount_x10` and `swap_v1_v2` barely move it. Input drift can go unseen by output
  monitoring, which is why the drift share is the primary alert.
- `amount_x10` crosses the 3% threshold by one column. It is the most fragile detection.
- Schema changes never reach the model or the prediction log, so Evidently cannot see them.
  They show up only as 422 responses in the API logs.

## Files

Evidently reports (open in a browser):
- `before_clean_report.html`: clean traffic, 0% drift.
- `out_of_range_report.html`: after the `out_of_range` scenario, 93% drift (28 of 30 columns).

Screenshots, in the order of the story:
- `grafana_after_top.png`, `grafana_after_panels.png`: Grafana during the `out_of_range` run.
  Drift share tile 93.3%, drift-over-time panel stepping up, probability heatmap shifting.
- `prometheus_alert_firing.png`: `DataDriftDetected` FIRING (value 0.933).
- `evidently_out_of_range.png`: the Evidently drift summary for the same run.
- `grafana_clean_top.png`, `grafana_clean_panels.png`: Grafana after clean traffic is sent again
  and a fresh report is pushed. Drift share tile back to 0.0%, drift panel dropped to 0.
- `prometheus_alert_resolved.png`: all four alerts inactive again.

The clean-state screenshots were taken after the drift run, so the drift-over-time panel in them
shows the whole cycle: drift, alert, recovery.
