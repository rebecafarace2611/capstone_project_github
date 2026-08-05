# Runtime Inputs Required for Phase 1/2

The Phase 1/2 backend structure is ready. To enable true live scoring, the
runtime artifacts supplied to the prototype must match the current report and
the latest reviewed GitHub workflow.

## 1. Chosen model artifact

The fitted model artifact should be the model selected for the prototype track
and should reproduce the corresponding report metrics when evaluated at the
fixed q-star threshold.

Needed file for XGBoost prototype route:

- `xgboost_tuned.json`
- `xgboost_best_params.json` if the tuned model must be regenerated
- the same training/test parquet and `fold_assignments.parquet` used by the
  final XGBoost workflow

Needed file for LightGBM prototype route:

- one or more current-report LightGBM `model.txt` files whose outputs match the
  report

## 2. Current train/test split

The report states:

```text
train rows = 444,074
test rows  = 111,020
test fraud = 465
```

Needed files:

- current `train_model_dataset.parquet`
- current `test_model_dataset.parquet`

These should match the report row counts before generating the prototype claim
pool.
