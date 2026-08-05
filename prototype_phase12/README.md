# Phase 1/2 Prototype Backend

This folder contains the Phase 1 and Phase 2 handoff for the fraud-scoring
prototype.

The backend supports the two Python tree-model tracks used in the
project:

- XGBoost, including `xgboost_tuned.json` when that is the chosen prototype
  track.
- LightGBM, using `model.txt` files when the team chooses a LightGBM track.

The current shared operating rule is q-star:

```text
q_star_threshold = 0.0041884911
test set         = 111,020 claims, 465 fraud cases in the report
```

Use only model and data artifacts whose outputs match the current report. The
prototype should reproduce the reported operating threshold and model outputs
before it is connected to the user interface.

## What is included

- `artifacts/model_config.json`: one readable place for q-star, model track,
  target column, model paths, and claim-pool paths.
- `artifacts/approved_features.json`: the 144 leakage-approved model features.
- `artifacts/label_map.json`: human-readable labels used by the prototype and
  explanation output.
- `backend/service.py`: scoring/classification/explanation backend API for the
  UI team. It can load either XGBoost or LightGBM depending on
  `model_config.json`.
- `scripts/prepare_phase12_assets.py`: validates the supplied split, copies the
  chosen fitted model files, and prepares a user-safe claim pool.
- `examples/demo_backend.py`: small command-line smoke test once model artifacts
  and claim pool are present.

## Required runtime inputs

The source code is present in the latest GitHub version reviewed for this
handoff. The confidential runtime artifacts must be supplied separately before
true live scoring can be enabled:

- Chosen model artifact. For the XGBoost route, this is usually
  `xgboost_tuned.json`. For a LightGBM route, this is one or more `model.txt`
  files.
- Current train/test split files matching the report metadata:
  `train_model_dataset.parquet` with 444,074 rows and
  `test_model_dataset.parquet` with 111,020 rows and 465 fraud cases.

## Setup

Create an environment with the runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r prototype_phase12\requirements.txt
```

Prepare assets after the correct model and split files are available:
The command below uses placeholder paths for the confidential runtime artifacts. Before running the prototype, replace each C:\path\to\... value with the actual path to the approved train/test split and the selected fitted model artifact. These files are not bundled with this handoff because they are confidential runtime inputs.
```powershell
.\.venv\Scripts\python prototype_phase12\scripts\prepare_phase12_assets.py `
  --train C:\path\to\train_model_dataset.parquet `
  --test C:\path\to\test_model_dataset.parquet `
  --model C:\path\to\xgboost_tuned.json `
  --model-family xgboost `
  --out prototype_phase12\artifacts
```

Run a smoke test:

```powershell
.\.venv\Scripts\python prototype_phase12\examples\demo_backend.py
```

## Backend contract

The UI can call `FraudScoringBackend.score_claim(claim_id)` and receives:

- `fraud_probability`
- `threshold`
- `flagged`
- `classification_label`
- `top_factors`, each with a raw feature name, readable label, value,
  contribution, and direction

The explanation direction is based on native LightGBM TreeSHAP contribution
signs for LightGBM, or XGBoost's native `pred_contribs=True` contribution signs
for XGBoost: positive values push the raw score toward fraud; negative values
push it toward non-fraud. These are model-behaviour explanations, not causal
claims.
