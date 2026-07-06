import pandas as pd
import numpy as np
import xgboost as xgb

# ----------------------------------------------------------------------
# PART 2 (grouped): Baseline XGBoost, early stopping on a grouped fold
# Uses fold_assignments.parquet — same grouped folds as RFQC/LightGBM.
# Option 1: hold out fold 0 as the early-stopping validation set.
# ----------------------------------------------------------------------

TARGET = "respuesta_dicot_c"
VAL_FOLD = 0  

# 1. Load data + folds
train = pd.read_parquet("train_model_dataset.parquet")
test = pd.read_parquet("test_model_dataset.parquet")
folds = pd.read_parquet("fold_assignments.parquet")

# Align folds to train row order (row_index maps to train rows)
folds = folds.sort_values("row_index").reset_index(drop=True)
assert len(folds) == len(train), "Fold file and train set row counts differ!"

# 2. Features / target
X = train.drop(columns=[TARGET])
y = train[TARGET]
X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

# 3. Native categorical handling (same as before — matches RFQC factors)
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
print(f"Categorical columns ({len(cat_cols)}):", cat_cols)
for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories)

# 4. Split train into (4 folds) train / (1 fold) validation using folds of new file
val_mask = folds["fold"].values == VAL_FOLD
X_tr, y_tr = X[~val_mask], y[~val_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"\nHold-out validation = fold {VAL_FOLD}")
print(f"Train: {X_tr.shape} ({y_tr.sum()} fraud)")
print(f"Val:   {X_val.shape} ({y_val.sum()} fraud)")
print(f"Fraud rate — train: {y_tr.mean():.4%}, val: {y_val.mean():.4%}")

# 5. Baseline XGBoost — identical settings to the single-split version,
#    NO scale_pos_weight (imbalance handled at the q* threshold).
model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    early_stopping_rounds=50,
    enable_categorical=True,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
print(f"\nBest iteration: {model.best_iteration}")

# 6. Test-set fraud probabilities
test_proba = model.predict_proba(X_test)[:, 1]

# 7. Save — NEW filenames so the single-split outputs are preserved
model.save_model("xgboost_grouped.json")
pd.DataFrame({
    "y_true": y_test.values,
    "y_proba": test_proba,
}).to_parquet("xgboost_grouped_test_predictions.parquet")

print("\nSaved: xgboost_grouped.json + xgboost_grouped_test_predictions.parquet")
print("Part 2 (grouped) complete.")
