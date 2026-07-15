import pandas as pd
import numpy as np
import json
import time
import xgboost as xgb

TARGET = "respuesta_dicot_c"

# Load best params from the search
with open("xgboost_best_params.json") as f:
    best = json.load(f)
print("Loaded tuned params:", best)

# Fix: these params must be integers, but were saved as floats (3.0 -> 3)
for k in ["max_depth", "min_child_weight", "n_estimators"]:
    if k in best:
        best[k] = int(best[k])
print("Corrected params:", best)

# Load train + test + folds
train = pd.read_parquet("train_model_dataset.parquet")
test = pd.read_parquet("test_model_dataset.parquet")
folds = pd.read_parquet("fold_assignments.parquet").sort_values("row_index").reset_index(drop=True)

X = train.drop(columns=[TARGET]); y = train[TARGET]
X_test = test.drop(columns=[TARGET]); y_test = test[TARGET]

cat_cols = X.select_dtypes(include=["object","string"]).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories)

# Hold out fold 0 for early stopping, as before
val_mask = folds["fold"].values == 0
X_tr, y_tr = X[~val_mask], y[~val_mask]
X_val, y_val = X[val_mask], y[val_mask]

model = xgb.XGBClassifier(
    **best,
    enable_categorical=True,
    tree_method="hist",
    eval_metric="aucpr",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)

start = time.perf_counter()
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
train_time = time.perf_counter() - start
print(f"\nTuned model training time: {train_time:.1f}s ({train_time/60:.2f} min)")
print(f"Best iteration: {model.best_iteration}")

test_proba = model.predict_proba(X_test)[:, 1]
model.save_model("xgboost_tuned.json")
pd.DataFrame({"y_true": y_test.values, "y_proba": test_proba}).to_parquet("xgboost_tuned_test_predictions.parquet")
print("Saved tuned model + predictions.")