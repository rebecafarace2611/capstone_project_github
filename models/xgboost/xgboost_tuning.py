import pandas as pd
import numpy as np
import time
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
import xgboost as xgb

TARGET = "respuesta_dicot_c"

# 1. Load training data + folds (TEST SET IS NEVER LOADED HERE — training only)
train = pd.read_parquet("train_model_dataset.parquet")
folds = pd.read_parquet("fold_assignments.parquet").sort_values("row_index").reset_index(drop=True)
assert len(folds) == len(train), "Fold/train row mismatch"

X = train.drop(columns=[TARGET])
y = train[TARGET]
groups = folds["fold"].values  # Jenny's grouped fold assignments

# 2. Native categorical handling (same as baseline)
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")

# 3. Grouped CV — same folds as the baseline and the other models
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Base estimator (no early stopping here — the CV handles validation)
base = xgb.XGBClassifier(
    enable_categorical=True,
    tree_method="hist",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)

# 5. Random search space 
    "n_estimators":     randint(100, 800),
    "learning_rate":    uniform(0.01, 0.19),   # 0.01 to 0.20
    "max_depth":        randint(3, 10),
    "min_child_weight": randint(1, 10),
    "subsample":        uniform(0.6, 0.4),      # 0.6 to 1.0
    "colsample_bytree": uniform(0.6, 0.4),      # 0.6 to 1.0
    "gamma":            uniform(0, 5),
    "reg_alpha":        uniform(0, 5),
    "reg_lambda":       uniform(0, 5),
}

# 6. Random search, scoring on PR-AUC, over the grouped folds
search = RandomizedSearchCV(
    estimator=base,
    param_distributions=param_dist,
    n_iter=40,                    # number of random configs to try
    scoring="average_precision",  # PR-AUC
    cv=cv.split(X, y, groups),    # grouped folds
    verbose=2,
    random_state=42,
    n_jobs=1,                     # each fit already uses all cores
    refit=True,
)

print("Starting random search — training data only, grouped folds, PR-AUC scoring...")
start = time.perf_counter()
search.fit(X, y)
elapsed = time.perf_counter() - start

print(f"\nSearch completed in {elapsed/60:.1f} minutes")
print(f"Best PR-AUC (CV): {search.best_score_:.4f}")
print("Best params:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

# 7. Save the best params so you can retrain + test cleanly in a separate step
pd.Series(search.best_params_).to_json("xgboost_best_params.json")
print("\nSaved: xgboost_best_params.json")