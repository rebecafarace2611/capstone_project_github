import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

TARGET = "respuesta_dicot_c"

# 1. Load the tuned model and the test data
model = xgb.XGBClassifier()
model.load_model("xgboost_tuned.json")

test = pd.read_parquet("test_model_dataset.parquet")
train = pd.read_parquet("train_model_dataset.parquet")  # for category alignment

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

# 2. Restore native categorical dtypes (must match how the model was trained)
X_train = train.drop(columns=[TARGET])
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

# 3. Score all test rows, then take the TOP 500 by fraud probability
#    (same approach as RFQC and LightGBM — explain the highest-risk alerts)
proba = model.predict_proba(X_test)[:, 1]
top_idx = np.argsort(proba)[::-1][:500]   # indices of 500 highest-scoring claims
X_top = X_test.iloc[top_idx].copy()
print(f"Explaining top {len(X_top)} highest-scoring claims.")

# 4. TreeSHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_top)

# 5. Global feature importance = mean absolute SHAP value per feature
importance = np.abs(shap_values).mean(axis=0)
imp_df = pd.DataFrame({
    "feature": X_top.columns,
    "mean_abs_shap": importance,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 10 features by mean |SHAP|:")
print(imp_df.head(10).to_string(index=False))
imp_df.to_csv("xgboost_shap_importance.csv", index=False)

# 6. Summary plot (like RFQC / LightGBM produced)
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_top, max_display=10, show=False)
plt.xlabel("SHAP value (impact on model output)")
plt.tight_layout()
plt.savefig("xgboost_shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()

# 7. Bar plot of top features
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_top, plot_type="bar", max_display=10, show=False)
plt.xlabel("mean(|SHAP value|)")
plt.tight_layout()
plt.savefig("xgboost_shap_bar.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nSaved: xgboost_shap_importance.csv, xgboost_shap_summary.png, xgboost_shap_bar.png")

# RFQC-style SHAP table: top 10 by mean|SHAP|, with positive share
mean_abs = np.abs(shap_values).mean(axis=0)
positive_share = (shap_values > 0).mean(axis=0)  # fraction of rows pushing toward fraud

shap_table = pd.DataFrame({
    "feature": X_top.columns,
    "mean_abs_shap": mean_abs,
    "positive_share": positive_share,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 10 SHAP risk drivers (RFQC-style table):")
print(shap_table.head(10).to_string(index=False))
shap_table.head(10).to_csv("xgboost_shap_table_top10.csv", index=False)
print("\nSaved: xgboost_shap_table_top10.csv")