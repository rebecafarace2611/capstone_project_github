import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer

TARGET = "respuesta_dicot_c"

# Verified against variables_ddbb_fraud_original.xlsx. "scoring" is included
# with a label, but its meaning ("risk_scoring_group") is still unconfirmed
# with the sponsor as of this run — see the printed reminder at the end of
# this script before treating these results as final.
LABEL_MAP = {
    "antiguedad_poliza": "Policy tenure (days)",
    "garantia": "Coverage type",
    "garantia_agrupada": "Grouped coverage type",
    "tomadornivel": "Policyholder customer tier",
    "tomadorcodigopostal": "Policyholder postal code",
    "tomadormunicipioid": "Policyholder municipality",
    "edad_conductor1": "Driver age",
    "aceptoculpasinantecedentes": "Accepted fault, no prior record",
    "dias_notificacion": "Notification delay (days)",
    "anyomatricula": "Vehicle registration year",
    "comarcaid": "Region / district",
    "zona": "Geographic area",
    "ine_008": "% household income from other benefits",
    "ine_011": "Average household size (region)",
    "ine_021": "% non-Spanish European population (region)",
    "ine_026": "% population under 16 (region)",
    "sea_005": "% vote share, regional political party",
    "subramo": "Insurance sub-branch",
    "formapago": "Payment method",
    "tipo": "Vehicle type",
    "pesomaxautorizado": "Maximum authorised vehicle weight (kg)",
    "canalnombre": "Sales channel",
    "scoring": "Risk scoring group",  # CONFIRM with sponsor — see note
    "motor": "Engine type",
    "tomadornacionalidadid": "Policyholder nationality",
    "categoria": "Vehicle category",
    "flota": "Fleet flag",
    "tomadorexperian": "Policyholder credit score group (Experian)",
    "contratadanyospropios": "Own damages insurance",
}

MODEL_COLOR = "#D2691E"

SAMPLE_SIZE = 500
NUM_LIME_SAMPLES = 500

# 1. Load the tuned model and the test data
model = xgb.XGBClassifier()
model.load_model("xgboost_tuned.json")

test = pd.read_parquet("test_model_dataset.parquet")
train = pd.read_parquet("train_model_dataset.parquet")

X_test = test.drop(columns=[TARGET])
X_train = train.drop(columns=[TARGET])

# 2. Restore native categorical dtypes on a working copy used for XGBoost scoring
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
X_test_xgb = X_test.copy()
X_train_xgb = X_train.copy()
for col in cat_cols:
    X_train_xgb[col] = X_train_xgb[col].astype("category")
    X_test_xgb[col] = pd.Categorical(X_test_xgb[col], categories=X_train_xgb[col].cat.categories)

# 3. LIME needs a purely numeric array, not pandas categoricals, so build a
#    separate encoded version for LIME's explainer only. Each categorical
#    column is label-encoded; we keep the category-name mapping so LIME's
#    output can be decoded back to readable labels.
X_train_lime = X_train.copy()
X_test_lime = X_test.copy()
categorical_names = {}
feature_names = list(X_train.columns)
categorical_feature_idx = []

for col in cat_cols:
    idx = feature_names.index(col)
    categorical_feature_idx.append(idx)
    categories = X_train[col].astype("category").cat.categories
    mapping = {cat: i for i, cat in enumerate(categories)}
    categorical_names[idx] = list(categories)
    X_train_lime[col] = X_train[col].map(mapping).fillna(-1).astype(int)
    # Any test-set category not seen in training maps to -1 (unseen category)
    X_test_lime[col] = X_test[col].map(mapping).fillna(-1).astype(int)

X_train_lime_arr = X_train_lime.to_numpy(dtype=float)
X_test_lime_arr = X_test_lime.to_numpy(dtype=float)

# 4. Score all test rows, take the same top-scoring claims used for SHAP
proba = model.predict_proba(X_test_xgb)[:, 1]
top_idx = np.argsort(proba)[::-1][:SAMPLE_SIZE]
print(f"Explaining top {SAMPLE_SIZE} highest-scoring claims with LIME.")

# 5. Prediction function LIME will call repeatedly. It receives the numeric
#    (label-encoded) array, so we decode categoricals back before scoring.
def predict_fn(numeric_rows):
    df = pd.DataFrame(numeric_rows, columns=feature_names)
    for col in cat_cols:
        idx = feature_names.index(col)
        categories = categorical_names[idx]
        df[col] = df[col].round().astype(int).map(
            {i: cat for i, cat in enumerate(categories)}
        )
        df[col] = pd.Categorical(df[col], categories=categories)
    return model.predict_proba(df)

# 6. Build the LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train_lime_arr,
    feature_names=feature_names,
    categorical_features=categorical_feature_idx,
    categorical_names=categorical_names,
    class_names=["Non-fraud", "Fraud"],
    mode="classification",
)

# 7. Explain each instance, timed
records = []
start = time.time()
for i, row_idx in enumerate(top_idx):
    row_start = time.time()
    exp = explainer.explain_instance(
        X_test_lime_arr[row_idx],
        predict_fn,
        num_features=10,
        num_samples=NUM_LIME_SAMPLES,
    )
    row_time = time.time() - row_start
    for feature_desc, weight in exp.as_list():
        records.append({"claim_index": int(row_idx), "feature_rule": feature_desc,
                        "weight": weight, "seconds": row_time})
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        print(f"  {i+1}/{SAMPLE_SIZE} done, {elapsed:.1f}s elapsed")

total_time = time.time() - start
print(f"\nDone. Total: {total_time:.1f}s for {SAMPLE_SIZE} instances "
      f"({total_time/SAMPLE_SIZE:.2f}s/instance average).")

lime_df = pd.DataFrame(records)
lime_df.to_csv("xgboost_lime_raw_top500.csv", index=False)
print("Saved: xgboost_lime_raw_top500.csv")

# 8. Aggregate: mean absolute weight per underlying feature (strip the LIME
#    rule text like "antiguedad_poliza <= 120" down to the base feature name)
def base_feature(rule_text):
    for f in feature_names:
        if f in rule_text:
            return f
    return rule_text

lime_df["base_feature"] = lime_df["feature_rule"].apply(base_feature)
agg = (lime_df.groupby("base_feature")["weight"]
       .apply(lambda x: x.abs().mean())
       .sort_values(ascending=False)
       .reset_index())
agg.columns = ["feature", "mean_abs_lime_weight"]
agg["label"] = agg["feature"].map(LABEL_MAP).fillna(agg["feature"])

# Flag anything in the FINAL top-10 ranking that's still unmapped. Checked
# against the actual top 10 by aggregated weight, i.e. exactly what gets
# plotted below, not a separate frequency count that can miss features
# that only rank highly after aggregation.
top10_by_weight = agg.head(10)["feature"].tolist()
unmapped = [f for f in top10_by_weight if f not in LABEL_MAP]
if unmapped:
    print(f"\nWARNING: no human-readable label found for: {unmapped}")
    print("Check variables_ddbb_fraud_original.xlsx and add them to LABEL_MAP.\n")

print("\nTop LIME features (mean |weight|), n=500:")
print(agg.head(10).to_string(index=False))
agg.to_csv("xgboost_lime_importance_top500.csv", index=False)
print("Saved: xgboost_lime_importance_top500.csv")

# 9. Bar chart, same visual style as the SHAP bar chart
plt.figure(figsize=(9, 6))
top10 = agg.head(10).iloc[::-1]
plt.barh(top10["label"], top10["mean_abs_lime_weight"], color=MODEL_COLOR)
plt.xlabel("mean(|LIME weight|)")
plt.title(f"XGBoost \u2014 Top LIME Feature Importance (n={SAMPLE_SIZE})",
          fontsize=13, fontweight="bold", color=MODEL_COLOR)
plt.tight_layout()
plt.savefig("xgboost_lime_bar_top500.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: xgboost_lime_bar_top500.png")

if "scoring" in top10_by_weight:
    print("\nREMINDER: 'scoring' (risk_scoring_group) is in the top 10 and its")
    print("exact meaning is not yet confirmed with the sponsor. Do not treat")
    print("this chart as final for the report until that's resolved.")