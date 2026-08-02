import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shap
import xgboost as xgb

TARGET = "respuesta_dicot_c"

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
}

MODEL_COLOR = "#D2691E"

# 1. Load the tuned model and the test data
model = xgb.XGBClassifier()
model.load_model("xgboost_tuned.json")

test = pd.read_parquet("test_model_dataset.parquet")
train = pd.read_parquet("train_model_dataset.parquet")

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

X_train = train.drop(columns=[TARGET])
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

proba = model.predict_proba(X_test)[:, 1]
top_idx = np.argsort(proba)[::-1][:500]
X_top = X_test.iloc[top_idx].copy()
print(f"Explaining top {len(X_top)} highest-scoring claims.")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_top)

importance = np.abs(shap_values).mean(axis=0)
imp_df = pd.DataFrame({
    "feature": X_top.columns,
    "mean_abs_shap": importance,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 10 features by mean |SHAP|:")
print(imp_df.head(10).to_string(index=False))
imp_df.to_csv("xgboost_shap_importance.csv", index=False)

top10_features = imp_df.head(10)["feature"].tolist()
unmapped = [f for f in top10_features if f not in LABEL_MAP]
if unmapped:
    print(f"\nWARNING: no human-readable label found for: {unmapped}")
    print("These will show as their raw variable name in the plots below.")
    print("Check variables_ddbb_fraud_original.xlsx and add them to LABEL_MAP.\n")

X_top_display = X_top.rename(columns=LABEL_MAP)

# 6. Summary (beeswarm) plot — UNCHANGED, default shap styling and font kept
# exactly as before. Its colour scale encodes feature value (high/low), not
# decoration, so it is intentionally left alone.
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_top_display, max_display=10, show=False)
plt.xlabel("SHAP value (impact on model output)")
plt.tight_layout()
plt.savefig("xgboost_shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()

# 7. Bar plot — orange theme, Verdana font, new title
available_fonts = {f.name for f in fm.fontManager.ttflist}
if "Verdana" in available_fonts:
    bar_font = "Verdana"
    print("Using Verdana for bar chart.")
else:
    bar_font = "DejaVu Sans"
    print("WARNING: Verdana not found, bar chart falling back to DejaVu Sans.")
    print("To install: sudo apt install ttf-mscorefonts-installer && fc-cache -f -v")

with plt.rc_context({"font.family": bar_font}):
    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(shap_values, X_top_display, plot_type="bar", max_display=10, show=False)

    # Recolor bars to the XGBoost orange, regardless of installed shap version's
    # default bar-plot color handling
    ax = plt.gca()
    for patch in ax.patches:
        patch.set_facecolor(MODEL_COLOR)

    plt.xlabel("mean(|SHAP value|)")
    plt.title("XGBoost \u2014 Top SHAP Feature Importance (Tuned Model)",
              fontsize=14, fontweight="bold", color=MODEL_COLOR, pad=12)
    plt.tight_layout()
    plt.savefig("xgboost_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

print("\nSaved: xgboost_shap_importance.csv, xgboost_shap_summary.png, xgboost_shap_bar.png")

mean_abs = np.abs(shap_values).mean(axis=0)
positive_share = (shap_values > 0).mean(axis=0)
shap_table = pd.DataFrame({
    "feature": X_top.columns,
    "mean_abs_shap": mean_abs,
    "positive_share": positive_share,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 10 SHAP risk drivers (RFQC-style table):")
print(shap_table.head(10).to_string(index=False))
shap_table.head(10).to_csv("xgboost_shap_table_top10.csv", index=False)
print("\nSaved: xgboost_shap_table_top10.csv")