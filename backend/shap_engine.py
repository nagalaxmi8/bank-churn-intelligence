"""
========================================================
Customer Churn SHAP + Dynamic Retention Engine
========================================================

Flow:
1. Load trained model
2. Load & preprocess dataset
3. Align feature columns
4. Compute SHAP values (sampled for speed)
5. Compute churn probability
6. Segment risk
7. Generate retention actions only if needed
"""

import pandas as pd
import shap
import joblib
import numpy as np

from backend.retention_engine import get_top_shap_reasons, map_rules_to_actions


# ======================================================
# 1️⃣ LOAD MODEL + FEATURE COLUMNS
# ======================================================

model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

print("✅ Model loaded.")


# ======================================================
# 2️⃣ LOAD AND PREPROCESS DATA
# ======================================================

df = pd.read_csv("data/Bank Customer Churn Prediction.csv")

df["balance_salary_ratio"] = df["balance"] / (df["estimated_salary"] + 1)
df["low_tenure"] = (df["tenure"] <= 2).astype(int)
df["single_product"] = (df["products_number"] == 1).astype(int)

df = df.drop(columns=["customer_id"])
df = pd.get_dummies(df, columns=["country", "gender"], drop_first=True)

X = df.drop("churn", axis=1).astype(float)

# Align feature columns exactly as training
X = X.reindex(columns=feature_columns, fill_value=0)

print("✅ Dataset prepared.")
print("Dataset shape:", X.shape)


# ======================================================
# 3️⃣ SAMPLE FOR SHAP (FAST)
# ======================================================

X_sample = X.sample(300, random_state=42)


# ======================================================
# 4️⃣ SHAP EXPLAINER
# ======================================================

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Handle SHAP output format safely
if isinstance(shap_values, list):
    shap_class1 = shap_values[1]

elif isinstance(shap_values, np.ndarray):
    if shap_values.ndim == 3:
        shap_class1 = shap_values[:, :, 1]
    elif shap_values.ndim == 2:
        shap_class1 = shap_values
    else:
        raise ValueError("Unexpected SHAP ndarray shape")
else:
    raise ValueError("Unexpected SHAP output format")

print("✅ SHAP values computed.")


# ======================================================
# 5️⃣ GLOBAL FEATURE IMPORTANCE
# ======================================================

mean_abs_shap = np.abs(shap_class1).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "MeanAbsSHAP": mean_abs_shap
}).sort_values(by="MeanAbsSHAP", ascending=False)

print("\n🔎 Top 10 Global Important Features:")
print(importance_df.head(10))


# ======================================================
# 6️⃣ EXPLAIN ONE CUSTOMER
# ======================================================

customer_index = 0
customer_features = X_sample.iloc[[customer_index]]

# ---- Churn Probability ----
churn_prob = model.predict_proba(customer_features)[0][1]

print(f"\n📊 Churn Probability: {round(churn_prob,4)}")

# ---- Risk Segmentation ----
if churn_prob < 0.3:
    risk = "Low"
elif churn_prob < 0.7:
    risk = "Medium"
else:
    risk = "High"

print("🚦 Risk Level:", risk)


# ======================================================
# 7️⃣ RETENTION DECISION LOGIC
# ======================================================

if risk in ["Medium", "High"]:

    # Extract top SHAP reasons
    top_reasons = get_top_shap_reasons(
        shap_class1[customer_index],
        X_sample.columns,
        top_n=3
    )

    print("\n🚨 Top SHAP churn drivers:")
    print(top_reasons)

    # Generate dynamic retention actions
    actions = map_rules_to_actions(top_reasons)

    print("\n🎯 Recommended Retention Actions:")
    for act in actions:
        print("-", act)

else:
    print("\n✅ Customer is low risk. No retention action required.")
