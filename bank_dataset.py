"""
========================================================
Customer Churn Model Training Script
========================================================

This script performs:

1. Data loading
2. Feature engineering
3. Train-test split
4. Model training
5. Model evaluation
6. Saving model + feature schema

Artifacts produced:
- churn_model.pkl
- feature_columns.pkl
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# ======================================================
# 1️⃣ LOAD DATA
# ======================================================

print("Loading dataset...")

df = pd.read_csv("data/Bank Customer Churn Prediction.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ======================================================
# 2️⃣ FEATURE ENGINEERING
# ======================================================

print("\nPerforming feature engineering...")

# Create new meaningful features
df["balance_salary_ratio"] = df["balance"] / (df["estimated_salary"] + 1)
df["low_tenure"] = (df["tenure"] <= 2).astype(int)
df["single_product"] = (df["products_number"] == 1).astype(int)

# Remove non-informative ID column
df = df.drop(columns=["customer_id"])

# Encode categorical variables
df = pd.get_dummies(df, columns=["country", "gender"], drop_first=True)

print("Feature engineering completed.")
print("Updated dataset shape:", df.shape)


# ======================================================
# 3️⃣ DEFINE FEATURES & TARGET
# ======================================================

X = df.drop("churn", axis=1)
y = df["churn"]

# Save feature column order for inference alignment
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")

print("\nFeature schema saved as feature_columns.pkl")


# ======================================================
# 4️⃣ TRAIN-TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train-test split completed.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ======================================================
# 5️⃣ MODEL TRAINING
# ======================================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed.")


# ======================================================
# 6️⃣ MODEL EVALUATION
# ======================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


# ======================================================
# 7️⃣ SAVE PRODUCTION MODEL
# ======================================================

joblib.dump(model, "churn_model.pkl")

print("\nProduction model saved as churn_model.pkl")
print("Training pipeline completed successfully.")
