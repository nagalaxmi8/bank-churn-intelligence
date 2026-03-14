
# ======================================================
# Customer Churn Intelligence API (LLM-PRIMARY VERSION)
# ======================================================

import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
from dotenv import load_dotenv
from google import genai

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retention_engine import get_top_shap_reasons, map_rules_to_actions


# ======================================================
# 1️⃣ FastAPI Setup
# ======================================================

app = FastAPI(title="Customer Churn Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# 2️⃣ Load Model + SHAP
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
explainer = shap.TreeExplainer(model)

print("✅ Model and SHAP loaded.")


# ======================================================
# 3️⃣ LLM Setup
# ======================================================

load_dotenv(os.path.join(BASE_DIR, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini LLM initialized.")
else:
    client = None
    print("⚠️ Gemini API key missing. Using rule fallback.")


# ======================================================
# 4️⃣ Input Schema
# ======================================================

class CustomerInput(BaseModel):
    credit_score: float
    age: float
    tenure: float
    balance: float
    products_number: float
    credit_card: float
    active_member: float
    estimated_salary: float
    country_Germany: float = 0
    country_Spain: float = 0
    gender_Male: float = 0


# ======================================================
# 5️⃣ Risk Segmentation
# ======================================================

def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


# ======================================================
# 6️⃣ LLM Generator
# ======================================================

def generate_llm_strategy(prompt_text):

    if client is None:
        return None

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text
        )

        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        first = raw.find("{")
        last = raw.rfind("}")

        if first != -1 and last != -1:
            raw = raw[first:last+1]

        return json.loads(raw)

    except Exception as e:
        print("❌ LLM ERROR:", e)
        return None


# ======================================================
# 7️⃣ Single Prediction
# ======================================================

@app.post("/predict")
def predict_churn(customer: CustomerInput):

    input_df = pd.DataFrame([customer.dict()])

    # Feature engineering
    input_df["balance_salary_ratio"] = input_df["balance"] / (input_df["estimated_salary"] + 1)
    input_df["low_tenure"] = (input_df["tenure"] <= 2).astype(int)
    input_df["single_product"] = (input_df["products_number"] == 1).astype(int)

    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Prediction
    probability = float(model.predict_proba(input_df)[0][1])
    risk_level = get_risk_level(probability)

    # SHAP
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        shap_class1 = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_class1 = shap_values[:, :, 1]
    else:
        shap_class1 = shap_values

    top_reasons_df = get_top_shap_reasons(
        shap_class1[0],
        input_df.columns,
        top_n=3
    )

    top_drivers = [
        {
            "feature": row["Feature"],
            "impact": round(float(row["SHAP_Value"]), 4)
        }
        for _, row in top_reasons_df.iterrows()
    ]

    source = "LLM"

    if risk_level in ["Medium", "High"]:

        prompt = f"""
You are a senior banking retention strategist.

Customer churn probability: {probability:.4f}
Risk level: {risk_level}

Top churn drivers:
{json.dumps(top_drivers, indent=2)}

Return ONLY JSON:

{{
  "strategy_summary": "...",
  "recommended_actions": ["...", "..."],
  "business_reasoning": "...",
  "projected_probability_after_retention": 0.45
}}
"""

        strategy = generate_llm_strategy(prompt)

        if strategy is None:
            rule_actions = map_rules_to_actions(top_reasons_df)

            strategy = {
                "strategy_summary": "Rule-based retention strategy (LLM unavailable).",
                "recommended_actions": rule_actions,
                "business_reasoning": "Mapped SHAP drivers to retention policies.",
                "projected_probability_after_retention": round(probability * 0.8, 4)
            }

            source = "RULE"

    else:
        strategy = {
            "strategy_summary": "Customer is low risk. No retention required.",
            "recommended_actions": [],
            "business_reasoning": "Low churn probability.",
            "projected_probability_after_retention": probability
        }

    return {
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
        "top_drivers": top_drivers,
        "strategy": {
            "source": source,
            **strategy
        }
    }


# ======================================================
# 8️⃣ Batch Prediction
# ======================================================


@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):

    try:
        print("Batch request received")

        # -----------------------------
        # Load dataset
        # -----------------------------
        df = pd.read_csv(file.file)

        # -----------------------------
        # Feature engineering
        # -----------------------------
        df["balance_salary_ratio"] = df["balance"] / (df["estimated_salary"] + 1)
        df["low_tenure"] = (df["tenure"] <= 2).astype(int)
        df["single_product"] = (df["products_number"] == 1).astype(int)

        df_model = df.reindex(columns=feature_columns, fill_value=0)

        # -----------------------------
        # Prediction
        # -----------------------------
        print("Running predictions")

        probs = model.predict_proba(df_model)[:, 1]

        df["churn_probability"] = probs
        df["risk_level"] = df["churn_probability"].apply(get_risk_level)

        avg_churn = float(np.mean(probs))
        total_customers = len(df)
        predicted_churners = int((probs > 0.5).sum())
        high_risk_count = int((df["risk_level"] == "High").sum())
        risk_distribution = df["risk_level"].value_counts().to_dict()

        # -----------------------------
        # FAST SHAP (sample only)
        # -----------------------------
        print("Running SHAP")

        sample_size = min(200, len(df_model))
        sample_df = df_model.sample(sample_size, random_state=42)

        shap_values = explainer.shap_values(sample_df)

        if isinstance(shap_values, list):
            shap_class1 = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_class1 = shap_values[:, :, 1]
        else:
            shap_class1 = shap_values

        mean_abs_shap = np.abs(shap_class1).mean(axis=0)

        shap_df = pd.DataFrame({
            "feature": sample_df.columns,
            "importance": mean_abs_shap
        }).sort_values(by="importance", ascending=False)

        top_portfolio_drivers = shap_df.head(5).to_dict(orient="records")

        # -----------------------------
        # LLM Strategy
        # -----------------------------
        print("Calling LLM")

        prompt = f"""
You are a senior banking strategist.

Portfolio Summary:
Total Customers: {total_customers}
Average Churn Probability: {avg_churn:.4f}
Predicted Churners (>0.5): {predicted_churners}
High Risk Customers: {high_risk_count}

Top Portfolio Drivers:
{json.dumps(top_portfolio_drivers, indent=2)}

Return ONLY JSON:

{{
  "portfolio_summary": "...",
  "recommended_actions": ["...", "..."],
  "business_impact": "...",
  "projected_portfolio_churn_after_retention": 0.35
}}
"""

        source = "LLM"

        try:
            strategy = generate_llm_strategy(prompt)
        except Exception:
            strategy = None

        # fallback strategy if LLM fails
        if not strategy:
            source = "RULE"
            strategy = {
                "portfolio_summary": "Rule-based portfolio retention strategy.",
                "recommended_actions": [
                    "Target high-risk customers with personalized retention offers",
                    "Provide loyalty rewards for long-tenure customers",
                    "Promote bundled financial products"
                ],
                "business_impact": "Focused retention strategy reduces predicted churn.",
                "projected_portfolio_churn_after_retention": round(avg_churn * 0.8, 4)
            }

        print("Returning response")

        # -----------------------------
        # Response
        # -----------------------------
        return {
            "portfolio_metrics": {
                "total_customers": total_customers,
                "average_churn_probability": round(avg_churn, 4),
                "predicted_churners": predicted_churners,
                "high_risk_customers": high_risk_count,
                "risk_distribution": risk_distribution
            },
            "top_portfolio_drivers": top_portfolio_drivers,
            "portfolio_strategy": {
                "source": source,
                **strategy
            }
        }

    except Exception as e:
        print("Batch error:", str(e))
        return {"error": str(e)}
# ======================================================
# 9️⃣ Health
# ======================================================

@app.get("/")
def health():
    return {"message": "Churn Intelligence API running"}