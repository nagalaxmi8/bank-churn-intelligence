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
from impact_engine import compute_retention_impact


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
# 7️⃣ Single Prediction (UNCHANGED)
# ======================================================

@app.post("/predict")
def predict_churn(customer: CustomerInput):

    input_df = pd.DataFrame([customer.dict()])

    input_df["balance_salary_ratio"] = input_df["balance"] / (input_df["estimated_salary"] + 1)
    input_df["low_tenure"] = (input_df["tenure"] <= 2).astype(int)
    input_df["single_product"] = (input_df["products_number"] == 1).astype(int)

    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    probability = float(model.predict_proba(input_df)[0][1])
    risk_level = get_risk_level(probability)

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
        {"feature": row["Feature"], "impact": round(float(row["SHAP_Value"]), 4)}
        for _, row in top_reasons_df.iterrows()
    ]

    impact_result = compute_retention_impact(probability, [], top_drivers)

    source = "LLM"

    prompt = f"""
Customer churn probability: {probability}
Risk level: {risk_level}
Drivers: {top_drivers}

Return JSON with strategy_summary and recommended_actions
"""

    strategy = generate_llm_strategy(prompt)

    if strategy is None:
        strategy = {
            "strategy_summary": "Fallback strategy",
            "recommended_actions": ["Improve engagement", "Offer incentives"]
        }
        source = "RULE"

    return {
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
        "top_drivers": top_drivers,
        "retention_impact": impact_result,
        "strategy": {
            "source": source,
            **strategy
        }
    }


# ======================================================
# 8️⃣ Batch Prediction (UPDATED ONLY)
# ======================================================

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):

    try:
        df = pd.read_csv(file.file)

        df["balance_salary_ratio"] = df["balance"] / (df["estimated_salary"] + 1)
        df["low_tenure"] = (df["tenure"] <= 2).astype(int)
        df["single_product"] = (df["products_number"] == 1).astype(int)

        df_model = df.reindex(columns=feature_columns, fill_value=0)

        probs = model.predict_proba(df_model)[:, 1]

        # ===== SHAP ADDED =====
        try:
            shap_values = explainer.shap_values(df_model)
            shap_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values
            mean_shap = np.abs(shap_class1).mean(axis=0)

            top_indices = np.argsort(mean_shap)[::-1][:5]

            top_portfolio_drivers = [
                {"feature": df_model.columns[i], "importance": float(mean_shap[i])}
                for i in top_indices
            ]
        except:
            top_portfolio_drivers = []

        df["churn_probability"] = probs
        df["risk_level"] = df["churn_probability"].apply(get_risk_level)

        avg_churn = float(np.mean(probs))
        total_customers = len(df)
        predicted_churners = int((probs > 0.5).sum())
        high_risk_count = int((df["risk_level"] == "High").sum())
        risk_distribution = df["risk_level"].value_counts().to_dict()

        # ===== IMPACT ADDED =====
        portfolio_impact = {
            "before": round(avg_churn, 4),
            "after": round(max(avg_churn - 0.15, 0), 4),
            "improvement": round(min(0.15, avg_churn), 4)
        }

        # ===== GEMINI STRATEGY ADDED =====
        prompt = f"""
Portfolio churn: {avg_churn}
Risk distribution: {risk_distribution}
Drivers: {top_portfolio_drivers}

Return JSON:
{{"portfolio_summary":"...","recommended_actions":["..."]}}
"""

        strategy = generate_llm_strategy(prompt)

        if strategy is None:
            strategy = {
                "portfolio_summary": "Fallback strategy",
                "recommended_actions": [
                    "Target high-risk customers",
                    "Offer incentives",
                    "Improve engagement"
                ]
            }

        return {
            "portfolio_metrics": {
                "total_customers": total_customers,
                "average_churn_probability": round(avg_churn, 4),
                "predicted_churners": predicted_churners,
                "high_risk_customers": high_risk_count,
                "risk_distribution": risk_distribution
            },
            "portfolio_impact": portfolio_impact,
            "top_portfolio_drivers": top_portfolio_drivers,
            "portfolio_strategy": strategy
        }

    except Exception as e:
        return {"error": str(e)}


# ======================================================
# 9️⃣ Health
# ======================================================

@app.get("/")
def health():
    return {"message": "Churn Intelligence API running"}