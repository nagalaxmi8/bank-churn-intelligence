# Bank Churn Intelligence System

An AI-powered system that predicts customer churn and recommends personalized retention strategies using machine learning, SHAP explainability, and generative AI.

---

## Project Overview

Customer churn is a major challenge for banks and financial institutions. This project builds an intelligent churn prediction system that:

• Predicts the probability of a customer leaving the bank
• Explains the key factors causing churn using SHAP
• Generates targeted retention strategies for at-risk customers
• Provides batch analysis for entire customer datasets

The system combines machine learning, explainable AI, and LLM-powered insights to help businesses proactively retain customers.

---

## Key Features

### Customer Churn Prediction

Predicts the probability that a customer will leave the bank using a trained Random Forest model.

### Explainable AI (SHAP)

Identifies the top features contributing to churn risk for each customer.

### Risk Segmentation

Customers are classified into risk levels:

* Low Risk
* Medium Risk
* High Risk

### AI-Powered Retention Strategy

Generates intelligent retention actions using Google's Gemini model.

### Batch Dataset Analysis

Allows businesses to upload a full dataset and get portfolio-level churn insights.

---

## Tech Stack

### Backend

* Python
* FastAPI
* Scikit-learn
* SHAP
* Pandas
* NumPy

### Frontend

* React.js
* Axios

### AI / ML

* Random Forest Model
* SHAP Explainability
* Google Gemini Generative AI

---

## Project Structure

```
bank-churn-intelligence
│
├── backend
│   ├── main.py
│   ├── retention_engine.py
│   ├── shap_engine.py
│   ├── bank_dataset.py
│   ├── churn_model.pkl
│   └── feature_columns.pkl
│
├── frontend
│   ├── src
│   ├── public
│   └── package.json
│
├── data
│   └── Bank Customer Churn Prediction.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```
git clone https://github.com/nagalaxmi8/bank-churn-intelligence.git
```

Move into the project folder:

```
cd bank-churn-intelligence
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment:

Windows

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Backend API

Start the FastAPI server:

```
uvicorn backend.main:app --reload
```

The API will run at:

```
http://127.0.0.1:8000
```

API documentation:

```
http://127.0.0.1:8000/docs
```

---

## Running the Frontend

Move to the frontend folder:

```
cd frontend
```

Install dependencies:

```
npm install
```

Run the React app:

```
npm start
```

Frontend will run at:

```
http://localhost:3000
```

---

## Example API Endpoint

POST request:

```
/predict
```

Example request body:

```
{
  "credit_score": 650,
  "age": 45,
  "tenure": 5,
  "balance": 120000,
  "products_number": 1,
  "credit_card": 1,
  "active_member": 0,
  "estimated_salary": 60000
}
```

Example response:

```
{
  "churn_probability": 0.72,
  "risk_level": "High",
  "top_drivers": [
    {"feature": "balance", "impact": 0.21},
    {"feature": "age", "impact": 0.18},
    {"feature": "single_product", "impact": 0.12}
  ],
  "rule_based_actions": [
    "Offer bundle discount or cross-sell products",
    "Provide premium financial benefits"
  ]
}
```

---

## Future Improvements

• Deploy the system on cloud platforms
• Add dashboard analytics for churn monitoring
• Integrate real banking datasets
• Improve retention strategy recommendations using advanced LLM prompts

---

## Author

Naga Laxmi

Machine Learning & AI Enthusiast
