import React, { useState } from "react";
import ProbabilityCard from "./ProbabilityCard";
import ShapChart from "./ShapChart";
import RetentionComparisonChart from "./RetentionComparisonChart";
import StrategyPanel from "./StrategyPanel";

export default function SinglePrediction() {

  const [formData, setFormData] = useState({
    credit_score: "",
    age: "",
    tenure: "",
    balance: "",
    products_number: "",
    credit_card: "",
    active_member: "",
    estimated_salary: ""
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: Number(e.target.value)
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData)
    });

    const data = await response.json();

  if (data) {
    setResult(data);
  }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input name="credit_score" placeholder="Credit Score" onChange={handleChange}/>
        <input name="age" placeholder="Age" onChange={handleChange}/>
        <input name="tenure" placeholder="Tenure" onChange={handleChange}/>
        <input name="balance" placeholder="Balance" onChange={handleChange}/>
        <input name="products_number" placeholder="Products Number" onChange={handleChange}/>
        <input name="credit_card" placeholder="Credit Card (0/1)" onChange={handleChange}/>
        <input name="active_member" placeholder="Active Member (0/1)" onChange={handleChange}/>
        <input name="estimated_salary" placeholder="Estimated Salary" onChange={handleChange}/>
        <button type="submit">Predict</button>
      </form>

      {result && (
        <>
          <ProbabilityCard probability={result.churn_probability} risk={result.risk_level} />

          <ShapChart drivers={result.top_drivers} />

          <RetentionComparisonChart
            before={result.retention_impact.before}
            after={result.retention_impact.after} />


          <StrategyPanel strategy={result.strategy} />
        </>
      )}
    </div>
  );
} 