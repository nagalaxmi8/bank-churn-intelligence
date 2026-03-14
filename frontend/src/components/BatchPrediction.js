import React, { useState } from "react";
import axios from "axios";

import {
PieChart,
Pie,
Cell,
Tooltip,
Legend,
BarChart,
Bar,
XAxis,
YAxis,
CartesianGrid
} from "recharts";

/* Colors for risk chart */
const COLORS = ["#52c41a", "#faad14", "#ff4d4f"];

export default function BatchPrediction() {

const [file, setFile] = useState(null);
const [result, setResult] = useState(null);
const [loading, setLoading] = useState(false);

/* ---------- FILE SELECTION ---------- */

const handleFileChange = (event) => {
setFile(event.target.files[0]);
};

/* ---------- SEND FILE TO BACKEND ---------- */

const handleUpload = async () => {


if (!file) {
  alert("Please upload a CSV file first.");
  return;
}

const formData = new FormData();
formData.append("file", file);

try {

  setLoading(true);

  const response = await axios.post(
    "http://127.0.0.1:8000/batch_predict",
    formData
  );

  setResult(response.data);

} catch (error) {

  console.error("Batch prediction error:", error);

} finally {

  setLoading(false);

}


};

/* ---------- RISK PIE DATA ---------- */

const riskData = result
? Object.entries(result.portfolio_metrics.risk_distribution).map(
([key, value]) => ({
name: key,
value: value
})
)
: [];

/* ---------- SHAP DRIVER DATA ---------- */

const shapData = result
? result.top_portfolio_drivers.map((driver) => ({
feature: driver.feature,
importance: driver.importance
}))
: [];

/* ---------- SEGMENTATION DATA ---------- */

const segmentationData = result
? [
{
segment: "Low Risk",
customers: result.portfolio_metrics.risk_distribution.Low || 0
},
{
segment: "Medium Risk",
customers: result.portfolio_metrics.risk_distribution.Medium || 0
},
{
segment: "High Risk",
customers: result.portfolio_metrics.risk_distribution.High || 0
}
]
: [];

return (
<div style={{ padding: "20px" }}>


  <h2>Batch Churn Prediction</h2>

  {/* Upload CSV */}
  <input type="file" accept=".csv" onChange={handleFileChange} />

  <br /><br />

  <button onClick={handleUpload}>
    Upload & Predict
  </button>

  {loading && <p>Processing dataset...</p>}

  {result && (

    <div style={{ marginTop: "20px" }}>

      {/* PORTFOLIO METRICS */}

      <h3>Portfolio Metrics</h3>

      <p>Total Customers: {result.portfolio_metrics.total_customers}</p>

      <p>
        Average Churn Probability:
        {" "}
        {(result.portfolio_metrics.average_churn_probability * 100).toFixed(2)}%
      </p>

      <p>
        Predicted Churners:
        {" "}
        {result.portfolio_metrics.predicted_churners}
      </p>

      <p>
        High Risk Customers:
        {" "}
        {result.portfolio_metrics.high_risk_customers}
      </p>


      {/* RISK DISTRIBUTION PIE CHART */}

      <h3>Portfolio Risk Distribution</h3>

      <PieChart width={400} height={300}>

        <Pie
          data={riskData}
          cx="50%"
          cy="50%"
          outerRadius={100}
          dataKey="value"
          label
        >

          {riskData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={COLORS[index % COLORS.length]}
            />
          ))}

        </Pie>

        <Tooltip />
        <Legend />

      </PieChart>


      {/* SHAP DRIVER CHART */}

      <h3>Top Churn Drivers</h3>

      <BarChart
        width={500}
        height={300}
        data={shapData}
      >

        <CartesianGrid strokeDasharray="3 3" />

        <XAxis dataKey="feature" />

        <YAxis />

        <Tooltip />

        <Bar dataKey="importance" fill="#8884d8" />

      </BarChart>


      {/* CUSTOMER SEGMENTATION */}

      <h3>Customer Segmentation</h3>

      <BarChart
        width={500}
        height={300}
        data={segmentationData}
      >

        <CartesianGrid strokeDasharray="3 3" />

        <XAxis dataKey="segment" />

        <YAxis />

        <Tooltip />

        <Bar dataKey="customers" fill="#00C49F" />

      </BarChart>


      {/* STRATEGY OUTPUT */}

      <h3>Portfolio Strategy</h3>

      <p>{result.portfolio_strategy.portfolio_summary}</p>

      <ul>
        {result.portfolio_strategy.recommended_actions.map((a, i) => (
          <li key={i}>{a}</li>
        ))}
      </ul>

    </div>

  )}

</div>


);
}
