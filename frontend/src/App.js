import React, { useState } from "react";
import ModeToggle from "./components/ModeToggle";
import SinglePrediction from "./components/SinglePrediction";
import BatchPrediction from "./components/BatchPrediction";
import "./App.css";

function App() {
  const [mode, setMode] = useState("single");

  return (
    <div className="container">
      <h1>Customer Churn Intelligence Dashboard</h1>

      <ModeToggle mode={mode} setMode={setMode} />

      {mode === "single" && <SinglePrediction />}
      {mode === "batch" && <BatchPrediction />}
    </div>
  );
}

export default App;