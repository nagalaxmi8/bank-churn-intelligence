export default function ProbabilityCard({ probability = 0, risk = "Unknown" }) {

  const percentage = (probability * 100).toFixed(2);

  return (
    <div className="card">
      <h2>Churn Probability</h2>
      <h1>{percentage}%</h1>
      <p>Risk Level: {risk}</p>
    </div>
  );
}