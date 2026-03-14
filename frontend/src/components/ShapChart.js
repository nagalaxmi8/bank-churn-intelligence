import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function ShapChart({ drivers }) {

  if (!drivers || drivers.length === 0) {
  return <p>No SHAP data available</p>;
}

  const data = drivers.map(d => ({
  name: d.feature,
  impact: d.impact ?? d.importance
}));

  return (
    <div className="card">
      <h3>Top Churn Drivers (SHAP)</h3>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart layout="vertical" data={data}>
          <XAxis type="number" />
          <YAxis type="category" dataKey="name" />
          <Tooltip />
          <Bar dataKey="impact" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}