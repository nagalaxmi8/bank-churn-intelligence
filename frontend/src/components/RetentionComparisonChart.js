import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function RetentionComparisonChart({ before, after }) {

  const data = [
    { name: "Before Retention", value: before *100 },
    { name: "After Retention", value: after *100 }
  ];

  return (
    <div className="card">
      <h3>Retention Impact Simulation</h3>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}