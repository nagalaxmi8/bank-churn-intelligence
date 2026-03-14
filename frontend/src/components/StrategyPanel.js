export default function StrategyPanel({ strategy }) {

  return (
    <div className="card">
      <h3>Retention Strategy ({strategy.source})</h3>

      <p><strong>Summary:</strong> {strategy.strategy_summary ?? strategy.portfolio_summary}</p>

      <h4>Recommended Actions</h4>
      <ul>
        {strategy.recommended_actions.map((action, i) => (
          <li key={i}>{action}</li>
        ))}
      </ul>

      <h4>Business Reasoning</h4>
      <p>{strategy.business_reasoning ?? strategy.business_impact}</p>
    </div>
  );
}