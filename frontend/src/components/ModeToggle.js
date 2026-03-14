export default function ModeToggle({ mode, setMode }) {
  return (
    <div className="toggle">
      <button
        className={mode === "single" ? "active" : ""}
        onClick={() => setMode("single")}
      >
        Single Prediction
      </button>

      <button
        className={mode === "batch" ? "active" : ""}
        onClick={() => setMode("batch")}
      >
        Batch Prediction
      </button>
    </div>
  );
}