export default function PredictionBar({ result, loading }) {
  if (loading) {
    return (
      <div className="prediction-bar state-msg">
        Analysing…
      </div>
    )
  }

  if (!result) {
    return (
      <div className="prediction-bar state-msg">
        Draw a digit to see predictions
      </div>
    )
  }

  const { prediction, activations } = result
  const outputs = activations[3]   // softmax values, length 10
  const confidence = outputs[prediction]

  return (
    <div className="prediction-bar">
      <div className="big-digit">
        <span className="digit-num">{prediction}</span>
        <span className="digit-conf">{(confidence * 100).toFixed(1)}%</span>
      </div>

      <div className="bars">
        {outputs.map((v, i) => (
          <div key={i} className={`bar-row ${i === prediction ? 'winner' : ''}`}>
            <span className="bar-label">{i}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(v * 100).toFixed(2)}%` }} />
            </div>
            <span className="bar-pct">{(v * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}
