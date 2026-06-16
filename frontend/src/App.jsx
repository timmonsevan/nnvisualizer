import { useState, useRef, useCallback } from 'react'
import DrawCanvas from './components/DrawCanvas'
import NetworkGraph from './components/NetworkGraph'
import PredictionBar from './components/PredictionBar'
import './App.css'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const canvasRef = useRef(null)

  const handleStrokeEnd = useCallback(async (dataUrl) => {
    setLoading(true)
    setError(null)
    try {
      const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setResult(await res.json())
    } catch {
      setError('Could not reach the backend. It may be waking up — try again in a moment.')
    } finally {
      setLoading(false)
    }
  }, [])

  function handleClear() {
    canvasRef.current?.clear()
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <h1>Neural Net Visualizer</h1>

      <div className="top-panel">
        <div className="canvas-section">
          <DrawCanvas ref={canvasRef} onStrokeEnd={handleStrokeEnd} />
          <button onClick={handleClear}>Clear</button>
          {error && <p className="error-msg">{error}</p>}
        </div>

        <PredictionBar result={result} loading={loading} />
      </div>

      <NetworkGraph activations={result?.activations ?? null} />
    </div>
  )
}
