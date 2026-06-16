import { useRef, useEffect, useImperativeHandle, forwardRef } from 'react'

const SIZE = 280
const BRUSH = 20

const DrawCanvas = forwardRef(function DrawCanvas({ onStrokeEnd }, ref) {
  const canvasEl = useRef(null)
  const drawing = useRef(false)
  const lastPos = useRef(null)

  useImperativeHandle(ref, () => ({
    clear() {
      fill(canvasEl.current.getContext('2d'))
    },
  }))

  useEffect(() => {
    fill(canvasEl.current.getContext('2d'))
  }, [])

  function fill(ctx) {
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, SIZE, SIZE)
  }

  function getXY(e) {
    const rect = canvasEl.current.getBoundingClientRect()
    const src = e.touches?.[0] ?? e
    return [src.clientX - rect.left, src.clientY - rect.top]
  }

  function onDown(e) {
    e.preventDefault()
    drawing.current = true
    const [x, y] = getXY(e)
    lastPos.current = [x, y]
    // paint a dot so single taps register
    const ctx = canvasEl.current.getContext('2d')
    ctx.fillStyle = '#fff'
    ctx.beginPath()
    ctx.arc(x, y, BRUSH / 2, 0, Math.PI * 2)
    ctx.fill()
  }

  function onMove(e) {
    if (!drawing.current) return
    e.preventDefault()
    const [x, y] = getXY(e)
    const ctx = canvasEl.current.getContext('2d')
    ctx.lineWidth = BRUSH
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.strokeStyle = '#fff'
    ctx.beginPath()
    ctx.moveTo(...lastPos.current)
    ctx.lineTo(x, y)
    ctx.stroke()
    lastPos.current = [x, y]
  }

  function onUp() {
    if (!drawing.current) return
    drawing.current = false
    lastPos.current = null
    onStrokeEnd(canvasEl.current.toDataURL('image/png'))
  }

  return (
    <canvas
      ref={canvasEl}
      width={SIZE}
      height={SIZE}
      onMouseDown={onDown}
      onMouseMove={onMove}
      onMouseUp={onUp}
      onMouseLeave={onUp}
      onTouchStart={onDown}
      onTouchMove={onMove}
      onTouchEnd={onUp}
      style={{ cursor: 'crosshair', borderRadius: 8, display: 'block' }}
    />
  )
})

export default DrawCanvas
