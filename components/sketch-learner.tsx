"use client"

import { useEffect, useRef, useState } from "react"
import * as tf from "@tensorflow/tfjs"

export function formatEpochStatus(logs, epoch, totalEpochs = 10) {
  const lossVal = logs && typeof logs.loss === "number" ? logs.loss : null
  const accVal =
    (logs && (typeof logs.acc === "number" ? logs.acc : undefined)) ??
    (logs && (typeof logs.accuracy === "number" ? logs.accuracy : undefined)) ??
    null
  const lossStr = lossVal !== null ? lossVal.toFixed(4) : "N/A"
  const accStr = accVal !== null ? accVal.toFixed(3) : "N/A"
  return `Epoch ${epoch + 1}/${totalEpochs} - loss: ${lossStr} acc: ${accStr}`
}

export default function SketchLearner() {
  const canvasRef = useRef(null)
  const ctxRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [brushSize, setBrushSize] = useState(18)
  const [brushColor, setBrushColor] = useState("black")
  const [labels, setLabels] = useState(["circle", "square"])
  const [currentLabel, setCurrentLabel] = useState("circle")
  const [examples, setExamples] = useState({}) // {label: count}
  const [model, setModel] = useState(null)
  const [status, setStatus] = useState("Ready")
  const [prediction, setPrediction] = useState(null)
  const [imageSize] = useState(28)

  const datasetRef = useRef({ xs: [], ys: [] })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    ctxRef.current = ctx

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    canvas.style.width = rect.width + "px"
    canvas.style.height = rect.height + "px"

    ctx.lineCap = "round"
    ctx.lineJoin = "round"
    ctx.fillStyle = "white"
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    loadModel()
  }, [])

  function startDrawing(e) {
    setIsDrawing(true)
    const ctx = ctxRef.current
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    ctx.globalCompositeOperation = brushColor === "white" ? "destination-out" : "source-over"
    ctx.strokeStyle = brushColor
    ctx.lineWidth = brushSize
    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.lineTo(x, y)
    ctx.stroke()
  }

  function draw(e) {
    if (!isDrawing) return
    const ctx = ctxRef.current
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    ctx.lineTo(x, y)
    ctx.stroke()
  }

  function stopDrawing() {
    setIsDrawing(false)
  }

  function clearCanvas() {
    const ctx = ctxRef.current
    const canvas = canvasRef.current
    ctx.fillStyle = "white"
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    setPrediction(null)
  }

  function canvasToTensor() {
    const canvas = canvasRef.current
    if (!canvas) throw new Error("Canvas not ready")

    return tf.tidy(() => {
      const img = tf.browser.fromPixels(canvas, 1).toFloat().div(255.0) // [H, W, 1]
      const resized = tf.image.resizeBilinear(img, [imageSize, imageSize]) // [H, W, 1]
      const batched = resized.expandDims(0) // [1, H, W, 1]
      return batched
    })
  }

  function addExample() {
    try {
      const x = canvasToTensor() // [1, H, W, 1]
      const labelIdx = labels.indexOf(currentLabel)
      if (labelIdx === -1) throw new Error(`Label "${currentLabel}" not found`)

      const y = tf.tidy(() => tf.oneHot(labelIdx, labels.length).expandDims(0)) // [1, C]

      datasetRef.current.xs.push(x)
      datasetRef.current.ys.push(y)

      setExamples((prev) => ({ ...prev, [currentLabel]: (prev[currentLabel] || 0) + 1 }))
      setStatus(`Added example for "${currentLabel}"`)
      clearCanvas()
    } catch (err) {
      console.error(err)
      setStatus(`Failed to add example: ${err.message}`)
    }
  }

  function buildModel(numClasses) {
    const m = tf.sequential()
    m.add(tf.layers.conv2d({ inputShape: [imageSize, imageSize, 1], filters: 16, kernelSize: 3, activation: "relu" }))
    m.add(tf.layers.maxPool2d({ poolSize: 2 }))
    m.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu" }))
    m.add(tf.layers.maxPool2d({ poolSize: 2 }))
    m.add(tf.layers.flatten())
    m.add(tf.layers.dense({ units: 64, activation: "relu" }))
    m.add(tf.layers.dropout({ rate: 0.25 }))
    m.add(tf.layers.dense({ units: numClasses, activation: "softmax" }))

    m.compile({ optimizer: tf.train.adam(0.001), loss: "categoricalCrossentropy", metrics: ["accuracy"] })
    return m
  }

  async function train() {
    const { xs, ys } = datasetRef.current
    if (xs.length < labels.length) {
      setStatus(`Need at least 1 example per class (${labels.length} total)`)
      return
    }

    setStatus("Preparing tensors...")
    let X = null
    let Y = null

    try {
      X = tf.tidy(() => tf.concat(xs, 0)) // [N, H, W, 1]
      Y = tf.tidy(() => tf.concat(ys, 0)) // [N, C]

      const m = buildModel(labels.length)
      setModel(m)
      setStatus("Training model...")

      const hist = await m.fit(X, Y, {
        epochs: 10,
        batchSize: 16,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            const acc = logs.acc || logs.accuracy || 0
            setStatus(`Epoch ${epoch + 1}/10 - accuracy: ${(acc * 100).toFixed(1)}%`)
          },
        },
      })

      setStatus("Training complete!")
    } catch (err) {
      console.error(err)
      setStatus(`Training failed: ${err.message}`)
    } finally {
      if (X) X.dispose()
      if (Y) Y.dispose()
    }
  }

  async function predict() {
    if (!model) {
      setStatus("Train (or load) a model first")
      return
    }

    try {
      const x = canvasToTensor()
      const probs = model.predict(x)
      const outTensor = Array.isArray(probs) ? probs[0] : probs
      const data = await outTensor.data()
      const arr = Array.from(data)
      const maxIdx = arr.indexOf(Math.max(...arr))

      setPrediction({ label: labels[maxIdx], probs: arr })

      x.dispose()
      tf.dispose(probs)
    } catch (err) {
      console.error(err)
      setStatus(`Prediction failed: ${err.message}`)
    }
  }

  async function saveModel() {
    if (!model) return setStatus("No trained model to save")

    try {
      await model.save("indexeddb://sketch-learner-model")
      localStorage.setItem("sketch-learner-labels", JSON.stringify(labels))
      setStatus("Model saved to IndexedDB")
    } catch (err) {
      console.error(err)
      setStatus(`Save failed: ${err.message}`)
    }
  }

  async function loadModel() {
    try {
      const m = await tf.loadLayersModel("indexeddb://sketch-learner-model")
      setModel(m)

      const savedLabels = JSON.parse(localStorage.getItem("sketch-learner-labels") || "null")
      if (Array.isArray(savedLabels) && savedLabels.length > 0) {
        setLabels(savedLabels)
        setCurrentLabel(savedLabels[0])
      }

      setStatus("Model loaded from IndexedDB")
    } catch (e) {
      console.warn(e)
      setStatus("No saved model found")
    }
  }

  function addLabel() {
    const newLabel = prompt("Enter new label:")
    if (newLabel && !labels.includes(newLabel)) {
      setLabels((prev) => [...prev, newLabel])
      setCurrentLabel(newLabel)
    }
  }

  function removeLabel() {
    if (labels.length <= 1) return
    const updatedLabels = labels.filter((l) => l !== currentLabel)
    setLabels(updatedLabels)
    setCurrentLabel(updatedLabels[0])
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="bg-card rounded-lg p-6">
        <h2 className="text-2xl font-semibold mb-4">Drawing Canvas</h2>
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1">
            <canvas
              ref={canvasRef}
              width={400}
              height={400}
              className="border-2 border-border rounded-lg cursor-crosshair bg-white"
              style={{ width: "400px", height: "400px" }}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
            />
          </div>

          <div className="space-y-4">
            <label className="text-sm font-medium">Brush Size: {brushSize}px</label>
            <input
              type="range"
              min="1"
              max="50"
              value={brushSize}
              onChange={(e) => setBrushSize(Number(e.target.value))}
              className="w-full"
            />

            <div className="flex gap-2">
              <button
                onClick={() => setBrushColor("black")}
                className={`px-4 py-2 rounded ${brushColor === "black" ? "bg-black text-white" : "bg-gray-200"}`}
              >
                Draw
              </button>
              <button
                onClick={() => setBrushColor("white")}
                className={`px-4 py-2 rounded border ${brushColor === "white" ? "bg-gray-100" : "bg-white"}`}
              >
                Erase
              </button>
              <button onClick={clearCanvas} className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600">
                Clear
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-card rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">Labels & Training</h3>

          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">Current Label:</label>
              <select
                value={currentLabel}
                onChange={(e) => setCurrentLabel(e.target.value)}
                className="w-full mt-1 p-2 border rounded"
              >
                {labels.map((label) => (
                  <option key={label} value={label}>
                    {label}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex gap-2">
              <button onClick={addLabel} className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600">
                Add Label
              </button>
              <button
                onClick={removeLabel}
                className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
              >
                Remove Label
              </button>
            </div>

            <div>
              <h4 className="font-medium mb-2">Training Examples:</h4>
              {labels.map((label) => (
                <div key={label} className="flex justify-between items-center py-1">
                  <span>
                    {label}: {examples[label] || 0} examples
                  </span>
                </div>
              ))}
            </div>

            <div className="flex gap-2">
              <button onClick={addExample} className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600">
                Add Example
              </button>
            </div>
          </div>
        </div>

        <div className="bg-card rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">Model Training & Prediction</h3>

          <div className="space-y-4">
            <div className="flex gap-2">
              <button onClick={train} className="px-4 py-2 bg-orange-500 text-white rounded hover:bg-orange-600">
                Train Model
              </button>
              <button onClick={predict} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
                Predict
              </button>
            </div>

            <div className="flex gap-2">
              <button onClick={saveModel} className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600">
                Save Model
              </button>
              <button onClick={loadModel} className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600">
                Load Model
              </button>
            </div>

            {prediction && (
              <div className="space-y-2">
                <h4 className="font-medium">Prediction:</h4>
                <div className="text-lg font-semibold text-blue-600">{prediction.label}</div>
                <div className="text-sm space-y-1">
                  {labels.map((label, idx) => (
                    <div key={label} className="flex justify-between">
                      <span>{label}:</span>
                      <span>{(prediction.probs[idx] * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="bg-muted rounded-lg p-4">
        <div className="text-sm font-mono">{status}</div>
      </div>
    </div>
  )
}
