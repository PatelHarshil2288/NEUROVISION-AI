"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import Link from "next/link";

import DrawingCanvas, {
  type DrawingCanvasHandle,
  type DrawingTool,
  type HistoryAvailability,
} from "./DrawingCanvas";
import {
  getBackendStatus,
  predictDigit,
  type PredictionResult,
} from "../lib/api";
import {
  clearCurrentModelInput,
  saveCurrentModelInput,
} from "../../../lib/current-model-input";

type BackendState = "checking" | "ready" | "missing" | "offline";

interface PredictionHistoryItem {
  id: string;
  predictedClass: number;
  confidence: number;
  inferenceMs: number;
  createdAt: string;
}

function messageFromError(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Something went wrong while processing this image.";
}

function formatTime(timestamp: string): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(timestamp));
}

function ProcessedPreview({ pixels }: { pixels: number[][] | null }) {
  const previewRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!pixels) {
      return;
    }
    const canvas = previewRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) {
      return;
    }
    const imageData = context.createImageData(28, 28);
    for (let y = 0; y < 28; y += 1) {
      for (let x = 0; x < 28; x += 1) {
        const offset = (y * 28 + x) * 4;
        const intensity = Math.round(pixels[y][x] * 255);
        imageData.data[offset] = intensity;
        imageData.data[offset + 1] = intensity;
        imageData.data[offset + 2] = intensity;
        imageData.data[offset + 3] = 255;
      }
    }
    context.putImageData(imageData, 0, 0);
  }, [pixels]);

  if (!pixels) {
    return (
      <div className="processed-preview-empty" aria-label="No processed image">
        <span>28</span>
        <span>× 28</span>
        <small>awaiting input</small>
      </div>
    );
  }

  return (
    <canvas
      ref={previewRef}
      className="processed-preview"
      width={28}
      height={28}
      aria-label="Processed 28 by 28 grayscale image sent to the MNIST model"
    />
  );
}

function ProbabilityBars({ result }: { result: PredictionResult | null }) {
  const ranked = useMemo(() => {
    if (!result) {
      return [];
    }
    return result.probabilities
      .map((probability, digit) => ({ digit, probability }))
      .sort((left, right) => right.probability - left.probability)
      .slice(0, 5);
  }, [result]);

  if (!result) {
    return (
      <div className="empty-chart">
        <span className="empty-chart-mark" aria-hidden="true">
          0.00
        </span>
        <p>Run a prediction to reveal the model&apos;s class probabilities.</p>
      </div>
    );
  }

  return (
    <ol className="probability-list" aria-label="Top five class probabilities">
      {ranked.map(({ digit, probability }, index) => (
        <li className="probability-row" key={digit}>
          <span className="probability-rank">
            {String(index + 1).padStart(2, "0")}
          </span>
          <span className="probability-digit">{digit}</span>
          <span
            className="probability-meter"
            role="progressbar"
            aria-label={`Probability for digit ${digit}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Number((probability * 100).toFixed(1))}
          >
            <span
              className={`probability-fill${index === 0 ? " is-leading" : ""}`}
              style={{ width: `${probability * 100}%` }}
            />
          </span>
          <span className="probability-value">
            {(probability * 100).toFixed(1)}%
          </span>
        </li>
      ))}
    </ol>
  );
}

export default function DigitStudio() {
  const drawingRef = useRef<DrawingCanvasHandle>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inputRevision = useRef(0);
  const [pixels, setPixels] = useState<number[][] | null>(null);
  const [tool, setTool] = useState<DrawingTool>("draw");
  const [brushSize, setBrushSize] = useState(14);
  const [historyAvailability, setHistoryAvailability] =
    useState<HistoryAvailability>({ canUndo: false, canRedo: false });
  const [backendState, setBackendState] = useState<BackendState>("checking");
  const [modelVersion, setModelVersion] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<
    PredictionHistoryItem[]
  >([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const checkBackend = useCallback(async (signal?: AbortSignal) => {
    try {
      const status = await getBackendStatus(signal);
      setBackendState(status.state);
      setModelVersion(status.modelVersion);
    } catch (error) {
      if (signal?.aborted) {
        return;
      }
      setBackendState("offline");
      setModelVersion(null);
      setErrorMessage(messageFromError(error));
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void getBackendStatus(controller.signal)
      .then((status) => {
        setBackendState(status.state);
        setModelVersion(status.modelVersion);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setBackendState("offline");
        setModelVersion(null);
        setErrorMessage(messageFromError(error));
      });
    return () => controller.abort();
  }, []);

  const handlePixelsChange = useCallback((nextPixels: number[][] | null) => {
    inputRevision.current += 1;
    setPixels(nextPixels);
    setPrediction(null);
    try {
      if (nextPixels) {
        saveCurrentModelInput(nextPixels);
      } else {
        clearCurrentModelInput();
      }
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(messageFromError(error));
    }
  }, []);

  const handleHistoryChange = useCallback(
    (availability: HistoryAvailability) => {
      setHistoryAvailability(availability);
    },
    [],
  );

  const handleFileChange = async (
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) {
      return;
    }
    setErrorMessage(null);
    setIsUploading(true);
    try {
      await drawingRef.current?.loadUpload(file);
    } catch (error) {
      setErrorMessage(messageFromError(error));
    } finally {
      setIsUploading(false);
      input.value = "";
    }
  };

  const handlePredict = async (): Promise<void> => {
    if (!pixels) {
      setErrorMessage("Draw or upload a digit before requesting a prediction.");
      return;
    }
    if (backendState !== "ready") {
      setErrorMessage(
        backendState === "missing"
          ? "The backend is running, but no trained MNIST checkpoint is loaded."
          : "The AI backend is not available. Check the service connection and retry.",
      );
      return;
    }

    setIsPredicting(true);
    setErrorMessage(null);
    const revision = inputRevision.current;
    try {
      const result = await predictDigit(pixels);
      if (revision !== inputRevision.current) {
        return;
      }
      setPrediction(result);
      setPredictionHistory((current) =>
        [
          {
            id: `${Date.now()}-${current.length}`,
            predictedClass: result.predicted_class,
            confidence: result.probabilities[result.predicted_class],
            inferenceMs: result.inference_ms,
            createdAt: new Date().toISOString(),
          },
          ...current,
        ].slice(0, 6),
      );
    } catch (error) {
      if (revision === inputRevision.current) {
        setErrorMessage(messageFromError(error));
      }
    } finally {
      setIsPredicting(false);
    }
  };

  const backendLabel = {
    checking: "Checking AI service",
    ready: "Model ready",
    missing: "Checkpoint not loaded",
    offline: "Service unavailable",
  }[backendState];

  return (
    <div className="studio-shell">
      <header className="topbar">
        <Link className="brand-lockup" href="/" aria-label="NEUROVISION AI home">
          <span className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span className="brand-wordmark">
            <strong>NEUROVISION</strong>
            <span>INTELLIGENCE LAB</span>
          </span>
        </Link>
        <div className="topbar-right">
          <Link className="phase-link" href="/observatory">
            3D Observatory
          </Link>
          <Link className="phase-link" href="/training">Training Lab</Link>
          <Link className="phase-link" href="/explorer">Explorer</Link>
          <Link className="phase-link" href="/analytics">Analytics</Link>
          <span className="phase-label">PHASE 02 / DIGIT STUDIO</span>
          <span
            className={`service-status status-${backendState}`}
            role="status"
          >
            <span className="status-light" aria-hidden="true" />
            {backendLabel}
          </span>
        </div>
      </header>

      <main className="studio-main">
        <section className="page-intro" aria-labelledby="page-title">
          <div>
            <p className="eyebrow">
              <span className="eyebrow-line" />
              INTERACTIVE INFERENCE
            </p>
            <h1 id="page-title">
              Give the network
              <br />
              <span>something to see.</span>
            </h1>
            <p className="intro-copy">
              Draw a digit or bring your own image. Watch the trained MNIST
              network turn pixels into probabilities.
            </p>
          </div>
          <div className="model-chip">
            <span className="model-chip-label">ACTIVE MODEL</span>
            <span className="model-chip-name">784 → 128 → 64 → 10</span>
            <span className="model-chip-foot">
              {modelVersion ?? "MNIST multilayer perceptron"}
            </span>
          </div>
        </section>

        <section className="studio-grid" aria-label="Digit inference workspace">
          <article className="panel input-panel">
            <div className="panel-heading">
              <div>
                <p className="panel-index">01 / INPUT</p>
                <h2>Digit canvas</h2>
              </div>
              <span className="panel-meta">28 × 28 MODEL INPUT</span>
            </div>

            <div className="tool-row" aria-label="Drawing tools">
              <div className="tool-toggle" role="group" aria-label="Canvas tool">
                <button
                  type="button"
                  className={`tool-button${tool === "draw" ? " is-active" : ""}`}
                  aria-pressed={tool === "draw"}
                  onClick={() => setTool("draw")}
                >
                  Draw
                </button>
                <button
                  type="button"
                  className={`tool-button${tool === "erase" ? " is-active" : ""}`}
                  aria-pressed={tool === "erase"}
                  onClick={() => setTool("erase")}
                >
                  Erase
                </button>
              </div>
              <div className="history-tools">
                <button
                  type="button"
                  className="quiet-button"
                  onClick={() => drawingRef.current?.undo()}
                  disabled={!historyAvailability.canUndo}
                >
                  Undo
                </button>
                <button
                  type="button"
                  className="quiet-button"
                  onClick={() => drawingRef.current?.redo()}
                  disabled={!historyAvailability.canRedo}
                >
                  Redo
                </button>
                <button
                  type="button"
                  className="quiet-button clear-button"
                  onClick={() => drawingRef.current?.clear()}
                >
                  Clear
                </button>
              </div>
            </div>

            <div className="canvas-layout">
              <div className="canvas-column">
                <div className="canvas-frame">
                  <DrawingCanvas
                    ref={drawingRef}
                    tool={tool}
                    brushSize={brushSize}
                    onPixelsChange={handlePixelsChange}
                    onHistoryChange={handleHistoryChange}
                  />
                </div>
                <div className="brush-control">
                  <label htmlFor="brush-size">
                    Brush size <span>{brushSize}px</span>
                  </label>
                  <input
                    id="brush-size"
                    type="range"
                    min={6}
                    max={30}
                    step={2}
                    value={brushSize}
                    onChange={(event) =>
                      setBrushSize(Number(event.currentTarget.value))
                    }
                  />
                </div>
              </div>

              <aside className="preprocess-card" aria-label="Processed model input">
                <div className="preprocess-heading">
                  <span className="mini-label">MODEL VIEW</span>
                  <span className="processing-dot" aria-hidden="true" />
                </div>
                <div className="preview-well">
                  <ProcessedPreview pixels={pixels} />
                </div>
                <p className="preview-caption">28 × 28 px</p>
                <div className="preprocess-divider" />
                <p className="preprocess-note">
                  Center-cropped
                  <br />
                  grayscale input
                </p>
              </aside>
            </div>

            <p className="canvas-hint">
              Use mouse or touch to draw. Keyboard: focus the canvas, move with
              arrow keys, press Enter to draw.
            </p>

            <div className="input-actions">
              <input
                ref={fileInputRef}
                className="sr-only"
                type="file"
                accept="image/*"
                aria-label="Upload a digit image"
                onChange={handleFileChange}
              />
              <button
                type="button"
                className="upload-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
              >
                <span aria-hidden="true">＋</span>
                {isUploading ? "Processing image…" : "Upload image"}
              </button>
              <span className="upload-note">PNG, JPEG or WebP · up to 10 MB</span>
              <button
                type="button"
                className="predict-button"
                onClick={() => void handlePredict()}
                disabled={
                  !pixels ||
                  backendState !== "ready" ||
                  isPredicting ||
                  isUploading
                }
              >
                {isPredicting ? (
                  <>
                    <span className="button-spinner" aria-hidden="true" />
                    Running inference
                  </>
                ) : (
                  <>
                    Run prediction
                    <span aria-hidden="true">→</span>
                  </>
                )}
              </button>
            </div>

            {backendState !== "ready" && (
              <div className="service-message">
                <span>
                  {backendState === "checking"
                    ? "Connecting to the local AI service…"
                    : backendState === "missing"
                      ? "Backend is reachable, but a trained checkpoint is not loaded."
                      : "Start the backend API to enable real predictions."}
                </span>
                <button
                  type="button"
                  className="text-button"
                  onClick={() => {
                    setErrorMessage(null);
                    setBackendState("checking");
                    void checkBackend();
                  }}
                  disabled={backendState === "checking"}
                >
                  Retry connection
                </button>
              </div>
            )}

            {errorMessage && (
              <div className="error-message" role="alert">
                <span className="error-mark" aria-hidden="true">
                  !
                </span>
                <span>{errorMessage}</span>
              </div>
            )}
          </article>

          <article className="panel results-panel">
            <div className="panel-heading">
              <div>
                <p className="panel-index">02 / INFERENCE</p>
                <h2>Prediction result</h2>
              </div>
              <span className="panel-meta">
                {prediction ? prediction.model_id : "AWAITING INPUT"}
              </span>
            </div>

            <div className={`prediction-hero${prediction ? " has-result" : ""}`}>
              <div className="prediction-digit-wrap" aria-live="polite">
                <span className="prediction-digit">
                  {prediction ? prediction.predicted_class : "—"}
                </span>
                <span className="prediction-digit-caption">
                  {prediction ? "PREDICTED DIGIT" : "NO PREDICTION YET"}
                </span>
              </div>
              <div className="prediction-stats">
                <div>
                  <span className="stat-label">CONFIDENCE</span>
                  <strong>
                    {prediction
                      ? `${(
                          prediction.probabilities[
                            prediction.predicted_class
                          ] * 100
                        ).toFixed(1)}%`
                      : "—"}
                  </strong>
                </div>
                <div>
                  <span className="stat-label">INFERENCE</span>
                  <strong>
                    {prediction
                      ? `${prediction.inference_ms.toFixed(2)} ms`
                      : "—"}
                  </strong>
                </div>
              </div>
            </div>

            <div className="probability-heading">
              <div>
                <h3>Probability distribution</h3>
                <p>Top five classes from the model softmax output</p>
              </div>
              <span className="distribution-mark">P(y | x)</span>
            </div>
            <ProbabilityBars result={prediction} />

            <div className="result-footnote">
              <span className="footnote-indicator" aria-hidden="true" />
              <p>
                Results are returned by the trained model. No sample or
                placeholder predictions are shown.
              </p>
            </div>
          </article>
        </section>

        <section className="history-panel panel" aria-labelledby="history-title">
          <div className="history-heading">
            <div>
              <p className="panel-index">03 / SESSION</p>
              <h2 id="history-title">Recent predictions</h2>
            </div>
            <span className="panel-meta">
              {predictionHistory.length}{" "}
              {predictionHistory.length === 1 ? "RUN" : "RUNS"}
            </span>
          </div>
          {predictionHistory.length === 0 ? (
            <div className="history-empty">
              <span className="history-empty-line" aria-hidden="true" />
              <p>Successful predictions from this session will appear here.</p>
            </div>
          ) : (
            <ol className="history-list">
              {predictionHistory.map((item, index) => (
                <li className="history-item" key={item.id}>
                  <span className="history-index">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <span className="history-digit">{item.predictedClass}</span>
                  <span className="history-confidence">
                    {(item.confidence * 100).toFixed(1)}% confidence
                  </span>
                  <span className="history-inference">
                    {item.inferenceMs.toFixed(2)} ms
                  </span>
                  <time className="history-time" dateTime={item.createdAt}>
                    {formatTime(item.createdAt)}
                  </time>
                </li>
              ))}
            </ol>
          )}
        </section>

        <footer className="studio-footer">
          <span>NEUROVISION AI</span>
          <span>MNIST · FULLY CONNECTED MLP · CPU INFERENCE</span>
        </footer>
      </main>
    </div>
  );
}
