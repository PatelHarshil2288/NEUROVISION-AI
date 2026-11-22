"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  getActivations,
  getModelWeights,
  type ActivationsResult,
  type ModelWeightsResult,
} from "../../digit-studio/lib/api";
import {
  readCurrentModelInput,
  type ModelInputPixels,
} from "../../../lib/current-model-input";
import {
  NETWORK_LAYERS,
  NETWORK_TOPOLOGY,
  type LayerId,
} from "../lib/network";
import {
  getConnectionContributions,
  getTopConnectionContributions,
  type TransitionIndex,
  type NeuronSelection,
} from "../lib/signal";
import NeuronInspector from "./NeuronInspector";

const NeuralScene = dynamic(() => import("./NeuralScene"), {
  ssr: false,
  loading: () => (
    <div className="observatory-scene-loading" role="status">
      Initializing the 3D network…
    </div>
  ),
});

interface LiveModelData {
  activations: ActivationsResult;
  weights: ModelWeightsResult;
}

function messageFromError(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "The model activity could not be loaded.";
}

function transitionForLayer(layerIndex: number): TransitionIndex {
  if (layerIndex <= 1) {
    return 0;
  }
  if (layerIndex === 2) {
    return 1;
  }
  return 2;
}

function layerActivitySummary(
  layerId: LayerId,
  activations: ActivationsResult,
  pixels: ModelInputPixels,
): string {
  if (layerId === "output") {
    return `Digit ${activations.predicted_class} · ${(
      activations.probabilities[activations.predicted_class]! * 100
    ).toFixed(1)}%`;
  }

  const values = activations.layers[layerId].activations;
  const activeCount =
    layerId === "input"
      ? pixels.flat().filter((value) => value >= 0.05).length
      : values.filter((value) => value > 0).length;
  const peak = Math.max(0, ...values);
  return layerId === "input"
    ? `${activeCount} lit pixels · normalized peak ${peak.toFixed(3)}`
    : `${activeCount} active · peak ${peak.toFixed(3)}`;
}

function InputThumbnail({ pixels }: { pixels: ModelInputPixels }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) {
      return;
    }

    const imageData = context.createImageData(28, 28);
    for (let index = 0; index < 28 * 28; index += 1) {
      const row = Math.floor(index / 28);
      const column = index % 28;
      const offset = index * 4;
      const value = Math.round(pixels[row]![column]! * 255);
      imageData.data[offset] = value;
      imageData.data[offset + 1] = value;
      imageData.data[offset + 2] = value;
      imageData.data[offset + 3] = 255;
    }
    context.putImageData(imageData, 0, 0);
  }, [pixels]);

  return (
    <canvas
      ref={canvasRef}
      className="observatory-input-thumbnail"
      width={28}
      height={28}
      aria-label="Latest processed 28 by 28 digit input"
    />
  );
}

export default function NeuralObservatory() {
  const [focusedLayerId, setFocusedLayerId] = useState<LayerId | null>(null);
  const [selectedNeuron, setSelectedNeuron] = useState<NeuronSelection>({ layerId: "input", index: 0 });
  const [resetSignal, setResetSignal] = useState(0);
  const [currentPixels, setCurrentPixels] = useState<ModelInputPixels | null>(
    null,
  );
  const [modelData, setModelData] = useState<LiveModelData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [retrySignal, setRetrySignal] = useState(0);
  const [activeLayerIndex, setActiveLayerIndex] = useState(
    NETWORK_LAYERS.length - 1,
  );
  const [playbackRequested, setPlaybackRequested] = useState(false);
  const [selectedTransitionOverride, setSelectedTransitionOverride] =
    useState<TransitionIndex | null>(null);
  const isPlaying =
    playbackRequested && activeLayerIndex < NETWORK_LAYERS.length - 1;
  const selectedTransition =
    selectedTransitionOverride ?? transitionForLayer(activeLayerIndex);

  useEffect(() => {
    const controller = new AbortController();

    async function loadModelData(): Promise<void> {
      setIsLoading(true);
      setLoadError(null);
      setModelData(null);
      setPlaybackRequested(false);

      try {
        const pixels = readCurrentModelInput();
        setCurrentPixels(pixels);
        if (!pixels) {
          return;
        }

        const [activations, weights] = await Promise.all([
          getActivations(pixels, controller.signal),
          getModelWeights(controller.signal),
        ]);
        if (activations.model_version !== weights.model_version) {
          throw new Error(
            "The activation data and trained weights belong to different model versions.",
          );
        }

        setModelData({ activations, weights });
        setActiveLayerIndex(NETWORK_LAYERS.length - 1);
        setSelectedTransitionOverride(null);
      } catch (error) {
        if (!controller.signal.aborted) {
          setLoadError(messageFromError(error));
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      }
    }

    void loadModelData();
    return () => controller.abort();
  }, [retrySignal]);

  useEffect(() => {
    if (!isPlaying) {
      return;
    }

    const timeout = window.setTimeout(() => {
      setActiveLayerIndex((current) =>
        Math.min(current + 1, NETWORK_LAYERS.length - 1),
      );
    }, 720);
    return () => window.clearTimeout(timeout);
  }, [activeLayerIndex, isPlaying]);

  const focusedLayer = NETWORK_LAYERS.find(
    (layer) => layer.id === focusedLayerId,
  );
  const transitionContributions = useMemo(
    () =>
      modelData
        ? getConnectionContributions(
            modelData.activations,
            modelData.weights,
            selectedTransition,
          )
        : [],
    [modelData, selectedTransition],
  );
  const topContributions = useMemo(
    () => getTopConnectionContributions(transitionContributions),
    [transitionContributions],
  );
  const sourceLayer = NETWORK_LAYERS[selectedTransition];
  const targetLayer = NETWORK_LAYERS[selectedTransition + 1];

  const resetCamera = () => {
    setFocusedLayerId(null);
    setResetSignal((current) => current + 1);
  };

  const toggleLayerFocus = (layerId: LayerId) => {
    setFocusedLayerId((current) => (current === layerId ? null : layerId));
  };

  const playSignal = () => {
    if (activeLayerIndex >= NETWORK_LAYERS.length - 1) {
      setActiveLayerIndex(0);
    }
    setSelectedTransitionOverride(null);
    setPlaybackRequested(true);
  };

  const stepSignal = (direction: -1 | 1) => {
    setPlaybackRequested(false);
    setActiveLayerIndex((current) =>
      Math.max(
        0,
        Math.min(current + direction, NETWORK_LAYERS.length - 1),
      ),
    );
  };

  return (
    <div className="studio-shell observatory-shell">
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
          <Link className="phase-link" href="/">
            Digit Studio
          </Link>
          <Link className="phase-link" href="/training">Training Lab</Link>
          <Link className="phase-link" href="/explorer">Explorer</Link>
          <Link className="phase-link" href="/analytics">Analytics</Link>
          <span className="phase-label">PHASE 05 / NEURON INSPECTOR</span>
        </div>
      </header>

      <main className="studio-main observatory-main">
        <section className="page-intro observatory-intro">
          <div>
            <p className="eyebrow">
              <span className="eyebrow-line" />
              LIVE MODEL ACTIVITY
            </p>
            <h1>
              Neural
              <br />
              <span>Observatory.</span>
            </h1>
            <p className="intro-copy">
              Inspect real activations, trained weights, and signal flow for the
              latest digit drawn or uploaded in Digit Studio.
            </p>
          </div>
          <div className="model-chip observatory-model-chip">
            <span className="model-chip-label">MNIST MULTILAYER PERCEPTRON</span>
            <span className="model-chip-name">986 NEURONS · 109,386 PARAMETERS</span>
            <span className="model-chip-foot">
              Three fully connected transitions
            </span>
          </div>
        </section>

        <section
          className="observatory-grid"
          aria-label="Three-dimensional neural network observatory"
        >
          <article className="panel observatory-visual-panel">
            <div className="panel-heading">
              <div>
                <p className="panel-index">01 / NETWORK SIGNAL</p>
                <h2>784 → 128 → 64 → 10</h2>
              </div>
              <span className="panel-meta">
                {modelData ? "LIVE ACTIVATIONS" : "MODEL TOPOLOGY"}
              </span>
            </div>

            <div className="observatory-canvas-frame">
              <ol
                className="observatory-layer-strip"
                aria-label="Network layers from input to output"
              >
                {NETWORK_LAYERS.map((layer, index) => (
                  <li className="observatory-layer-strip-item" key={layer.id}>
                    <span
                      className="observatory-layer-swatch"
                      style={{ backgroundColor: layer.color }}
                      aria-hidden="true"
                    />
                    <span>{layer.shortLabel}</span>
                    {index < NETWORK_LAYERS.length - 1 && (
                      <span
                        className="observatory-layer-strip-arrow"
                        aria-hidden="true"
                      >
                        →
                      </span>
                    )}
                  </li>
                ))}
              </ol>
              <NeuralScene
                focusedLayerId={focusedLayerId}
                resetSignal={resetSignal}
                activations={modelData?.activations ?? null}
                weights={modelData?.weights ?? null}
                selectedNeuron={selectedNeuron}
                onSelectNeuron={setSelectedNeuron}
                activeLayerIndex={
                  modelData ? activeLayerIndex : NETWORK_LAYERS.length - 1
                }
              />
              <p className="observatory-canvas-help">
                Drag to rotate · Scroll or pinch to zoom · Click a neuron to inspect
              </p>
            </div>

            <div className="observatory-playback-controls" role="group" aria-label="Signal playback">
              <button
                className="observatory-playback-button"
                disabled={!modelData}
                onClick={playSignal}
                type="button"
              >
                {activeLayerIndex === NETWORK_LAYERS.length - 1
                  ? "Replay signal"
                  : "Play signal"}
              </button>
              <button
                className="observatory-playback-button"
                disabled={!isPlaying}
                onClick={() => setPlaybackRequested(false)}
                type="button"
              >
                Pause
              </button>
              <button
                aria-label="Previous signal layer"
                className="observatory-step-button"
                disabled={!modelData || activeLayerIndex === 0 || isPlaying}
                onClick={() => stepSignal(-1)}
                type="button"
              >
                ←
              </button>
              <button
                aria-label="Next signal layer"
                className="observatory-step-button"
                disabled={
                  !modelData ||
                  activeLayerIndex === NETWORK_LAYERS.length - 1 ||
                  isPlaying
                }
                onClick={() => stepSignal(1)}
                type="button"
              >
                →
              </button>
              <span className="observatory-playback-status" aria-live="polite">
                {modelData
                  ? `Stage ${activeLayerIndex + 1} of ${NETWORK_LAYERS.length} · ${NETWORK_LAYERS[activeLayerIndex]?.shortLabel}`
                  : isLoading
                    ? "Loading real model signal…"
                    : "Waiting for a Digit Studio input"}
              </span>
            </div>

            {modelData && (
              <div className="observatory-signal-legend">
                <span>
                  <i className="is-positive" aria-hidden="true" />
                  Positive contribution
                </span>
                <span>
                  <i className="is-negative" aria-hidden="true" />
                  Negative contribution
                </span>
                <span>Line brightness scales to contribution within each transition</span>
              </div>
            )}

            <div className="observatory-controls">
              <button
                className="observatory-reset-button"
                onClick={resetCamera}
                type="button"
              >
                Reset camera
              </button>
              <span className="observatory-edge-count">
                {NETWORK_TOPOLOGY.connections.length.toLocaleString("en-US")} of{" "}
                {NETWORK_TOPOLOGY.totalConnectionCount.toLocaleString("en-US")}{" "}
                connections shown
              </span>
            </div>
            <p className="observatory-focus-status" aria-live="polite">
              {focusedLayer
                ? `Camera focused on ${focusedLayer.label}.`
                : "Showing the complete network."}
            </p>
          </article>

          <aside className="panel observatory-details-panel">
            <div className="panel-heading">
              <div>
                <p className="panel-index">02 / ACTIVATIONS</p>
                <h2>Network activity</h2>
              </div>
              <span className="panel-meta">986 TOTAL</span>
            </div>

            <div className="observatory-input-state">
              {isLoading ? (
                <p role="status">Loading the input and trained model data…</p>
              ) : loadError ? (
                <div className="observatory-load-error">
                  <p role="alert">{loadError}</p>
                  <button
                    className="observatory-retry-button"
                    onClick={() => setRetrySignal((current) => current + 1)}
                    type="button"
                  >
                    Retry model data
                  </button>
                </div>
              ) : !currentPixels ? (
                <div>
                  <p>
                    No digit is saved in this tab yet. Draw or upload one in
                    Digit Studio to see its real model activity.
                  </p>
                  <Link className="observatory-open-studio" href="/">
                    Open Digit Studio
                  </Link>
                </div>
              ) : modelData ? (
                <figure className="observatory-input-summary">
                  <InputThumbnail pixels={currentPixels} />
                  <figcaption>
                    <strong>
                      Predicted digit {modelData.activations.predicted_class}
                    </strong>
                    <span>
                      {(
                        modelData.activations.probabilities[
                          modelData.activations.predicted_class
                        ]! * 100
                      ).toFixed(1)}
                      % confidence
                    </span>
                    <small>Model {modelData.activations.model_version}</small>
                  </figcaption>
                </figure>
              ) : null}
            </div>

            <ol className="observatory-layer-list">
              {NETWORK_LAYERS.map((layer, index) => (
                <li className="observatory-layer-item" key={layer.id}>
                  <span
                    className="observatory-layer-swatch"
                    style={{ backgroundColor: layer.color }}
                    aria-hidden="true"
                  />
                  <span className="observatory-layer-copy">
                    <strong>{layer.label}</strong>
                    <small>
                      {layer.neuronCount.toLocaleString("en-US")} neurons
                    </small>
                    {modelData && currentPixels && (
                      <small className="observatory-layer-activity">
                        {layerActivitySummary(
                          layer.id,
                          modelData.activations,
                          currentPixels,
                        )}
                      </small>
                    )}
                  </span>
                  <button
                    aria-label={`${focusedLayerId === layer.id ? "Unfocus" : "Focus"} camera on ${layer.label}`}
                    aria-pressed={focusedLayerId === layer.id}
                    className="observatory-focus-button"
                    onClick={() => toggleLayerFocus(layer.id)}
                    type="button"
                  >
                    {focusedLayerId === layer.id ? "Focused" : "Focus"}
                  </button>
                  {index < NETWORK_LAYERS.length - 1 && (
                    <span className="observatory-layer-arrow" aria-hidden="true">
                      ↓
                    </span>
                  )}
                </li>
              ))}
            </ol>

            <section
              className="observatory-contribution-panel"
              aria-labelledby="contribution-title"
            >
              <div className="observatory-contribution-heading">
                <div>
                  <p className="observatory-note-label">03 / CONTRIBUTIONS</p>
                  <h3 id="contribution-title">Strongest sampled links</h3>
                </div>
                <label>
                  <span className="sr-only">Layer transition</span>
                  <select
                    aria-label="Layer transition"
                    disabled={!modelData}
                    onChange={(event) => {
                      const value = event.currentTarget.value;
                      if (value === "0") {
                        setSelectedTransitionOverride(0);
                      } else if (value === "1") {
                        setSelectedTransitionOverride(1);
                      } else if (value === "2") {
                        setSelectedTransitionOverride(2);
                      }
                    }}
                    value={selectedTransition}
                  >
                    {NETWORK_LAYERS.slice(0, -1).map((layer, index) => (
                      <option key={layer.id} value={index}>
                        {layer.shortLabel} → {NETWORK_LAYERS[index + 1]?.shortLabel}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {modelData ? (
                <>
                  <p className="observatory-contribution-explainer">
                    Contribution = source activation × weight; target bias is
                    shown separately. Ranked from the rendered connection sample.
                  </p>
                  <ol className="observatory-contribution-list">
                    {topContributions.map((item, index) => (
                      <li
                        className="observatory-contribution-item"
                        key={`${item.sourceIndex}-${item.targetIndex}`}
                      >
                        <span className="observatory-contribution-rank">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                        <span className="observatory-contribution-detail">
                          <strong>
                            {sourceLayer.shortLabel} #{item.sourceIndex + 1} →
                            {" "}
                            {targetLayer.shortLabel} #{item.targetIndex + 1}
                          </strong>
                          <small>
                            {item.sourceActivation.toFixed(3)} ×{" "}
                            {item.weight.toFixed(3)} · bias{" "}
                            {item.targetBias.toFixed(3)}
                          </small>
                        </span>
                        <strong
                          className={`observatory-contribution-value${
                            item.contribution < 0 ? " is-negative" : " is-positive"
                          }`}
                        >
                          {item.contribution.toFixed(3)}
                        </strong>
                      </li>
                    ))}
                  </ol>
                </>
              ) : (
                <p className="observatory-contribution-empty">
                  Contribution values will appear after a real model input is
                  loaded.
                </p>
              )}
            </section>

            <NeuronInspector
              selection={selectedNeuron}
              onSelect={setSelectedNeuron}
              activations={modelData?.activations ?? null}
              weights={modelData?.weights ?? null}
              pixels={currentPixels}
            />
            <div className="observatory-note">
              <p className="observatory-note-label">HOW TO READ THE VIEW</p>
              <p>
                Input/output brightness shows raw intensity or probability;
                hidden layers scale to their peak activation. Cyan links add to
                a target&apos;s weighted sum and rose links subtract. Playback
                stages the completed inference one layer at a time.
              </p>
            </div>
          </aside>
        </section>

        <footer className="studio-footer">
          <span>NEUROVISION AI</span>
          <span>WEBGL · 986 NEURONS · LIVE MODEL SIGNAL</span>
        </footer>
      </main>
    </div>
  );
}
