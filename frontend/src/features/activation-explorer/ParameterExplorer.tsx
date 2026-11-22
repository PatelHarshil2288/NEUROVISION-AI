"use client";

import { useEffect, useMemo, useState } from "react";
import { getModelWeights, type ModelWeightsResult } from "../digit-studio/lib/api";
import { rankConnections, summarize, type HistogramBin, type RankedWeight } from "./lib/parameters";

const LAYERS = [
  { key: "input_to_hidden_1", label: "Input → hidden 1" },
  { key: "hidden_1_to_hidden_2", label: "Hidden 1 → hidden 2" },
  { key: "hidden_2_to_output", label: "Hidden 2 → output" },
] as const;
type LayerKey = typeof LAYERS[number]["key"];

function Histogram({ bins }: { bins: HistogramBin[] }) {
  const maxCount = Math.max(...bins.map((bin) => bin.count), 1);
  return (
    <>
      <svg className="explorer-histogram" viewBox="0 0 400 150" role="img" aria-label="Parameter histogram; bin boundaries and counts are listed below">
        <line x1="25" x2="395" y1="120" y2="120" className="explorer-axis" />
        {bins.map((bin, index) => {
          const height = 100 * bin.count / maxCount;
          return <rect key={index} x={27 + index * 18.3} y={120 - height} width={bins.length === 1 ? 365 : 15} height={height} fill="#66e4e6" />;
        })}
        <text x="25" y="143">{bins[0]!.lower.toFixed(3)}</text>
        <text x="395" y="143" textAnchor="end">{bins[bins.length - 1]!.upper.toFixed(3)}</text>
      </svg>
      <p>Horizontal axis: parameter value; vertical axis: count (maximum {maxCount.toLocaleString("en-US")}). Bins include their lower bound; the last bin also includes the maximum.</p>
      <details>
        <summary>Show histogram bin counts</summary>
        <div className="explorer-table-wrap"><table>
          <thead><tr><th>Lower bound</th><th>Upper bound</th><th>Count</th></tr></thead>
          <tbody>{bins.map((bin, index) => <tr key={index}><td>{bin.lower.toPrecision(6)}</td><td>{bin.upper.toPrecision(6)}</td><td>{bin.count}</td></tr>)}</tbody>
        </table></div>
      </details>
    </>
  );
}

function ConnectionList({ title, items }: { title: string; items: RankedWeight[] }) {
  return (
    <div>
      <h3>{title}</h3>
      <ol className="explorer-connection-list">
        {items.map((item) => (
          <li key={`${item.targetIndex}-${item.sourceIndex}`}>
            Source {item.sourceIndex} → target {item.targetIndex}{" "}
            <strong>{item.value !== 0 && Math.abs(item.value) < 0.0001 ? item.value.toExponential(4) : item.value.toFixed(6)}</strong>
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function ParameterExplorer() {
  const [weights, setWeights] = useState<ModelWeightsResult | null>(null);
  const [layerKey, setLayerKey] = useState<LayerKey>("input_to_hidden_1");
  const [kind, setKind] = useState<"weights" | "biases">("weights");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    getModelWeights(controller.signal).then((result) => {
      if (!controller.signal.aborted) {
        setWeights(result);
        setError(null);
        setLoading(false);
      }
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) {
        setWeights(null);
        setError(reason instanceof Error ? reason.message : "Model parameters could not be loaded.");
        setLoading(false);
      }
    });
    return () => controller.abort();
  }, [retry]);

  const layer = weights?.layers[layerKey];
  const summary = useMemo(() => layer ? summarize(kind === "weights" ? layer.weights.flat() : layer.biases) : null, [layer, kind]);
  const ranked = useMemo(() => layer && kind === "weights" ? rankConnections(layer) : null, [layer, kind]);

  return (
    <section className="panel explorer-panel explorer-parameters" aria-labelledby="real-parameters-title">
      <p className="panel-index">04 / ACTUAL LOADED CHECKPOINT</p>
      <h2 id="real-parameters-title">Trained model parameters</h2>
      <p>These values come from the loaded inference checkpoint, not the sandbox or Training Lab. They do not depend on the current digit.</p>
      {loading ? <p role="status">Loading trained model parameters…</p> : error ? (
        <div role="alert" className="training-error">
          <p>{error}</p>
          <button type="button" className="observatory-retry-button" onClick={() => { setError(null); setLoading(true); setRetry((value) => value + 1); }}>Retry parameters</button>
        </div>
      ) : weights && summary ? (
        <>
          <p>Model <strong>{weights.model_id}</strong> · version <strong>{weights.model_version}</strong></p>
          <div className="explorer-selectors">
            <label>Layer transition
              <select aria-label="Parameter layer transition" value={layerKey} onChange={(event) => {
                const selectedLayer = LAYERS.find((item) => item.key === event.currentTarget.value);
                if (selectedLayer) setLayerKey(selectedLayer.key);
              }}>
                {LAYERS.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
              </select>
            </label>
            <label>Parameter type
              <select aria-label="Parameter type" value={kind} onChange={(event) => {
                if (event.currentTarget.value === "weights" || event.currentTarget.value === "biases") setKind(event.currentTarget.value);
              }}>
                <option value="weights">Weights</option><option value="biases">Biases</option>
              </select>
            </label>
          </div>
          <dl className="explorer-stat-grid">
            <div><dt>Count</dt><dd>{summary.count.toLocaleString("en-US")}</dd></div>
            <div><dt>Positive / negative / zero</dt><dd>{summary.positive.toLocaleString("en-US")} / {summary.negative.toLocaleString("en-US")} / {summary.zero.toLocaleString("en-US")}</dd></div>
            <div><dt>Minimum / maximum</dt><dd>{summary.min.toFixed(5)} / {summary.max.toFixed(5)}</dd></div>
            <div><dt>Mean</dt><dd>{summary.mean.toFixed(5)}</dd></div>
          </dl>
          <h3>{kind === "weights" ? "Weight" : "Bias"} distribution</h3>
          <Histogram bins={summary.bins} />
          {ranked && (
            <>
              <p>Ranked by absolute weight, not by activation or influence. Indices are zero-based.</p>
              <div className="explorer-ranked">
                <ConnectionList title="Strongest connections" items={ranked.strongest} />
                <ConnectionList title="Weakest connections" items={ranked.weakest} />
              </div>
            </>
          )}
        </>
      ) : null}
    </section>
  );
}
