"use client";

import { useMemo } from "react";

import type { ActivationsResult, ModelWeightsResult } from "../../digit-studio/lib/api";
import type { ModelInputPixels } from "../../../lib/current-model-input";
import { NETWORK_LAYERS } from "../lib/network";
import { getNeuronDetails, type NeuronSelection } from "../lib/signal";

interface NeuronInspectorProps {
  selection: NeuronSelection;
  onSelect: (selection: NeuronSelection) => void;
  activations: ActivationsResult | null;
  weights: ModelWeightsResult | null;
  pixels: ModelInputPixels | null;
}

function format(value: number): string {
  return value.toFixed(5);
}

export default function NeuronInspector({
  selection,
  onSelect,
  activations,
  weights,
  pixels,
}: NeuronInspectorProps) {
  const layer = NETWORK_LAYERS.find((item) => item.id === selection.layerId)!;
  const details = useMemo(
    () => activations && weights ? getNeuronDetails(selection, activations, weights) : null,
    [selection, activations, weights],
  );
  const row = Math.floor(selection.index / 28);
  const column = selection.index % 28;
  const weightedSum = details?.incoming.reduce((sum, item) => sum + item.contribution, 0);

  return (
    <section className="neuron-inspector" aria-labelledby="inspector-title">
      <p className="observatory-note-label">04 / NEURON INSPECTOR</p>
      <h3 id="inspector-title">Inspect a neuron</h3>
      <p>Click a neuron in the 3D view, or select one below. Indices start at 0.</p>
      <div className="neuron-inspector-selectors">
        <label>
          Layer
          <select
            aria-label="Inspector layer"
            value={selection.layerId}
            onChange={(event) => {
              const nextLayer = NETWORK_LAYERS.find((item) => item.id === event.currentTarget.value);
              if (nextLayer) onSelect({ layerId: nextLayer.id, index: 0 });
            }}
          >
            {NETWORK_LAYERS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </select>
        </label>
        <label>
          Neuron index
          <select
            aria-label="Neuron index"
            value={selection.index}
            onChange={(event) => onSelect({ layerId: selection.layerId, index: Number(event.currentTarget.value) })}
          >
            {Array.from({ length: layer.neuronCount }, (_, index) => (
              <option key={index} value={index}>{index}</option>
            ))}
          </select>
        </label>
      </div>
      <h4>{layer.label} · neuron {selection.index}</h4>
      {selection.layerId === "input" && (
        <p>Pixel location: row {row}, column {column} (0-based).
          {pixels && <> Raw pixel intensity: {format(pixels[row]![column]!)}.</>}
        </p>
      )}
      {!details ? (
        <p>Load a digit in Digit Studio to inspect real model values.</p>
      ) : (
        <>
          <dl className="neuron-inspector-values">
            <div><dt>Activation</dt><dd>{format(details.activation)}</dd></div>
            {details.preActivation !== null && <div><dt>Pre-activation z</dt><dd>{format(details.preActivation)}</dd></div>}
            {details.bias !== null && <div><dt>Bias b</dt><dd>{format(details.bias)}</dd></div>}
          </dl>
          {selection.layerId === "input" ? (
            <p>Input activation = (pixel intensity − 0.1307) / 0.3081. Input neurons have no learned bias or incoming connections.</p>
          ) : (
            <>
              <p>z = Σ(source activation × weight) + bias = {format(weightedSum!)} + {format(details.bias!)} ≈ {format(details.preActivation!)}.
                Small differences can arise from floating-point rounding.</p>
              <p>
                {selection.layerId === "output"
                  ? "Softmax converts all 10 logits together into probabilities; this activation is the probability for this digit."
                  : "ReLU activation = max(0, z)."}
              </p>
              <h4>Incoming connections ({details.incoming.length})</h4>
              <p>Every incoming connection is listed, including those not drawn in the sampled 3D view.</p>
              <div className="neuron-inspector-table-wrap">
                <table>
                  <thead><tr><th>Source index</th><th>Activation</th><th>Weight</th><th>Contribution</th></tr></thead>
                  <tbody>
                    {details.incoming.map((item) => (
                      <tr key={item.sourceIndex}>
                        <td>{item.sourceIndex}</td><td>{format(item.sourceActivation)}</td><td>{format(item.weight)}</td><td>{format(item.contribution)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {details.outgoing.length > 0 ? (
            <>
              <h4>Direct contribution to output logits</h4>
              <p>Activation × output weight; these are logit contributions, not independent changes in softmax probabilities.</p>
              <div className="neuron-inspector-table-wrap">
                <table>
                  <thead><tr><th>Digit</th><th>Weight</th><th>Contribution</th></tr></thead>
                  <tbody>{details.outgoing.map((item) => (
                    <tr key={item.digit}><td>{item.digit}</td><td>{format(item.weight)}</td><td>{format(item.contribution)}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </>
          ) : selection.layerId === "output" ? (
            <p>This output activation is the probability for this digit; there is no later model layer.</p>
          ) : (
            <p>Individual downstream output influence is not shown for this layer; later nonlinearities and softmax couple the result.</p>
          )}
        </>
      )}
    </section>
  );
}
