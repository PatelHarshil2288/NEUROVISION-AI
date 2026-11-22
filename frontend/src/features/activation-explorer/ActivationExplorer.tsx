"use client";

import Link from "next/link";
import { useState } from "react";

import { ACTIVATIONS, activation, derivative, type ActivationId } from "./lib/functions";
import ParameterExplorer from "./ParameterExplorer";

function FunctionGraph({ id, x, showDerivative }: { id: ActivationId; x: number; showDerivative: boolean }) {
  const points = Array.from({ length: 201 }, (_, index) => {
    const input = (index - 100) / 20;
    return { input, value: showDerivative ? derivative(id, input) : activation(id, input) };
  });
  const values = points.flatMap((point) => point.value === null ? [] : [point.value]);
  const minimum = Math.min(0, ...values);
  const maximum = Math.max(0, ...values);
  const span = Math.max(maximum - minimum, 0.01);
  const low = minimum - span * 0.08;
  const high = maximum + span * 0.08;
  const plotX = (value: number) => 38 + (value + 5) * 25;
  const plotY = (value: number) => 124 - (value - low) / (high - low) * 104;
  let drawing = false;
  const path = points.map((point) => {
    if (point.value === null) {
      drawing = false;
      return "";
    }
    const command = drawing ? "L" : "M";
    drawing = true;
    return `${command}${plotX(point.input).toFixed(2)},${plotY(point.value).toFixed(2)}`;
  }).join(" ");
  const selected = showDerivative ? derivative(id, x) : activation(id, x);
  const label = showDerivative ? "Derivative" : "Function";

  return (
    <figure className="explorer-graph">
      <figcaption>{label} · x from −5 to 5 · y from {low.toFixed(2)} to {high.toFixed(2)}</figcaption>
      <svg viewBox="0 0 310 150" role="img" aria-label={`${label} graph for ${ACTIVATIONS.find((item) => item.id === id)!.name}, x from minus 5 to 5`}>
        <line x1="38" x2="288" y1={plotY(0)} y2={plotY(0)} className="explorer-axis" />
        <line x1={plotX(0)} x2={plotX(0)} y1="20" y2="124" className="explorer-axis" />
        <path d={path} fill="none" stroke={showDerivative ? "#c2a8ff" : "#66e4e6"} strokeWidth="2" />
        {selected !== null && <circle cx={plotX(x)} cy={plotY(selected)} r="3" fill="#ffffff" />}
        <text x="38" y="144">−5</text><text x="163" y="144" textAnchor="middle">0</text><text x="288" y="144" textAnchor="end">5</text>
      </svg>
    </figure>
  );
}

function RangeControl({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="explorer-range">
      <span>{label}: <strong>{value.toFixed(2)}</strong></span>
      <input type="range" min="-5" max="5" step="0.05" value={value}
        onChange={(event) => onChange(Number(event.currentTarget.value))} />
    </label>
  );
}

export default function ActivationExplorer() {
  const [selected, setSelected] = useState<ActivationId>("relu");
  const [x, setX] = useState(1);
  const [x1, setX1] = useState(1);
  const [x2, setX2] = useState(-1);
  const [w1, setW1] = useState(0.5);
  const [w2, setW2] = useState(-0.5);
  const [bias, setBias] = useState(0);
  const definition = ACTIVATIONS.find((item) => item.id === selected)!;
  const slope = derivative(selected, x);
  const z = x1 * w1 + x2 * w2 + bias;

  return (
    <div className="studio-shell">
      <header className="topbar">
        <Link className="brand-lockup" href="/" aria-label="NEUROVISION AI home">
          <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
          <span className="brand-wordmark"><strong>NEUROVISION</strong><span>INTELLIGENCE LAB</span></span>
        </Link>
        <div className="topbar-right">
          <Link className="phase-link" href="/">Digit Studio</Link>
          <Link className="phase-link" href="/observatory">Observatory</Link>
          <Link className="phase-link" href="/training">Training Lab</Link>
          <Link className="phase-link" href="/analytics">Analytics</Link>
          <span className="phase-label">PHASE 07 / EXPLORER</span>
        </div>
      </header>
      <main className="studio-main">
        <section className="page-intro">
          <div>
            <p className="eyebrow"><span className="eyebrow-line" /> INTERACTIVE MATHEMATICS</p>
            <h1>Activation<br /><span>&amp; Weight Explorer.</span></h1>
            <p className="intro-copy">Explore activation functions with hypothetical inputs, then inspect the loaded model&apos;s actual weights and biases.</p>
          </div>
          <div className="model-chip">
            <span className="model-chip-label">IMPORTANT DISTINCTION</span>
            <span className="model-chip-name">SANDBOX ≠ TRAINED MODEL</span>
            <span className="model-chip-foot">Sandbox changes never alter the model</span>
          </div>
        </section>

        <section className="explorer-grid" aria-label="Activation function sandbox">
          <article className="panel explorer-panel">
            <p className="panel-index">01 / HYPOTHETICAL FUNCTION</p>
            <h2>Activation functions</h2>
            <label className="explorer-select-label">Function
              <select aria-label="Activation function" value={selected} onChange={(event) => {
                const definition = ACTIVATIONS.find((item) => item.id === event.currentTarget.value);
                if (definition) setSelected(definition.id);
              }}>
                {ACTIVATIONS.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
            </label>
            <p className="explorer-equation">{definition.equation}</p>
            <p className="explorer-equation">{definition.derivativeEquation}</p>
            <p><strong>Advantage:</strong> {definition.advantage}</p>
            <p><strong>Limitation:</strong> {definition.limitation}</p>
            <RangeControl label="Example input x" value={x} onChange={setX} />
            <p aria-live="polite">f({x.toFixed(2)}) = <strong>{activation(selected, x).toFixed(5)}</strong> · derivative = <strong>{slope === null ? "undefined at 0" : slope.toFixed(5)}</strong></p>
            <p>The MNIST model uses ReLU in its hidden layers and softmax for output probabilities; other choices here are hypothetical.</p>
          </article>
          <article className="panel explorer-panel">
            <p className="panel-index">02 / FUNCTION AND DERIVATIVE</p>
            <h2>Function graphs</h2>
            <FunctionGraph id={selected} x={x} showDerivative={false} />
            <FunctionGraph id={selected} x={x} showDerivative />
            <p>The white marker shows the selected input. At a ReLU or Leaky ReLU kink, the mathematical derivative is undefined.</p>
          </article>
          <article className="panel explorer-panel explorer-sandbox">
            <p className="panel-index">03 / HYPOTHETICAL NEURON</p>
            <h2>Weight and bias sandbox</h2>
            <p>These sliders are independent examples, not parameters or predictions from the trained model.</p>
            <div className="explorer-range-grid">
              <RangeControl label="Input x1" value={x1} onChange={setX1} />
              <RangeControl label="Weight w1" value={w1} onChange={setW1} />
              <RangeControl label="Input x2" value={x2} onChange={setX2} />
              <RangeControl label="Weight w2" value={w2} onChange={setW2} />
              <RangeControl label="Bias b" value={bias} onChange={setBias} />
            </div>
            <p aria-live="polite">z = x1 × w1 + x2 × w2 + b = <strong>{z.toFixed(5)}</strong> · {definition.name}(z) = <strong>{activation(selected, z).toFixed(5)}</strong></p>
          </article>
        </section>
        <ParameterExplorer />
        <footer className="studio-footer"><span>NEUROVISION AI</span><span>HYPOTHETICAL FUNCTIONS · REAL MODEL PARAMETERS</span></footer>
      </main>
    </div>
  );
}
