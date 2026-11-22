"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getTrainingStatus, startTraining, type EpochMetrics, type TrainingStatus } from "./api";

function message(error: unknown): string {
  return error instanceof Error ? error.message : "The training service could not be reached.";
}

function format(value: number | null, digits = 5): string {
  return value === null ? "—" : value.toFixed(digits);
}

function LossChart({ history }: { history: EpochMetrics[] }) {
  const maximum = Math.max(...history.flatMap((item) => [item.train_loss, item.validation_loss]), 0.000001);
  const x = (index: number) => history.length === 1 ? 150 : 20 + index * 260 / (history.length - 1);
  const y = (value: number) => 110 - value / maximum * 90;
  return (
    <figure className="training-chart">
      <svg viewBox="0 0 300 125" role="img" aria-label="Training and validation loss by epoch; values are listed in the table below">
        <line x1="20" y1="110" x2="280" y2="110" stroke="#435366" />
        {(["train_loss", "validation_loss"] as const).map((key) => (
          <g key={key} fill={key === "train_loss" ? "#66e4e6" : "#c2a8ff"} stroke={key === "train_loss" ? "#66e4e6" : "#c2a8ff"}>
            <polyline fill="none" strokeWidth="2" points={history.map((item, index) => `${x(index)},${y(item[key])}`).join(" ")} />
            {history.map((item, index) => <circle key={item.epoch} cx={x(index)} cy={y(item[key])} r="3" />)}
          </g>
        ))}
      </svg>
      <figcaption><span>● Training loss</span> <span>● Validation loss</span> · horizontal axis: epoch; vertical axis: loss (zero at baseline)</figcaption>
    </figure>
  );
}

export default function TrainingLab() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [epochs, setEpochs] = useState(1);
  const [starting, setStarting] = useState(false);
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function poll() {
      try {
        const next = await getTrainingStatus(controller.signal);
        if (!controller.signal.aborted) {
          setStatus(next);
          setError(null);
        }
      } catch (reason) {
        if (!controller.signal.aborted) setError(message(reason));
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 1500);
      }
    }
    void poll();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [refresh]);

  const job = status?.job ?? null;
  const active = job?.status === "queued" || job?.status === "running";
  async function submit() {
    setStarting(true);
    setActionError(null);
    try {
      await startTraining(epochs);
      setRefresh((value) => value + 1);
    } catch (reason) {
      setActionError(message(reason));
      setRefresh((value) => value + 1);
    } finally {
      setStarting(false);
    }
  }

  return (
    <div className="studio-shell">
      <header className="topbar">
        <Link className="brand-lockup" href="/" aria-label="NEUROVISION AI home">
          <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
          <span className="brand-wordmark"><strong>NEUROVISION</strong><span>INTELLIGENCE LAB</span></span>
        </Link>
        <div className="topbar-right">
          <Link className="phase-link" href="/">Digit Studio</Link>
          <Link className="phase-link" href="/observatory">3D Observatory</Link>
          <Link className="phase-link" href="/explorer">Explorer</Link>
          <Link className="phase-link" href="/analytics">Analytics</Link>
          <span className="phase-label">PHASE 06 / TRAINING LAB</span>
        </div>
      </header>
      <main className="studio-main training-main">
        <section className="page-intro">
          <div>
            <p className="eyebrow"><span className="eyebrow-line" /> BACKEND TRAINING</p>
            <h1>Training<br /><span>Laboratory.</span></h1>
            <p className="intro-copy">Watch a real neural network learn from MNIST. Each experiment starts with new weights and runs on the backend CPU.</p>
          </div>
          <div className="model-chip">
            <span className="model-chip-label">BOUNDED LOCAL EXPERIMENT</span>
            <span className="model-chip-name">2,048 TRAIN · 512 VALIDATION</span>
            <span className="model-chip-foot">Separate from the inference checkpoint</span>
          </div>
        </section>

        <section className="training-grid" aria-label="Training laboratory">
          <article className="panel training-panel">
            <p className="panel-index">01 / EXPERIMENT</p>
            <h2>Start a training job</h2>
            <p>CPU only · Adam · learning rate 0.001 · batch size 128 · no dataset download. Limit: one job at a time, up to 3 epochs and 3 jobs per backend process.</p>
            {!status && !error && <p role="status">Connecting to the training service…</p>}
            {error && <p className="training-error" role="alert">{error} {status && "Displayed values may be stale."}</p>}
            {status && !status.enabled && (
              <div className="training-notice">
                <strong>Local training is disabled.</strong>
                <p>Set <code>NEUROVISION_TRAINING_ENABLED=1</code> before starting the backend. Use only a direct, single-process local server; never enable this lab on a public or proxied deployment.</p>
              </div>
            )}
            {status && <p>Jobs remaining this process: <strong>{status.remaining_jobs}</strong></p>}
            <div className="training-controls">
              <label>Epochs
                <select value={epochs} onChange={(event) => setEpochs(Number(event.currentTarget.value))}>
                  <option value={1}>1</option><option value={2}>2</option><option value={3}>3</option>
                </select>
              </label>
              <button type="button" className="observatory-playback-button" onClick={() => void submit()}
                disabled={!status?.enabled || Boolean(error) || starting || active || status.remaining_jobs <= 0}>
                {starting ? "Starting…" : active ? "Training in progress" : "Start training"}
              </button>
            </div>
            {actionError && <p className="training-error" role="alert">{actionError}</p>}
            <p className="training-footnote">Experiments are temporary process state. Restarting the backend clears the job status and quota; lab checkpoints are overwritten by the next experiment and never replace the prediction model.</p>
          </article>

          <article className="panel training-panel">
            <p className="panel-index">02 / LIVE PROGRESS</p>
            <h2>Training progress</h2>
            {!job ? <p>No training job has been started in this backend process.</p> : (
              <>
                <p role="status"><strong>{job.status.toUpperCase()}</strong> · Epoch {job.epoch} of {job.epochs} · batch {job.batch} of {job.batches}</p>
                <progress aria-label="Current epoch batch progress" value={job.batch} max={job.batches} />
                <dl className="training-metrics">
                  <div><dt>Training loss (current epoch)</dt><dd>{format(job.train_loss)}</dd></div>
                  <div><dt>Validation loss (last completed epoch)</dt><dd>{format(job.validation_loss)}</dd></div>
                  <div><dt>Validation accuracy</dt><dd>{job.validation_accuracy === null ? "—" : `${(job.validation_accuracy * 100).toFixed(2)}%`}</dd></div>
                  <div><dt>Optimizer / learning rate</dt><dd>{job.optimizer} / {job.learning_rate}</dd></div>
                </dl>
                {job.status === "failed" && <p className="training-error" role="alert">Training failed: {job.error ?? "Unknown error"}</p>}
                {job.checkpoint && <p>Latest checkpoint: epoch {job.checkpoint.latest_epoch} · best validation accuracy: epoch {job.checkpoint.best_epoch}. Saved in <code>{job.checkpoint.directory}</code>.</p>}
              </>
            )}
          </article>

          <article className="panel training-panel">
            <p className="panel-index">03 / LEARNING CURVE</p>
            <h2>Epoch metrics</h2>
            {!job?.history.length ? <p>Metrics appear after the first validation pass.</p> : (
              <>
                <LossChart history={job.history} />
                <div className="training-table-wrap">
                  <table><thead><tr><th>Epoch</th><th>Train loss</th><th>Validation loss</th><th>Validation accuracy</th></tr></thead>
                    <tbody>{job.history.map((item) => <tr key={item.epoch}><td>{item.epoch}</td><td>{format(item.train_loss)}</td><td>{format(item.validation_loss)}</td><td>{(item.validation_accuracy * 100).toFixed(2)}%</td></tr>)}</tbody>
                  </table>
                </div>
              </>
            )}
          </article>

          <article className="panel training-panel">
            <p className="panel-index">04 / BACKPROPAGATION</p>
            <h2>What changed?</h2>
            <p>Backpropagation computes how the batch loss changes with each parameter. Adam uses those gradients and its running moments to update the weights; the update is not simply learning rate × gradient.</p>
            {!job || job.gradient_norm === null ? <p>Gradient and weight information appears after the first batch.</p> : (
              <>
                <p>Latest batch · first input-to-hidden weight <code>fc1.weight[0, 0]</code></p>
                <dl className="training-metrics">
                  <div><dt>All-parameter gradient L2 norm</dt><dd>{format(job.gradient_norm)}</dd></div>
                  <div><dt>Selected weight gradient</dt><dd>{format(job.sample_gradient)}</dd></div>
                  <div><dt>Weight before update</dt><dd>{format(job.sample_before)}</dd></div>
                  <div><dt>Weight after update</dt><dd>{format(job.sample_after)}</dd></div>
                  <div><dt>Actual parameter change</dt><dd>{job.sample_before === null || job.sample_after === null ? "—" : format(job.sample_after - job.sample_before)}</dd></div>
                </dl>
              </>
            )}
          </article>
        </section>
        <footer className="studio-footer"><span>NEUROVISION AI</span><span>LOCAL CPU TRAINING · REAL MODEL METRICS</span></footer>
      </main>
    </div>
  );
}
