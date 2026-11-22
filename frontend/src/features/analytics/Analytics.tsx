"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { predictDigit, type PredictionResult } from "../digit-studio/lib/api";
import { readCurrentModelInput } from "../../lib/current-model-input";
import {
  activateModel,
  compareModels,
  evaluateCheckpoint,
  getComparisonHistory,
  getComparisonReports,
  getErrorImage,
  getEvaluation,
  getModelRegistry,
  getSensitivity,
  rollbackModel,
  type AvailableCheckpoint,
  type Evaluation,
  type ModelComparison,
  type ModelRegistrySnapshot,
  type Sensitivity,
} from "./api";

const digits = Array.from({ length: 10 }, (_, index) => index);
const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
const message = (error: unknown) => error instanceof Error ? error.message : "The request failed.";

function PixelMap({ values, sensitivity = false }: { values: number[][]; sensitivity?: boolean }) {
  const maximum = Math.max(...values.flat().map(Math.abs));
  return <svg className="analytics-pixels" viewBox="0 0 28 28" role="img"
    aria-label={sensitivity ? "Signed input sensitivity map: cyan positive, coral negative" : "28 by 28 input digit"}>
    {values.flatMap((row, y) => row.map((value, x) => {
      const opacity = sensitivity ? (maximum === 0 ? 0 : Math.abs(value) / maximum) : value;
      return <rect key={`${x}-${y}`} x={x} y={y} width="1" height="1"
        fill={sensitivity ? (value >= 0 ? "#57e4e6" : "#ff8c83") : "#f1f8fa"} opacity={opacity} />;
    }))}
  </svg>;
}

export default function Analytics() {
  const [report, setReport] = useState<Evaluation | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);
  const [filterTrue, setFilterTrue] = useState("all");
  const [filterPredicted, setFilterPredicted] = useState("all");
  const [sample, setSample] = useState<number | null>(null);
  const [image, setImage] = useState<{ pixels: number[][]; label: string } | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [target, setTarget] = useState("predicted");
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [sensitivity, setSensitivity] = useState<Sensitivity | null>(null);
  const [sensitivityError, setSensitivityError] = useState<string | null>(null);
  const [comparisonReports, setComparisonReports] = useState<Evaluation[]>([]);
  const [availableCheckpoints, setAvailableCheckpoints] = useState<AvailableCheckpoint[]>([]);
  const [selectedCheckpointHash, setSelectedCheckpointHash] = useState("");
  const [modelRegistry, setModelRegistry] = useState<ModelRegistrySnapshot | null>(null);
  const [comparisonLoaded, setComparisonLoaded] = useState(false);
  const [comparisonHistory, setComparisonHistory] = useState<ModelComparison[]>([]);
  const [comparison, setComparison] = useState<ModelComparison | null>(null);
  const [selectedModelA, setSelectedModelA] = useState("");
  const [selectedModelB, setSelectedModelB] = useState("");
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [comparisonBusy, setComparisonBusy] = useState(false);
  const [evaluationBusy, setEvaluationBusy] = useState(false);
  const [activationBusy, setActivationBusy] = useState(false);
  const [activationMessage, setActivationMessage] = useState<string | null>(null);
  const [activationError, setActivationError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void getEvaluation(controller.signal).then((value) => {
      if (!controller.signal.aborted) {
        setReport(value);
        setReportError(null);
      }
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setReportError(message(error));
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void Promise.all([
      getModelRegistry(controller.signal),
      getComparisonReports(controller.signal),
      getComparisonHistory(controller.signal),
    ]).then(([registry, reports, history]) => {
        if (controller.signal.aborted) return;
        setModelRegistry(registry);
        setAvailableCheckpoints(registry.available_checkpoints);
        setSelectedCheckpointHash(
          registry.active?.checkpoint_hash
          ?? registry.available_checkpoints[0]?.checkpoint_hash
          ?? "",
        );
        setComparisonReports(reports);
        setComparisonHistory(history);
        setComparison(history[0] ?? null);
        setComparisonError(null);
        setComparisonLoaded(true);
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setComparisonError(message(error));
          setComparisonLoaded(true);
        }
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!report) return;
    const controller = new AbortController();
    void getComparisonReports(controller.signal).then((reports) => {
      if (!controller.signal.aborted) setComparisonReports(reports);
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setComparisonError(message(error));
    });
    return () => controller.abort();
  }, [report]);

  useEffect(() => {
    let active = true;
    Promise.resolve().then(() => readCurrentModelInput()).then((pixels) => {
      if (active && pixels) setImage({ pixels, label: "Current Digit Studio input" });
    }).catch((error: unknown) => { if (active) setImageError(message(error)); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (sample === null || !report) return;
    const controller = new AbortController();
    void getErrorImage(sample, report.checkpoint_sha256, controller.signal).then((pixels) => {
      if (!controller.signal.aborted) setImage({ pixels, label: `MNIST test sample ${sample}` });
    }).catch((error: unknown) => { if (!controller.signal.aborted) setImageError(message(error)); });
    return () => controller.abort();
  }, [sample, report]);

  useEffect(() => {
    if (!image) return;
    const controller = new AbortController();
    void predictDigit(image.pixels, controller.signal).then((value) => {
      if (!controller.signal.aborted) setPrediction(value);
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setPredictionError(message(error));
    });
    return () => controller.abort();
  }, [image]);

  useEffect(() => {
    if (!image) return;
    const controller = new AbortController();
    void getSensitivity(image.pixels, target === "predicted" ? null : Number(target), controller.signal)
      .then((value) => { if (!controller.signal.aborted) setSensitivity(value); }).catch((error: unknown) => {
        if (!controller.signal.aborted) setSensitivityError(message(error));
      });
    return () => controller.abort();
  }, [image, target]);

  function clearResults() {
    setPrediction(null); setPredictionError(null); setSensitivity(null); setSensitivityError(null); setImageError(null);
  }

  function loadCurrent() {
    setSample(null); setImage(null); clearResults();
    try {
      const pixels = readCurrentModelInput();
      if (pixels) setImage({ pixels, label: "Current Digit Studio input" });
      else setImageError("No current digit in this tab. Draw or upload one in Digit Studio first.");
    } catch (error) { setImageError(message(error)); }
  }

  const modelA = comparisonReports.find((item) => item.checkpoint_sha256 === selectedModelA)
    ?? comparisonReports.find((item) => item.checkpoint_sha256 === report?.checkpoint_sha256)
    ?? comparisonReports[0];
  const modelB = comparisonReports.find((item) =>
    item.checkpoint_sha256 === selectedModelB && item.checkpoint_sha256 !== modelA?.checkpoint_sha256)
    ?? comparisonReports.find((item) => item.checkpoint_sha256 !== modelA?.checkpoint_sha256);

  async function runCheckpointEvaluation() {
    if (!selectedCheckpointHash) return;
    setEvaluationBusy(true);
    setComparisonError(null);
    try {
      const evaluated = await evaluateCheckpoint(selectedCheckpointHash);
      if (availableCheckpoints.find((item) => item.checkpoint_hash === selectedCheckpointHash)?.is_loaded) {
        setReport(evaluated);
        setReportError(null);
      }
      const reports = await getComparisonReports();
      setComparisonReports(reports);
      setComparisonError(null);
    } catch (error) {
      setComparisonError(message(error));
    } finally {
      setEvaluationBusy(false);
    }
  }

  async function finishModelChange(registry: ModelRegistrySnapshot, action: string) {
    setModelRegistry(registry);
    setAvailableCheckpoints(registry.available_checkpoints);
    setSelectedCheckpointHash(registry.active?.checkpoint_hash ?? "");
    setActivationMessage(registry.active ? `${action}: ${registry.active.model_version}.` : null);
    setActivationError(null);
    if (!registry.active) return;
    try {
      const currentReport = await getEvaluation();
      setReport(currentReport);
      setReportError(null);
      setComparisonReports(await getComparisonReports());
    } catch (error) {
      const detail = message(error);
      setReportError(detail);
      setActivationError(`Model changed, but its evaluation could not be refreshed: ${detail}`);
    }
  }

  async function runModelActivation() {
    if (!selectedCheckpointHash) return;
    setActivationBusy(true);
    setActivationError(null);
    setActivationMessage(null);
    try {
      await finishModelChange(
        await activateModel(selectedCheckpointHash),
        "Activated model",
      );
    } catch (error) {
      setActivationError(message(error));
    } finally {
      setActivationBusy(false);
    }
  }

  async function runModelRollback() {
    setActivationBusy(true);
    setActivationError(null);
    setActivationMessage(null);
    try {
      await finishModelChange(await rollbackModel(), "Rolled back to");
    } catch (error) {
      setActivationError(message(error));
    } finally {
      setActivationBusy(false);
    }
  }

  async function runComparison() {
    if (!modelA || !modelB || modelA.checkpoint_sha256 === modelB.checkpoint_sha256) return;
    setComparisonBusy(true);
    setComparisonError(null);
    try {
      const result = await compareModels(modelA, modelB);
      setComparison(result);
      setComparisonHistory(await getComparisonHistory());
    } catch (error) {
      setComparisonError(message(error));
    } finally {
      setComparisonBusy(false);
    }
  }

  const errors = report?.misclassified_examples.filter((item) =>
    (filterTrue === "all" || item.true_label === Number(filterTrue)) &&
    (filterPredicted === "all" || item.predicted_label === Number(filterPredicted))) ?? [];
  const activeHash = modelRegistry?.active?.checkpoint_hash;
  const hasRollback = activeHash !== undefined && Boolean(
    modelRegistry?.history.some((entry) => entry.checkpoint_hash !== activeHash),
  );
  return <div className="studio-shell">
    <header className="topbar">
      <Link className="brand-lockup" href="/" aria-label="NEUROVISION AI home">
        <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
        <span className="brand-wordmark"><strong>NEUROVISION</strong><span>INTELLIGENCE LAB</span></span>
      </Link>
      <div className="topbar-right">
        <Link className="phase-link" href="/">Digit Studio</Link>
        <Link className="phase-link" href="/observatory">Observatory</Link>
        <Link className="phase-link" href="/training">Training Lab</Link>
        <Link className="phase-link" href="/explorer">Explorer</Link>
        <span className="phase-label">PHASE 10 / MODEL VERSIONING</span>
      </div>
    </header>
    <main className="studio-main">
      <section className="page-intro">
        <div><p className="eyebrow"><span className="eyebrow-line" /> MODEL EVIDENCE</p>
          <h1>Performance<br /><span>&amp; Sensitivity.</span></h1>
          <p className="intro-copy">Explore the official test results, examine real errors, and see how a selected class responds to small pixel changes.</p>
        </div>
        <div className="model-chip"><span className="model-chip-label">EVALUATION SOURCE</span>
          <span className="model-chip-name">MNIST / TEST SPLIT</span>
          <span className="model-chip-foot">Checkpoint-matched report</span></div>
      </section>

      <section className="panel explorer-panel analytics-section" aria-label="Evaluation results">
        <p className="panel-index">01 / VERIFIED TEST RESULTS</p><h2>Evaluation overview</h2>
        {reportError ? <p role="alert">{reportError}</p> : !report ? <p>Loading evaluation report…</p> : <>
          <p>{report.model_version} · {report.num_samples.toLocaleString()} official test images · generated {report.generated_at}</p>
          <p className="analytics-hash">Checkpoint SHA-256: {report.checkpoint_sha256}</p>
          <div className="analytics-stats"><div><strong>{percent(report.accuracy)}</strong><span>Accuracy</span></div>
            <div><strong>{report.loss.toFixed(4)}</strong><span>Average loss</span></div>
            <div><strong>{report.misclassified_examples.length.toLocaleString()}</strong><span>Misclassifications</span></div></div>
          <h3>Confusion matrix</h3><p>Rows are true labels; columns are predicted labels. Diagonal cells are correct predictions.</p>
          <div className="analytics-scroll"><table className="analytics-matrix"><caption>Counts by true and predicted digit</caption>
            <thead><tr><th scope="col">True ↓ / Pred →</th>{digits.map((digit) => <th scope="col" key={digit}>{digit}</th>)}</tr></thead>
            <tbody>{report.confusion_matrix.map((row, trueLabel) => <tr key={trueLabel}><th scope="row">{trueLabel}</th>
              {row.map((count, predicted) => <td key={predicted} className={trueLabel === predicted ? "diagonal" : ""}>{count}</td>)}</tr>)}</tbody>
          </table></div>
          <h3>Per-class metrics</h3>
          <div className="analytics-scroll"><table className="analytics-table"><thead><tr><th>Digit</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
            <tbody>{digits.map((digit) => <tr key={digit}><th scope="row">{digit}</th><td>{percent(report.precision_per_class[digit])}</td>
              <td>{percent(report.recall_per_class[digit])}</td><td>{percent(report.f1_per_class[digit])}</td><td>{report.support_per_class[digit]}</td></tr>)}</tbody>
          </table></div>
        </>}
      </section>

      <section className="panel explorer-panel analytics-section" aria-label="Model version control">
        <p className="panel-index">02 / MODEL VERSIONING</p><h2>Activate or roll back</h2>
        <p>Only approved local checkpoints can be activated. The active model and activation history persist across API restarts.</p>
        <p>Active model: <strong>{modelRegistry?.active
          ? `${modelRegistry.active.model_version} · ${modelRegistry.active.architecture}`
          : "No approved active checkpoint"}</strong></p>
        <div className="comparison-selectors">
          <label>Checkpoint to activate<select value={selectedCheckpointHash}
            onChange={(event) => setSelectedCheckpointHash(event.target.value)}>
            <option value="">Select a checkpoint</option>
            {availableCheckpoints.map((checkpoint) => <option key={checkpoint.checkpoint_hash} value={checkpoint.checkpoint_hash}>
              {checkpoint.model_version} · {checkpoint.architecture} · {checkpoint.checkpoint_name}
              {checkpoint.is_loaded ? " · active" : ""}
            </option>)}
          </select></label>
        </div>
        <button type="button" className="analytics-button" disabled={!selectedCheckpointHash ||
          selectedCheckpointHash === modelRegistry?.active?.checkpoint_hash || activationBusy}
          onClick={() => void runModelActivation()}>
          {activationBusy ? "Changing model…" : "Activate checkpoint"}
        </button>
        <button type="button" className="analytics-button" disabled={!hasRollback || activationBusy}
          onClick={() => void runModelRollback()}>
          Roll back to previous model
        </button>
        {activationMessage && <p role="status">{activationMessage}</p>}
        {activationError && <p role="alert">{activationError}</p>}
        {modelRegistry?.history.length ? <div className="comparison-history">
          <h3>Recent model activations</h3>
          <ol>{modelRegistry.history.map((entry, index) => <li key={`${entry.checkpoint_hash}-${index}`}>
            {entry.model_version} · {entry.architecture} · {new Date(entry.activated_at).toLocaleString()}
          </li>)}</ol>
        </div> : null}
      </section>

      <div className="analytics-grid">
        <section className="panel explorer-panel analytics-section" aria-label="Misclassification explorer">
          <p className="panel-index">03 / ERROR ANALYSIS</p><h2>Misclassification explorer</h2>
          {!report ? <p>Evaluation data is unavailable.</p> : <>
            <p>Choose an actual MNIST test error to inspect its image and sensitivity.</p>
            <div className="explorer-selectors">
              <label>True digit<select value={filterTrue} onChange={(event) => setFilterTrue(event.target.value)}><option value="all">All</option>{digits.map((d) => <option key={d}>{d}</option>)}</select></label>
              <label>Predicted digit<select value={filterPredicted} onChange={(event) => setFilterPredicted(event.target.value)}><option value="all">All</option>{digits.map((d) => <option key={d}>{d}</option>)}</select></label>
            </div>
            <p>{errors.length} matching errors</p>
            <div className="analytics-errors">{errors.map((item) => <button key={item.sample_index} type="button"
              aria-pressed={sample === item.sample_index} onClick={() => {
                if (sample === item.sample_index) return;
                setImage(null); clearResults(); setSample(item.sample_index);
              }}>#{item.sample_index} · {item.true_label} → {item.predicted_label} · {percent(item.confidence)}</button>)}
              {errors.length === 0 && <p>No errors match these filters.</p>}
            </div>
          </>}
        </section>
        <section className="panel explorer-panel analytics-section" aria-label="Prediction and sensitivity">
          <p className="panel-index">04 / INPUT ANALYSIS</p><h2>Prediction &amp; sensitivity</h2>
          <button type="button" className="analytics-button" onClick={loadCurrent}>Load current Digit Studio input</button>
          {!image && !imageError && <p>Select an error or draw a digit in Digit Studio to begin.</p>}
          {imageError && <p role="alert">{imageError}</p>}
          {image && <>
            <p><strong>{image.label}</strong></p><PixelMap values={image.pixels} />
            {predictionError ? <p role="alert">{predictionError}</p> : prediction ? <>
              <p>Predicted <strong>{prediction.predicted_class}</strong> · model {prediction.model_version} · inference {prediction.inference_ms.toFixed(2)} ms (backend model inference, not network time)</p>
              <ol className="analytics-probabilities">{prediction.probabilities.map((value, digit) => <li key={digit}>
                <span>{digit}</span><meter min="0" max="1" value={value} aria-label={`Probability for digit ${digit}`} />
                <span>{percent(value)}</span></li>)}</ol>
            </> : <p>Running prediction…</p>}
            <label className="explorer-select-label">Sensitivity target
              <select value={target} onChange={(event) => { setSensitivity(null); setSensitivityError(null); setTarget(event.target.value); }}>
                <option value="predicted">Predicted class</option>{digits.map((digit) => <option key={digit} value={digit}>Digit {digit}</option>)}
              </select>
            </label>
            {sensitivityError ? <p role="alert">{sensitivityError}</p> : sensitivity ? <>
              <p>Class {sensitivity.target_class} logit · model {sensitivity.model_version}</p>
              <PixelMap values={sensitivity.gradient} sensitivity />
              <p>Cyan: positive gradient · coral: negative gradient · brightness: relative magnitude. A zero gradient map is blank.</p>
            </> : <p>Calculating sensitivity…</p>}
            <p>This gradient is a local approximation for small changes to raw pixel intensity, not a definitive explanation of the model&apos;s reasoning.</p>
          </>}
        </section>
      </div>
      <section className="panel explorer-panel analytics-section" aria-label="Model comparison">
        <p className="panel-index">05 / MODEL COMPARISON</p><h2>Compare evaluated checkpoints</h2>
        <p>Reports are generated by the backend from approved local checkpoints and the official MNIST test split.</p>
        <label>Approved checkpoint<select value={selectedCheckpointHash}
          onChange={(event) => setSelectedCheckpointHash(event.target.value)}>
          <option value="">Select a checkpoint</option>
          {availableCheckpoints.map((checkpoint) => <option key={checkpoint.checkpoint_hash} value={checkpoint.checkpoint_hash}>
            {checkpoint.model_version} · {checkpoint.architecture} · {checkpoint.checkpoint_name}
            {checkpoint.is_loaded ? " · loaded" : ""}
          </option>)}
        </select></label>
        <button type="button" className="analytics-button" disabled={!selectedCheckpointHash || evaluationBusy}
          onClick={() => void runCheckpointEvaluation()}>
          {evaluationBusy ? "Evaluating…" : "Evaluate checkpoint on official test split"}
        </button>
        {comparisonError && <p role="alert">{comparisonError}</p>}
        {!comparisonLoaded ? <p>Loading saved evaluation reports…</p> : comparisonReports.length === 0
          ? <p>No evaluation reports are stored yet.</p> : <>
          <div className="comparison-selectors">
            <label>Model A<select value={modelA?.checkpoint_sha256 ?? ""} onChange={(event) => {
              const nextModelA = event.target.value;
              setSelectedModelA(nextModelA);
              setSelectedModelB((current) => current === nextModelA
                ? comparisonReports.find((item) => item.checkpoint_sha256 !== nextModelA)?.checkpoint_sha256 ?? ""
                : current);
            }}>
              {comparisonReports.map((item) => <option key={item.checkpoint_sha256} value={item.checkpoint_sha256}>
                {item.model_version} · {item.checkpoint_sha256.slice(0, 10)}
              </option>)}
            </select></label>
            <label>Model B<select value={modelB?.checkpoint_sha256 ?? ""} onChange={(event) => setSelectedModelB(event.target.value)}>
              <option value="">Select a different checkpoint</option>
              {comparisonReports.filter((item) => item.checkpoint_sha256 !== selectedModelA).map((item) =>
                <option key={item.checkpoint_sha256} value={item.checkpoint_sha256}>
                  {item.model_version} · {item.checkpoint_sha256.slice(0, 10)}
                </option>)}
            </select></label>
          </div>
          <p>{comparisonReports.length} stored evaluation report{comparisonReports.length === 1 ? "" : "s"}</p>
          <button type="button" className="analytics-button"
            disabled={!modelA || !modelB || modelB.checkpoint_sha256 === modelA.checkpoint_sha256 || comparisonBusy}
            onClick={() => void runComparison()}>
            {comparisonBusy ? "Comparing…" : "Compare checkpoints"}
          </button>
        </>}
        {comparison && <div className="comparison-result" aria-live="polite">
          <p><strong>{comparison.model_a_name} vs {comparison.model_b_name}</strong></p>
          <p>{comparison.summary}</p>
          <div className="analytics-scroll"><table className="analytics-table">
            <thead><tr><th>Metric</th><th>{comparison.model_a_version}</th><th>{comparison.model_b_version}</th><th>Change</th></tr></thead>
            <tbody>{comparison.metrics.map((metric) => <tr key={metric.metric_name}>
              <th scope="row">{metric.metric_name}</th>
              <td>{metric.model_a_value.toFixed(4)}</td><td>{metric.model_b_value.toFixed(4)}</td>
              <td>{metric.difference > 0 ? "+" : ""}{metric.difference.toFixed(4)} ({metric.percent_change.toFixed(2)}%)</td>
            </tr>)}</tbody>
          </table></div>
        </div>}
        {comparisonHistory.length > 0 && <div className="comparison-history">
          <h3>Saved comparisons</h3>
          <div className="analytics-errors">{comparisonHistory.map((item) =>
            <button key={item.comparison_id} type="button" aria-pressed={comparison?.comparison_id === item.comparison_id}
              onClick={() => setComparison(item)}>
              {item.model_a_name} vs {item.model_b_name} · {new Date(item.compared_at).toLocaleString()}
            </button>)}</div>
        </div>}
      </section>
    </main>
  </div>;
}
