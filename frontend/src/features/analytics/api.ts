import { getApiBaseUrl, requestJson } from "../digit-studio/lib/api";
import { isModelInputPixels } from "../../lib/current-model-input";

export interface ErrorSample {
  sample_index: number;
  true_label: number;
  predicted_label: number;
  confidence: number;
}

export interface Evaluation {
  model_version: string;
  checkpoint_sha256: string;
  dataset: string;
  dataset_split: string;
  dataset_images_sha256: string;
  dataset_labels_sha256: string;
  generated_at: string;
  accuracy: number;
  loss: number;
  num_samples: number;
  confusion_matrix: number[][];
  precision_per_class: number[];
  recall_per_class: number[];
  f1_per_class: number[];
  support_per_class: number[];
  misclassified_examples: ErrorSample[];
}

export interface Sensitivity {
  model_version: string;
  predicted_class: number;
  target_class: number;
  probabilities: number[];
  gradient: number[][];
}

export interface ComparisonMetric {
  metric_name: string;
  model_a_value: number;
  model_b_value: number;
  difference: number;
  percent_change: number;
}

export interface ModelComparison {
  comparison_id: string;
  model_a_name: string;
  model_b_name: string;
  model_a_version: string;
  model_b_version: string;
  compared_at: string;
  metrics: ComparisonMetric[];
  summary: string;
}

export interface AvailableCheckpoint {
  checkpoint_hash: string;
  model_version: string;
  architecture: string;
  checkpoint_name: string;
  is_loaded: boolean;
}

export interface ModelActivation {
  checkpoint_hash: string;
  model_version: string;
  architecture: string;
  checkpoint_name: string;
  activated_at: string;
}

export interface ModelRegistrySnapshot {
  active: ModelActivation | null;
  available_checkpoints: AvailableCheckpoint[];
  history: ModelActivation[];
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}

function isComparisonMetric(value: unknown): value is ComparisonMetric {
  return record(value) && typeof value.metric_name === "string" &&
    finite(value.model_a_value) && finite(value.model_b_value) &&
    finite(value.difference) && finite(value.percent_change);
}

function isModelComparison(value: unknown): value is ModelComparison {
  return record(value) && typeof value.comparison_id === "string" &&
    typeof value.model_a_name === "string" && typeof value.model_b_name === "string" &&
    typeof value.model_a_version === "string" && typeof value.model_b_version === "string" &&
    typeof value.compared_at === "string" && typeof value.summary === "string" &&
    Array.isArray(value.metrics) && value.metrics.every(isComparisonMetric);
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function digit(value: unknown): value is number {
  return finite(value) && Number.isInteger(value) && value >= 0 && value <= 9;
}

function vector(value: unknown, length: number, predicate: (value: unknown) => boolean): value is number[] {
  return Array.isArray(value) && value.length === length && value.every(predicate);
}

const probability = (value: unknown) => finite(value) && value >= 0 && value <= 1;
const count = (value: unknown) => finite(value) && Number.isInteger(value) && value >= 0;
const hash = (value: unknown) => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);

function isEvaluation(value: unknown): value is Evaluation {
  if (!record(value) || typeof value.model_version !== "string" || !hash(value.checkpoint_sha256) ||
      value.dataset !== "MNIST" || value.dataset_split !== "official-test" ||
      !hash(value.dataset_images_sha256) || !hash(value.dataset_labels_sha256) ||
      typeof value.generated_at !== "string" || !probability(value.accuracy) ||
      !finite(value.loss) || value.loss < 0 || !count(value.num_samples) || value.num_samples === 0 ||
      !Array.isArray(value.confusion_matrix) || value.confusion_matrix.length !== 10 ||
      !value.confusion_matrix.every((row) => vector(row, 10, count)) ||
      !vector(value.precision_per_class, 10, probability) ||
      !vector(value.recall_per_class, 10, probability) ||
      !vector(value.f1_per_class, 10, probability) ||
      !vector(value.support_per_class, 10, count) ||
      !Array.isArray(value.misclassified_examples) ||
      !value.misclassified_examples.every((item: unknown) => record(item) && count(item.sample_index) &&
        digit(item.true_label) && digit(item.predicted_label) && probability(item.confidence))) {
    return false;
  }
  return true;
}

export async function getEvaluation(signal?: AbortSignal): Promise<Evaluation> {
  const value = await requestJson(`${getApiBaseUrl()}/api/v1/analytics/evaluation`, { signal });
  if (!isEvaluation(value)) throw new Error("The AI service returned an invalid evaluation report.");
  return value;
}

export async function getComparisonReports(signal?: AbortSignal): Promise<Evaluation[]> {
  const value = await requestJson(`${getApiBaseUrl()}/api/v1/comparison/evaluation-reports`, { signal });
  if (!Array.isArray(value) || !value.every(isEvaluation)) {
    throw new Error("The AI service returned invalid saved evaluation reports.");
  }
  return value;
}

function isAvailableCheckpoint(value: unknown): value is AvailableCheckpoint {
  return record(value) && hash(value.checkpoint_hash) &&
    typeof value.model_version === "string" && typeof value.architecture === "string" &&
    typeof value.checkpoint_name === "string" && typeof value.is_loaded === "boolean";
}

function isModelActivation(value: unknown): value is ModelActivation {
  return record(value) && hash(value.checkpoint_hash) &&
    typeof value.model_version === "string" && typeof value.architecture === "string" &&
    typeof value.checkpoint_name === "string" && typeof value.activated_at === "string";
}

function isModelRegistrySnapshot(value: unknown): value is ModelRegistrySnapshot {
  return record(value) && (value.active === null || isModelActivation(value.active)) &&
    Array.isArray(value.available_checkpoints) && value.available_checkpoints.every(isAvailableCheckpoint) &&
    Array.isArray(value.history) && value.history.every(isModelActivation);
}

export async function getAvailableCheckpoints(signal?: AbortSignal): Promise<AvailableCheckpoint[]> {
  const value = await requestJson(`${getApiBaseUrl()}/api/v1/comparison/checkpoints`, { signal });
  if (!Array.isArray(value) || !value.every(isAvailableCheckpoint)) {
    throw new Error("The AI service returned invalid checkpoint information.");
  }
  return value;
}

export async function getModelRegistry(signal?: AbortSignal): Promise<ModelRegistrySnapshot> {
  const value = await requestJson(`${getApiBaseUrl()}/api/v1/model-registry`, { signal });
  if (!isModelRegistrySnapshot(value)) {
    throw new Error("The AI service returned an invalid model registry.");
  }
  return value;
}

export async function activateModel(
  checkpointHash: string,
  signal?: AbortSignal,
): Promise<ModelRegistrySnapshot> {
  if (!hash(checkpointHash)) throw new Error("Choose an approved checkpoint.");
  const value = await requestJson(`${getApiBaseUrl()}/api/v1/model-registry/activate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint_hash: checkpointHash }),
    signal,
  });
  if (!isModelRegistrySnapshot(value) || value.active?.checkpoint_hash !== checkpointHash) {
    throw new Error("The AI service could not activate the selected checkpoint.");
  }
  return value;
}

export async function rollbackModel(signal?: AbortSignal): Promise<ModelRegistrySnapshot> {
  const value = await requestJson(`${getApiBaseUrl()}/api/v1/model-registry/rollback`, {
    method: "POST",
    signal,
  });
  if (!isModelRegistrySnapshot(value) || !value.active) {
    throw new Error("The AI service could not roll back to the previous model.");
  }
  return value;
}

export async function evaluateCheckpoint(checkpointHash: string, signal?: AbortSignal): Promise<Evaluation> {
  if (!hash(checkpointHash)) throw new Error("Choose an approved checkpoint.");
  const value = await requestJson(`${getApiBaseUrl()}/api/v1/comparison/evaluation-reports`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint_hash: checkpointHash }),
    signal,
  });
  if (!isEvaluation(value) || value.checkpoint_sha256 !== checkpointHash) {
    throw new Error("The AI service returned an invalid checkpoint evaluation.");
  }
  return value;
}

export async function compareModels(
  modelA: Evaluation,
  modelB: Evaluation,
  signal?: AbortSignal,
): Promise<ModelComparison> {
  const value = await requestJson(`${getApiBaseUrl()}/api/v1/comparison/reports`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_a_checkpoint_hash: modelA.checkpoint_sha256,
      model_b_checkpoint_hash: modelB.checkpoint_sha256,
      model_a_name: modelA.model_version,
      model_b_name: modelB.model_version,
    }),
    signal,
  });
  if (!isModelComparison(value)) {
    throw new Error("The AI service returned an invalid model comparison.");
  }
  return value;
}

export async function getComparisonHistory(signal?: AbortSignal): Promise<ModelComparison[]> {
  const value = await requestJson(`${getApiBaseUrl()}/api/v1/comparison/reports`, { signal });
  if (!Array.isArray(value) || !value.every(isModelComparison)) {
    throw new Error("The AI service returned invalid comparison history.");
  }
  return value;
}

export async function getErrorImage(index: number, checkpoint: string, signal?: AbortSignal): Promise<number[][]> {
  const value = await requestJson(`${getApiBaseUrl()}/api/v1/analytics/misclassifications/${index}`, { signal });
  if (!record(value) || value.sample_index !== index || value.checkpoint_sha256 !== checkpoint ||
      !isModelInputPixels(value.pixels)) {
    throw new Error("The AI service returned an invalid or mismatched example image.");
  }
  return value.pixels;
}

function isSensitivity(value: unknown, target: number | null): value is Sensitivity {
  if (!record(value) || typeof value.model_version !== "string" ||
      !digit(value.predicted_class) || !digit(value.target_class) ||
      (target !== null && value.target_class !== target) ||
      !vector(value.probabilities, 10, probability) ||
      Math.abs(value.probabilities.reduce((sum, item) => sum + item, 0) - 1) > 0.01 ||
      !Array.isArray(value.gradient) || value.gradient.length !== 28 ||
      !value.gradient.every((row) => vector(row, 28, finite))) {
    return false;
  }
  return true;
}

export async function getSensitivity(pixels: number[][], target: number | null, signal?: AbortSignal): Promise<Sensitivity> {
  const value = await requestJson(`${getApiBaseUrl()}/api/v1/saliency`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pixels, target_class: target }), signal,
  });
  if (!isSensitivity(value, target)) throw new Error("The AI service returned invalid sensitivity data.");
  return value;
}
