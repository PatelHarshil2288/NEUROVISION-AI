import { getApiBaseUrl, requestJson } from "../digit-studio/lib/api";

export interface EpochMetrics {
  epoch: number;
  train_loss: number;
  validation_loss: number;
  validation_accuracy: number;
}

export interface TrainingJob {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  epochs: number;
  epoch: number;
  batch: number;
  batches: number;
  train_loss: number | null;
  validation_loss: number | null;
  validation_accuracy: number | null;
  optimizer: string;
  learning_rate: number;
  train_samples: number;
  validation_samples: number;
  gradient_norm: number | null;
  sample_gradient: number | null;
  sample_before: number | null;
  sample_after: number | null;
  history: EpochMetrics[];
  checkpoint: { latest_epoch: number; best_epoch: number; directory: string } | null;
  error: string | null;
}

export interface TrainingStatus {
  enabled: boolean;
  remaining_jobs: number;
  job: TrainingJob | null;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function nullableNumber(value: unknown): boolean {
  return value === null || finite(value);
}

function isJob(value: unknown): value is TrainingJob {
  if (!record(value) || typeof value.id !== "string" ||
      !["queued", "running", "completed", "failed"].includes(String(value.status)) ||
      typeof value.optimizer !== "string" ||
      !["epochs", "epoch", "batch", "batches", "learning_rate", "train_samples", "validation_samples"].every((key) => finite(value[key])) ||
      !["train_loss", "validation_loss", "validation_accuracy", "gradient_norm", "sample_gradient", "sample_before", "sample_after"].every((key) => nullableNumber(value[key])) ||
      !(value.error === null || typeof value.error === "string") ||
      !Array.isArray(value.history) ||
      !value.history.every((item: unknown) => record(item) && ["epoch", "train_loss", "validation_loss", "validation_accuracy"].every((key) => finite(item[key])))) {
    return false;
  }
  return value.checkpoint === null ||
    (record(value.checkpoint) && finite(value.checkpoint.latest_epoch) &&
      finite(value.checkpoint.best_epoch) && typeof value.checkpoint.directory === "string");
}

export async function getTrainingStatus(signal?: AbortSignal): Promise<TrainingStatus> {
  const result = await requestJson(`${getApiBaseUrl()}/api/v1/training`, { signal });
  if (!record(result) || typeof result.enabled !== "boolean" ||
      !Number.isInteger(result.remaining_jobs) || !finite(result.remaining_jobs) ||
      !(result.job === null || isJob(result.job))) {
    throw new Error("The AI service returned invalid training status.");
  }
  return { enabled: result.enabled, remaining_jobs: result.remaining_jobs, job: result.job };
}

export async function startTraining(epochs: number): Promise<void> {
  const result = await requestJson(`${getApiBaseUrl()}/api/v1/training`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ epochs }),
  });
  if (!isJob(result)) {
    throw new Error("The AI service returned an invalid training job.");
  }
}
