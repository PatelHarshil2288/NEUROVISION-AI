export interface BackendStatus {
  state: "ready" | "missing";
  modelVersion: string | null;
}

export interface PredictionResult {
  model_id: string;
  model_version: string;
  predicted_class: number;
  logits: number[];
  probabilities: number[];
  inference_ms: number;
}

export interface LayerActivationData {
  pre_activations: number[];
  activations: number[];
}

export interface ActivationsResult {
  model_id: string;
  model_version: string;
  predicted_class: number;
  logits: number[];
  probabilities: number[];
  layers: {
    input: LayerActivationData;
    hidden_1: LayerActivationData;
    hidden_2: LayerActivationData;
    output: LayerActivationData;
  };
}

export interface LayerWeightsData {
  weights: number[][];
  biases: number[];
}

export interface ModelWeightsResult {
  model_id: string;
  model_version: string;
  architecture: [784, 128, 64, 10];
  layers: {
    input_to_hidden_1: LayerWeightsData;
    hidden_1_to_hidden_2: LayerWeightsData;
    hidden_2_to_output: LayerWeightsData;
  };
}

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export function getApiBaseUrl(): string {
  return (
    process.env.NEXT_PUBLIC_API_BASE_URL?.trim().replace(/\/+$/, "") ||
    DEFAULT_API_BASE_URL
  );
}

export async function getBackendStatus(
  signal?: AbortSignal,
): Promise<BackendStatus> {
  const baseUrl = getApiBaseUrl();
  const [health, modelList] = await Promise.all([
    requestJson(`${baseUrl}/api/v1/health`, { signal }),
    requestJson(`${baseUrl}/api/v1/models`, { signal }),
  ]);
  if (!isRecord(health) || !isRecord(modelList)) {
    throw new Error("The AI service returned an invalid status response.");
  }

  const model = Array.isArray(modelList.models)
    ? modelList.models.find(
        (candidate) =>
          isRecord(candidate) && candidate.model_id === "mnist-mlp-v1",
      )
    : undefined;
  if (
    typeof health.model_ready !== "boolean" ||
    !isRecord(model) ||
    typeof model.available !== "boolean"
  ) {
    throw new Error("The AI service returned an invalid model status.");
  }

  return {
    state: health.model_ready && model.available ? "ready" : "missing",
    modelVersion:
      typeof model.model_version === "string" ? model.model_version : null,
  };
}

export async function predictDigit(
  pixels: number[][],
  signal?: AbortSignal,
): Promise<PredictionResult> {
  const payload = await requestJson(`${getApiBaseUrl()}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pixels }),
    signal,
  });
  if (!isPredictionResult(payload)) {
    throw new Error("The AI service returned an invalid prediction.");
  }
  return payload;
}

export async function getActivations(
  pixels: number[][],
  signal?: AbortSignal,
): Promise<ActivationsResult> {
  const payload = await requestJson(`${getApiBaseUrl()}/api/v1/activations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pixels }),
    signal,
  });
  if (!isActivationsResult(payload)) {
    throw new Error("The AI service returned invalid activation data.");
  }
  return payload;
}

export async function getModelWeights(
  signal?: AbortSignal,
): Promise<ModelWeightsResult> {
  const payload = await requestJson(`${getApiBaseUrl()}/api/v1/weights`, {
    signal,
  });
  if (!isModelWeightsResult(payload)) {
    throw new Error("The AI service returned invalid model weights.");
  }
  return payload;
}

export async function requestJson(
  url: string,
  init?: RequestInit,
): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error(
      "Cannot reach the AI service. Start the backend and check its URL.",
    );
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error(
      response.ok
        ? "The AI service returned an unreadable response."
        : `The AI service request failed (${response.status}).`,
    );
  }

  if (!response.ok) {
    throw new Error(readErrorMessage(payload, response.status));
  }
  return payload;
}

function readErrorMessage(payload: unknown, status: number): string {
  if (isRecord(payload) && typeof payload.detail === "string") {
    return payload.detail;
  }
  if (isRecord(payload) && Array.isArray(payload.detail)) {
    const messages = payload.detail
      .filter(
        (item): item is Record<string, unknown> & { msg: string } =>
          isRecord(item) && typeof item.msg === "string",
      )
      .map((item) => item.msg);
    if (messages.length > 0) {
      return messages.join(" ");
    }
  }
  return `The AI service request failed (${status}).`;
}

function isPredictionResult(value: unknown): value is PredictionResult {
  if (!isRecord(value)) {
    return false;
  }
  const probabilities = value.probabilities;
  const logits = value.logits;
  if (
    typeof value.model_id !== "string" ||
    typeof value.model_version !== "string" ||
    typeof value.predicted_class !== "number" ||
    !Number.isInteger(value.predicted_class) ||
    value.predicted_class < 0 ||
    value.predicted_class > 9 ||
    !Array.isArray(probabilities) ||
    !Array.isArray(logits) ||
    probabilities.length !== 10 ||
    logits.length !== 10 ||
    !probabilities.every(
      (probability) =>
        typeof probability === "number" &&
        Number.isFinite(probability) &&
        probability >= 0 &&
        probability <= 1,
    ) ||
    !logits.every(
      (logit) => typeof logit === "number" && Number.isFinite(logit),
    ) ||
    typeof value.inference_ms !== "number" ||
    !Number.isFinite(value.inference_ms) ||
    value.inference_ms < 0
  ) {
    return false;
  }
  const probabilitySum = probabilities.reduce<number>(
    (total, probability) => total + probability,
    0,
  );
  return Math.abs(probabilitySum - 1) < 0.01;
}

function isActivationsResult(value: unknown): value is ActivationsResult {
  if (
    !isRecord(value) ||
    value.model_id !== "mnist-mlp-v1" ||
    typeof value.model_version !== "string" ||
    !isPredictedClass(value.predicted_class) ||
    !isFiniteNumberArray(value.logits, 10) ||
    !isProbabilities(value.probabilities) ||
    !isRecord(value.layers)
  ) {
    return false;
  }
  const { input, hidden_1, hidden_2, output } = value.layers;
  return (
    isLayerActivation(input, 0, 784) &&
    isLayerActivation(hidden_1, 128, 128) &&
    isLayerActivation(hidden_2, 64, 64) &&
    isLayerActivation(output, 10, 10)
  );
}

function isModelWeightsResult(value: unknown): value is ModelWeightsResult {
  if (
    !isRecord(value) ||
    value.model_id !== "mnist-mlp-v1" ||
    typeof value.model_version !== "string" ||
    !Array.isArray(value.architecture) ||
    value.architecture.length !== 4 ||
    value.architecture[0] !== 784 ||
    value.architecture[1] !== 128 ||
    value.architecture[2] !== 64 ||
    value.architecture[3] !== 10 ||
    !isRecord(value.layers)
  ) {
    return false;
  }
  return (
    isLayerWeights(value.layers.input_to_hidden_1, 128, 784) &&
    isLayerWeights(value.layers.hidden_1_to_hidden_2, 64, 128) &&
    isLayerWeights(value.layers.hidden_2_to_output, 10, 64)
  );
}

function isLayerActivation(
  value: unknown,
  preActivationCount: number,
  activationCount: number,
): value is LayerActivationData {
  return (
    isRecord(value) &&
    isFiniteNumberArray(value.pre_activations, preActivationCount) &&
    isFiniteNumberArray(value.activations, activationCount)
  );
}

function isLayerWeights(
  value: unknown,
  outputCount: number,
  inputCount: number,
): value is LayerWeightsData {
  return (
    isRecord(value) &&
    Array.isArray(value.weights) &&
    value.weights.length === outputCount &&
    value.weights.every((row) => isFiniteNumberArray(row, inputCount)) &&
    isFiniteNumberArray(value.biases, outputCount)
  );
}

function isPredictedClass(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= 9
  );
}

function isProbabilities(value: unknown): value is number[] {
  if (
    !isFiniteNumberArray(value, 10) ||
    value.some((probability) => probability < 0 || probability > 1)
  ) {
    return false;
  }
  const total = value.reduce((sum, probability) => sum + probability, 0);
  return Math.abs(total - 1) < 0.01;
}

function isFiniteNumberArray(value: unknown, length: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === length &&
    value.every((item) => typeof item === "number" && Number.isFinite(item))
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
