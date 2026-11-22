import { afterEach, describe, expect, it, vi } from "vitest";

import {
  getActivations,
  getBackendStatus,
  getModelWeights,
  predictDigit,
} from "./api";

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const readyHealth = { status: "ok", model_ready: true };
const readyModels = {
  models: [
    {
      model_id: "mnist-mlp-v1",
      model_version: "mlp-mnist-v1",
      available: true,
    },
  ],
};

function validActivations() {
  return {
    model_id: "mnist-mlp-v1",
    model_version: "mlp-mnist-v1",
    predicted_class: 7,
    logits: Array<number>(10).fill(0),
    probabilities: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    layers: {
      input: { pre_activations: [], activations: Array<number>(784).fill(0) },
      hidden_1: {
        pre_activations: Array<number>(128).fill(0),
        activations: Array<number>(128).fill(0),
      },
      hidden_2: {
        pre_activations: Array<number>(64).fill(0),
        activations: Array<number>(64).fill(0),
      },
      output: {
        pre_activations: Array<number>(10).fill(0),
        activations: Array<number>(10).fill(0.1),
      },
    },
  };
}

function layerWeights(inputCount: number, outputCount: number) {
  return {
    weights: Array.from({ length: outputCount }, () =>
      Array<number>(inputCount).fill(0.01),
    ),
    biases: Array<number>(outputCount).fill(0),
  };
}

function validModelWeights() {
  return {
    model_id: "mnist-mlp-v1",
    model_version: "mlp-mnist-v1",
    architecture: [784, 128, 64, 10],
    layers: {
      input_to_hidden_1: layerWeights(784, 128),
      hidden_1_to_hidden_2: layerWeights(128, 64),
      hidden_2_to_output: layerWeights(64, 10),
    },
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("AI API client", () => {
  it("reports the loaded model version from the backend", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(readyHealth))
      .mockResolvedValueOnce(jsonResponse(readyModels));
    vi.stubGlobal("fetch", fetchMock);

    await expect(getBackendStatus()).resolves.toEqual({
      state: "ready",
      modelVersion: "mlp-mnist-v1",
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("posts the 28 by 28 image and validates a real prediction response", async () => {
    const probabilities = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.65, 0.04, 0.03];
    const result = {
      model_id: "mnist-mlp-v1",
      model_version: "mlp-mnist-v1",
      predicted_class: 7,
      logits: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      probabilities,
      inference_ms: 1.4,
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(result));
    vi.stubGlobal("fetch", fetchMock);
    const pixels = Array.from({ length: 28 }, () => Array<number>(28).fill(0));

    await expect(predictDigit(pixels)).resolves.toEqual(result);
    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(request.body as string)).toEqual({ pixels });
    expect(request.method).toBe("POST");
  });

  it("validates real activations and versioned model weights", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(validActivations()))
      .mockResolvedValueOnce(jsonResponse(validModelWeights()));
    vi.stubGlobal("fetch", fetchMock);
    const pixels = Array.from({ length: 28 }, () => Array<number>(28).fill(0));

    const activations = await getActivations(pixels);
    const weights = await getModelWeights();

    expect(activations.layers.hidden_1.activations).toHaveLength(128);
    expect(weights.layers.input_to_hidden_1.weights[0]).toHaveLength(784);
    expect(fetchMock.mock.calls[0][0]).toContain("/api/v1/activations");
    expect(fetchMock.mock.calls[1][0]).toContain("/api/v1/weights");
  });

  it("rejects activation or weight payloads with invalid tensor shapes", async () => {
    const invalidActivations = validActivations();
    invalidActivations.layers.hidden_1.activations.pop();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      jsonResponse(invalidActivations),
    ));
    await expect(getActivations([])).rejects.toThrow("invalid activation data");

    const invalidWeights = validModelWeights();
    invalidWeights.layers.input_to_hidden_1.weights.pop();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      jsonResponse(invalidWeights),
    ));
    await expect(getModelWeights()).rejects.toThrow("invalid model weights");
  });

  it("surfaces API errors and network failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: "No trained MNIST checkpoint is available." }, 503),
      ),
    );
    await expect(predictDigit([])).rejects.toThrow(
      "No trained MNIST checkpoint is available.",
    );

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    await expect(getBackendStatus()).rejects.toThrow(
      "Cannot reach the AI service.",
    );
  });
});
