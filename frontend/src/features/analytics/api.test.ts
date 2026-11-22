import { afterEach, expect, it, vi } from "vitest";
import {
  activateModel,
  evaluateCheckpoint,
  getModelRegistry,
  getAvailableCheckpoints,
  getErrorImage,
  getEvaluation,
  rollbackModel,
  getSensitivity,
} from "./api";

const pixels = Array.from({ length: 28 }, () => Array<number>(28).fill(0));
function respond(value: unknown) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(value), {
    headers: { "Content-Type": "application/json" },
  })));
}

afterEach(() => vi.unstubAllGlobals());

it("rejects malformed reports and mismatched example images", async () => {
  respond({ accuracy: 1 });
  await expect(getEvaluation()).rejects.toThrow("invalid evaluation report");
  respond({ sample_index: 2, checkpoint_sha256: "b".repeat(64), pixels });
  await expect(getErrorImage(2, "a".repeat(64))).rejects.toThrow("mismatched example");
});

it("rejects sensitivity for a different target or malformed gradient", async () => {
  const result = {
    model_version: "test", predicted_class: 1, target_class: 1,
    probabilities: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], gradient: pixels,
  };
  respond(result);
  await expect(getSensitivity(pixels, 2)).rejects.toThrow("invalid sensitivity");
  respond({ ...result, target_class: 2, gradient: [[0]] });
  await expect(getSensitivity(pixels, 2)).rejects.toThrow("invalid sensitivity");
  respond({ ...result, target_class: 2 });
  await expect(getSensitivity(pixels, 2)).resolves.toMatchObject({ target_class: 2 });
});

it("lists approved checkpoints and requests server-side evaluation by hash", async () => {
  const checkpoint = {
    checkpoint_hash: "a".repeat(64), model_version: "test-model", architecture: "mlp",
    checkpoint_name: "best.pt", is_loaded: true,
  };
  respond([checkpoint]);
  await expect(getAvailableCheckpoints()).resolves.toEqual([checkpoint]);

  const report = {
    schema_version: 1, model_version: "test-model", checkpoint_sha256: checkpoint.checkpoint_hash,
    dataset: "MNIST", dataset_split: "official-test", generated_at: "2026-01-01",
    dataset_images_sha256: "b".repeat(64), dataset_labels_sha256: "c".repeat(64),
    accuracy: 1, loss: 0, num_samples: 10,
    confusion_matrix: Array.from({ length: 10 }, (_, row) =>
      Array.from({ length: 10 }, (_, column) => row === column ? 1 : 0)),
    precision_per_class: Array<number>(10).fill(1), recall_per_class: Array<number>(10).fill(1),
    f1_per_class: Array<number>(10).fill(1), support_per_class: Array<number>(10).fill(1),
    misclassified_examples: [],
  };
  respond(report);
  await expect(evaluateCheckpoint(checkpoint.checkpoint_hash)).resolves.toMatchObject({
    checkpoint_sha256: checkpoint.checkpoint_hash,
  });
  const request = vi.mocked(fetch).mock.calls.at(-1)?.[1] as RequestInit;
  expect(JSON.parse(String(request.body))).toEqual({ checkpoint_hash: checkpoint.checkpoint_hash });
});

it("loads model activation history and sends activation and rollback requests", async () => {
  const active = {
    checkpoint_hash: "a".repeat(64), model_version: "v1", architecture: "mlp",
    checkpoint_name: "best.pt", activated_at: "2026-01-01T00:00:00Z",
  };
  const candidate = {
    checkpoint_hash: "d".repeat(64), model_version: "v2", architecture: "cnn",
    checkpoint_name: "models/cnn.pt", is_loaded: false,
  };
  const snapshot = {
    active,
    available_checkpoints: [{ ...active, is_loaded: true }, candidate],
    history: [active],
  };
  respond(snapshot);
  await expect(getModelRegistry()).resolves.toEqual(snapshot);

  const activated = { ...snapshot, active: { ...candidate, activated_at: "2026-01-02T00:00:00Z" } };
  respond(activated);
  await expect(activateModel(candidate.checkpoint_hash)).resolves.toMatchObject({
    active: { checkpoint_hash: candidate.checkpoint_hash },
  });
  let request = vi.mocked(fetch).mock.calls.at(-1)?.[1] as RequestInit;
  expect(JSON.parse(String(request.body))).toEqual({ checkpoint_hash: candidate.checkpoint_hash });

  respond(snapshot);
  await expect(rollbackModel()).resolves.toMatchObject({ active: { checkpoint_hash: active.checkpoint_hash } });
  request = vi.mocked(fetch).mock.calls.at(-1)?.[1] as RequestInit;
  expect(request.method).toBe("POST");
});
