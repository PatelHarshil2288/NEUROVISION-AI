import { describe, expect, it } from "vitest";

import type {
  ActivationsResult,
  ModelWeightsResult,
} from "../../digit-studio/lib/api";
import {
  getConnectionContributions,
  getLayerDisplayIntensities,
  getNeuronDetails,
  getTopConnectionContributions,
} from "./signal";

function createActivations(): ActivationsResult {
  return {
    model_id: "mnist-mlp-v1",
    model_version: "mlp-mnist-v1",
    predicted_class: 0,
    logits: Array<number>(10).fill(0),
    probabilities: Array<number>(10).fill(0.1),
    layers: {
      input: {
        pre_activations: [],
        activations: Array<number>(784).fill(0),
      },
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

function createWeights(): ModelWeightsResult {
  const makeLayer = (inputCount: number, outputCount: number) => ({
    weights: Array.from({ length: outputCount }, () =>
      Array<number>(inputCount).fill(0),
    ),
    biases: Array<number>(outputCount).fill(0),
  });
  return {
    model_id: "mnist-mlp-v1",
    model_version: "mlp-mnist-v1",
    architecture: [784, 128, 64, 10],
    layers: {
      input_to_hidden_1: makeLayer(784, 128),
      hidden_1_to_hidden_2: makeLayer(128, 64),
      hidden_2_to_output: makeLayer(64, 10),
    },
  };
}

describe("neural signal data", () => {
  it("uses raw input intensities and normalizes hidden activations per layer", () => {
    const activations = createActivations();
    activations.layers.input.activations[0] =
      (0.25 - 0.1307) / 0.3081;
    activations.layers.hidden_1.activations[0] = 2;
    activations.layers.hidden_1.activations[1] = 1;

    expect(getLayerDisplayIntensities(activations, "input")[0]).toBeCloseTo(0.25);
    expect(getLayerDisplayIntensities(activations, "hidden_1").slice(0, 3)).toEqual([
      1, 0.5, 0,
    ]);
  });

  it("computes signed, sampled source-activation contributions with target bias", () => {
    const activations = createActivations();
    const weights = createWeights();
    activations.layers.input.activations[0] = 0.5;
    weights.layers.input_to_hidden_1.weights[0][0] = -2.4;
    weights.layers.input_to_hidden_1.biases[0] = 0.1;

    const contributions = getConnectionContributions(
      activations,
      weights,
      0,
    );
    const first = contributions.find(
      (contribution) =>
        contribution.sourceIndex === 0 && contribution.targetIndex === 0,
    );

    expect(first).toMatchObject({
      weight: -2.4,
      contribution: -1.2,
      targetBias: 0.1,
    });
  });

  it("ranks contributions by absolute value without mutating their source list", () => {
    const contributions = getConnectionContributions(
      createActivations(),
      createWeights(),
      0,
    );
    contributions[0]!.contribution = 0.2;
    contributions[1]!.contribution = -0.8;
    const originalFirst = contributions[0];

    const top = getTopConnectionContributions(contributions, 1);

    expect(top).toHaveLength(1);
    expect(top[0]).toBe(contributions[1]);
    expect(contributions[0]).toBe(originalFirst);
  });

  it("uses every incoming weight and the real pre-activation, not the rendered sample", () => {
    const activations = createActivations();
    const weights = createWeights();
    activations.layers.input.activations[783] = 2;
    activations.layers.hidden_1.pre_activations[0] = -2.5;
    weights.layers.input_to_hidden_1.weights[0][783] = -1.5;
    weights.layers.input_to_hidden_1.biases[0] = 0.5;

    const details = getNeuronDetails({ layerId: "hidden_1", index: 0 }, activations, weights);
    expect(details.incoming).toHaveLength(784);
    expect(details.incoming[783]).toMatchObject({ weight: -1.5, contribution: -3 });
    expect(details.bias).toBe(0.5);
    expect(details.preActivation).toBe(-2.5);
  });

  it("reports direct hidden-to-output logit contributions and validates selections", () => {
    const activations = createActivations();
    const weights = createWeights();
    activations.layers.hidden_2.activations[4] = 2;
    weights.layers.hidden_2_to_output.weights[7][4] = -0.75;
    const details = getNeuronDetails({ layerId: "hidden_2", index: 4 }, activations, weights);
    expect(details.incoming).toHaveLength(128);
    expect(details.outgoing[7]).toEqual({ digit: 7, weight: -0.75, contribution: -1.5 });
    expect(() => getNeuronDetails({ layerId: "output", index: 10 }, activations, weights)).toThrow();
    expect(getNeuronDetails({ layerId: "input", index: 783 }, activations, weights).bias).toBeNull();
    expect(getNeuronDetails({ layerId: "output", index: 0 }, activations, weights).incoming).toHaveLength(64);
  });
});
