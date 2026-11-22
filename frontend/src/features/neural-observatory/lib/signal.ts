import type {
  ActivationsResult,
  ModelWeightsResult,
} from "../../digit-studio/lib/api";
import {
  NETWORK_LAYERS,
  NETWORK_TOPOLOGY,
  type LayerId,
  type NetworkConnection,
} from "./network";

const WEIGHT_LAYER_KEYS = [
  "input_to_hidden_1",
  "hidden_1_to_hidden_2",
  "hidden_2_to_output",
] as const;
const MNIST_INPUT_MEAN = 0.1307;
const MNIST_INPUT_STANDARD_DEVIATION = 0.3081;

export type TransitionIndex = 0 | 1 | 2;

export interface ConnectionContribution extends NetworkConnection {
  sourceActivation: number;
  weight: number;
  contribution: number;
  targetBias: number;
}

export interface NeuronSelection {
  layerId: LayerId;
  index: number;
}

export function getNeuronDetails(
  selection: NeuronSelection,
  activations: ActivationsResult,
  weights: ModelWeightsResult,
) {
  const layerIndex = NETWORK_LAYERS.findIndex((layer) => layer.id === selection.layerId);
  const layer = NETWORK_LAYERS[layerIndex];
  if (!layer || !Number.isInteger(selection.index) || selection.index < 0 || selection.index >= layer.neuronCount) {
    throw new Error("The selected neuron is outside the network.");
  }
  const activation = activations.layers[layer.id].activations[selection.index];
  if (activation === undefined) {
    throw new Error("The selected neuron has no activation value.");
  }
  if (layerIndex === 0) {
    return { layer, activation, preActivation: null, bias: null, incoming: [], outgoing: [] };
  }
  const sourceLayer = NETWORK_LAYERS[layerIndex - 1];
  const weightLayer = weights.layers[WEIGHT_LAYER_KEYS[layerIndex - 1]];
  const row = weightLayer.weights[selection.index];
  const bias = weightLayer.biases[selection.index];
  const preActivation = activations.layers[layer.id].pre_activations[selection.index];
  if (!row || bias === undefined || preActivation === undefined || row.length !== sourceLayer.neuronCount) {
    throw new Error("The selected neuron data does not match the network.");
  }
  const incoming = row.map((weight, sourceIndex) => {
    const sourceActivation = activations.layers[sourceLayer.id].activations[sourceIndex];
    if (sourceActivation === undefined) {
      throw new Error("A source activation is missing.");
    }
    return { sourceIndex, sourceActivation, weight, contribution: sourceActivation * weight };
  });
  const outgoing = layer.id === "hidden_2"
    ? weights.layers.hidden_2_to_output.weights.map((outputWeights, digit) => {
        const weight = outputWeights[selection.index];
        if (weight === undefined) {
          throw new Error("An output weight is missing.");
        }
        return { digit, weight, contribution: activation * weight };
      })
    : [];
  return { layer, activation, preActivation, bias, incoming, outgoing };
}

export function getLayerDisplayIntensities(
  activations: ActivationsResult,
  layerId: LayerId,
): number[] {
  const values = activations.layers[layerId].activations;
  if (layerId === "input") {
    return values.map((value) =>
      Math.max(
        0,
        Math.min(
          1,
          value * MNIST_INPUT_STANDARD_DEVIATION + MNIST_INPUT_MEAN,
        ),
      ),
    );
  }
  if (layerId === "output") {
    return values;
  }

  const maximum = Math.max(0, ...values);
  if (maximum === 0) {
    return values.map(() => 0);
  }
  return values.map((value) => Math.max(0, Math.min(1, value / maximum)));
}

export function getConnectionContributions(
  activations: ActivationsResult,
  weights: ModelWeightsResult,
  transition: TransitionIndex,
): ConnectionContribution[] {
  const sourceLayer = NETWORK_LAYERS[transition];
  const weightLayerKey = WEIGHT_LAYER_KEYS[transition];
  const sourceActivations = activations.layers[sourceLayer.id].activations;
  const layerWeights = weights.layers[weightLayerKey];

  return NETWORK_TOPOLOGY.connections
    .filter((connection) => connection.transition === transition)
    .map((connection) => {
      const sourceActivation = sourceActivations[connection.sourceIndex];
      const weight = layerWeights.weights[connection.targetIndex]?.[
        connection.sourceIndex
      ];
      const targetBias = layerWeights.biases[connection.targetIndex];

      if (
        sourceActivation === undefined ||
        weight === undefined ||
        targetBias === undefined
      ) {
        throw new Error("The activation and weight tensors do not match the network.");
      }

      return {
        ...connection,
        sourceActivation,
        weight,
        contribution: sourceActivation * weight,
        targetBias,
      };
    });
}

export function getTopConnectionContributions(
  contributions: ConnectionContribution[],
  limit = 5,
): ConnectionContribution[] {
  return [...contributions]
    .sort(
      (left, right) =>
        Math.abs(right.contribution) - Math.abs(left.contribution),
    )
    .slice(0, limit);
}
